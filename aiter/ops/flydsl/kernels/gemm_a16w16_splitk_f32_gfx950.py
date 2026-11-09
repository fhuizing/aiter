# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K A16W16 GEMM with FP32 output for gfx950 (the FlyDSL HGEMM ``sr`` policy).

    C[m, n] = sum_k A[m, k] * B[n, k],  bf16/fp16 in, fp32 out

A (M, K) and B (N, K) are row-major; C is a row-major FP32 (M, N). Each
workgroup computes one ``block_m`` x ``block_n`` tile over one of ``split_k``
slices of K. The main loop streams A and B into a ``stages``-deep LDS ring with
global->LDS DMA (``buffer_load_dwordx4 ... lds``, XOR-swizzled 16-byte chunks)
and multiplies with 16x16x32 MFMAs.

The K slices are reduced in the kernel, without FP32 atomics (those resolve past
the L2 on gfx950 at about one lane per channel per clock):

* Workgroups are dealt round-robin over the XCDs. The block id is decoded so
  that every split of a tile lands on one XCD, and an XCD owns an
  ``XM x (N_BLOCKS / XN)`` group of tiles, so the splits and their neighbours
  share A and B in one L2.
* Each split stores its FP32 partial tile in accumulator-fragment order (one
  16-byte store per lane per fragment) to a workspace, then advances its own
  flag. Once all ``split_k`` flags of the tile agree, split s sums slice s of
  the tile over all splits and writes it to C. Every launch advances all flags
  of a tile by one, so the flag buffer needs no reset pass; it must be private
  to one compiled kernel and stream.

Rows of A at or above m read as zero and rows of C at or above m are dropped by
the buffer-descriptor bounds, so one compiled kernel serves every m.

Low-level boundaries kept deliberately, as in ``gemm_a16w16_skinny_gfx950``:
  * the global->LDS DMA is inline asm. ``fx.rocdl.raw_ptr_buffer_load_lds``
    emits the same instruction but tells the compiler that it writes LDS, and
    the compiler then drains ``vmcnt`` in front of every ``ds_read`` of the
    ring. The asm form is opaque on purpose; the hand-counted
    ``s_waitcnt vmcnt(N)`` before each barrier orders the ring instead;
  * the MFMAs call ``rocdl.mfma_*`` directly: the operands are raw fragments
    read out of the swizzled LDS ring, with no partition form for
    ``fx.make_mma_atom``/``fx.gemm`` to take;
  * workspace and C go through ``buffer_ops`` with explicit descriptors: the
    workspace needs the ``sc1`` cache policy on both sides of the cross-XCD
    handoff, and C needs ``num_records`` at its real extent to drop rows past m.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from . import buffer_ops
from .kernels_common import create_llvm_ptr

WAVE_SIZE = 64
ELEM_BYTES = 2
DMA_BYTES = 16
DMA_VEC = DMA_BYTES // ELEM_BYTES
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
ACC_VALUES = 4
# XCDs split the tile grid into XM row groups of XN column groups each.
XCD_N_GROUPS = 4
# gfx950 buffer cache policy (LLVM CPol) SC1: system-coherent past the XCD L2.
CPOL_SC1 = 16
# Outer extent of layouts whose outermost mode never wraps: idx2crd lowers a
# bounded outer mode to a redundant remainder (aiter-flydsl-gotchas).
UNBOUNDED = 2**31 - 1


def xcd_tile_groups(num_xcds: int) -> tuple[int, int]:
    """(XM, XN): the tile rows and tile-column groups the XCDs are arranged in."""
    xn = XCD_N_GROUPS if num_xcds % XCD_N_GROUPS == 0 else 1
    return num_xcds // xn, xn


def splitk_f32_tiles(m: int, n: int, block_m: int, block_n: int, num_xcds: int) -> int:
    """Tiles of one launch, M blocks padded to a multiple of XM."""
    xm, _ = xcd_tile_groups(num_xcds)
    m_blocks = -(-m // block_m)
    return -(-m_blocks // xm) * xm * (n // block_n)


@functools.lru_cache(maxsize=1024)
def splitk_f32_config_error(
    n, k, block_m, block_n, block_k, stages, split_k, m_waves, n_waves, num_xcds
):
    """Return why (N, K, config) cannot run on this kernel, or None. Any M runs."""
    threads = m_waves * n_waves * WAVE_SIZE
    chunks_per_row = block_k // DMA_VEC
    _, xn = xcd_tile_groups(num_xcds)
    if block_k < MFMA_K or block_k & (block_k - 1):
        return f"block_k={block_k} must be a power of two >= {MFMA_K}"
    if block_m % (m_waves * MFMA_M) or block_n % (n_waves * MFMA_N):
        return (
            f"block_m={block_m} and block_n={block_n} must be multiples of "
            f"m_waves*{MFMA_M} and n_waves*{MFMA_N}"
        )
    if block_m % chunks_per_row or block_n % chunks_per_row:
        return (
            f"block_m and block_n must be multiples of {chunks_per_row}, the "
            "period of the LDS swizzle"
        )
    if (block_m * block_k) % (DMA_VEC * threads) or (block_n * block_k) % (
        DMA_VEC * threads
    ):
        return "each tile must take a whole number of DMA instructions per thread"
    if n % block_n or (n // block_n) % xn:
        return f"N={n} must be a multiple of block_n * {xn}"
    if split_k < 2 or k % split_k or (k // split_k) % block_k:
        return f"K={k} must split into {split_k} >= 2 multiples of block_k"
    if stages < 2 or (k // split_k) // block_k < stages:
        return f"stages={stages} must be in [2, K tiles per split]"
    dma_insts = (block_m + block_n) * block_k // DMA_VEC // threads
    if (stages - 2) * dma_insts >= 63:
        return f"stages={stages} needs more DMAs in flight than vmcnt can count"
    if (block_m * block_n // ACC_VALUES) % split_k:
        return "the tile's accumulator vectors must split evenly over split_k"
    lds_bytes = stages * (block_m + block_n) * block_k * ELEM_BYTES
    if lds_bytes > get_lds_capacity_bytes("gfx950"):
        return f"needs {lds_bytes} bytes of LDS"
    return None


@functools.lru_cache(maxsize=256)
def compile_gemm_a16w16_splitk_f32_gfx950(
    dtype: str,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    num_xcds: int,
):
    err = splitk_f32_config_error(
        n, k, block_m, block_n, block_k, stages, split_k, m_waves, n_waves, num_xcds
    )
    if err is not None:
        raise ValueError(f"split-K FP32 HGEMM cannot run N={n}, K={k}: {err}")
    if dtype not in ("bf16", "fp16"):
        raise ValueError(f"split-K FP32 HGEMM supports bf16/fp16, got {dtype}")
    elem_ty = fx.BFloat16 if dtype == "bf16" else fx.Float16
    mfma_op = (
        rocdl.mfma_f32_16x16x32_bf16 if dtype == "bf16" else rocdl.mfma_f32_16x16x32_f16
    )

    TM, TN, TK = block_m, block_n, block_k
    N_BLOCKS = n // TN
    XM, XN = xcd_tile_groups(num_xcds)
    NG = N_BLOCKS // XN
    KS = k // split_k
    K_LOOPS = KS // TK
    NWAVES = m_waves * n_waves
    BLOCK_THREADS = NWAVES * WAVE_SIZE
    WMS = TM // m_waves // MFMA_M
    WNS = TN // n_waves // MFMA_N
    WARP_M = WMS * MFMA_M
    WARP_N = WNS * MFMA_N
    K_STEPS = TK // MFMA_K
    NFRAG = WMS * WNS
    CPR = TK // DMA_VEC
    SWZ_BITS = CPR.bit_length() - 1
    A_DMA = TM * TK // DMA_VEC // BLOCK_THREADS
    B_DMA = TN * TK // DMA_VEC // BLOCK_THREADS
    DMA_PER_STAGE = A_DMA + B_DMA
    ITEMS = NFRAG * BLOCK_THREADS
    PER_SPLIT = ITEMS // split_k
    ROUNDS = -(-PER_SPLIT // BLOCK_THREADS)
    TILE_ELEMS = ITEMS * ACC_VALUES

    @fx.struct
    class SplitKLds:
        a: fx.Array[elem_ty, stages * TM * TK, 16]
        b: fx.Array[elem_ty, stages * TN * TK, 16]

    KERNEL_NAME = (
        f"hgemm_{dtype}_splitk_f32_t{TM}x{TN}x{TK}x{stages}_ks{split_k}"
        f"_w{m_waves}x{n_waves}x1_xcd{num_xcds}"
    )

    def coords(index, layout):
        crd = fx.idx2crd(index, layout)
        return [fx.get_scalar(fx.get_(crd, i)) for i in range(fx.rank(crd))]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def splitk_f32_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        WS: fx.Pointer,
        FLAGS: fx.Pointer,
        m: fx.Int32,
    ):
        # (stage, row, k) -> element offset in an operand's LDS ring; the
        # 16-byte chunk of k is XORed with row % CPR so the 16 rows of an MFMA
        # fragment read land in distinct banks. The write view is the DMA
        # below, the read view is read_frag.
        a_ring = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(SWZ_BITS, 3, SWZ_BITS)),
            fx.make_ordered_layout((stages, TM, TK), (2, 1, 0)),
        )
        b_ring = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(SWZ_BITS, 3, SWZ_BITS)),
            fx.make_ordered_layout((stages, TN, TK), (2, 1, 0)),
        )
        # (row, k) -> element offset of A and B in global memory
        g_layout = fx.make_layout((UNBOUNDED, k), (k, 1))
        # thread -> (lane, wave); wave -> (N wave, M wave)
        thread_layout = fx.make_layout((WAVE_SIZE, NWAVES), (1, WAVE_SIZE))
        wave_layout = fx.make_layout((n_waves, m_waves), (1, n_waves))
        # MFMA operand lane -> (row of the 16-row fragment, 8-element K group)
        lane_layout = fx.make_layout((MFMA_M, WAVE_SIZE // MFMA_M), (1, MFMA_M))
        # DMA slot of a stage (16 bytes each, in LDS order) -> (chunk, row)
        dma_slot_layout = fx.make_layout((CPR, UNBOUNDED), (1, CPR))
        # block id -> (XCD, split, tile column in the XCD group, tile row group).
        # Not mfma_preshuffle_pipeline.xcd_remap_bx_by: that remaps a 2D tile
        # grid and cannot keep the split_k workgroups of a tile on one XCD.
        bid_layout = fx.make_layout(
            (num_xcds, split_k, NG, UNBOUNDED),
            (1, num_xcds, num_xcds * split_k, num_xcds * split_k * NG),
        )
        # XCD -> (column group, row in the row group)
        xcd_layout = fx.make_layout((XN, XM), (1, XN))
        # workspace item, i.e. (lane, fragment, wave) as stored after the main
        # loop -> C element offset from the tile origin; one item holds the 4
        # accumulator rows of a 16x16 MFMA tile, n apart.
        c_item_layout = fx.make_layout(
            ((MFMA_N, WAVE_SIZE // MFMA_N), (WNS, WMS), (n_waves, m_waves)),
            ((1, ACC_VALUES * n), (MFMA_N, MFMA_M * n), (WARP_N, WARP_M * n)),
        )
        # (accumulator vector, split, tile) -> workspace element offset
        ws_layout = fx.make_layout(
            (ITEMS, split_k, UNBOUNDED),
            (ACC_VALUES, TILE_ELEMS, split_k * TILE_ELEMS),
        )

        lds = fx.SharedAllocator().allocate(SplitKLds).peek()
        tid = fx.thread_idx.x
        lane, wid = coords(tid, thread_layout)
        wn, wm = coords(wid, wave_layout)
        frag_row, frag_kg = coords(lane, lane_layout)

        xcd, ks, jn, jm = coords(fx.block_idx.x, bid_layout)
        xn, xm = coords(xcd, xcd_layout)
        block_m_idx = jm * XM + xm
        block_n_idx = xn * NG + jn
        tile_idx = block_m_idx * N_BLOCKS + block_n_idx
        m_offset = block_m_idx * TM
        n_offset = block_n_idx * TN
        ks_begin = ks * KS

        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(A)), num_records_bytes=fx.Int64(m) * (k * ELEM_BYTES)
        )
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(B)), num_records_bytes=n * k * ELEM_BYTES
        )
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(C)), num_records_bytes=fx.Int64(m) * (n * 4)
        )
        tiles = (m + TM - 1) // TM
        tiles = (tiles + XM - 1) // XM * XM * N_BLOCKS
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(WS)),
            num_records_bytes=fx.Int64(tiles) * (split_k * TILE_ELEMS * 4),
        )

        # Fetched early so its latency hides under the main loop.
        tile_flags = FLAGS + fx.Int64(tile_idx * split_k)
        own_flag = fx.generic_load(tile_flags + fx.Int64(ks))

        # m0 takes the LDS destination, so the wave's base has to be uniform.
        warp_dma_offset = rocdl.readfirstlane(
            T.i64, fx.Int64(wid * (WAVE_SIZE * DMA_BYTES)).ir_value()
        )
        a_lds_i64 = fx.Int64(fx.ptrtoint(lds.a.ptr))
        b_lds_i64 = fx.Int64(fx.ptrtoint(lds.b.ptr))

        def dma_tile(rsrc, lds_i64, ring, rows, row_base, count, k_offset, stage):
            # The DMA writes each instruction's 64 lanes to consecutive 16-byte
            # slots, so the stage base is unswizzled; the swizzle picks which K
            # chunk of global memory each slot fetches.
            stage_bytes = fx.Int64(stage * (rows * TK * ELEM_BYTES))
            lds_ptr = buffer_ops.get_element_ptr(
                create_llvm_ptr(lds_i64 + stage_bytes, fx.AddressSpace.Shared),
                warp_dma_offset,
            )
            for i in range_constexpr(count):
                chunk, r = coords(tid + i * BLOCK_THREADS, dma_slot_layout)
                col = fx.get_scalar(fx.crd2idx((0, r, chunk * DMA_VEC), ring)) % TK
                goff = (
                    fx.get_scalar(fx.crd2idx((row_base + r, k_offset + col), g_layout))
                    * ELEM_BYTES
                )
                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                    )
                llvm.InlineAsmOp(
                    None,
                    [lds_ptr, fx.Int32(goff).ir_value(), rsrc],
                    "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds",
                    "s,v,s",
                    has_side_effects=True,
                )

        def dma_stage(k_offset, stage):
            dma_tile(b_rsrc, b_lds_i64, b_ring, TN, n_offset, B_DMA, k_offset, stage)
            dma_tile(a_rsrc, a_lds_i64, a_ring, TM, m_offset, A_DMA, k_offset, stage)

        def read_frag(ptr, ring, stage, row, k_col):
            off = fx.get_scalar(fx.crd2idx((stage, row, k_col), ring))
            return fx.ptr_load(
                ptr + fx.Int64(off), result_type=fx.Vector.make_type(DMA_VEC, elem_ty)
            )

        def compute_stage(stage, c_frags):
            out = list(c_frags)
            for kk in range_constexpr(K_STEPS):
                k_col = frag_kg * DMA_VEC + kk * MFMA_K
                b_frags = [
                    read_frag(
                        lds.b.ptr,
                        b_ring,
                        stage,
                        wn * WARP_N + jj * MFMA_N + frag_row,
                        k_col,
                    )
                    for jj in range_constexpr(WNS)
                ]
                a_frags = [
                    read_frag(
                        lds.a.ptr,
                        a_ring,
                        stage,
                        wm * WARP_M + ii * MFMA_M + frag_row,
                        k_col,
                    )
                    for ii in range_constexpr(WMS)
                ]
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WMS):
                    for jj in range_constexpr(WNS):
                        f = ii * WNS + jj
                        out[f] = mfma_op(
                            T.f32x4, [a_frags[ii], b_frags[jj], out[f], 0, 0, 0]
                        )
            return out

        def hot_loop_scheduler():
            for _ in range_constexpr(DMA_PER_STAGE):
                rocdl.sched_vmem(1)
            for _ in range_constexpr(K_STEPS):
                for _ in range_constexpr(WNS + WMS):
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(WMS):
                    rocdl.sched_mfma(WNS)
            rocdl.sched_barrier(0)

        def wait_ring(younger_dmas):
            # The DMAs are opaque asm, so only this count orders a stage's
            # fill before its reads; s_barrier then publishes it to all waves.
            fx.rocdl.s_waitcnt(vmcnt=younger_dmas)
            rocdl.s_barrier()

        for s in range_constexpr(stages - 1):
            dma_stage(ks_begin + s * TK, s)
        rocdl.sched_barrier(0)

        c_frags = [fx.Vector.filled(ACC_VALUES, 0.0, fx.Float32)] * NFRAG
        init_state = [ks_begin, fx.Int32(0)] + c_frags
        for _, state in range(0, K_LOOPS - (stages - 1), 1, init=init_state):
            k_offset = state[0]
            cur = state[1]
            wait_ring((stages - 2) * DMA_PER_STAGE)
            dma_stage(k_offset + (stages - 1) * TK, (cur + stages - 1) % stages)
            c_new = compute_stage(cur, state[2:])
            hot_loop_scheduler()
            results = yield [k_offset + TK, (cur + 1) % stages] + c_new
        cur = results[1]
        c_frags = results[2:]
        for s in range_constexpr(stages - 1):
            wait_ring((stages - 2 - s) * DMA_PER_STAGE)
            c_frags = compute_stage(cur, c_frags)
            cur = (cur + 1) % stages

        # Split-K reduction.
        for f in range_constexpr(NFRAG):
            item = (wid * NFRAG + f) * WAVE_SIZE + lane
            off = fx.get_scalar(fx.crd2idx((item, ks, tile_idx), ws_layout))
            buffer_ops.buffer_store(
                c_frags[f], ws_rsrc, fx.Int32(off), cache_modifier=CPOL_SC1
            )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        gpu.barrier()

        # Rendezvous of the tile's split_k workgroups. Agent-scope monotonic
        # flag accesses lower to sc1 global loads/stores, coherent across XCDs.
        new_flag = own_flag + 1
        if tid == 0:
            fx.generic_store(
                tile_flags + fx.Int64(ks),
                new_flag,
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
            done = fx.Int32(0)
            while done == 0:
                arrived = None
                for s in range_constexpr(split_k):
                    flag = fx.generic_load(
                        tile_flags + s,
                        memory_order=fx.AtomicOrdering.Monotonic,
                        syncscope="agent",
                    )
                    eq = flag == new_flag
                    arrived = eq if const_expr(arrived is None) else arrived & eq
                done = arrived.select(fx.Int32(1), fx.Int32(0))
        rocdl.sched_barrier(0)
        gpu.barrier()

        # Split ks sums items [ks * PER_SPLIT, (ks + 1) * PER_SPLIT) of the tile.
        c_tile = m_offset * n + n_offset

        def reduce_item(item):
            acc = None
            for s in range_constexpr(split_k):
                off = fx.get_scalar(fx.crd2idx((item, s, tile_idx), ws_layout))
                part = fx.Vector(
                    buffer_ops.buffer_load(
                        ws_rsrc,
                        fx.Int32(off),
                        vec_width=ACC_VALUES,
                        dtype=T.f32,
                        cache_modifier=CPOL_SC1,
                    )
                )
                acc = part if const_expr(acc is None) else acc + part
            c_off = c_tile + fx.get_scalar(fx.crd2idx(item, c_item_layout))
            for r in range_constexpr(ACC_VALUES):
                buffer_ops.buffer_store(acc[r], c_rsrc, fx.Int32(c_off + r * n))

        for rr in range_constexpr(ROUNDS):
            local_item = tid + rr * BLOCK_THREADS
            if const_expr((rr + 1) * BLOCK_THREADS <= PER_SPLIT):
                reduce_item(ks * PER_SPLIT + local_item)
            else:
                if local_item < PER_SPLIT:
                    reduce_item(ks * PER_SPLIT + local_item)

    @flyc.jit
    def launch_splitk_f32_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        WS: fx.Pointer,
        FLAGS: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        m_blocks = (m + TM - 1) // TM
        tiles = (m_blocks + XM - 1) // XM * XM * N_BLOCKS
        splitk_f32_kernel._func.__name__ = KERNEL_NAME
        splitk_f32_kernel(C, A, B, WS, FLAGS, m).launch(
            grid=(tiles * split_k, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_splitk_f32_kernel
