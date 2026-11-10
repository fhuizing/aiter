# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense BF16/FP16 GEMM for gfx950 specialized on the weight shape.

C[M, N] = A[M, K] @ B, with B either W[N, K] (``b_layout="nk"``, the model's
``W.mT``) or a contiguous B[K, N] (``b_layout="kn"``). It runs the
half-tile-interleaved 8-wave ping-pong of ``gemm_a16w16_gfx950`` (256x256x64
tile, two LDS stages, 16x16x32 MFMA) with:

- N, K and the operand strides as compile-time constants (M stays dynamic),
  so tile scheduling and DMA addressing need no runtime division;
- a prologue that waits only for the first LDS stage, not part of the second;
- an epilogue that drops the ``vmcnt(0)`` round trips between the C quadrant
  stores, which the LDS regions they use never required.

There is no split-K, bias or fp32 output. On a 256-CU MI355X this shape class
is power-limited: these changes match, not beat, the generic kernel there.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from .gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    AsyncLoadTile,
    async_load_to_lds,
    make_gemm_a16w16_gfx950_param,
    make_gemm_ab_lds_layouts,
    make_gemm_ab_load_context,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    BlockSwizzle,
    wait_vmcnt_and_barrier,
)
from .kernels_common import run_cached, set_slp_vectorize_once_used
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

HTI_DENSE_TILE = {
    "block_m": 256,
    "block_n": 256,
    "block_k": 64,
    "stages": 2,
    "m_waves": 2,
    "n_waves": 4,
}


@fx.struct
class GemmA16W16HtiDenseParam:
    in_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    n: fx.Constexpr[int]
    k: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


@functools.cache
def make_gemm_a16w16_hti_dense_param(
    in_dtype_id: int, n: int, k: int, b_is_transposed: bool, group_m: int
) -> GemmA16W16HtiDenseParam:
    base = make_gemm_a16w16_gfx950_param(
        in_dtype_id,
        in_dtype_id,
        group_m=group_m,
        use_half_tile_interleaved=True,
        b_is_transposed=b_is_transposed,
        **HTI_DENSE_TILE,
    )
    k_tiles = k // base.block_k
    if k % base.block_k or k_tiles < 2 or k_tiles % 2:
        raise ValueError(
            f"K={k} must be a multiple of {2 * base.block_k} for the HTI pipeline"
        )
    if n % base.cshuffle_r2g_vec_size:
        raise ValueError(f"N={n} must be a multiple of {base.cshuffle_r2g_vec_size}")
    if not b_is_transposed and n % (GFX950_DMA_BYTES // base.in_data_bytes):
        raise ValueError(f"N={n} must be 16-byte aligned for a [K, N] B")
    return GemmA16W16HtiDenseParam(
        in_dtype_id=in_dtype_id,
        block_m=base.block_m,
        block_n=base.block_n,
        block_k=base.block_k,
        m_waves=base.m_waves,
        n_waves=base.n_waves,
        group_m=group_m,
        a_is_transposed=False,
        b_is_transposed=b_is_transposed,
        n=n,
        k=k,
        mma_m=base.mma_m,
        mma_n=base.mma_n,
        mma_k=base.mma_k,
        async_load_bytes=base.async_load_bytes,
        in_data_bytes=base.in_data_bytes,
        cshuffle_r2g_vec_size=base.cshuffle_r2g_vec_size,
        ldg_x_threads=base.ldg_x_threads,
        block_threads=base.block_threads,
        ldg_a_iters=base.ldg_a_iters,
        ldg_b_iters=base.ldg_b_iters,
    )


def make_gemm_a16w16_hti_dense_kernel_name(param: GemmA16W16HtiDenseParam):
    dtype = "fp16" if param.in_dtype_id == GEMM_A16W16_DTYPE_FP16 else "bf16"
    b_layout = "nk" if param.b_is_transposed else "kn"
    return (
        f"hgemm_hti_dense_{dtype}_t{param.block_m}x{param.block_n}x{param.block_k}"
        f"_n{param.n}_k{param.k}_gm{param.group_m}_{b_layout}"
    )


@flyc.kernel
def gemm_a16w16_hti_dense_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16HtiDenseParam,
):
    n = param.n
    k = param.k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    a_leading_stride = k
    b_leading_stride = k if const_expr(param.b_is_transposed) else n
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = fx.Int32(k // block_k)
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, 2 * block_m * block_k, 16]
        b: fx.Array[elem_dtype, 2 * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)

    ab_load_context = make_gemm_ab_load_context(
        elem_dtype, load_tid=tid, ks_begin=0, param=param
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(
        ab_load_context.a_tiled_copy_atom, tiled_mma
    ).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(
        ab_load_context.b_tiled_copy_atom, tiled_mma
    ).get_slice(tid)

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        half_block_m, half_block_n, block_k, False, param.b_is_transposed
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=half_a_base(stage, m_part),
                src_base=fx.get_iter(a_buf),
                lds_layout=a_lds_layout,
                outer_tile_size=half_block_m,
                outer_bound=m,
                global_outer_offset=block_m_offset + m_part * half_block_m,
                leading_stride=a_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=half_ldg_a_iters,
            is_k_major=False,
        )

    def async_load_b_to_lds(n_part, k_tile, stage):
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=half_b_base(stage, n_part),
                src_base=fx.get_iter(b_buf),
                lds_layout=b_lds_layout,
                outer_tile_size=half_block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset + n_part * half_block_n,
                leading_stride=b_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=half_ldg_b_iters,
            is_k_major=not param.b_is_transposed,
        )

    row_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    def make_c_fragment(m_part, n_part):
        gC = fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]
        frag = thr_mma.make_fragment_C(gC)
        frag.fill(0.0)
        return frag

    def load_a_fragment(m_part, read_stage):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
        return frag_A

    def load_b_fragment(n_part, read_stage):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
        return frag_B

    def consume(frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)

    def half_c_base(m_part, n_part):
        return smem_c + (m_part * 2 + n_part) * half_block_m * half_block_n

    def store_half_tile_to_lds(m_part, n_part, frag_C):
        sC = fx.make_view(half_c_base(m_part, n_part), c_lds_layout)
        frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
        frag_C_out.store(frag_C.load().to(elem_dtype))
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

    def store_half_tile_to_global(m_part, n_part):
        sC_base = half_c_base(m_part, n_part)
        x_threads = half_block_n // cshuffle_r2g_vec_size
        vectors = half_block_m * half_block_n // cshuffle_r2g_vec_size
        for i in range_constexpr((vectors + block_threads - 1) // block_threads):
            vector_idx = block_threads * i + tid
            if vector_idx < vectors:
                local_row = vector_idx // x_threads
                local_col = vector_idx % x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + m_part * half_block_m + local_row
                global_col = block_n_offset + n_part * half_block_n + local_col
                if (global_row < m) and (global_col < n):
                    c_vec = fx.ptr_load(
                        sC_base + local_row * half_block_n + local_col,
                        result_type=fx.Vector.make_type(
                            cshuffle_r2g_vec_size, elem_dtype
                        ),
                    )
                    fx.ptr_store(c_vec, fx.get_iter(out) + global_row * n + global_col)

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    # Prologue. The second wave group runs one barrier behind the first, so the
    # two groups alternate between MFMA and LDS/DMA phases for the whole loop.
    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
        rocdl.s_waitcnt(vmcnt=0)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    rocdl.sched_barrier(0)
    # The first loop phases read only stage 0; the loop waits for stage 1.
    wait_vmcnt_and_barrier(2 * half_ldg_b_iters + half_ldg_a_iters)

    main_loop_end = k_tiles - 2
    for k_tile in range(0, main_loop_end, 2):
        next_k_tile = k_tile + 2
        b0 = load_b_fragment(0, 0)
        a0 = load_a_fragment(0, 0)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c00, a0, b0, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 0)
        async_load_b_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c01, a0, b1, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 0)
        async_load_a_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c10, a1, b0, True)
        rocdl.s_barrier()
        b0 = load_b_fragment(0, 1)
        async_load_b_to_lds(1, next_k_tile, 0)
        wait_vmcnt_and_barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(c11, a1, b1, True)
        rocdl.s_barrier()
        a0 = load_a_fragment(0, 1)
        async_load_a_to_lds(1, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c00, a0, b0, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 1)
        async_load_b_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c01, a0, b1, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 1)
        async_load_a_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c10, a1, b0, True)
        rocdl.s_barrier()
        async_load_b_to_lds(1, next_k_tile + 1, 1)
        wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(c11, a1, b1, True)
        rocdl.s_barrier()

    k_tile = main_loop_end
    b0 = load_b_fragment(0, 0)
    a0 = load_a_fragment(0, 0)
    async_load_a_to_lds(1, k_tile + 1, 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 0)
    rocdl.s_barrier()
    consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 0)
    rocdl.s_barrier()
    consume(c10, a1, b0, True)
    rocdl.s_barrier()
    b0 = load_b_fragment(0, 1)
    if wid // n_waves == 1:
        # The next barrier releases the first group to read stage 1 A half 0.
        rocdl.s_waitcnt(vmcnt=half_ldg_b_iters + half_ldg_a_iters)
    rocdl.s_barrier()
    consume(c11, a1, b1, True)
    wait_vmcnt_and_barrier(0)
    a0 = load_a_fragment(0, 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 1)
    rocdl.s_barrier()
    consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 1)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 0, c00)
    consume(c10, a1, b0, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 1, c01)
    consume(c11, a1, b1, False)
    rocdl.sched_barrier(0)
    # C halves 0 and 1 overlay the A and B stages: write the second pair only
    # after every wave has finished its last B fragment reads.
    gpu.barrier()
    store_half_tile_to_lds(1, 0, c10)
    store_half_tile_to_lds(1, 1, c11)
    gpu.barrier()
    store_half_tile_to_global(0, 0)
    store_half_tile_to_global(0, 1)
    store_half_tile_to_global(1, 0)
    store_half_tile_to_global(1, 1)


@flyc.jit
def gemm_a16w16_hti_dense_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    param: GemmA16W16HtiDenseParam,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    k_per_mfma_group = param.mma_k // 4
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(
            None,
            None,
            fx.make_layout((k_per_mfma_group, 4), (1, k_per_mfma_group)),
        ),
    )
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (param.n + param.block_n - 1) // param.block_n
    kernel = gemm_a16w16_hti_dense_gfx950_kernel
    kernel._known_block_size = [param.block_threads, 1, 1]
    kernel._func.__name__ = make_gemm_a16w16_hti_dense_kernel_name(param)
    kernel(out, a, b, m, tiled_mma, param).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


gemm_a16w16_hti_dense_gfx950.compile_hints = {
    "llvm_options": {
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    },
}
set_slp_vectorize_once_used(gemm_a16w16_hti_dense_gfx950, False)


def gemm_a16w16_hti_dense(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    b_layout: str = "nk",
    group_m: int = 4,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ B for a contiguous A.

    ``b_layout="nk"``: ``b`` is W[N, K] and C = A @ W.T (the model's ``W.mT``).
    ``b_layout="kn"``: ``b`` is a contiguous B[K, N] and C = A @ B.
    Inputs are BF16 or FP16 and C has the input dtype. K must be a multiple
    of 128. Each distinct (N, K, layout) compiles its own kernel.
    """
    if b_layout not in ("nk", "kn"):
        raise ValueError(f"b_layout must be 'nk' or 'kn', got {b_layout!r}")
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
        raise ValueError("A and B must both be bfloat16 or both float16")
    if a.ndim != 2 or b.ndim != 2 or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous 2-D tensors")
    for t in (a, b):
        if t.data_ptr() % GFX950_DMA_BYTES:
            raise ValueError("A and B must be 16-byte aligned")
    m, k = a.shape
    n = b.shape[0] if b_layout == "nk" else b.shape[1]
    if (b.shape[1] if b_layout == "nk" else b.shape[0]) != k:
        raise ValueError(f"shape mismatch: A {tuple(a.shape)}, B {tuple(b.shape)}")
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    elif out.shape != (m, n) or out.dtype != a.dtype or not out.is_contiguous():
        raise ValueError("out must be a contiguous [M, N] tensor of the input dtype")
    if m == 0:
        return out
    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    param = make_gemm_a16w16_hti_dense_param(
        in_dtype_id, n, k, b_layout == "nk", group_m
    )
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    # The kernel reads B as a logical [K, N]: W[N, K] is its column-major form.
    b_kn = b.t() if b_layout == "nk" else b
    dispatch_args = (
        flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(b_kn).mark_layout_dynamic(
            leading_dim=0 if b_layout == "nk" else 1
        ),
        param,
        stream,
    )
    run_cached(
        gemm_a16w16_hti_dense_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out
