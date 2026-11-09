# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch contract and scratch buffers of the split-K FP32-output HGEMM."""

import functools

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx, get_num_xcds
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .kernels.gemm_a16w16_splitk_f32_gfx950 import (
    compile_gemm_a16w16_splitk_f32_gfx950,
    splitk_f32_config_error,
    splitk_f32_tiles,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg

SPLIT_REDUCE_POLICY = "sr"
_TORCH_DTYPES = {
    torch.bfloat16: ("bf16", fx.BFloat16),
    torch.float16: ("fp16", fx.Float16),
}
# The kernel addresses A, B, C and the workspace with 32-bit buffer offsets.
_MAX_BUFFER_BYTES = 2**31


def splitk_f32_name_supported(params):
    """Name fields independent of the runtime M/N/K, for tuned-name parsing."""
    return (
        params["dtype"] in ("bf16", "fp16")
        and params["out_dtype"] == "fp32"
        and params["target_gfx"] == "gfx950"
        and not params["has_bias"]
        and params["split_k"] >= 2
        and params["k_waves"] == 1
        and params["group_m"] == 0
        and not params["k_stagger"]
        and not params["b_non_temporal"]
    )


@functools.cache
def _workspace(device, stream, elems):
    # Fully overwritten by every launch, so it needs no initialization and is
    # shared with captures. Never evicted: a captured graph may point at it.
    with persistent_alloc(device):
        return torch.empty(elems, dtype=torch.float32, device=device)


@capture_scoped_cache
def _flags(stream, device, kernel, count):
    # A tile's flags must start equal and advance only through launches of one
    # compiled kernel on one stream, hence the key.
    with persistent_alloc(device):
        return torch.zeros(count, dtype=torch.int32, device=device)


def gemm_a16w16_splitk_f32(a, b, out, bias, out_dtype, stream, config):
    """out[m, n] = a[m, k] @ b[n, k]^T in FP32 on the split-K reduce kernel."""
    if get_gfx() != "gfx950":
        raise RuntimeError("The FlyDSL split-K FP32 HGEMM supports gfx950 only")
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.device.type != "cuda"
        or b.device != a.device
        or a.dtype not in _TORCH_DTYPES
        or b.dtype != a.dtype
        or a.shape[1] != b.shape[1]
        or bias is not None
        or out_dtype not in (None, torch.float32)
    ):
        raise ValueError(
            "split-K FP32 HGEMM requires CUDA A[M, K] and B[N, K] of one "
            "bf16/fp16 dtype, FP32 output and no bias"
        )
    if (
        config.get("k_waves", 1) != 1
        or config.get("group_m")
        or config.get("k_stagger")
        or config.get("b_non_temporal")
    ):
        raise ValueError(
            "split-K FP32 HGEMM requires k_waves=1, group_m=0, k_stagger=0 and "
            "b_non_temporal=False"
        )
    m, k = a.shape
    n = b.shape[0]
    num_xcds = get_num_xcds()
    cfg = (
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["stages"],
        config["split_k"],
        config["m_waves"],
        config["n_waves"],
    )
    err = splitk_f32_config_error(n, k, *cfg, num_xcds)
    if err is None and m < 1:
        err = "M must be at least 1"
    if err is not None:
        raise ValueError(f"split-K FP32 HGEMM cannot run M={m}, N={n}, K={k}: {err}")
    tiles = splitk_f32_tiles(m, n, cfg[0], cfg[1], num_xcds)
    split_k = cfg[4]
    ws_elems = tiles * split_k * cfg[0] * cfg[1]
    if max(m, n) * k * a.element_size() >= _MAX_BUFFER_BYTES or (
        max(ws_elems, m * n) * 4 >= _MAX_BUFFER_BYTES
    ):
        raise ValueError(
            f"split-K FP32 HGEMM M={m}, N={n}, K={k} exceeds 32-bit buffer offsets"
        )
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("split-K FP32 HGEMM A/B pointers must be 16-byte aligned")
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.float32
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be a contiguous FP32 [{m}, {n}] tensor on A's device"
        )

    dtype, fx_dtype = _TORCH_DTYPES[a.dtype]
    exe = compile_gemm_a16w16_splitk_f32_gfx950(dtype, n, k, *cfg, num_xcds)
    # Capacities round up to powers of two, so the buffers a stream holds stay
    # few however many M it sees.
    flags = _flags(
        stream,
        a.device,
        (dtype, n, k, cfg, num_xcds),
        1 << (tiles * split_k - 1).bit_length(),
    )
    workspace = _workspace(a.device, stream, 1 << (ws_elems - 1).bit_length())
    _run_compiled(
        exe,
        ptr_arg(out, fx.Float32),
        ptr_arg(a, fx_dtype),
        ptr_arg(b, fx_dtype),
        ptr_arg(workspace, fx.Float32),
        ptr_arg(flags, fx.Int32),
        int(m),
        fx.Stream(stream),
    )
    return out
