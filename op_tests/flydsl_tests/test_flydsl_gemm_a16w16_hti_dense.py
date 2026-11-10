# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16
from aiter.ops.flydsl.kernels.gemm_a16w16_hti_dense_gfx950 import (
    gemm_a16w16_hti_dense,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
B_LAYOUTS = ["nk", "kn"]
# 16384x1024x4096 is one group of the DeepSeek-V4.1-Flash grouped output
# projection (prefill chunk); the others cover partial M and N tiles and the
# shortest K the two-stage pipeline accepts.
DEFAULT_MNK = [
    (16384, 1024, 4096),
    (300, 1024, 4096),
    (512, 640, 1024),
    (256, 256, 128),
]
HTI_CONFIG = {
    "block_m": 256,
    "block_n": 256,
    "block_k": 64,
    "stages": 2,
    "m_waves": 2,
    "n_waves": 4,
    "group_m": 4,
    "use_half_tile_interleaved": True,
}


def run_torch(x, w, dtype):
    return (x.to(dtypes.fp32) @ w.to(dtypes.fp32).t()).to(dtype)


@benchmark()
def test_gemm_a16w16_hti_dense(m, n, k, dtype, b_layout):
    x = torch.randn(m, k, dtype=dtype)
    w = torch.randn(n, k, dtype=dtype)
    # "nk" is the model's W[N, K] (used as W.mT); "kn" is a contiguous B[K, N].
    b = w if b_layout == "nk" else w.t().contiguous()
    ref = run_torch(x, w, dtype)
    out = torch.empty(m, n, dtype=dtype)
    generic_layout = "nt" if b_layout == "nk" else "nn"
    generic_b = b.t() if b_layout == "nk" else b
    candidates = {
        "hti_dense": lambda: gemm_a16w16_hti_dense(x, b, out, b_layout=b_layout),
        "flydsl_ht": lambda: gemm_a16w16(
            x, generic_b, user_kwargs=HTI_CONFIG, layout=generic_layout
        ),
    }

    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * x.element_size()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            result.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: gemm a16w16 hti dense",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("hti dense GEMM unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
        help="""Input and output dtype.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_MNK,
        help="""M, N, K; K must be a multiple of 128.
        e.g.: -s 16384,1024,4096""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        choices=B_LAYOUTS,
        default=B_LAYOUTS,
        help="""B layout: nk is W[N, K] (C = A @ W.T), kn is B[K, N].
        e.g.: -l nk""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = pd.DataFrame(
            [
                test_gemm_a16w16_hti_dense(m, n, k, dtype, b_layout)
                for b_layout, (m, n, k) in itertools.product(args.layout, args.mnk)
            ]
        )
        aiter.logger.info(
            "gemm_a16w16 hti dense summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
