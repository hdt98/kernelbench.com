"""SOTA reference for FP8 GEMM on AMD ROCm.

On ROCm, torch._scaled_mm dispatches to hipBLASLt's FP8 GEMM kernel,
which is the vendor-optimized ceiling. If AITER is installed, we try its
FP8 GEMM path first (it may have shape-specialized tiles that beat the
general hipBLASLt path).

Agents are FORBIDDEN from using torch._scaled_mm in their solution (see
problem.yaml.forbidden). This file is only for the benchmark's reference line.
"""
from __future__ import annotations

import torch


def _try_aiter(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    try:
        import aiter  # AITER's public API may differ; adapt if needed
        return None
    except ImportError:
        return None


def _scaled_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    scale_a = torch.tensor(1.0, device=x.device)
    scale_b = torch.tensor(1.0, device=x.device)
    out = torch._scaled_mm(
        x,
        w.T,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    return out if not isinstance(out, tuple) else out[0]


def sota_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    result = _try_aiter(x, w)
    if result is not None:
        return result
    return _scaled_mm(x, w)


def is_available() -> bool:
    try:
        x = torch.randn(16, 16, dtype=torch.float8_e4m3fn, device="cuda")
        w = torch.randn(16, 16, dtype=torch.float8_e4m3fn, device="cuda")
        _scaled_mm(x, w)
        return True
    except Exception:
        return False
