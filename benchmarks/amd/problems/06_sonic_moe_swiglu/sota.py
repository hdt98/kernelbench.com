"""SOTA reference for Sonic-MoE up-projection: Tri Dao's sonic-moe.

sonic-moe is an NVIDIA-only library (CuTeDSL grouped GEMM kernels) and is not
available on AMD MI325X. is_available() returns False on ROCm, and the
benchmark scores the agent against the documented MI325X reference throughput
(see problem.yaml.sota.reference_throughput_tflops_mi325x). Agents are
FORBIDDEN from importing sonic_moe in solution.py (see problem.yaml.forbidden).
"""
from __future__ import annotations

import torch


def _try_sonic_moe(
    hidden_states: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor | None:
    try:
        import sonic_moe  # type: ignore  # noqa: F401
    except Exception:
        return None
    try:
        # Public sonic-moe API surface is still stabilizing. The expected entry
        # point bundles gate+up weights as a single (E, H, 2*I) tensor and fuses
        # SwiGLU. Adapt to the actual signature once SM120 lands.
        W = torch.cat([W_gate, W_up], dim=-1).contiguous()  # (E, H, 2*I)
        from sonic_moe import fused_moe_up  # type: ignore
        return fused_moe_up(hidden_states, W, expert_offsets)
    except Exception:
        return None


def sota_forward(
    hidden_states: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Best-available grouped-GEMM + SwiGLU reference."""
    out = _try_sonic_moe(hidden_states, W_gate, W_up, expert_offsets)
    if out is not None:
        return out
    raise RuntimeError("sonic-moe SOTA path unavailable on this hardware")


def is_available() -> bool:
    # sonic-moe is NVIDIA-only; on AMD ROCm the import fails and we return
    # False. The benchmark uses the MI325X reference throughput instead.
    try:
        import sonic_moe  # type: ignore  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    # sonic-moe targets NVIDIA GPUs only (CuTeDSL, sm_90/sm_100).
    # On AMD ROCm, the import above fails so we never reach here.
    # benchmark.py wraps sota_forward in try/except and treats failures as
    # "SOTA unavailable" -- see problem.yaml.sota.reference_throughput_tflops_mi325x
    # for the documented MI325X ceiling used in that case.
    return True
