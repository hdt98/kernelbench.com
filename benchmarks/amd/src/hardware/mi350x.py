"""AMD Instinct MI350X — CDNA 4, gfx950 (next-gen, Blackwell competitor).

Supports FP4 and FP6 data types (new in CDNA 4). Peak numbers below are
ESTIMATES based on AMD's public claims and architecture analysis.
VERIFY on-node with rocprof before publishing any results.

Accessible nodes: do-sonle5-mi350-gpu, do-vunguyen-mi350-gpu,
do-huyhoang-mi350-gpu, do-longnguyen-mi350-gpu.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

MI350X = HardwareTarget(
    name="AMD Instinct MI350X",
    sm="gfx950",
    vram_gb=288,
    peak_bandwidth_gb_s=8000.0,  # HBM3E, verify on-node
    peak_tflops_dense={
        # ESTIMATED — verify on-node with rocprof before publishing.
        # CDNA 4 adds FP4/FP6 matrix core support.
        "fp4": 2500.0,       # estimate
        "fp6": 1250.0,       # estimate
        "fp8": 626.0,        # estimate (~2.4x MI325X)
        "bf16": 313.0,       # estimate (~2.4x MI325X)
        "fp16": 313.0,       # estimate
        "fp32": 200.0,       # estimate, vector ALU
        "int8": 626.0,       # estimate
    },
)
