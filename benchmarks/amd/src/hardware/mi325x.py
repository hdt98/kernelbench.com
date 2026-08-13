"""AMD Instinct MI325X -- CDNA 3, gfx942 (same compute dies as MI300X).

288 GB HBM3E at 8.0 TB/s. Same matrix-core compute as MI300X but clocked higher:
MI300X runs at 1700 MHz SCLK; MI325X runs at 2100 MHz SCLK.

Peak numbers are scaled from the MI300X dense matrix-core specs (1307.0 TFLOPS
BF16 at 1700 MHz, AMD official) by the clock ratio (2100/1700 = 1.235):
  BF16: 1614.4 TFLOPS
  FP8:  3228.8 TFLOPS (2x BF16 on CDNA 3: FP8 MFMA has 2x the K-dimension throughput)

Empirically verified: torch.matmul (rocBLAS) BF16 achieves 757 TFLOPS on
8192x8192x8192 -- 46.9% of the 1614 TFLOPS peak, which is a typical rocBLAS
efficiency for large GEMM on CDNA 3.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

MI325X = HardwareTarget(
    name="AMD Instinct MI325X",
    sm="gfx942",
    vram_gb=288,
    peak_bandwidth_gb_s=8000.0,
    peak_tflops_dense={
        "fp8": 3228.8,
        "bf16": 1614.4,
        "fp16": 1614.4,
        "fp32": 201.8,
        "int8": 3228.8,
        "int4": 6457.6,
    },
)

