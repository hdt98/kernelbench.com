# KernelBench-AMD: Design Specification

Last updated: 2026-08-12.

## Purpose

KernelBench-AMD evaluates frontier coding agents on writing competitive GPU
kernels for AMD Instinct GPUs using ROCm/HIP. It reuses the same harness,
grading pipeline, and roofline methodology as KernelBench-Hard/CUDA — only
the hardware target, kernel language surface, and SOTA libraries change.

The scientific question: **how well does the same agent write a competitive
kernel on AMD silicon vs NVIDIA silicon?** Same problems, same grading,
different vendor.

## Hardware targets

| GPU | Architecture | gfx | VRAM | Bandwidth | Nodes | Status |
| --- | --- | --- | --- | --- | --- | --- |
| MI325X | CDNA 3 | gfx942 | 288 GB HBM3E | 8.0 TB/s | cr7 (8x) | Primary |
| MI350X | CDNA 4 | gfx950 | 288 GB HBM3E | 8.0 TB/s (est) | 4 nodes | Specs pending on-node verification |

Peak TFLOPS (dense, matrix core):

| Precision | MI325X | MI350X (est) |
| --- | --- | --- |
| FP8 | 3228.8 | TBD |
| BF16 | 1614.4 | TBD |
| FP4 | n/a | TBD |
| FP6 | n/a | TBD |
| INT8 | 261.5 | 626.0 |

MI350X numbers are estimates — verify with rocprof before publishing.

## Node access

| Node | GPU | Access | Notes |
| --- | --- | --- | --- |
| cr7 | 8x MI325X | ssh cr7 | Primary eval node. ROCm installed, torch/uv need bootstrap. |
| do-sonle5-mi350-gpu | MI350X | ssh do-sonle5-mi350-gpu | Specs pending. |
| do-vunguyen-mi350-gpu | MI350X | ssh do-vunguyen-mi350-gpu | Specs pending. |
| do-huyhoang-mi350-gpu | MI350X | ssh do-huyhoang-mi350-gpu | Specs pending. |
| do-longnguyen-mi350-gpu | MI350X | ssh do-longnguyen-mi350-gpu | Specs pending. |
| kimlong-mi325 | 8x MI325X | OFF-LIMITS | Do not touch. |
| messi-gpu-8xmi325x | 8x MI325X | OFF-LIMITS | Do not touch. |

**Shared-node rules:** always check GPU idleness with rocm-smi before
launching work. Never kill another tenant's process. Pin GPU visibility
with HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES.

## Language policy

Unlike KernelBench-CUDA (which forbids Triton), KernelBench-AMD **allows
both HIP and Triton-on-ROCm**. Triton's ROCm backend is a legitimate
production kernel path, not a cheat.

The language gate (src/eval/rocm_language.py) requires evidence of a real
GPU kernel:
- HIP C++ via torch.utils.cpp_extension.load_inline
- Composable Kernels (CK)
- rocWMMA / MFMA intrinsics
- Triton @jit kernels
- AITER

Pure PyTorch op chains (torch.nn.functional) are rejected. Kernel DSLs
(ThunderKittens, TileLang) are also rejected.

## Problem deck

Problems are ported from KernelBench-Hard to enable cross-vendor comparison.
Same shapes, same tolerances, same forbidden ops — different hardware and
SOTA references.

| NN | Problem | Status |
| --- | --- | --- |
| 01 | FP8 e4m3 GEMM | Ported, SOTA = hipBLASLt |
| 03 | Paged Attention | Planned |
| 06 | Sonic MoE SwiGLU | Planned |
| 07 | W4A16 GEMM | Planned |

## SOTA references

| Problem | NVIDIA SOTA | AMD SOTA |
| --- | --- | --- |
| FP8 GEMM | FlashInfer / cuBLAS | hipBLASLt (via torch._scaled_mm on ROCm) |
| Paged Attention | FlashInfer | AITER / CK paged attention |
| MoE SwiGLU | vLLM / SGLang | AITER MoE kernels |
| W4A16 GEMM | Machete / Marlin | CK mixed-precision GEMM |

## Key differences from NVIDIA benches

1. **No CUDA toolkit.** ROCm at /opt/rocm provides HIP, hipBLASLt, CK, etc.
   The shared harness runner skips CUDA setup when KBH_CUDA_HOME points to a
   non-existent path.
2. **Triton is allowed.** Triton-on-ROCm is a first-class kernel path. The
   language gate checks for kernel evidence, not CUDA specifically.
3. **Profiling with rocprof** instead of ncu/nsys. rocm-smi instead of
   nvidia-smi.
4. **L2 cache is 256 MB** on MI325X (vs 96 MB on Blackwell). The timing
   module flushes 384 MB to evict L2 between benchmark trials.
5. **Device strings stay "cuda".** PyTorch ROCm maps "cuda" to HIP
   internally — no code changes needed in reference.py or benchmark.py.

## Bootstrapping a node

On a fresh AMD node (cr7 or MI350 nodes):

1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2. Clone or sync the repo to the node.
3. cd benchmarks/amd && uv sync
4. Install ROCm torch: uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
5. Run patch_torch.sh if needed (same inductor issues as NVIDIA benches).
6. Smoke test: uv run python -c "import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available())"

## Harness

cd benchmarks/amd
./scripts/run_hard.sh <harness> <model> problems/01_fp8_gemm

Or from the monorepo root (once kb CLI support is added):
kb -b amd run <harness> <model> problems/01_fp8_gemm
