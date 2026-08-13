"""ROCm language gate for KernelBench-AMD.

Unlike KernelBench-CUDA (which forbids Triton), this bench ALLOWS both HIP
and Triton-on-ROCm, since Triton's ROCm backend is a legitimate production
kernel path. Pure PyTorch op chains with no kernel evidence are rejected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

KERNEL_EVIDENCE_PATTERNS: list[tuple[str, str]] = [
    ("load_inline", r"torch\.utils\.cpp_extension\.load_inline|cpp_extension\.load\b"),
    ("hip_global", r"__global__\s+(?:void|__launch_bounds__)"),
    ("hip_header", r"#include\s*<hip(?:_runtime)?\.h>|#include\s*<hip/hip_runtime\.h>"),
    ("composable_kernel", r"ck::|composable_kernel|#include\s*<ck/"),
    ("rocwmma", r"rocwmma::|__mfma"),
    ("triton_jit", r"@triton\.jit\b"),
    ("triton_import", r"(?m)^\s*import\s+triton\b|from\s+triton\b"),
    ("triton_tl", r"\btl\.(?:program_id|load|store|dot)\b"),
    ("aiter", r"\baiter\b"),
    ("hip_file", r"load(?:_inline)?\s*\([^)]*['\"][^'\"]+\.(?:hip|cpp|cc)['\"]"),
]

DSL_PATTERNS: list[tuple[str, str]] = [
    ("thunderkittens", r"\bimport\s+thunderkittens\b|from\s+thunderkittens\b"),
    ("tilelang", r"\bimport\s+tilelang\b|from\s+tilelang\b"),
]

FRAMEWORK_PRIORITY: list[tuple[str, str]] = [
    ("composable_kernel", r"ck::|composable_kernel|#include\s*<ck/"),
    ("hip_raw", r"load_inline|__global__\s+void|hip_runtime"),
    ("rocwmma", r"rocwmma::|__mfma"),
    ("aiter", r"\baiter\b"),
    ("triton", r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
    ("pytorch_only", r"torch\.(?:nn\.functional|ops)"),
]


def _scan(code: str, patterns: list[tuple[str, str]]) -> list[str]:
    return [name for name, pat in patterns if re.search(pat, code)]


def detect_framework(code: str) -> str:
    for name, pat in FRAMEWORK_PRIORITY:
        if re.search(pat, code):
            return name
    return "unknown"


def check_rocm_language(
    sol_src: str,
    meta: dict[str, Any] | None = None,
    *,
    require_kernel_evidence: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    meta = meta or {}
    if "require_kernel_evidence" in meta:
        require_kernel_evidence = bool(meta["require_kernel_evidence"])
    elif "require_cuda_evidence" in meta:
        require_kernel_evidence = bool(meta["require_cuda_evidence"])

    dsl_hits = _scan(sol_src, DSL_PATTERNS)
    kernel_hits = _scan(sol_src, KERNEL_EVIDENCE_PATTERNS)
    framework = detect_framework(sol_src)

    messages: list[str] = []
    ok = True

    if dsl_hits:
        ok = False
        joined = ", ".join(dsl_hits)
        messages.append("FAIL: kernel DSL forbidden on KernelBench-AMD (hits: " + joined + "). Write a HIP C++ or Triton kernel.")
    if require_kernel_evidence and not kernel_hits:
        ok = False
        messages.append("FAIL: no GPU kernel evidence in solution.py. Expected load_inline / __global__ / .hip / Triton @jit / CK / AITER.")

    report = {
        "framework": framework,
        "has_kernel_evidence": bool(kernel_hits),
        "kernel_evidence": kernel_hits,
        "dsl_cheat": bool(dsl_hits),
        "forbidden_hits": dsl_hits,
        "ok": ok,
    }
    return ok, messages, report


def enforce_and_write(
    sol_path: Path = Path("solution.py"),
    meta: dict[str, Any] | None = None,
    report_path: Path = Path("rocm_language.json"),
    framework_path: Path = Path("framework.txt"),
) -> None:
    import sys

    sol_src = sol_path.read_text() if sol_path.exists() else ""
    ok, messages, report = check_rocm_language(sol_src, meta)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    framework_path.write_text(report["framework"] + "\n")
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    evidence = ",".join(report["kernel_evidence"]) or "none"
    print("rocm_language: ok framework=" + report["framework"] + " evidence=" + evidence)


def collect_solution_sources(root: Path = Path(".")) -> str:
    chunks: list[str] = []
    sol = root / "solution.py"
    if sol.exists():
        chunks.append(sol.read_text())
    for pattern in ("*.hip", "*.cpp", "*.cc", "*.cuh", "*.h", "*.hpp"):
        for p in sorted(root.glob(pattern)):
            try:
                chunks.append(p.read_text(errors="ignore"))
            except OSError:
                pass
    return "\n".join(chunks)
