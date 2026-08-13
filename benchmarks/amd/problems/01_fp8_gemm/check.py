"""Correctness runner for FP8 GEMM (AMD ROCm).

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3 seeds
each, with per-dtype atol/rtol. Also rejects forbidden ops by grep and enforces
the ROCm language gate (HIP or Triton kernel evidence required).
"""
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)
from src.eval.rocm_language import check_rocm_language, collect_solution_sources  # noqa: E402


def main():
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    problem_yaml = Path("problem.yaml")
    meta = yaml.safe_load(problem_yaml.read_text()) if problem_yaml.exists() else {}

    # --- Forbidden-op check ------------------------------------------------
    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    for forbidden in meta.get("forbidden", []):
        pat = re.escape(forbidden)
        if re.search(pat, sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    # --- ROCm language gate ------------------------------------------------
    all_src = collect_solution_sources()
    ok, messages, report = check_rocm_language(all_src, meta)
    Path("rocm_language.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("framework.txt").write_text(report["framework"] + "\n")
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    print(f"rocm_language: ok framework={report['framework']}")

    # --- Per-shape correctness --------------------------------------------
    device = torch.device("cuda:0")
    tol_override = meta.get("tolerance") or None

    all_shapes = shapes.SHAPES
    for shape_idx, shape in enumerate(all_shapes):
        reference.M = shape["M"]
        reference.N = shape["N"]
        reference.K = shape["K"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()

        sd = ref_model.state_dict()
        try:
            sol_model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch at shape {shape_idx} ({shape}): {e}")
            sys.exit(1)

        for seed in (42, 123, 456):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_inputs = [t.to(device) for t in reference.get_inputs()]

            for case in numeric_stress_cases(meta.get("name", "")):
                with numeric_stress_context(ref_model, sol_model, base_inputs, case) as inputs:
                    with torch.no_grad():
                        ref_out = ref_model(*inputs)
                        sol_out = sol_model(*inputs)

                ok, msg = check_correctness(
                    ref_out, sol_out,
                    dtype=ref_out.dtype,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} case {case.name}: {msg}")
                    sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
