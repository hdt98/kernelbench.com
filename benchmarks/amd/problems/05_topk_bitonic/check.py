"""Correctness runner for TopK (AMD ROCm).

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3 seeds
each. Top-k correctness has two parts:

  1. VALUES: sol_values must match ref_values within fp32 tol. Both are
     returned sorted descending, so positional comparison is well-defined.
  2. INDICES: lenient — we do NOT require sol_indices == ref_indices because
     ties in x can yield multiple valid index sets. Instead we gather x at
     sol_indices and check those values match ref_values within tol. This
     catches "wrong indices" without false-failing on legitimate tie-breaks.

Also rejects forbidden ops by grep and enforces the ROCm language gate.
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
from src.eval.property_stress import (  # noqa: E402
    
    generate_property_cases,
    property_shape_index,
    tolerance_for_property,
)


def main():
    problem_yaml = Path("problem.yaml")
    meta = yaml.safe_load(problem_yaml.read_text()) if problem_yaml.exists() else {}
    try:
        import reference
        import shapes
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    _, property_cases = generate_property_cases(meta.get("name", ""))
    try:
        import solution
    except Exception as e:
        print(f"FAIL: solution import error: {e}")
        sys.exit(1)

    # --- Forbidden-op check ------------------------------------------------
    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    for forbidden in meta.get("forbidden", []):
        pat = re.escape(forbidden)
        if re.search(pat, sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    device = torch.device("cuda:0")
    tol_override = meta.get("tolerance") or None

    all_shapes = shapes.SHAPES
    for shape_idx, shape in enumerate(all_shapes):
        reference.batch = shape["batch"]
        reference.n = shape["n"]
        reference.k = shape["k"]

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
                        ref_values, ref_indices = ref_model(*inputs)
                        sol_out = sol_model(*inputs)

                if not (isinstance(sol_out, (tuple, list)) and len(sol_out) == 2):
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} case {case.name}: "
                          f"solution must return (values, indices); got {type(sol_out)}")
                    sys.exit(1)
                sol_values, sol_indices = sol_out

                expected_shape = (shape["batch"], shape["k"])
                if tuple(sol_values.shape) != expected_shape:
                    print(f"FAIL: shape {shape_idx} case {case.name} values shape "
                          f"{tuple(sol_values.shape)} != expected {expected_shape}")
                    sys.exit(1)
                if tuple(sol_indices.shape) != expected_shape:
                    print(f"FAIL: shape {shape_idx} case {case.name} indices shape "
                          f"{tuple(sol_indices.shape)} != expected {expected_shape}")
                    sys.exit(1)

                ok, msg = check_correctness(
                    ref_values.float(), sol_values.float(),
                    dtype=torch.float32,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} "
                          f"case {case.name} values: {msg}")
                    sys.exit(1)

                x = inputs[0]
                sol_idx_long = sol_indices.to(torch.int64)
                if sol_idx_long.min() < 0 or sol_idx_long.max() >= shape["n"]:
                    print(f"FAIL: shape {shape_idx} case {case.name} indices out of range "
                          f"[{int(sol_idx_long.min())}, {int(sol_idx_long.max())}]")
                    sys.exit(1)
                gathered = torch.gather(x, dim=-1, index=sol_idx_long)
                ok, msg = check_correctness(
                    ref_values.float(), gathered.float(),
                    dtype=torch.float32,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} "
                          f"case {case.name} indices (gather mismatch): {msg}")
                    sys.exit(1)

        if shape_idx == property_shape_index(meta.get("name", "")):
            torch.manual_seed(0xC0DE)
            torch.cuda.manual_seed_all(0xC0DE)
            base_inputs = [value.to(device) for value in reference.get_inputs()]
            try:
                check_topk_properties(
                    ref_model, sol_model, base_inputs,
                    k=shape["k"],
                    tolerance=tolerance_for_property(meta.get("name", ""), tol_override),
                    cases=property_cases,
                )
            except Exception as e:
                print(f"FAIL: shape {shape_idx} {shape} property stress: {e}")
                sys.exit(1)

    _emit_framework_label()
    print("PASS")


def _emit_framework_label():
    sol_src = collect_solution_sources(Path("."))
    ok, messages, report = check_rocm_language(sol_src)
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    evidence = ",".join(report["kernel_evidence"]) or "none"
    print("rocm_language: ok framework=" + report["framework"] + " evidence=" + evidence)
    Path("framework.txt").write_text(report["framework"] + "\n")
    Path("rocm_language.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

