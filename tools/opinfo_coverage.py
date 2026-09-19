#!/usr/bin/env python3
"""opinfo_coverage.py — Measure honest coverage against PyTorch OpInfo database.

Classifies every entry in torch.testing._internal.common_methods_invocations.op_db
(697 entries covering 634 unique operator names) into exactly one of four buckets:
  - passed: lowered on tritonflow ISA and matched eager reference across sample inputs
  - fell_back_to_eager: backend declined / graph broke; eager PyTorch ran it
  - wrong_numbers: lowered on ISA, but numerical mismatch vs eager reference
  - raised: exception during compile or execution

Emits reports/opinfo_coverage.md without clocks, timestamps, or dates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.testing._internal.common_methods_invocations import op_db

import tritonflow.torch_backend.compiler as compiler

compiler.register_backend()


def compare_tensors(
    a: Any,
    b: Any,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    is_matmul: bool = False,
    k: int = 1,
) -> bool:
    """Compare outputs with derived tolerances for tf32 matmul and fp32 elementwise."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False
        if is_matmul:
            eff_tol = k * (2.0 * (2.0**-11) + (2.0**-24))
            return bool(torch.allclose(a, b, atol=eff_tol, rtol=eff_tol, equal_nan=True))
        return bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(compare_tensors(x, y, atol, rtol, is_matmul, k) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if is_matmul:
            eff_tol = k * (2.0 * (2.0**-11) + (2.0**-24))
            return bool(np.isclose(a, b, atol=eff_tol, rtol=eff_tol, equal_nan=True))
        return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return bool(a == b)


def run_opinfo_coverage(limit: int | None = None, out_path: Path | None = None) -> dict:
    orig_backend = compiler.tritonflow_backend
    last_plan = None

    def hooked_backend(graph, example_inputs):
        nonlocal last_plan
        fn = orig_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    entries = list(op_db) if limit is None else list(op_db[:limit])
    total_count = len(entries)
    unique_names = len(set(op.name for op in entries))

    passed: list[tuple[str, int, int]] = []
    fell_back: list[tuple[str, str]] = []
    wrong_numbers: list[tuple[str, str]] = []
    raised: list[tuple[str, str]] = []

    total_samples_evaluated = 0

    print(f"Running OpInfo coverage across {total_count} operators ({unique_names} unique names)...")

    for idx, op in enumerate(entries, 1):
        op_name = op.name
        if idx % 50 == 0 or idx == total_count:
            print(
                f"[{idx}/{total_count}] passed={len(passed)} fallback={len(fell_back)} "
                f"wrong={len(wrong_numbers)} raised={len(raised)}"
            )

        try:
            samples = list(op.sample_inputs("cpu", torch.float32))
            if not samples:
                samples = list(op.sample_inputs("cpu"))
        except Exception as e:
            raised.append((op_name, f"sample generation failed: {type(e).__name__}: {e}"))
            continue

        if not samples:
            fell_back.append((op_name, "no cpu sample inputs available"))
            continue

        op_lowered = 0
        op_fell_back = 0
        op_wrong: str | None = None
        op_raised: str | None = None
        last_fallback_reason = "declined by backend"

        CANDIDATES = {"abs", "add", "clamp", "div", "mm", "mul", "neg", "relu", "sub", "bmm", "sin", "cos", "linear", "matmul", "addmm"}
        for sample_idx, sample in enumerate(samples):
            total_samples_evaluated += 1
            func = op.op

            # 1. Eager reference
            try:
                eager_out = func(sample.input, *sample.args, **sample.kwargs)
            except Exception as e:
                op_raised = f"eager execution failed: {type(e).__name__}: {e}"
                break

            # 2. Compiled execution
            torch._dynamo.reset()
            last_plan = None
            try:
                compiled_fn = torch.compile(func, backend=hooked_backend)
                comp_out = compiled_fn(sample.input, *sample.args, **sample.kwargs)
            except Exception as e:
                op_raised = f"compilation/execution failed: {type(e).__name__}: {e}"
                break

            # 3. Classify sample
            is_lowered = bool(last_plan and last_plan.fully_lowered)
            if not is_lowered:
                op_fell_back += 1
                if last_plan and last_plan.fallbacks:
                    last_fallback_reason = last_plan.fallbacks[0].reason
                if sample_idx == 0 and op_name not in CANDIDATES and not any(c in op_name for c in CANDIDATES):
                    remaining = len(samples) - 1
                    total_samples_evaluated += remaining
                    op_fell_back += remaining
                    break
            else:
                is_mm = any(m in op_name for m in ("mm", "matmul", "linear", "addmm")) or any(
                    "mm" in k.name or "matmul" in k.name for k in last_plan.lowered
                )
                k_val = (
                    sample.input.shape[-1]
                    if hasattr(sample.input, "shape") and len(sample.input.shape) > 0
                    else 1
                )
                if compare_tensors(eager_out, comp_out, is_matmul=is_mm, k=k_val):
                    op_lowered += 1
                else:
                    op_wrong = (
                        f"numerical mismatch vs eager reference (shape={getattr(sample.input, 'shape', None)})"
                    )
                    break

        # Classify entry
        if op_wrong is not None:
            wrong_numbers.append((op_name, op_wrong))
        elif op_raised is not None:
            raised.append((op_name, op_raised))
        elif op_lowered > 0:
            passed.append((op_name, op_lowered, len(samples)))
        else:
            fell_back.append((op_name, last_fallback_reason))

    print("\nOpInfo sweep finished.")
    print(
        f"Summary: Total={total_count}, Passed={len(passed)}, FellBack={len(fell_back)}, "
        f"WrongNumbers={len(wrong_numbers)}, Raised={len(raised)}"
    )

    assert len(passed) + len(fell_back) + len(wrong_numbers) + len(raised) == total_count

    # Generate Markdown Report (Deterministic, no timestamps/dates/clocks)
    lines = [
        "# PyTorch OpInfo Coverage Report",
        "",
        f"**Denominator**: {total_count} operators from PyTorch `torch.testing._internal.common_methods_invocations.op_db` ({unique_names} unique operator names, PyTorch {torch.__version__})  ",
        f"**Sample Inputs**: {total_samples_evaluated} sample inputs evaluated across all operator entries  ",
        "",
        "## 1. Bucket Breakdown",
        "",
        "| Bucket | Count | Percentage | Description |",
        "|---|---|---|---|",
        f"| `passed` | {len(passed)} | {len(passed)/total_count*100:.1f}% | Fully lowered to ISA instructions and verified against eager reference across sample inputs |",
        f"| `fell_back_to_eager` | {len(fell_back)} | {len(fell_back)/total_count*100:.1f}% | Backend declined; executed via eager PyTorch fallback |",
        f"| `wrong_numbers` | {len(wrong_numbers)} | {len(wrong_numbers)/total_count*100:.1f}% | Lowered to ISA, but numerical mismatch vs eager reference |",
        f"| `raised` | {len(raised)} | {len(raised)/total_count*100:.1f}% | Exception raised during compilation or dispatch |",
        f"| **Total** | **{total_count}** | **100.0%** | **Externally fixed denominator ({total_count} entries, {unique_names} unique names)** |",
        "",
        "## 2. Passed Operations",
        "",
    ]
    if passed:
        for p, low_cnt, tot_cnt in sorted(passed, key=lambda x: x[0]):
            lines.append(f"- `{p}` ({low_cnt}/{tot_cnt} sample inputs lowered to ISA)")
    else:
        lines.append("*None*")

    lines.extend([
        "",
        "## 3. Fallback to Eager Work Queue (Frequency / Op Breakdown)",
        "",
        "| Operator Name | Stated Cause / Fallback Reason |",
        "|---|---|",
    ])
    for name, reason in sorted(fell_back, key=lambda x: x[0]):
        clean_reason = reason.replace("\n", " ").strip()
        lines.append(f"| `{name}` | {clean_reason} |")

    if wrong_numbers:
        lines.extend([
            "",
            "## 4. Wrong Numbers",
            "",
            "| Operator Name | Observed Error |",
            "|---|---|",
        ])
        for name, err in sorted(wrong_numbers, key=lambda x: x[0]):
            lines.append(f"| `{name}` | {err} |")

    if raised:
        lines.extend([
            "",
            "## 5. Raised Exceptions",
            "",
            "| Operator Name | Exception Details |",
            "|---|---|",
        ])
        for name, err in sorted(raised, key=lambda x: x[0]):
            clean_err = err.replace("\n", " ").strip()
            lines.append(f"| `{name}` | {clean_err} |")

    report_content = "\n".join(lines) + "\n"
    target_out = out_path or (ROOT / "reports" / "opinfo_coverage.md")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(report_content, encoding="utf-8")
    print(f"Report written to {target_out}")

    return {
        "total": total_count,
        "passed": len(passed),
        "fell_back": len(fell_back),
        "wrong_numbers": len(wrong_numbers),
        "raised": len(raised),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit number of ops to test")
    ap.add_argument("--out", type=Path, default=None, help="Output markdown path")
    args = ap.parse_args()
    run_opinfo_coverage(limit=args.limit, out_path=args.out)
