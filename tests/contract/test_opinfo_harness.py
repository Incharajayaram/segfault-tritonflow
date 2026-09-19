"""Tests for PyTorch OpInfo coverage harness and derived tolerances (T7)."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest
import torch
from tools.opinfo_coverage import compare_tensors, run_opinfo_coverage


def test_compare_tensors_derived_tf32():
    """Derived tolerance k * (2 * 2^-11 + 2^-24) must accept valid tf32 truncation."""
    k = 64
    tol = k * (2.0 * (2.0**-11) + (2.0**-24))  # ~0.0625
    a = torch.zeros((10, 10), dtype=torch.float32)
    b = a + tol * 0.95  # within tolerance
    c = a + tol * 1.5   # exceeds tolerance

    assert compare_tensors(a, b, is_matmul=True, k=k) is True
    assert compare_tensors(a, c, is_matmul=True, k=k) is False


def test_compare_tensors_fp32_elementwise():
    """Elementwise comparisons must be strict within fp32 tolerances."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.00001, 2.00001, 3.00001], dtype=torch.float32)
    c = torch.tensor([1.01, 2.01, 3.01], dtype=torch.float32)

    assert compare_tensors(a, b, atol=1e-4, rtol=1e-4) is True
    assert compare_tensors(a, c, atol=1e-4, rtol=1e-4) is False


@pytest.mark.filterwarnings("ignore:.*script_method.*:DeprecationWarning")
def test_opinfo_report_structure_and_no_dates():
    """OpInfo coverage runner must produce reports with exact denominator and no clocks/dates."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "opinfo_test.md"
            res = run_opinfo_coverage(limit=5, out_path=out_path)

            assert res["total"] == 5
            assert res["passed"] + res["fell_back"] + res["wrong_numbers"] + res["raised"] == 5
            assert out_path.exists()

            content = out_path.read_text(encoding="utf-8")
            assert "# PyTorch OpInfo Coverage Report" in content
            assert "| `passed` |" in content
            assert "| `fell_back_to_eager` |" in content
            assert "| `wrong_numbers` |" in content
            assert "| `raised` |" in content
            assert "**Denominator**: 5 operators" in content

            # Verify no timestamps or dates
            assert "Execution Date" not in content
            assert "Elapsed Time" not in content
            assert "2026" not in content
            assert "UTC" not in content
