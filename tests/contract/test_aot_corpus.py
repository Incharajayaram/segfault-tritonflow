"""test_aot_corpus.py — Verification contract for Task E1: AOT compilation and kernel corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from bench.corpus.kernels import get_corpus

from tritonflow.extract.aot import compile_aot

ROOT = Path(__file__).resolve().parent.parent.parent


def test_aot_entry_point_lowers_unseen_kernel():
    """Verify that compile_aot compiles and fully lowers a kernel not in fixtures."""
    corpus = get_corpus()
    k0 = next(k for k in corpus if k.name == "k00_dynamic_add_64")
    res = compile_aot(
        fn=k0.fn,
        signature=k0.signature,
        constexprs=k0.constexprs,
        name=k0.name,
        isa_name="tritonflow1",
        env=k0.env,
        inputs=k0.make_inputs(),
        reference_fn=k0.reference,
    )
    assert res.outcomes["extract"].status == "OK"
    assert res.outcomes["parse"].status == "OK"
    assert res.outcomes["recognise"].status == "OK"
    assert res.outcomes["select"].status == "OK"
    assert res.outcomes["assemble"].status == "OK"
    assert res.outcomes["execute"].status == "OK"
    assert res.outcomes["parity"].status == "PASS"
    assert res.parity_error is not None and res.parity_error <= 1e-4
    assert res.program is not None and len(res.program.instructions()) > 0


def test_aot_corpus_status_report_exists():
    """Verify that the kernel corpus covers 23 kernels."""
    corpus = get_corpus()
    assert len(corpus) == 23
    for k in corpus:
        assert k.name and k.category and k.description


def test_aot_reduction_parses_cleanly():
    """Verify that reduction kernels parse into clean TTIR with regions and inherent attributes."""
    corpus = get_corpus()
    k_red = next(k for k in corpus if k.name == "k01_reduction_sum_1d")
    res = compile_aot(
        fn=k_red.fn,
        signature=k_red.signature,
        constexprs=k_red.constexprs,
        name=k_red.name,
        isa_name="vortex_rvgpu",
        env=k_red.env,
    )
    assert res.outcomes["extract"].status == "OK"
    assert res.outcomes["parse"].status == "OK"


@pytest.mark.parametrize("isa_name", ["tritonflow1", "tritonflow2", "vortex_rvgpu"])
@pytest.mark.parametrize("kernel_name", ["k15_mixed_dtypes_f16_f32", "k16_mixed_dtypes_i32_f32"])
def test_aot_mixed_dtypes_all_isas(isa_name: str, kernel_name: str):
    """Verify that cast ops (arith.extf, arith.sitofp) lower, execute, and match NumPy parity across all ISAs."""
    corpus = get_corpus()
    k = next(item for item in corpus if item.name == kernel_name)
    res = compile_aot(
        fn=k.fn,
        signature=k.signature,
        constexprs=k.constexprs,
        name=k.name,
        isa_name=isa_name,
        env=k.env,
        inputs=k.make_inputs(),
        reference_fn=k.reference,
    )
    assert res.outcomes["extract"].status == "OK"
    assert res.outcomes["parse"].status == "OK"
    assert res.outcomes["recognise"].status == "OK"
    assert res.outcomes["select"].status == "OK"
    assert res.outcomes["assemble"].status == "OK"
    assert res.outcomes["execute"].status == "OK"
    assert res.outcomes["parity"].status == "PASS"
    assert res.parity_error is not None and res.parity_error <= 1e-4
