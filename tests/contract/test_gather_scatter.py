"""Contract test for Task E4: Gather/Scatter Descriptor Bifurcation.

Verifies:
1. Indirect gather (k07) and indirect scatter (k08) recover gather/scatter descriptors
   (is_gather_scatter=True, indices_name specified) rather than refusing.
2. Both kernels compile with 0 markers to vortex_rvgpu.
3. Both kernels execute and match reference output with 0 diff.
4. Python and C++ emulators produce bit-identical results.
5. Affine/strided kernels remain structured (is_gather_scatter=False).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import numpy as np
from bench.corpus.kernels import get_corpus

from tritonflow.emu.exec import emulate
from tritonflow.extract.aot import compile_aot


def test_indirect_gather_k07():
    corpus = get_corpus()
    k07 = next(k for k in corpus if k.name == "k07_indirect_gather_1d")
    inputs = k07.make_inputs()
    res = compile_aot(
        k07.fn,
        k07.signature,
        k07.constexprs,
        env=k07.env,
        inputs=inputs,
        reference_fn=k07.reference,
    )

    assert res.outcomes["extract"].status == "OK"
    assert res.outcomes["parse"].status == "OK"
    assert res.outcomes["recognise"].status == "OK"
    assert res.outcomes["select"].status == "OK"
    assert res.outcomes["assemble"].status == "OK"
    assert res.outcomes["execute"].status == "OK"
    assert res.outcomes["parity"].status == "PASS"
    assert res.program is not None
    assert len(res.program.markers()) == 0

    # Verify descriptor has is_gather=True
    gather_instrs = [
        inst for inst in res.program.instructions()
        if inst.name in ("LDG", "VDMA_READ", "VLOAD") and "is_gather=True" in str(inst.operand("src"))
    ]
    assert len(gather_instrs) >= 1, "Expected at least one gather instruction"

    # Differential test: Python vs C++ emulator
    out_py = emulate(res.program, inputs, use_cpp=False)
    out_cpp = emulate(res.program, inputs, use_cpp=True)
    for k in out_py:
        np.testing.assert_array_equal(out_py[k], out_cpp[k])


def test_indirect_scatter_k08():
    corpus = get_corpus()
    k08 = next(k for k in corpus if k.name == "k08_indirect_scatter_1d")
    inputs = k08.make_inputs()
    res = compile_aot(
        k08.fn,
        k08.signature,
        k08.constexprs,
        env=k08.env,
        inputs=inputs,
        reference_fn=k08.reference,
    )

    assert res.outcomes["extract"].status == "OK"
    assert res.outcomes["parse"].status == "OK"
    assert res.outcomes["recognise"].status == "OK"
    assert res.outcomes["select"].status == "OK"
    assert res.outcomes["assemble"].status == "OK"
    assert res.outcomes["execute"].status == "OK"
    assert res.outcomes["parity"].status == "PASS"
    assert res.program is not None
    assert len(res.program.markers()) == 0

    # Verify descriptor has is_gather=True on dst
    scatter_instrs = [
        inst for inst in res.program.instructions()
        if inst.name in ("STG", "VDMA_WRITE", "VSTORE") and "is_gather=True" in str(inst.operand("dst"))
    ]
    assert len(scatter_instrs) >= 1, "Expected at least one scatter instruction"

    # Differential test: Python vs C++ emulator
    out_py = emulate(res.program, inputs, use_cpp=False)
    out_cpp = emulate(res.program, inputs, use_cpp=True)
    for k in out_py:
        np.testing.assert_array_equal(out_py[k], out_cpp[k])


def test_affine_kernel_structured_descriptor():
    corpus = get_corpus()
    k00 = next(k for k in corpus if k.name == "k00_dynamic_add_64")
    inputs = k00.make_inputs()
    res = compile_aot(
        k00.fn,
        k00.signature,
        k00.constexprs,
        env=k00.env,
        inputs=inputs,
        reference_fn=k00.reference,
    )
    assert res.program is not None
    assert len(res.program.markers()) == 0
    # No gather tags on purely affine descriptors
    for inst in res.program.instructions():
        assert "is_gather" not in str(inst.operand("src"))
        assert "is_gather" not in str(inst.operand("dst"))
