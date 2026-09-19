"""Differential test between Python and C++ emulators and fp64 NumPy oracle (Task A1, T5).

For each of the 4 fixtures x each of the 3 schemas, assemble the program once,
then execute through emulate(..., use_cpp=False) and emulate(..., use_cpp=True),
verifying:
1. Exact agreement between Python and C++ emulator outputs (or identical refusals).
2. Parity against an independent fp64 NumPy reference within derived arithmetic tolerances.
3. Strict adherence to an explicit manifest of expected refusals: unexpected refusals
   fail, and manifest entries that unexpectedly pass also fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tritonflow.emu._emu_cpp import ProgramNotExecutable as CppPNE

from tritonflow.emit.assemble import assemble
from tritonflow.emu.exec import ProgramNotExecutable as PyPNE
from tritonflow.emu.exec import emulate
from tritonflow.emu.precision import PrecisionPolicy
from tritonflow.idioms.detect import annotate
from tritonflow.isa.schema import load_builtin
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT / "fixtures"
LAUNCH = json.loads((FIXTURES_DIR / "launch_env.json").read_text())

SCHEMAS = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
TIERS = ["t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo"]

#: Explicit manifest of expected refusals per (ISA, tier).
#: A refusal not in this manifest fails the test.
#: A manifest entry that no longer refuses also fails the test.
EXPECTED_REFUSALS: dict[tuple[str, str], str] = {
    ("tritonflow1", "t3_modulo"): "no elementwise instruction is admissible",
    ("tritonflow2", "t3_modulo"): "no elementwise instruction is admissible",
    ("vortex_rvgpu", "t3_modulo"): "no elementwise instruction is admissible",
    # Vortex has no scalar-constant instruction and no tensor-shape ops (see the comment in
    # vortex_rvgpu.yaml), so the matmul tiers are refused rather than lowered through pseudo-instructions.
    ("vortex_rvgpu", "t1_matmul"): "no elementwise instruction is admissible",
    ("vortex_rvgpu", "t2_matmul_relu"): "no elementwise instruction is admissible",
}


def _tf32_bound(k: int) -> float:
    return float(k * (2.0 * 2.0**-11 + 2.0**-24))


@pytest.mark.parametrize("schema_name", SCHEMAS)
@pytest.mark.parametrize("tier", TIERS)
def test_emulator_differential(schema_name: str, tier: str) -> None:
    schema = load_builtin(schema_name)
    ttir_path = FIXTURES_DIR / f"{tier}.ttir"
    res = parse_module(ttir_path.read_text(), source_path=str(ttir_path))
    module = res.module
    graph = build_def_use(module)
    annotations = annotate(module, graph)
    program = assemble(module, graph, annotations, schema, env=dict(LAUNCH[tier]))

    rng = np.random.default_rng(24173)
    if tier == "t0_vecadd":
        x = rng.standard_normal(1024).astype(np.float32)
        y = rng.standard_normal(1024).astype(np.float32)
        inputs = {
            "%x_ptr": x,
            "%y_ptr": y,
            "%out_ptr": np.zeros(1024, dtype=np.float32),
            "%n": 1024,
        }
        policy = PrecisionPolicy(input_precision="ieee")
        # Independent fp64 oracle
        oracle_ref = x.astype(np.float64) + y.astype(np.float64)
        oracle_key = "%out_ptr"
        oracle_slice = slice(None)
        derived_tolerance = 2.0**-24

    elif tier == "t1_matmul":
        a = rng.standard_normal((128, 64)).astype(np.float32)
        b = rng.standard_normal((64, 128)).astype(np.float32)
        inputs = {
            "%a_ptr": a,
            "%b_ptr": b,
            "%c_ptr": np.zeros((128, 128), dtype=np.float32),
            "%M": 128, "%N": 128, "%K": 64,
            "%sam": 64, "%sak": 1, "%sbk": 128, "%sbn": 1, "%scm": 128, "%scn": 1,
        }
        policy = PrecisionPolicy(input_precision="tf32")
        # Independent fp64 oracle (fixture computes top-left 64x64 tile)
        oracle_ref = a[:64].astype(np.float64) @ b[:, :64].astype(np.float64)
        oracle_key = "%c_ptr"
        oracle_slice = (slice(0, 64), slice(0, 64))
        derived_tolerance = _tf32_bound(64)

    elif tier == "t2_matmul_relu":
        a = rng.standard_normal((128, 64)).astype(np.float32)
        b = rng.standard_normal((64, 128)).astype(np.float32)
        bias = rng.standard_normal(128).astype(np.float32)
        inputs = {
            "%a_ptr": a,
            "%b_ptr": b,
            "%bias_ptr": bias,
            "%c_ptr": np.zeros((128, 128), dtype=np.float32),
            "%M": 128, "%N": 128, "%K": 64,
            "%sam": 64, "%sak": 1, "%sbk": 128, "%sbn": 1, "%scm": 128, "%scn": 1,
        }
        policy = PrecisionPolicy(input_precision="tf32")
        # Independent fp64 oracle with bias add and relu
        matmul_ref = a[:64].astype(np.float64) @ b[:, :64].astype(np.float64)
        oracle_ref = np.maximum(matmul_ref + bias[:64].astype(np.float64)[None, :], 0.0)
        oracle_key = "%c_ptr"
        oracle_slice = (slice(0, 64), slice(0, 64))
        derived_tolerance = _tf32_bound(64)

    else:  # t3_modulo
        inputs = {
            "%x_ptr": rng.standard_normal((16, 16)).astype(np.float32),
            "%out_ptr": np.zeros((16, 16), dtype=np.float32),
            "%M": 16, "%N": 16,
            "%stride_xm": 16, "%stride_xn": 1, "%stride_om": 16, "%stride_on": 1,
        }
        policy = PrecisionPolicy(input_precision="ieee")
        oracle_ref = None
        oracle_key = "%out_ptr"
        oracle_slice = slice(None)
        derived_tolerance = 2.0**-24

    markers = program.markers()
    is_expected_refusal = (schema_name, tier) in EXPECTED_REFUSALS

    if is_expected_refusal:
        # Must refuse! A manifest entry that no longer refuses fails the test.
        assert markers, (
            f"Expected refusal for ({schema_name}, {tier}) according to manifest, "
            f"but program lowered with no markers"
        )
        assert any(EXPECTED_REFUSALS[(schema_name, tier)] in str(m) for m in markers), (
            f"({schema_name}, {tier}) is refused, but not for the manifest's reason "
            f"{EXPECTED_REFUSALS[(schema_name, tier)]!r}: {[str(m)[:120] for m in markers[:3]]}"
        )
        py_msg, cpp_msg = None, None
        try:
            emulate(program, {k: v.copy() if hasattr(v, "copy") else v for k, v in inputs.items()}, policy=policy, use_cpp=False)
        except PyPNE as e:
            py_msg = str(e)
        try:
            emulate(program, {k: v.copy() if hasattr(v, "copy") else v for k, v in inputs.items()}, policy=policy, use_cpp=True)
        except CppPNE as e:
            cpp_msg = str(e)
        assert py_msg is not None, "Python emulator should have refused on markers"
        assert cpp_msg is not None, "C++ emulator should have refused on markers"
        assert "UNSUPPORTED" in py_msg
        assert "UNSUPPORTED" in cpp_msg
    else:
        # Must NOT refuse! A refusal that is not in the manifest fails the test.
        assert not markers, (
            f"Unexpected refusal for ({schema_name}, {tier}) not in manifest: {markers}"
        )
        out_py = emulate(
            program,
            {k: v.copy() if hasattr(v, "copy") else v for k, v in inputs.items()},
            policy=policy,
            use_cpp=False,
        )
        out_cpp = emulate(
            program,
            {k: v.copy() if hasattr(v, "copy") else v for k, v in inputs.items()},
            policy=policy,
            use_cpp=True,
        )

        # 1. Python vs C++ differential: bit-identical outputs
        for k in out_py:
            assert k in out_cpp, f"Buffer {k} missing from C++ output"
            np.testing.assert_array_equal(out_py[k], out_cpp[k], err_msg=f"Disagreement on buffer {k}")

        # 2. Independent fp64 NumPy oracle check
        if oracle_ref is not None:
            got_py = np.asarray(out_py[oracle_key], dtype=np.float64)[oracle_slice]
            got_cpp = np.asarray(out_cpp[oracle_key], dtype=np.float64)[oracle_slice]
            scale = max(float(np.max(np.abs(oracle_ref))), 1e-30)

            rel_err_py = float(np.max(np.abs(got_py - oracle_ref))) / scale
            rel_err_cpp = float(np.max(np.abs(got_cpp - oracle_ref))) / scale

            assert rel_err_py <= derived_tolerance, (
                f"Python emulator output for ({schema_name}, {tier}) exceeds derived tolerance: "
                f"rel_err={rel_err_py:g} > tol={derived_tolerance:g}"
            )
            assert rel_err_cpp <= derived_tolerance, (
                f"C++ emulator output for ({schema_name}, {tier}) exceeds derived tolerance: "
                f"rel_err={rel_err_cpp:g} > tol={derived_tolerance:g}"
            )


NEW_OPS = ["add", "sub", "mul", "div", "neg", "abs", "clamp"]


@pytest.mark.parametrize("op", NEW_OPS)
def test_newly_supported_ops_differential(op: str) -> None:
    pytest.importorskip("triton")
    from tritonflow.extract.dynamic_extract import extract_elementwise

    schema = load_builtin("tritonflow1")
    rng = np.random.default_rng(24173)

    ext = extract_elementwise(op, 64)
    res = parse_module(ext.ttir)
    graph = build_def_use(res.module)
    ann = annotate(res.module, graph)
    program = assemble(res.module, graph, ann, schema, env=dict(ext.env))

    if op in ("relu", "neg", "abs"):
        mem = {
            "%x_ptr": rng.standard_normal(64).astype(np.float32),
            "%out_ptr": np.zeros(64, dtype=np.float32),
            "%n": 64, "n": 64,
        }
    elif op == "clamp":
        mem = {
            "%x_ptr": rng.standard_normal(64).astype(np.float32),
            "%min_ptr": np.full(64, -0.5, dtype=np.float32),
            "%max_ptr": np.full(64, 0.5, dtype=np.float32),
            "%out_ptr": np.zeros(64, dtype=np.float32),
            "%n": 64, "n": 64,
        }
    else:
        mem = {
            "%x_ptr": rng.standard_normal(64).astype(np.float32),
            "%y_ptr": rng.standard_normal(64).astype(np.float32) + 0.1,
            "%out_ptr": np.zeros(64, dtype=np.float32),
            "%n": 64, "n": 64,
        }

    out_py = emulate(
        program,
        {k: v.copy() if hasattr(v, "copy") else v for k, v in mem.items()},
        use_cpp=False,
    )
    out_cpp = emulate(
        program,
        {k: v.copy() if hasattr(v, "copy") else v for k, v in mem.items()},
        use_cpp=True,
    )
    np.testing.assert_array_equal(out_py["%out_ptr"], out_cpp["%out_ptr"])
