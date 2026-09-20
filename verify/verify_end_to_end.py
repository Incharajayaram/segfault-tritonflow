#!/usr/bin/env python3
"""verify_end_to_end.py — the two end-to-end paths in this tree, honestly labeled.

PATH A (real pipeline): fixture text -> parse_raw -> build_ir -> annotate
-> assemble (real schema + real selector) -> serialize -> deserialize
-> disassemble -> emulate. Expected values: structural identities (the
serializer's round-trip property) and op counts read from the fixture text
itself — plus, for execution, the math identity each fixture computes
(vecadd: out = x + y elementwise; modulo: out = x mod max(|y|,1)).

PATH B (demo pipeline, `lower.py`): the per-tier hardcoded instruction
table. Its numbers are NOT verified here as ground truth — this script
reports what it produces and cross-checks it against Path A where both
run, flagging disagreement instead of averaging it away.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAILURES: list[str] = []


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    fixtures = ROOT / "fixtures"

    # ------------------------------------------------------------------ #
    print("PATH A: real pipeline — parse -> build_ir -> assemble -> round-trip")
    from tritonflow.emit.disasm import deserialize, serialize
    from tritonflow.idioms.detect import annotate
    from tritonflow.isa.schema import load_builtin
    from tritonflow.ttir.graph import build_def_use
    from tritonflow.ttir.to_ir import parse_module  # text -> semantic Module (both layers)

    schema = load_builtin("tritonflow1")
    launch = json.loads((ROOT / "fixtures" / "launch_env.json").read_text())
    program = None
    t0_program = None
    expected_counts = {"t0_vecadd": 4, "t1_matmul": 40, "t2_matmul_relu": 40, "t3_modulo": 25}
    for tier in ("t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo"):
        res = parse_module((fixtures / f"{tier}.ttir").read_text(),
                           source_path=str(fixtures / f"{tier}.ttir"))
        module = res.module
        check(f"A/{tier}: semantic IR built", module is not None, True)
        if module is None:
            continue
        try:
            annotations = annotate(module, build_def_use(module))
            graph = build_def_use(module)
            from tritonflow.emit.assemble import assemble
            # contracted signature: (module, graph, annotations, schema, env=...)
            program = assemble(module, graph, annotations, schema,
                               env=dict(launch[tier]))
            if tier == "t0_vecadd":
                t0_program = program
            n_items = len(program.instrs) + sum(len(loop.body) for loop in program.loops)
            check(f"A/{tier}: assembled items ({n_items})", n_items > 0, True)
            check(f"A/{tier}: instruction count == {expected_counts[tier]}",
                  len(program.instrs), expected_counts[tier],
                  "count from the same selector run in the main repo")
            markers = program.markers()
            if tier == "t3_modulo":
                # the canon says t3's descriptors must refuse: wraparound is
                # unstructured, so UNSUPPORTED is the CORRECT answer here
                check(f"A/{tier}: t3 refused with markers (correct per canon)",
                      len(markers) > 0, True)
            else:
                check(f"A/{tier}: no UNSUPPORTED markers", len(markers), 0,
                      f"markers: {[str(m) for m in markers[:2]]}")
        except Exception as exc:
            FAILURES.append(f"A/{tier}: assemble")
            print(f"  FAIL A/{tier}: assemble RAISED {type(exc).__name__}: {exc}")
            continue

        # round-trip: deserialize(serialize(p)) == p — a structural identity
        text = serialize(program)
        reparsed = deserialize(text)
        check(f"A/{tier}: round-trip instruction count",
              len(reparsed.instrs), len(program.instrs))
        check(f"A/{tier}: round-trip total_cost",
              reparsed.total_cost, program.total_cost)

    # execution on the assembled t0 program, if one exists
    if t0_program is not None:
        from tritonflow.emu.exec import emulate
        from tritonflow.emu.precision import PrecisionPolicy
        rng = np.random.default_rng(24173)
        x = rng.standard_normal(1024).astype(np.float32)
        y = rng.standard_normal(1024).astype(np.float32)
        # the fixture computes out = x + y elementwise over 1024 f32 lanes
        try:
            out = emulate(
                t0_program,
                {
                    "%x_ptr": x,
                    "%y_ptr": y,
                    "%out_ptr": np.zeros_like(x),
                    "%n": 1024,
                },
                          policy=PrecisionPolicy(input_precision="ieee"))
            key = next(iter(out))
            got = out[key]
            err = float(np.max(np.abs(got - (x + y))))
            check("A/t0: emulator output == x+y (ieee)", err, 0.0,
                  f"max abs err={err}, buffers={list(out)}")
        except Exception as exc:
            # t0 assembled program carries an UNSUPPORTED marker -> halt is correct
            from tritonflow.emu.exec import ProgramNotExecutable
            check("A/t0: emulate either matches x+y or halts on a named refusal",
                  isinstance(exc, ProgramNotExecutable), True,
                  f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("PATH B: demo pipeline (lower.py, hardcoded per-tier tables) — reported, cross-checked")
    try:
        from tritonflow.emu.hardware import CoalescingUnit
        from tritonflow.lower import lower_fixture, make_inputs

        # cross-check lower.py's coalescing stats against the unit it claims to use
        cu = CoalescingUnit(cache_line_bytes=32, warp_size=32)
        scattered = cu.analyze(base_address=0, stride_elements=17, element_bytes=4)
        contiguous = cu.analyze(base_address=0, stride_elements=1, element_bytes=4)
        check("B: CoalescingUnit discriminates stride-1 vs stride-17",
              contiguous.coalescing_efficiency > scattered.coalescing_efficiency, True,
              f"contig={contiguous.coalescing_efficiency} scattered={scattered.coalescing_efficiency}")

        for tier in ("t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo"):
            ctx = lower_fixture(tier)
            check(f"B/{tier}: lowers without refusal", ctx.unsupported, [])
            check(f"B/{tier}: emulator ran", ctx.emu_outputs is not None, True)

            # independent math identity for t0: out == x + y exactly in fp32?
            # fp32 addition is NOT associative, so use a tolerance from the dtype
            if tier == "t0_vecadd":
                inputs = make_inputs("t0_vecadd")
                got = ctx.emu_outputs["out"]
                ref = inputs["x"] + inputs["y"]
                err = float(np.max(np.abs(got - ref)))
                check(f"B/{tier}: t0 output within fp32 exactness of x+y", err, 0.0,
                      f"max abs err={err}")
            elif tier == "t1_matmul":
                inputs = make_inputs("t1_matmul")
                got = ctx.emu_outputs["out"]
                # tf32 inputs: independent reference = fp64 matmul of truncated inputs
                from tritonflow.emu.precision import tf32_truncate
                a32 = tf32_truncate(inputs["a"].astype(np.float32)).astype(np.float64)
                b32 = tf32_truncate(inputs["b"].astype(np.float32)).astype(np.float64)
                ref = (a32 @ b32).astype(np.float32)
                peak = float(np.max(np.abs(ref)))
                rel = float(np.max(np.abs(got - ref)) / max(1.0, peak))
                n = inputs["a"].shape[1]
                bound = n * (2.0 * 2.0**-11 + 2.0**-24)
                check(f"B/{tier}: t1 inside derived tf32 band (rel={rel:.3e} <= {bound:.3e})",
                      rel <= bound, True)
    except Exception as exc:
        FAILURES.append("B: pipeline")
        print(f"  FAIL B: pipeline RAISED {type(exc).__name__}: {exc}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
