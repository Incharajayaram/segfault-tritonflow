#!/usr/bin/env python3
"""Validation gates for the TritonFlow cost model.

Gates:
  G1 Totality: every instruction x descriptor produces CostResult or named CostUnknown
  G2 Monotonicity: workload increase -> cost/resources never decrease
  G3 Unit Soundness: cost/time expressions parse, reference registered terms, valid units
  G4 Declared-param Fidelity: machine parameters valid, match declared vendor facts
  G8 Selection Stability: deterministic lowering on reference descriptors
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure PYTHONPATH contains src
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tritonflow.isa.cost import (
    CostQuery,
    CostUnknown,
    evaluate_cost,
    load_machine_by_name,
    term_registry,
)
from tritonflow.isa.schema import (
    is_known_cost_term,
    load_builtin,
)


class MockDescriptor:
    def __init__(
        self,
        base_num: int = 0,
        sizes: tuple[int, ...] = (64,),
        strides: tuple[int, ...] = (1,),
        offsets: tuple[int, ...] = (0,),
        shape: tuple[int, ...] = (64,),
        dtype: str = "f32",
    ) -> None:
        self.base_num = base_num
        self.base = f"%ptr_{base_num}"
        self.sizes = sizes
        self.strides = strides
        self.offsets = offsets
        self.shape = shape
        self.dtype = dtype


def gate_g1_totality() -> tuple[bool, str]:
    """G1 Totality: every instruction x corpus operand -> CostResult or named CostUnknown."""
    schemas_to_test = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
    descriptors = [
        MockDescriptor(sizes=(64,), strides=(1,)),
        MockDescriptor(sizes=(128,), strides=(1,)),
        MockDescriptor(sizes=(16, 16), strides=(16, 1)),
        MockDescriptor(sizes=(16, 16), strides=(64, 1)),
        MockDescriptor(sizes=(16, 16), strides=(16, 2)),  # non-contiguous
    ]
    tiles = [None, (16, 16), (32, 32), (16, 16, 16)]

    total_evaluated = 0
    total_admissible = 0
    total_unknown = 0

    for sname in schemas_to_test:
        schema = load_builtin(sname)
        mach = load_machine_by_name(sname) if (repo_root / f"src/tritonflow/isa/machines/{sname}.json").exists() else None

        for instr in schema.instructions.values():
            for desc in descriptors:
                for tile in tiles:
                    total_evaluated += 1
                    query = CostQuery(
                        instruction=instr,
                        access=desc,
                        tile=tile,
                        env={"words": desc.sizes[0], "trip_count": 1},
                        machine=mach,
                    )
                    try:
                        res = evaluate_cost(query, mach)
                        total_admissible += 1
                        if res.select_cost < 0.0:
                            return False, f"G1 FAIL: {sname}:{instr.name} returned negative cost {res.select_cost}"
                        if desc.sizes[0] > 0 and res.select_cost == 0.0 and instr.kind == "memory":
                            return False, f"G1 FAIL: {sname}:{instr.name} returned silent zero cost for non-empty access"
                    except CostUnknown:
                        total_unknown += 1
                    except Exception as exc:
                        return False, f"G1 FAIL: {sname}:{instr.name} crashed with unexpected exception: {type(exc).__name__}: {exc}"

    return True, f"G1 Totality OK ({total_evaluated} evaluations: {total_admissible} valid, {total_unknown} fail-closed)"


def gate_g2_monotonicity() -> tuple[bool, str]:
    """G2 Monotonicity: workload increase -> cost/resources never decrease."""
    schema = load_builtin("tritonflow1")
    mach = load_machine_by_name("tritonflow1")

    # 1. Elements monotonicity: DMA1D cost on 64 words <= DMA1D cost on 128 words
    dma1d = schema.instruction("DMA1D")
    q_small = CostQuery(instruction=dma1d, access=MockDescriptor(sizes=(64,)), env={"words": 64}, machine=mach)
    q_large = CostQuery(instruction=dma1d, access=MockDescriptor(sizes=(128,)), env={"words": 128}, machine=mach)
    c_small = evaluate_cost(q_small, mach).select_cost
    c_large = evaluate_cost(q_large, mach).select_cost
    if not (c_small < c_large):
        return False, f"G2 FAIL: DMA1D 64 words cost ({c_small}) >= 128 words cost ({c_large})"

    # 2. Coalescing / Stride monotonicity: Contiguous (stride=1) <= Strided (stride=17)
    vx_schema = load_builtin("vortex_rvgpu")
    vx_mach = load_machine_by_name("vortex_rvgpu")
    ldg = vx_schema.instruction("LDG")
    q_contig = CostQuery(
        instruction=ldg,
        access=MockDescriptor(sizes=(32,), strides=(1,)),
        tile=(32,),
        machine=vx_mach,
    )
    q_strided = CostQuery(
        instruction=ldg,
        access=MockDescriptor(sizes=(32,), strides=(17,)),
        tile=(32,),
        machine=vx_mach,
    )
    res_contig = evaluate_cost(q_contig, vx_mach)
    res_strided = evaluate_cost(q_strided, vx_mach)
    if not res_contig.resources.transactions < res_strided.resources.transactions:
        return False, (
            f"G2 FAIL: stride-1 transactions ({res_contig.resources.transactions}) are not strictly "
            f"fewer than stride-17 transactions ({res_strided.resources.transactions})"
        )
    if not res_contig.select_cost < res_strided.select_cost:
        return False, (
            f"G2 FAIL: stride-1 cost ({res_contig.select_cost}) is not strictly below "
            f"stride-17 cost ({res_strided.select_cost}); the cost ignores the access shape"
        )

    # 3. Compute monotonicity: MAC16 tile (16,16,16) < (32,32,32)
    mac16 = schema.instruction("MAC16")
    q_mac16 = CostQuery(instruction=mac16, tile=(16, 16, 16), machine=mach)
    q_mac32 = CostQuery(instruction=mac16, tile=(32, 32, 32), machine=mach)
    c_mac16 = evaluate_cost(q_mac16, mach).select_cost
    c_mac32 = evaluate_cost(q_mac32, mach).select_cost
    if not (c_mac16 < c_mac32):
        return False, f"G2 FAIL: MAC tile 16x16x16 cost ({c_mac16}) >= 32x32x32 cost ({c_mac32})"

    return True, "G2 Monotonicity OK (element, stride, and compute monotonicity verified)"


def gate_g3_unit_soundness() -> tuple[bool, str]:
    """G3 Unit Soundness: cost/time expressions parse, reference registered terms, valid units."""
    schemas_to_test = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
    assert set(term_registry().keys())

    for sname in schemas_to_test:
        schema = load_builtin(sname)
        for instr in schema.instructions.values():
            for cexpr, name in [(instr.cost, "cost"), (instr.time, "time"), (getattr(instr, "select_cost", None), "select_cost")]:
                if cexpr is None:
                    continue
                for t in cexpr.terms:
                    if not is_known_cost_term(t):
                        return False, f"G3 FAIL: {sname}:{instr.name} references unregistered term {t!r}"

    return True, "G3 Unit Soundness OK (all terms validated against registered term providers)"


def gate_g4_declared_fidelity() -> tuple[bool, str]:
    """G4 Declared-param Fidelity: machine parameters valid, match declared vendor facts."""
    machines = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
    for mname in machines:
        m = load_machine_by_name(mname)
        if not m.params:
            return False, f"G4 FAIL: {mname} has empty parameters"
        for pname, p in m.params.items():
            if p.value < 0.0:
                return False, f"G4 FAIL: {mname}:{pname} has negative value {p.value}"
            if p.basis not in ("declared", "fitted", "analytic", "measured"):
                return False, f"G4 FAIL: {mname}:{pname} has invalid basis {p.basis!r}"

    # Vendor-derived values are checked against the vendored upstream files, never against
    # numbers typed into this gate.
    sys.path.insert(0, str(repo_root))
    from tools.check_vortex_constants import verify_constants

    ok, errors = verify_constants()
    if not ok:
        return False, "G4 FAIL: " + "; ".join(errors)

    return True, "G4 Declared-param Fidelity OK (machine parameters sound; vortex values match the vendored upstream config)"


def gate_g8_selection_stability() -> tuple[bool, str]:
    """G8 Selection Stability: what the compiler chooses matches the reviewed golden file."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "snapshot_selection.py"), "--check"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, "G8 FAIL: " + (proc.stdout + proc.stderr).strip().splitlines()[-1]
    return True, "G8 Selection Stability OK (instruction selection matches fixtures/selection_golden.json)"


def main() -> int:
    gates = [
        ("G1 Totality", gate_g1_totality),
        ("G2 Monotonicity", gate_g2_monotonicity),
        ("G3 Unit Soundness", gate_g3_unit_soundness),
        ("G4 Declared Fidelity", gate_g4_declared_fidelity),
        ("G8 Selection Stability", gate_g8_selection_stability),
    ]

    all_passed = True
    print("==================================================")
    print("      TritonFlow Cost Model Gate Validation       ")
    print("==================================================")
    for name, gate_fn in gates:
        passed, msg = gate_fn()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if not passed:
            all_passed = False

    print("==================================================")
    if all_passed:
        print("ALL GATES PASSED.")
        return 0
    else:
        print("SOME GATES FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
