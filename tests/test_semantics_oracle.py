"""test_semantics_oracle.py — Test suite using ISA semantics strings as executable oracle (Task A6).

Compares three independent sources for every checked operation:
1. The hardware emulator (apply / MachineState)
2. The ISA semantics oracle (eval_semantics)
3. A plain NumPy reference computation

Also includes mutation checks (flipping semantics strings and flipping emulator handlers)
to ensure disagreements are never silently ignored.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tritonflow.emit.ir import Imm, Instr, Program, SourceRef, SsaRef
from tritonflow.emu.exec import _ELEMENTWISE, MachineState, apply
from tritonflow.emu.precision import PrecisionPolicy
from tritonflow.isa.schema import load_builtin
from tritonflow.isa.semantics import (
    SemanticsEvalError,
    eval_semantics,
    is_parseable,
    parse_semantics,
)

SCHEMAS = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]


def test_all_instructions_have_parseable_semantics():
    """Every instruction in every schema must have a parseable semantics string."""
    for sname in SCHEMAS:
        schema = load_builtin(sname)
        for iname, instr in schema.instructions.items():
            assert instr.semantics, f"{sname}.{iname} has empty semantics"
            assert is_parseable(instr.semantics), (
                f"{sname}.{iname} semantics not parseable: {instr.semantics!r}"
            )


def test_semantics_oracle_matching_instructions():
    """Verify compute instructions by comparing 3 things: emulator, semantics oracle, and NumPy."""
    schema = load_builtin("vortex_rvgpu")
    policy = PrecisionPolicy()

    # 1. VADD with real source op arith.addf
    vadd = schema.instructions["VADD"]
    parsed_vadd = parse_semantics(vadd.semantics)
    x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    sref = SourceRef("arith.addf", 1, 1, "%out")
    instr = Instr(name="VADD", cost=1.0, operands={"in0": SsaRef("%x"), "in1": SsaRef("%y")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x", "%y"), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x, "%y": y})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vadd, {"src0": x, "src1": y})
    numpy_ref = x + y

    assert np.allclose(emu_val, sem_val), f"VADD emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VADD emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"

    # 2. VMUL with real source op arith.mulf
    vmul = schema.instructions["VMUL"]
    parsed_vmul = parse_semantics(vmul.semantics)
    sref = SourceRef("arith.mulf", 1, 1, "%out")
    instr = Instr(name="VMUL", cost=1.0, operands={"in0": SsaRef("%x"), "in1": SsaRef("%y")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x", "%y"), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x, "%y": y})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vmul, {"src0": x, "src1": y})
    numpy_ref = x * y

    assert np.allclose(emu_val, sem_val), f"VMUL emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VMUL emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"

    # 3. VSUB with real source op arith.subf
    vsub = schema.instructions["VSUB"]
    parsed_vsub = parse_semantics(vsub.semantics)
    sref = SourceRef("arith.subf", 1, 1, "%out")
    instr = Instr(name="VSUB", cost=1.0, operands={"in0": SsaRef("%x"), "in1": SsaRef("%y")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x", "%y"), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x, "%y": y})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vsub, {"src0": x, "src1": y})
    numpy_ref = x - y

    assert np.allclose(emu_val, sem_val), f"VSUB emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VSUB emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"

    # 4. VNEG with real source op arith.negf
    vneg = schema.instructions["VNEG"]
    parsed_vneg = parse_semantics(vneg.semantics)
    sref = SourceRef("arith.negf", 1, 1, "%out")
    instr = Instr(name="VNEG", cost=1.0, operands={"in0": SsaRef("%x")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x",), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vneg, {"src": x})
    numpy_ref = -x

    assert np.allclose(emu_val, sem_val), f"VNEG emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VNEG emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"

    # 5. VABS with real source op math.absf
    vabs = schema.instructions["VABS"]
    parsed_vabs = parse_semantics(vabs.semantics)
    sref = SourceRef("math.absf", 1, 1, "%out")
    x_mixed = np.array([-4.0, 0.0, 7.5], dtype=np.float32)
    instr = Instr(name="VABS", cost=1.0, operands={"in0": SsaRef("%x")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x",), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x_mixed})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vabs, {"src": x_mixed})
    numpy_ref = np.abs(x_mixed)

    assert np.allclose(emu_val, sem_val), f"VABS emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VABS emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"

    # 6. VMAX with real source op arith.maxnumf (operands in0 and clamp 0)
    vmax = schema.instructions["VMAX"]
    parsed_vmax = parse_semantics(vmax.semantics)
    xr = np.array([-3.0, 0.0, 5.0], dtype=np.float32)
    sref = SourceRef("arith.maxnumf", 1, 1, "%out")
    instr = Instr(name="VMAX", cost=1.0, operands={"in0": SsaRef("%x"), "in1": Imm(0.0)}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x",), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": xr})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, policy)
    emu_val = state.values.get("%out")
    sem_val = eval_semantics(parsed_vmax, {"src0": xr, "src1": 0.0})
    numpy_ref = np.maximum(0.0, xr)

    assert np.allclose(emu_val, sem_val), f"VMAX emu vs sem mismatch: emu={emu_val}, sem={sem_val}"
    assert np.allclose(emu_val, numpy_ref), f"VMAX emu vs numpy mismatch: emu={emu_val}, np={numpy_ref}"


def test_semantics_oracle_clamp_requires_bounds():
    """Clamp must take lo and hi from operands without defaults; missing bounds must raise."""
    schema2 = load_builtin("tritonflow2")
    clamp = schema2.instructions["CLAMP"]
    parsed = parse_semantics(clamp.semantics)
    x = np.array([-2.0, 0.5, 3.0, 8.0], dtype=np.float32)

    # 1. Missing bounds must raise SemanticsEvalError
    with pytest.raises(SemanticsEvalError):
        eval_semantics(parsed, {"src": x})

    with pytest.raises(SemanticsEvalError):
        eval_semantics(parsed, {"src": x, "lo": 0.0})

    with pytest.raises(SemanticsEvalError):
        eval_semantics(parsed, {"src": x, "hi": 6.0})

    # 2. When operands are provided, it matches the NumPy reference exactly
    sem_val = eval_semantics(parsed, {"src": x, "lo": 0.0, "hi": 6.0})
    numpy_ref = np.clip(x, 0.0, 6.0)
    assert np.allclose(sem_val, numpy_ref), f"Clamp mismatch: sem={sem_val}, np={numpy_ref}"


def test_semantics_oracle_mutation_flip_semantics_string():
    """Mutation check 1: mutate VADD semantics string to subtraction and confirm disagreement."""
    schema = load_builtin("vortex_rvgpu")
    mutated_vadd = dataclasses.replace(schema.instructions["VADD"], semantics="dst[i] = src0[i] - src1[i]")
    parsed = parse_semantics(mutated_vadd.semantics)

    x = np.array([5.0, 10.0], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)
    sref = SourceRef("arith.addf", 1, 1, "%out")
    instr = Instr(name="VADD", cost=1.0, operands={"in0": SsaRef("%x"), "in1": SsaRef("%y")}, defs=("%out",), source_ops=(sref,))
    prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x", "%y"), instrs=(instr,))
    state = MachineState.from_program(prog, {"%x": x, "%y": y})
    state._isa_name = "vortex_rvgpu"
    apply(instr, state, PrecisionPolicy())
    emu_val = state.values.get("%out")  # computes x + y = [7.0, 13.0]
    sem_val = eval_semantics(parsed, {"src0": x, "src1": y})  # computes x - y = [3.0, 7.0]

    assert not np.allclose(emu_val, sem_val), f"Semantics mutation was not caught: emu={emu_val}, sem={sem_val}"


def test_semantics_oracle_mutation_flip_emulator_handler():
    """Mutation check 2: mutate emulator handler for arith.addf to subtraction and confirm disagreement."""
    schema = load_builtin("vortex_rvgpu")
    vadd = schema.instructions["VADD"]
    parsed = parse_semantics(vadd.semantics)

    x = np.array([5.0, 10.0], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)

    orig_handler = _ELEMENTWISE["arith.addf"]
    try:
        # Mutate handler to subtraction
        _ELEMENTWISE["arith.addf"] = lambda instr, state, shape: state.resolve(instr.operand("in0")) - state.resolve(instr.operand("in1"))

        sref = SourceRef("arith.addf", 1, 1, "%out")
        instr = Instr(name="VADD", cost=1.0, operands={"in0": SsaRef("%x"), "in1": SsaRef("%y")}, defs=("%out",), source_ops=(sref,))
        prog = Program(isa_name="vortex_rvgpu", schema_version=1, kernel_name="k", total_cost=1.0, inputs=("%x", "%y"), instrs=(instr,))
        state = MachineState.from_program(prog, {"%x": x, "%y": y})
        state._isa_name = "vortex_rvgpu"
        apply(instr, state, PrecisionPolicy())
        emu_val = state.values.get("%out")  # mutated emulator computes x - y = [3.0, 7.0]
        sem_val = eval_semantics(parsed, {"src0": x, "src1": y})  # semantics computes x + y = [7.0, 13.0]

        assert not np.allclose(emu_val, sem_val), f"Emulator mutation was not caught: emu={emu_val}, sem={sem_val}"
    finally:
        _ELEMENTWISE["arith.addf"] = orig_handler
