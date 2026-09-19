"""Tests for the C++ emulator backend."""

import numpy as np
import pytest

from tritonflow.emit.ir import (
    Imm,
    Instr,
    Loop,
    MemRef,
    Program,
    SourceRef,
    SsaRef,
    UnsupportedMarker,
)
from tritonflow.emu import HAS_CPP
from tritonflow.emu.exec import emulate
from tritonflow.emu.precision import PrecisionPolicy, derive_tolerance, tf32_truncate

if not HAS_CPP:
    import unittest
    raise unittest.SkipTest("C++ backend is not installed")

from tritonflow.emu._emu_cpp import (
    PrecisionPolicy as CppPrecisionPolicy,
)
from tritonflow.emu._emu_cpp import (
    ProgramNotExecutable,
    UnsupportedInstruction,
)
from tritonflow.emu._emu_cpp import (
    derive_tolerance as cpp_derive_tolerance,
)
from tritonflow.emu._emu_cpp import (
    tf32_truncate as cpp_tf32_truncate,
)


def test_tf32_truncate():
    """Test that the C++ tf32_truncate produces exactly the same bits as the Python one."""
    # Some finite values, subnormals, and exactly halfway values to test rounding.
    vals = np.array([1.0, 1.5, 3.14159, 1e-10, 0.0, -0.0, 1.0 + 2**-11, 1.0 + 2**-12], dtype=np.float32)
    py_trunc = tf32_truncate(vals)
    cpp_trunc = cpp_tf32_truncate(vals)
    np.testing.assert_array_equal(py_trunc, cpp_trunc, strict=True)


def test_derive_tolerance():
    """Test that the C++ derived tolerance matches the Python one."""
    py_tol = derive_tolerance("tf32", 64, "f32")
    cpp_tol = cpp_derive_tolerance("tf32", 64, "f32")
    assert pytest.approx(py_tol.value) == cpp_tol.value
    assert py_tol.absolute == cpp_tol.absolute

    py_tol = derive_tolerance("ieee", 128, "f32")
    cpp_tol = cpp_derive_tolerance("ieee", 128, "f32")
    assert pytest.approx(py_tol.value) == cpp_tol.value


def test_precision_policy_mac():
    """Test C++ PrecisionPolicy.multiply_accumulate matches Python exactly."""
    a = np.random.rand(16, 32).astype(np.float32)
    b = np.random.rand(32, 16).astype(np.float32)
    acc = np.zeros((16, 16), dtype=np.float32)

    py_policy = PrecisionPolicy("tf32", "k_major_sequential", 32)
    cpp_policy = CppPrecisionPolicy("tf32", "k_major_sequential", 32)

    py_acc = py_policy.multiply_accumulate(a, b, acc.copy())
    cpp_acc = cpp_policy.multiply_accumulate(a, b, acc.copy())

    np.testing.assert_array_equal(py_acc, cpp_acc, strict=True)


def test_elementwise_addi():
    """Test a basic elementwise operation in the C++ emulator."""
    # Minimal program: arith.addi on two scalar inputs, then store
    prog = Program(
        isa_name="tritonflow1",
        schema_version=1,
        inputs=("A", "B", "Out"),
        instrs=(
            Instr(
                name="EPI_ADD",
                operands={"in1": SsaRef("A"), "in2": SsaRef("B")},
                defs=("C",),
                source_ops=(SourceRef(op_name="arith.addi"),),
            ),
            Instr(
                name="DMA1D",
                operands={"dst": MemRef(space="global", base="Out"), "value": SsaRef("C")},
                source_ops=(SourceRef(op_name="tt.store"),),
            ),
        ),
    )

    inputs = {
        "A": np.array([5], dtype=np.float32),
        "B": np.array([7], dtype=np.float32),
        "Out": np.array([0], dtype=np.float32),
    }

    # Run C++ emulate
    outputs = emulate(prog, inputs, use_cpp=True)
    assert "Out" in outputs
    np.testing.assert_array_equal(outputs["Out"], np.array([12], dtype=np.float32))

    # Run Python emulate to verify parity
    outputs_py = emulate(prog, inputs, use_cpp=False)
    np.testing.assert_array_equal(outputs["Out"], outputs_py["Out"])


def test_loop():
    """Test loop execution in the C++ emulator."""
    prog = Program(
        isa_name="tritonflow1",
        schema_version=1,
        inputs=("lower", "upper", "step", "init", "Out"),
        loops=(
            Loop(
                id=0,
                induction_var="i",
                lower=SsaRef("lower"),
                upper=SsaRef("upper"),
                step=SsaRef("step"),
                iter_args=("acc",),
                inits=("init",),
                yields=("acc_next",),
                results=("final_acc",),
                body=(
                    Instr(
                        name="EPI_ADD",
                        operands={"in1": SsaRef("acc"), "in2": Imm(1)},
                        defs=("acc_next",),
                        loop=0,
                        source_ops=(SourceRef(op_name="arith.addi"),),
                    ),
                ),
            ),
        ),
        epilogue=(
            Instr(
                name="DMA1D",
                operands={"dst": MemRef(space="global", base="Out"), "value": SsaRef("final_acc")},
                source_ops=(SourceRef(op_name="tt.store"),),
            ),
        )
    )
    # Provide the loop bounds via inputs (these would normally be SSA values)
    inputs = {
        "lower": np.array([0], dtype=np.float32),
        "upper": np.array([5], dtype=np.float32),
        "step": np.array([1], dtype=np.float32),
        "init": np.array([10], dtype=np.float32),
        "Out": np.array([0], dtype=np.float32),
    }

    outputs = emulate(prog, inputs, use_cpp=True)
    assert "Out" in outputs
    np.testing.assert_array_equal(outputs["Out"], np.array([15], dtype=np.float32))


def test_program_id():
    """Test get_program_id with grid parameters in the C++ emulator."""
    prog = Program(
        isa_name="tritonflow1",
        schema_version=1,
        inputs=("OutX", "OutY"),
        instrs=(
            Instr(
                name="EPI_ADD",
                operands={"value": Imm(0)},
                defs=("PIDX",),
                source_ops=(SourceRef(op_name="tt.get_program_id", line=1),),
            ),
            Instr(
                name="EPI_ADD",
                operands={"value": Imm(1)},
                defs=("PIDY",),
                source_ops=(SourceRef(op_name="tt.get_program_id", line=2),),
            ),
            Instr(
                name="DMA1D",
                operands={"dst": MemRef(space="global", base="OutX"), "value": SsaRef("PIDX")},
                source_ops=(SourceRef(op_name="tt.store", line=3),),
            ),
            Instr(
                name="DMA1D",
                operands={"dst": MemRef(space="global", base="OutY"), "value": SsaRef("PIDY")},
                source_ops=(SourceRef(op_name="tt.store", line=4),),
            ),
        ),
    )
    inputs = {
        "OutX": np.array([0], dtype=np.float32),
        "OutY": np.array([0], dtype=np.float32),
    }

    # Run with a specific grid
    outputs = emulate(prog, inputs, grid=(42, 17, 0), use_cpp=True)
    assert "OutX" in outputs
    assert "OutY" in outputs
    np.testing.assert_array_equal(outputs["OutX"], np.array([42], dtype=np.float32))
    np.testing.assert_array_equal(outputs["OutY"], np.array([17], dtype=np.float32))

    # Python parity
    outputs_py = emulate(prog, inputs, grid=(42, 17, 0), use_cpp=False)
    np.testing.assert_array_equal(outputs["OutX"], outputs_py["OutX"])
    np.testing.assert_array_equal(outputs["OutY"], outputs_py["OutY"])


def test_exceptions():
    """Test that unsupported ops raise UnsupportedInstruction, and markers raise ProgramNotExecutable."""
    prog_invalid_op = Program(
        isa_name="tritonflow1",
        schema_version=1,
        inputs=("Out",),
        instrs=(
            Instr(
                name="EPI_ADD",
                operands={"value": Imm(0)},
                defs=("Invalid",),
                source_ops=(SourceRef(op_name="tt.invalid_op_does_not_exist"),),
            ),
        ),
    )
    inputs = {"Out": np.array([0], dtype=np.float32)}

    # Invalid op raises UnsupportedInstruction (not implemented by machine)
    with pytest.raises(UnsupportedInstruction, match="tt.invalid_op_does_not_exist"):
        emulate(prog_invalid_op, inputs, use_cpp=True)

    prog_with_marker = Program(
        isa_name="tritonflow1",
        schema_version=1,
        inputs=("Out",),
        instrs=(),
        unsupported=(
            UnsupportedMarker(
                op_name="tt.something_hard",
                reason="not supported",
            ),
        )
    )

    # UNSUPPORTED marker raises ProgramNotExecutable
    with pytest.raises(ProgramNotExecutable, match="tt.something_hard"):
        emulate(prog_with_marker, inputs, use_cpp=True)

