"""Test for every schema instruction and every op it claims.

Runs programs in Python emulator, C++ emulator, and compares with NumPy oracle.
Also includes a mutation check confirming that altering an emulator handler causes a test failure.
"""

import numpy as np
import pytest

from tritonflow.emit.ir import Imm, Instr, MemRef, Program, SourceRef, SsaRef
from tritonflow.emu.exec import emulate
from tritonflow.isa.schema import load_builtin


def _get_schema_op_cases():
    cases = []
    for isa_name in ['tritonflow1', 'tritonflow2', 'vortex_rvgpu']:
        isa = load_builtin(isa_name)
        for iname, instr in isa.instructions.items():
            if not instr.ops:
                continue
            for op in instr.ops:
                cases.append((isa_name, iname, op))
    return cases


SCHEMA_OP_CASES = _get_schema_op_cases()


def _build_and_run(isa_name: str, iname: str, op: str, use_cpp: bool):
    store_name = 'DMA1D' if isa_name == 'tritonflow1' else 'STG'
    if op in ('addf', 'subf', 'mulf', 'divf'):
        A = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        B = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
        expected = {'addf': A + B, 'subf': A - B, 'mulf': A * B, 'divf': A / B}[op]
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'B', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'B': B, 'Out': np.zeros_like(expected)}
    elif op in ('addi', 'subi', 'muli', 'divsi', 'divui', 'remsi', 'remui'):
        A = np.array([10, 20, 31, 40], dtype=np.int32)
        B = np.array([3, 4, 5, 8], dtype=np.int32)
        if op == 'addi':
            expected = A + B
        elif op == 'subi':
            expected = A - B
        elif op == 'muli':
            expected = A * B
        elif op == 'divsi':
            expected = np.trunc(A.astype(np.float64) / B).astype(np.int64)
        elif op == 'divui':
            expected = (A.astype(np.uint64) // B.astype(np.uint64)).astype(np.int64)
        elif op == 'remsi':
            expected = A - np.trunc(A.astype(np.float64) / B).astype(np.int64) * B
        elif op == 'remui':
            expected = (A.astype(np.uint64) % B.astype(np.uint64)).astype(np.int64)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'B', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'B': B, 'Out': np.zeros(4, dtype=np.int64)}
    elif op in ('negf', 'absf'):
        A = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float32)
        expected = -A if op == 'negf' else np.abs(A)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}' if op=='negf' else f'math.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros_like(expected)}
    elif op in ('maxnumf', 'minnumf'):
        A = np.array([1.0, 5.0, 2.0, 8.0], dtype=np.float32)
        B = np.array([3.0, 2.0, 7.0, 4.0], dtype=np.float32)
        expected = np.maximum(A, B) if op == 'maxnumf' else np.minimum(A, B)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'B', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'B': B, 'Out': np.zeros_like(expected)}
    elif op in ('maxsi', 'minsi'):
        A = np.array([10, -5, 30, 15], dtype=np.int32)
        B = np.array([20, 0, 15, 25], dtype=np.int32)
        expected = np.maximum(A, B) if op == 'maxsi' else np.minimum(A, B)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'B', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'B': B, 'Out': np.zeros(4, dtype=np.int64)}
    elif op in ('maxui', 'minui'):
        A = np.array([10, -5, 30, 15], dtype=np.int32)
        B = np.array([20, 0, 15, 25], dtype=np.int32)
        expected = np.maximum(A.astype(np.uint64), B.astype(np.uint64)).astype(np.int64) if op == 'maxui' else np.minimum(A.astype(np.uint64), B.astype(np.uint64)).astype(np.int64)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'B', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'B': B, 'Out': np.zeros(4, dtype=np.int64)}
    elif op == 'clamp':
        A = np.array([-1.0, 0.5, 2.0, 0.2], dtype=np.float32)
        lo = 0.0
        hi = 1.0
        expected = np.clip(A, lo, hi)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A'), 'lo': Imm(lo), 'hi': Imm(hi)}, defs=('C',), source_ops=(SourceRef(op_name='tt.clamp'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros_like(expected)}
    elif op == 'constant':
        expected = np.full(4, 3.14, dtype=np.float32)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('Out',),
            instrs=(
                Instr(name=iname, operands={'value': Imm(3.14)}, defs=('C',), source_ops=(SourceRef(op_name='arith.constant'),), constrained_on='sizes=[4]'),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'Out': np.zeros(4, dtype=np.float32)}
    elif op == 'expand_dims':
        A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.expand_dims(A, 0)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A')}, defs=('C',), source_ops=(SourceRef(op_name='tt.expand_dims'),), constrained_on='sizes=[1, 4]'),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros((1, 4), dtype=np.float32)}
    elif op == 'broadcast':
        A = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        expected = np.broadcast_to(A, (2, 4))
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A')}, defs=('C',), source_ops=(SourceRef(op_name='tt.broadcast'),), constrained_on='sizes=[2, 4]'),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros((2, 4), dtype=np.float32)}
    elif op == 'splat':
        A = np.array([3.14], dtype=np.float32)
        expected = np.full((4,), 3.14, dtype=np.float32)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A')}, defs=('C',), source_ops=(SourceRef(op_name='tt.splat'),), constrained_on='sizes=[4]'),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros(4, dtype=np.float32)}
    elif op == 'make_range':
        expected = np.arange(4, dtype=np.int64)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('Out',),
            instrs=(
                Instr(name=iname, operands={'value': Imm(4)}, defs=('C',), source_ops=(SourceRef(op_name='tt.make_range'),), constrained_on='sizes=[4]'),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'Out': np.zeros(4, dtype=np.int64)}
    elif op == 'get_program_id':
        expected = np.zeros(1, dtype=np.int64)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('Out',),
            instrs=(
                Instr(name=iname, operands={'value': Imm(0)}, defs=('C',), source_ops=(SourceRef(op_name='tt.get_program_id'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'Out': np.zeros(1, dtype=np.int64)}
    elif op in ('cmpi', 'cmpf'):
        # The predicate travels as an integer operand carrying the MLIR enum value.
        # This entry pins one representative predicate so the matrix covers the
        # instruction; test_compare_every_predicate below exercises all of them.
        return _build_and_run_compare(isa_name, iname, op, 'ne' if op == 'cmpi' else 'one', use_cpp)
    elif op in ('extf', 'truncf', 'sitofp', 'fptosi', 'extsi', 'trunci'):
        # Independent oracles: numpy dtype casts, not the emulator's own helpers.
        if op == 'extf':
            src = np.array([1.5, -2.25, 0.0, 1024.0], dtype=np.float16)
            A = src.astype(np.float32)
            expected = src.astype(np.float32)
        elif op == 'truncf':
            # 1.0009765625 and 2048.5 are not representable in f16 and must round.
            A = np.array([1.0009765625, 2048.5, -0.30000001192, 6.0e-8], dtype=np.float32)
            expected = A.astype(np.float16).astype(np.float32)
        elif op == 'sitofp':
            A = np.array([-7, 0, 13, 4096], dtype=np.int32)
            expected = A.astype(np.float32)
        elif op == 'fptosi':
            # MLIR fptosi truncates toward zero, so -2.7 -> -2 (not -3).
            A = np.array([-2.7, 2.7, 0.5, -0.5], dtype=np.float32)
            expected = np.trunc(A).astype(np.int32)
        elif op == 'extsi':
            A = np.array([-5, 0, 17, 65535], dtype=np.int32)
            expected = A.astype(np.int64)
        else:  # trunci
            A = np.array([-5, 0, 17, 65535], dtype=np.int64)
            expected = A.astype(np.int32)
        p = Program(
            isa_name=isa_name,
            schema_version=1,
            inputs=('A', 'Out'),
            instrs=(
                Instr(name=iname, operands={'in1': SsaRef('A')}, defs=('C',), source_ops=(SourceRef(op_name=f'arith.{op}'),)),
                Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),))
            )
        )
        inputs = {'A': A, 'Out': np.zeros(A.shape, dtype=expected.dtype)}
    else:
        raise ValueError(f'Unknown op {op}')

    actual = emulate(p, {k: v.copy() for k, v in inputs.items()}, use_cpp=use_cpp)['Out']
    return actual, expected


# --- compares ------------------------------------------------------------- #
# Oracles below are written as plain numpy expressions on purpose. They must NOT call
# tritonflow.isa.predicates, or the test would be checking the implementation against
# itself -- which is exactly how the original "cmpi always computes slt" bug survived.

_CMPI_ORACLE = {
    'eq':  lambda a, b: a == b,
    'ne':  lambda a, b: a != b,
    'slt': lambda a, b: a < b,
    'sle': lambda a, b: a <= b,
    'sgt': lambda a, b: a > b,
    'sge': lambda a, b: a >= b,
    'ult': lambda a, b: _u32(a) < _u32(b),
    'ule': lambda a, b: _u32(a) <= _u32(b),
    'ugt': lambda a, b: _u32(a) > _u32(b),
    'uge': lambda a, b: _u32(a) >= _u32(b),
}

_CMPF_ORACLE = {
    'false': lambda a, b: np.zeros(a.shape, dtype=bool),
    'true':  lambda a, b: np.ones(a.shape, dtype=bool),
    'ord':   lambda a, b: ~(np.isnan(a) | np.isnan(b)),
    'uno':   lambda a, b: np.isnan(a) | np.isnan(b),
    'oeq':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a == b),
    'ogt':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a > b),
    'oge':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a >= b),
    'olt':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a < b),
    'ole':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a <= b),
    'one':   lambda a, b: ~(np.isnan(a) | np.isnan(b)) & (a != b),
    'ueq':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a == b),
    'ugt':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a > b),
    'uge':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a >= b),
    'ult':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a < b),
    'ule':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a <= b),
    'une':   lambda a, b: (np.isnan(a) | np.isnan(b)) | (a != b),
}

#: MLIR enum values. Duplicated here rather than imported, so a change to the
#: production table cannot silently change what the test expects.
_CMPI_CODES = {'eq': 0, 'ne': 1, 'slt': 2, 'sle': 3, 'sgt': 4, 'sge': 5,
               'ult': 6, 'ule': 7, 'ugt': 8, 'uge': 9}
_CMPF_CODES = {'false': 0, 'oeq': 1, 'ogt': 2, 'oge': 3, 'olt': 4, 'ole': 5, 'one': 6,
               'ord': 7, 'ueq': 8, 'ugt': 9, 'uge': 10, 'ult': 11, 'ule': 12, 'une': 13,
               'uno': 14, 'true': 15}


def _u32(values):
    """Reinterpret int32 as unsigned, the way an unsigned MLIR compare does."""
    return np.asarray(values).astype(np.int64) & 0xFFFFFFFF


def _compare_inputs(op: str):
    """Asymmetric operands: negatives, equal values, and (for cmpf) a NaN."""
    if op == 'cmpi':
        # -1 as unsigned is 0xFFFFFFFF, the largest u32: this separates signed
        # from unsigned predicates, which a non-negative corpus would not.
        a = np.array([1, 10, 5, -1, -7, 0], dtype=np.int32)
        b = np.array([5, 5, 5, 1, -7, 0], dtype=np.int32)
        return a, b
    a = np.array([1.0, 10.0, 5.0, -3.5, np.nan, 0.0], dtype=np.float32)
    b = np.array([5.0, 5.0, 5.0, -3.5, 1.0, np.nan], dtype=np.float32)
    return a, b


def _build_and_run_compare(isa_name: str, iname: str, op: str, predicate: str, use_cpp: bool):
    store_name = 'DMA1D' if isa_name == 'tritonflow1' else 'STG'
    a, b = _compare_inputs(op)
    codes = _CMPI_CODES if op == 'cmpi' else _CMPF_CODES
    oracle = _CMPI_ORACLE if op == 'cmpi' else _CMPF_ORACLE
    expected = oracle[predicate](a, b).astype(np.float32)
    p = Program(
        isa_name=isa_name,
        schema_version=1,
        inputs=('A', 'B', 'Out'),
        instrs=(
            Instr(
                name=iname,
                operands={'in1': SsaRef('A'), 'in2': SsaRef('B'), 'predicate': Imm(codes[predicate])},
                defs=('C',),
                source_ops=(SourceRef(op_name=f'arith.{op}'),),
            ),
            Instr(name=store_name, operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),)),
        ),
    )
    inputs = {'A': a, 'B': b, 'Out': np.zeros(a.shape, dtype=np.float32)}
    actual = emulate(p, {k: v.copy() for k, v in inputs.items()}, use_cpp=use_cpp)['Out']
    return actual, expected


def _compare_instructions():
    out = []
    for isa_name, iname, op in SCHEMA_OP_CASES:
        if op in ('cmpi', 'cmpf'):
            names = _CMPI_CODES if op == 'cmpi' else _CMPF_CODES
            out.extend((isa_name, iname, op, pred) for pred in names)
    return out


COMPARE_CASES = _compare_instructions()


@pytest.mark.parametrize('isa_name,iname,op,predicate', COMPARE_CASES)
def test_compare_every_predicate(isa_name: str, iname: str, op: str, predicate: str):
    """Every MLIR predicate, on both emulators, against an independent numpy oracle."""
    res_py, expected = _build_and_run_compare(isa_name, iname, op, predicate, use_cpp=False)
    res_cpp, _ = _build_and_run_compare(isa_name, iname, op, predicate, use_cpp=True)
    np.testing.assert_array_equal(res_py, expected, err_msg=f'{isa_name}:{iname}:{op}:{predicate} Python')
    np.testing.assert_array_equal(res_cpp, expected, err_msg=f'{isa_name}:{iname}:{op}:{predicate} C++')


def test_compare_without_a_predicate_is_refused():
    """A compare carrying no predicate must be refused, never defaulted to slt."""
    # The two emulators raise distinct exception classes for the same refusal
    # (the C++ one is defined by the pybind11 module), so accept either.
    from tritonflow.emu._emu_cpp import UnsupportedInstruction as CppUnsupported
    from tritonflow.emu.exec import UnsupportedInstruction
    a = np.array([1, 2], dtype=np.int32)
    p = Program(
        isa_name='tritonflow1',
        schema_version=1,
        inputs=('A', 'B', 'Out'),
        instrs=(
            Instr(name='EPI_CMP', operands={'in1': SsaRef('A'), 'in2': SsaRef('B')}, defs=('C',), source_ops=(SourceRef(op_name='arith.cmpi'),)),
            Instr(name='DMA1D', operands={'dst': MemRef('global', 'Out'), 'value': SsaRef('C')}, source_ops=(SourceRef(op_name='tt.store'),)),
        ),
    )
    inputs = {'A': a, 'B': a, 'Out': np.zeros(2, dtype=np.float32)}
    for use_cpp in (False, True):
        with pytest.raises((UnsupportedInstruction, CppUnsupported), match='no predicate'):
            emulate(p, {k: v.copy() for k, v in inputs.items()}, use_cpp=use_cpp)


def test_signed_and_unsigned_compares_actually_differ():
    """Guards the oracle itself: if these agreed, the unsigned cases would prove nothing."""
    slt, _ = _build_and_run_compare('tritonflow1', 'EPI_CMP', 'cmpi', 'slt', use_cpp=False)
    ult, _ = _build_and_run_compare('tritonflow1', 'EPI_CMP', 'cmpi', 'ult', use_cpp=False)
    assert not np.array_equal(slt, ult)


@pytest.mark.parametrize('isa_name,iname,op', SCHEMA_OP_CASES)
def test_schema_instruction_op(isa_name: str, iname: str, op: str):
    res_py, expected = _build_and_run(isa_name, iname, op, use_cpp=False)
    res_cpp, _ = _build_and_run(isa_name, iname, op, use_cpp=True)
    np.testing.assert_allclose(res_py, expected, rtol=1e-5, atol=1e-5, err_msg=f'{isa_name}:{iname}:{op} Python mismatch')
    np.testing.assert_allclose(res_cpp, expected, rtol=1e-5, atol=1e-5, err_msg=f'{isa_name}:{iname}:{op} C++ mismatch')


def test_schema_op_mutation():
    """Mutation check: corrupting an emulator handler must fail the comparison."""
    import tritonflow.emu.exec as exec_module
    orig_addf = exec_module._ELEMENTWISE.get('arith.addf')
    try:
        # Corrupt handler to compute subtraction instead of addition
        exec_module._ELEMENTWISE['arith.addf'] = lambda instr, state, shape: np.asarray(state.resolve(instr.operands['in1'])) - np.asarray(state.resolve(instr.operands['in2']))
        with pytest.raises(AssertionError):
            res_py, expected = _build_and_run('tritonflow1', 'EPI_ADD', 'addf', use_cpp=False)
            np.testing.assert_allclose(res_py, expected, rtol=1e-5, atol=1e-5)
    finally:
        if orig_addf is not None:
            exec_module._ELEMENTWISE['arith.addf'] = orig_addf
