"""MLIR `arith.cmpi` / `arith.cmpf` predicates: names, integer encodings, and their meaning.

The emitted program can only carry an integer immediate, so a compare's predicate travels
as the MLIR enum value (`arith::CmpIPredicate`, `arith::CmpFPredicate`). The two enums
overlap numerically (2 is `slt` for cmpi and `ogt` for cmpf), so the operation name, not
the number, says which table applies. The C++ emulator carries its own copy of these
tables (`machine.cpp`); `tests/test_schema_op_matrix.py` checks both against independent
NumPy expressions.

Integer compares treat unsigned predicates as 32-bit two's complement: Triton index math
is i32 and the emulators do not carry a bit width. An operand outside the 32-bit range is
refused instead of being compared as if it were wider or narrower.
"""

from __future__ import annotations

import numpy as np

CMPI_NAMES: dict[str, int] = {
    "eq": 0, "ne": 1, "slt": 2, "sle": 3, "sgt": 4, "sge": 5,
    "ult": 6, "ule": 7, "ugt": 8, "uge": 9,
}

CMPF_NAMES: dict[str, int] = {
    "false": 0, "oeq": 1, "ogt": 2, "oge": 3, "olt": 4, "ole": 5, "one": 6, "ord": 7,
    "ueq": 8, "ugt": 9, "uge": 10, "ult": 11, "ule": 12, "une": 13, "uno": 14, "true": 15,
}

_TABLES = {"arith.cmpi": CMPI_NAMES, "arith.cmpf": CMPF_NAMES}

COMPARE_OPS = tuple(_TABLES)


class PredicateError(ValueError):
    """A compare predicate that is not one MLIR defines for that operation."""


def predicate_code(op_name: str, token: str | None) -> int:
    """`("arith.cmpi", "slt")` -> 2. Raises `PredicateError` for anything MLIR does not define."""
    table = _TABLES.get(op_name)
    if table is None:
        raise PredicateError(f"{op_name!r} is not a compare operation")
    if token is None or token.strip().rstrip(",").lower() not in table:
        raise PredicateError(
            f"unparseable {op_name} predicate {token!r}; MLIR defines {sorted(table)}"
        )
    return table[token.strip().rstrip(",").lower()]


def _u32(values: np.ndarray) -> np.ndarray:
    wide = np.asarray(values).astype(np.int64)
    if wide.size and (wide.min() < -(2**31) or wide.max() > 2**32 - 1):
        raise PredicateError("unsigned compare operand does not fit 32 bits")
    return (wide & 0xFFFFFFFF).astype(np.uint32)


def evaluate_cmpi(code: int, left, right) -> np.ndarray:
    a = np.asarray(left)
    b = np.asarray(right)
    if code in (6, 7, 8, 9):
        a, b = _u32(a), _u32(b)
    if code == 0:
        return a == b
    if code == 1:
        return a != b
    if code in (2, 6):
        return a < b
    if code in (3, 7):
        return a <= b
    if code in (4, 8):
        return a > b
    if code in (5, 9):
        return a >= b
    raise PredicateError(f"arith.cmpi predicate code {code} is not defined")


def evaluate_cmpf(code: int, left, right) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    unordered = np.isnan(a) | np.isnan(b)
    ordered = ~unordered
    if code == 0:
        return np.zeros(np.broadcast(a, b).shape, dtype=bool)
    if code == 15:
        return np.ones(np.broadcast(a, b).shape, dtype=bool)
    if code == 7:
        return ordered
    if code == 14:
        return unordered
    ops = {
        1: lambda: a == b, 2: lambda: a > b, 3: lambda: a >= b, 4: lambda: a < b,
        5: lambda: a <= b, 6: lambda: a != b,
        8: lambda: a == b, 9: lambda: a > b, 10: lambda: a >= b, 11: lambda: a < b,
        12: lambda: a <= b, 13: lambda: a != b,
    }
    if code not in ops:
        raise PredicateError(f"arith.cmpf predicate code {code} is not defined")
    core = ops[code]()
    return (ordered & core) if code <= 6 else (unordered | core)


def evaluate(op_name: str, code: int, left, right) -> np.ndarray:
    if op_name == "arith.cmpi":
        return evaluate_cmpi(code, left, right)
    if op_name == "arith.cmpf":
        return evaluate_cmpf(code, left, right)
    raise PredicateError(f"{op_name!r} is not a compare operation")
