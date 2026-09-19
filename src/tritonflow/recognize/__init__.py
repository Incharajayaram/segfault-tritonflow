"""recognize: structure is recognised, not reconstructed.

Turns the def-use graph into `AccessDescriptor`s with a bounded, budgeted walk,
and reports `Unstructured` rather than guessing.

The three modules and what belongs in each:

| Module | Contains | Why separate |
|---|---|---|
| `op_shapes.py` | op-name tables, total predicates, constant decoding | data and predicates, no traversal — so "which ops do we understand" is answerable by reading one file |
| `walk.py` | `SymExpr`, `BoundedWalker`, `MAX_HOPS`, recurrence substitution | the budget and the algebra, shared by every fold |
| `descriptor.py` | `AccessDescriptor`, `describe`, the result union, conformance | the decision and its evidence |

`resolve_operand` is re-exported here as this package's entry point (the
implementation lives in `descriptor.py`, where the result union it returns is
defined).
"""

from . import op_shapes
from .descriptor import (
    CONFORMANCE_FIELDS,
    KEY_FIELDS,
    AccessDescriptor,
    AffineSpec,
    BudgetExhausted,
    ConformanceReport,
    DescriptorResult,
    Disagreement,
    Ok,
    Unstructured,
    canonicalize,
    conformance_check,
    describe,
    describe_operation,
    resolve_operand,
)
from .walk import MAX_HOPS, BoundedWalker, SymExpr

__all__ = [
    "CONFORMANCE_FIELDS",
    "KEY_FIELDS",
    "MAX_HOPS",
    "AccessDescriptor",
    "AffineSpec",
    "BoundedWalker",
    "BudgetExhausted",
    "ConformanceReport",
    "DescriptorResult",
    "Disagreement",
    "Ok",
    "SymExpr",
    "Unstructured",
    "canonicalize",
    "conformance_check",
    "describe",
    "describe_operation",
    "op_shapes",
    "resolve_operand",
]
