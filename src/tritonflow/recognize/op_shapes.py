"""Op-shape tables and predicates: which operations the walk understands, and how.

This module is deliberately *data plus total predicates*, with no traversal and no
descriptor type. It exists so that the two failure modes the recogniser has can be
told apart by name:

* an operation the walk **understands** — its shape is a rule about operands, and
  the walk can fold it into an access descriptor;
* an operation that makes the access **non-affine** — the walk knows the name and
  refuses by naming it, rather than falling through a `default: pass` and
  producing a plausible-looking offset.

That second case is the one v1 got wrong. An index expression whose op is not
recognised must produce `Unstructured(reason="non-affine index: <op>")`, not an
`Ok` descriptor missing a term — a missing term is a wrong address, and a wrong
address is a wrong answer with no marker on it.

**Nothing here imports Triton, and nothing here is a runtime dependency of it.**
The tables are keyed by op *name* strings, which is what `ttir` text gives us.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ttir.ssa import Operation, SsaValue
from ..ttir.types import TypeExpr

# --------------------------------------------------------------------------- #
# Op names
# --------------------------------------------------------------------------- #

#: The pointer-arithmetic operation. Either `base` (operand 0) is a splat of a
#: pointer, or recursion continues through it (a chain of `tt.addptr`s).
ADDPTR = "tt.addptr"

#: Rank manipulation that leaves the *value* alone and only changes the shape of
#: the container. All three are transparent to the per-dimension coefficients.
SPLAT = "tt.splat"
BROADCAST = "tt.broadcast"
EXPAND_DIMS = "tt.expand_dims"

#: The one operation that introduces an *index variable*: `make_range` produces
#: `start … end-1`, so dimension 0 of its result varies with coefficient 1.
MAKE_RANGE = "tt.make_range"

#: Integer arithmetic on index expressions.
MULI = "arith.muli"
ADDI = "arith.addi"
SUBI = "arith.subi"
REMSI = "arith.remsi"
CONSTANT = "arith.constant"

#: Memory operations and the reduction.
LOAD = "tt.load"
STORE = "tt.store"
DOT = "tt.dot"

#: Region terminators that are not lowerable instructions.
YIELD = "scf.yield"
RETURN = "tt.return"

#: Gather/scatter and friends: an access whose index is a *loaded* value rather
#: than an affine expression of index variables. Named here so the refusal says
#: which operation caused it instead of `unknown operation`.
NON_AFFINE_OPS = (
    "tt.gather",
    "tt.scatter",
    "tt.histogram",
    "tt.atomic_rmw",
    "tt.atomic_cas",
)

#: Operations the index walk folds. Anything outside this set inside an index
#: expression is a refusal.
INDEX_OPS = (
    ADDPTR,
    SPLAT,
    BROADCAST,
    EXPAND_DIMS,
    MAKE_RANGE,
    MULI,
    ADDI,
    SUBI,
    REMSI,
    CONSTANT,
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.index_castui",
)

#: Rank-preserving casts that are transparent to the coefficients.
CAST_OPS = ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast", "arith.index_castui")

#: Memory operations, in the order the recogniser reports them.
MEMORY_OPS = (LOAD, STORE)

#: Bare (non-tensor) memory operations: they take a `!tt.ptr<…>` directly rather
#: than a tensor of pointers, so the descriptor is a *scalar* access.
SCALAR_MEMORY_OPS = ("tt.load", "tt.store")

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #

#: `tt.dot`'s operand roles, positionally. `tt.dot %a, %b, %acc`.
DOT_ROLES = ("a", "b", "acc")

#: `tt.dot`'s precision attribute and its two legal values. Absent ⇒ `ieee`.
INPUT_PRECISION_ATTR = "inputPrecision"
INPUT_PRECISIONS = ("tf32", "ieee")

#: `expand_dims`'s axis attribute.
AXIS_ATTR = "axis"

# --------------------------------------------------------------------------- #
# Rule names
#
# These are the keys a rule module (`isa/rules/tritonflow1.py`) maps to candidate
# instruction sets. They are *not* the coarse `kind: memory|compute` axis of the
# schema document: "is this a MAC or an epilogue" is an α-attribute question, and
# a selector that enumerated both for a dot would be answering a different
# question. A stand-in selector uses these same three names,
# which is what lets the real recogniser be dropped into the emitter checks
# without touching them.
# --------------------------------------------------------------------------- #

RULE_MEMORY = "memory"
RULE_MAC = "mac"
RULE_ELEMENTWISE = "elementwise"


# --------------------------------------------------------------------------- #
# Predicates — total, no exceptions
# --------------------------------------------------------------------------- #


def is_memory_op(op: Operation) -> bool:
    return op.name in MEMORY_OPS


def is_dot(op: Operation) -> bool:
    return op.name == DOT


def is_index_op(op: Operation) -> bool:
    """Whether the index walk has a rule for this operation."""
    return op.name in INDEX_OPS


def is_transparent(op: Operation) -> bool:
    """Rank manipulation or a cast: it cannot change a coefficient's value."""
    return op.name in (SPLAT, BROADCAST, EXPAND_DIMS, *CAST_OPS)


def is_non_affine(op: Operation) -> bool:
    return op.name in NON_AFFINE_OPS


def pointer_operand(op: Operation) -> SsaValue | None:
    """The pointer a memory operation addresses. Operand 0 for load and store.

    Both `tt.load %p` and `tt.store %p, %v` put the pointer first, which is the
    printer's convention and not a discovery; naming it here means the walk does
    not silently take `operands[1]` if that ever changes.
    """
    if op.name not in MEMORY_OPS:
        return None
    return op.operands[0] if op.operands else None


def value_operand(op: Operation) -> SsaValue | None:
    """The value a `tt.store` writes. `None` for `tt.load`."""
    if op.name != STORE:
        return None
    return op.operands[1] if len(op.operands) > 1 else None


def base_operand(op: Operation) -> SsaValue | None:
    """A `tt.addptr`'s base. `None` for any other operation."""
    if op.name != ADDPTR:
        return None
    return op.operands[0] if op.operands else None


def index_operand(op: Operation) -> SsaValue | None:
    """A `tt.addptr`'s index tensor. `None` for any other operation."""
    if op.name != ADDPTR:
        return None
    return op.operands[1] if len(op.operands) > 1 else None


def expand_axis(op: Operation) -> int | None:
    """`tt.expand_dims`'s axis, or `None` when absent or not an integer.

    `None` is a *refusal signal*, not a default: guessing axis 0 for a missing
    attribute would transpose the access silently.
    """
    text = op.attr.get(AXIS_ATTR)
    if text is None:
        return None
    digits = "".join(ch for ch in text.split(":")[0] if ch.isdigit() or ch == "-")
    try:
        return int(digits)
    except ValueError:
        return None


def input_precision(op: Operation) -> str:
    """`tt.dot`'s input precision, defaulting to `ieee` when unstated.

    `ieee` is the honest default: `tf32` is a *reduced* precision, and assuming
    it would make the emulator's tolerance right for the wrong reason.
    """
    text = op.attr.get(INPUT_PRECISION_ATTR)
    return text if text in INPUT_PRECISIONS else "ieee"


def shape_of(type_: TypeExpr | None) -> tuple[int | None, ...]:
    """The shape of a tensor type; `()` for a scalar or a missing type."""
    if type_ is None or type_.kind != "tensor":
        return ()
    return tuple(type_.shape)


def scalar_dtype(type_: TypeExpr | None) -> str:
    """The innermost dtype, so both `f32` and `tensor<…x!tt.ptr<f32>>` give `f32`."""
    if type_ is None:
        return ""
    return type_.scalar_dtype


# --------------------------------------------------------------------------- #
# Constant decoding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Constant:
    """A decoded `arith.constant`, or the reason it could not be decoded.

    `value is None` with a non-empty `reason` is the honest form of "I saw a
    constant and could not read it" — as opposed to `0`, which is a value and
    would be folded into an address.
    """

    value: int | float | None = None
    reason: str = ""
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int)


def decode_constant(op: Operation) -> Constant:
    """Decode an `arith.constant`'s literal.

    Handles the two forms the corpus and the printer produce: a bare literal
    (`arith.constant 1024 : i32`, `arith.constant 0.000000e+00 : f32`) and a
    splat literal attribute (`arith.constant dense<2> : tensor<16xi32>`). A
    non-splat `dense<…>` vector is **refused** rather than reduced to its first
    element: a vector constant is not a scalar, and picking one lane of it
    because the rest were inconvenient is exactly the class of silent error this
    module's refusals exist to prevent.
    """
    if op.name != CONSTANT:
        return Constant(reason=f"not a constant: {op.name}")
    # Two places the literal can be, and both occur in the corpus text the
    # parser was given. `arith.constant 1024 : i32` puts it in the operand token
    # list (there is no `value =` in the generic form the printer emits for a
    # scalar), while `attributes {value = 1024 : i32}` is the form the parser
    # records when the op *is* written generically. Reading only one of the two
    # silently decoded nothing, so — before this was fixed — every constant in
    # the corpus looked undecodable and every descriptor was refused.
    text = op.attr.get("value", "").strip()
    if not text:
        literals = op.literal_tokens
        text = literals[0].strip() if literals else ""
    if not text:
        return Constant(reason="arith.constant with no value attribute or literal token", text=text)
    inner = text
    if text.startswith("dense<") and text.endswith(">"):
        inner = text[len("dense<") : -1].strip()
        if "," in inner:
            return Constant(reason=f"vector constant {text!r} is not a scalar", text=text)
    if _looks_float(inner):
        try:
            return Constant(value=float(inner), text=text)
        except ValueError:  # pragma: no cover - guarded by _looks_float
            return Constant(reason=f"unreadable float literal {inner!r}", text=text)
    try:
        return Constant(value=int(inner), text=text)
    except ValueError:
        return Constant(reason=f"unreadable literal {inner!r}", text=text)


def _looks_float(text: str) -> bool:
    lowered = text.lower().lstrip("+-")
    return any(marker in lowered for marker in (".", "e", "inf", "nan"))


__all__ = [
    "ADDI",
    "ADDPTR",
    "AXIS_ATTR",
    "BROADCAST",
    "CAST_OPS",
    "CONSTANT",
    "DOT",
    "DOT_ROLES",
    "EXPAND_DIMS",
    "INDEX_OPS",
    "INPUT_PRECISIONS",
    "INPUT_PRECISION_ATTR",
    "LOAD",
    "MAKE_RANGE",
    "MEMORY_OPS",
    "MULI",
    "NON_AFFINE_OPS",
    "REMSI",
    "RETURN",
    "RULE_ELEMENTWISE",
    "RULE_MAC",
    "RULE_MEMORY",
    "SCALAR_MEMORY_OPS",
    "SPLAT",
    "STORE",
    "SUBI",
    "YIELD",
    "Constant",
    "base_operand",
    "decode_constant",
    "expand_axis",
    "index_operand",
    "input_precision",
    "is_dot",
    "is_index_op",
    "is_memory_op",
    "is_non_affine",
    "is_transparent",
    "pointer_operand",
    "scalar_dtype",
    "shape_of",
    "value_operand",
]
