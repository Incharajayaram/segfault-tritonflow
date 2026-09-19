"""The bounded walk: symbolic values, budget accounting, loop-recurrence substitution.

Three separate jobs, kept in one module because they are all about *following a
pointer back to its base* and splitting them would mean three functions that each
half-understand the other two:

1. :class:`SymExpr` — a small closed algebra for `int | SymExpr`. The descriptor's
   `sizes`/`strides`/`offsets`/`increment` are `int | SymExpr`,
   and *whether a field is an `int`* is load-bearing: an
   integer stride is a fact the schema's predicate language can decide, and a
   symbolic one is a fact it must answer `unknown` to (fail-closed).

2. :class:`BoundedWalker` — the hop budget. `MAX_HOPS` is a named constant and
   exhausting it is a *result*, not a recursion limit reached at the wrong
   moment. The budget is checked *before* the visit, so the reported `hops` is
   the count actually performed and never `limit + 1`.

3. :func:`substitute_iter_arg` — the case v1's operation list omitted. Tier 1's
   `tt.load` operands are `scf.for` iter_args, so an operand's pointer is not
   defined anywhere in the module until the recurrence is resolved: the init
   value (`%a_ptrs_14`) plus the per-iteration advance from the yielded pointer.
   `increment` is decidable only for the latter, and the contract makes returning
   `Ok` with an *unknown* increment a violation rather than a weaker answer.

**What `SymExpr` deliberately cannot do.** It is a sum of scaled symbol products
— genuinely multivariate, because Tier 1's row offset is `pid_m * 64 * sam` and
Tier 2's store offset is `pid_m * 64 * scm + pid_n * 64 * scn`, both of which are
ordinary. What it cannot express is a product of two *index variables*: that is
the definition of a non-affine access, and :meth:`VarSpec.scale_by` refuses it
instead of producing a degree-2 term.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..ttir.graph import LoopInfo, iter_loops
from ..ttir.ssa import Module, Operation, SsaValue
from .op_shapes import ADDPTR, CONSTANT

#: The hop budget for the bounded walk. Named here, exported from
#: `recognize`, and never written as a literal at a call site.
MAX_HOPS = 32

#: The loop induction variable's coefficient in a per-iteration advance must be
#: zero for the increment to be decidable; this is the symbol set of a loop's
#: induction variable, used for that test.
IV_MARKER = "induction-variable"


# --------------------------------------------------------------------------- #
# Symbolic expressions
# --------------------------------------------------------------------------- #

#: A symbol product is an integer scale times a sorted tuple of symbol names.
#: The empty tuple is the constant term.
Product = tuple[int, tuple[str, ...]]


@dataclass(frozen=True)
class SymExpr:
    """A sum of scaled symbol products: `Σ scale_i · ∏ symbols_ij`.

    Normalised on construction — terms sorted, equal symbol products merged, a
    zero scale dropped — so that `SymExpr(sym("a")) + SymExpr(sym("a"))` is
    *equal* to `SymExpr.const(2) * SymExpr.sym("a")`. Equality is what makes the
    descriptor's determinism assertion meaningful: two runs that fold in
    a different order must produce the same descriptor, not merely the same
    address.
    """

    terms: tuple[Product, ...] = ((1, ()),)

    # -- constructors -------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", _normalise(self.terms))

    @classmethod
    def const(cls, value: int) -> SymExpr:
        return cls(terms=((int(value), ()),))

    @classmethod
    def symbol(cls, name: str, scale: int = 1) -> SymExpr:
        if scale == 0:
            return cls.const(0)
        return cls(terms=((int(scale), (name,)),))

    # -- predicates ---------------------------------------------------------

    @property
    def is_constant(self) -> bool:
        """Whether the whole expression is one constant term."""
        return all(not symbols for _, symbols in self.terms)

    @property
    def is_zero(self) -> bool:
        return self.terms == () or (len(self.terms) == 1 and self.terms[0][0] == 0)

    def symbols(self) -> frozenset[str]:
        return frozenset(name for _, symbols in self.terms for name in symbols)

    def as_int(self) -> int | None:
        """The integer value, when the expression *is* one. `None` otherwise."""
        if self.is_constant:
            return sum(scale for scale, _ in self.terms)
        return None

    def is_constant_multiple(self) -> bool:
        """Whether every term is a constant times a single symbol product.

        True for `32 * sak` (one term, scale 32) and for `2` (one constant term);
        false for `pid_m * sam` (a symbol product with two symbols) — which is
        why :meth:`integer_factor` is allowed to return the scale only when the
        expression has exactly one term.
        """
        return len(self.terms) == 1

    def integer_factor(self) -> int | None:
        """The `n` in `n · X`, when the expression has exactly one term."""
        if len(self.terms) == 1:
            scale, _ = self.terms[0]
            return scale
        return None

    # -- algebra ------------------------------------------------------------

    def __add__(self, other: SymExpr) -> SymExpr:
        return SymExpr(terms=(*self.terms, *other.terms))

    def __mul__(self, other: SymExpr) -> SymExpr:
        """Multiply two symbolic expressions. Any number of symbol products.

        Used where the *product is affine*: `splat(sam)` times an index
        expression whose variable coefficients are 1. A product of two index
        variables never reaches here — :meth:`VarSpec.scale_by` refuses it first
        — so this method may stay a plain distribution.
        """
        products: list[Product] = []
        for scale_a, syms_a in self.terms:
            for scale_b, syms_b in other.terms:
                products.append((scale_a * scale_b, tuple(sorted((*syms_a, *syms_b)))))
        return SymExpr(terms=tuple(products))

    def __neg__(self) -> SymExpr:
        return SymExpr(terms=tuple((-scale, symbols) for scale, symbols in self.terms))

    def scale(self, factor: int) -> SymExpr:
        return SymExpr(terms=tuple((scale * factor, symbols) for scale, symbols in self.terms))

    # -- presentation -------------------------------------------------------

    def to_public(self) -> int | SymExpr:
        """`int` when constant, else `self` — the `int | SymExpr` a descriptor field holds.

        The conversion happens at exactly one place so that "is this field an
        `int`?" is decided by the value, not by which code path built it.
        """
        value = self.as_int()
        return value if value is not None else self

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for scale, symbols in self.terms:
            if not symbols:
                parts.append(str(scale))
            elif scale == 1:
                parts.append("*".join(symbols))
            else:
                parts.append(f"{scale}*{'*'.join(symbols)}")
        return " + ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"SymExpr({self})"


def _normalise(terms: Iterable[Product]) -> tuple[Product, ...]:
    merged: dict[tuple[str, ...], int] = {}
    for scale, symbols in terms:
        key = tuple(sorted(symbols))
        merged[key] = merged.get(key, 0) + int(scale)
    kept = [(scale, symbols) for symbols, scale in merged.items() if scale != 0]
    kept.sort(key=lambda item: (len(item[1]), item[1], item[0]))
    return tuple(kept)


def public(value: SymExpr | int) -> int | SymExpr:
    """`int` when constant, else `SymExpr`. Accepts either form."""
    if isinstance(value, int):
        return value
    return value.to_public()


def public_list(values: Iterable[SymExpr | int]) -> tuple[int | SymExpr, ...]:
    return tuple(public(value) for value in values)


# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #


class BudgetReached(Exception):
    """Internal control flow for the hop budget.

    Private, and converted to `BudgetExhausted` by `describe` before it can
    escape: exhausting the budget is a *result*, never a crash. Raising
    internally and catching at one
    boundary is how the recursion stays readable without the budget being a
    value that every recursive call has to thread through and check.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(f"hop budget of {limit} exhausted")
        self.limit = limit


@dataclass
class BoundedWalker:
    """Counts the operations a walk visits and stops at `limit`.

    `provenance` is appended in visit order, which is what makes the contract's
    `provenance` field a *record of the walk* rather than a summarised string
    assembled after the fact.
    """

    limit: int = MAX_HOPS
    hops: int = 0
    provenance: list[str] = field(default_factory=list)

    def visit(self, op: Operation) -> None:
        """Account for one operation, refusing *before* exceeding the budget."""
        if self.hops >= self.limit:
            raise BudgetReached(self.limit)
        self.hops += 1
        self.provenance.append(op.name)

    def note(self, text: str) -> None:
        """Record a fact that is not an operation (a loop, a role)."""
        self.provenance.append(text)

    def remaining(self) -> int:
        return max(0, self.limit - self.hops)


# --------------------------------------------------------------------------- #
# Scalar folding
# --------------------------------------------------------------------------- #


def fold_constant(value: SsaValue) -> int | float | None:
    """The Python value of a value defined by `arith.constant`, else `None`.

    Kept separate from `op_shapes.decode_constant` so callers that only need
    "is this a literal" do not have to unpack a `Constant`.
    """
    from .op_shapes import decode_constant

    op = value.def_op
    if op is None:
        return None
    decoded = decode_constant(op)
    return decoded.value if decoded.ok else None


def fold_scalar(value: SsaValue, walker: BoundedWalker) -> SymExpr | None:
    """Fold a *scalar* value to a `SymExpr`: a constant, or a symbol.

    Three outcomes, and the third is a refusal rather than a guess:

    * an `arith.constant` → its value;
    * a function argument or a loop iter_arg of scalar type → a symbol named by
      the value's own SSA name, which is what makes `stride_am` appear in a
      descriptor under the name the Triton printer gave it;
    * anything else — a computed value like `arith.addi %K, %c31_i32` — → `None`,
      and the caller refuses with "non-decidable increment" or "non-affine index".

    `None` is the honest answer here. Folding `%K + 31` to `31` would be a
    number, and the descriptor would carry a wrong offset with no marker.
    """
    if value.def_op is not None and value.def_op.name == CONSTANT:
        walker.visit(value.def_op)
        literal = fold_constant(value)
        if isinstance(literal, bool):  # pragma: no cover - defensive
            return None
        if isinstance(literal, (int, float)):
            if isinstance(literal, float):
                # A float in an index expression is not an integer offset. It is
                # refused rather than truncated: `2.0` and `2` are different
                # values and only one of them is an address.
                return None
            return SymExpr.const(literal)
        return None
    if value.def_op is None:
        # A function argument or a block argument: symbolic, and named.
        return SymExpr.symbol(value.name)
    return None


# --------------------------------------------------------------------------- #
# Loop recurrence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recurrence:
    """One resolved `scf.for` iter_arg.

    `init` is the value fed in on entry, `advance` the index expression the body
    adds each iteration, and `yielded` the body's re-threaded value. All three are
    needed: `init` gives the descriptor, `advance` gives `increment`, and
    `yielded` is what makes the pairing *checked* rather than assumed.
    """

    loop: LoopInfo
    iter_arg: SsaValue
    init: SsaValue
    yielded: SsaValue
    advance_value: SsaValue | None

    @property
    def iv_name(self) -> str | None:
        return self.loop.iv.name if self.loop.iv is not None else None


def substitute_iter_arg(value: SsaValue, module: Module) -> Recurrence | None:
    """Resolve `value` if it is an `scf.for` iter_arg. `None` otherwise.

    Positional, both ways: `iter_args[i]` is fed by `inits[i]`, and `scf.yield`'s
    i-th operand re-threads it. Getting either pairing wrong produces a
    descriptor for a *different* pointer, which is why both are read from the
    structure rather than from a name match.
    """
    if value.def_op is not None:
        return None
    for loop in iter_loops(module):
        for index, iter_arg in enumerate(loop.iter_args):
            if iter_arg.name != value.name:
                continue
            init = loop.inits[index] if index < len(loop.inits) else None
            yield_op = loop.yield_op
            yielded = None
            if yield_op is not None and index < len(yield_op.operands):
                yielded = yield_op.operands[index]
            if init is None or yielded is None:
                return None
            return Recurrence(
                loop=loop,
                iter_arg=value,
                init=init,
                yielded=yielded,
                advance_value=advance_of(yielded, value),
            )
    return None


def advance_of(yielded: SsaValue, iter_arg: SsaValue) -> SsaValue | None:
    """The index expression a yielded pointer adds to its own iter_arg.

    `%a_ptrs_40 = tt.addptr %a_ptrs_34, %a_ptrs_39` where `%a_ptrs_34` is the
    iter_arg: the advance is `%a_ptrs_39`. If the yielded value is not an
    `addptr` on this iter_arg, `None` — which downstream becomes
    `Unstructured("non-decidable increment")` rather than an assumed step.
    """
    op = yielded.def_op
    if op is None or op.name != ADDPTR:
        return None
    base = op.operands[0] if op.operands else None
    if base is None or base.name != iter_arg.name:
        return None
    return op.operands[1] if len(op.operands) > 1 else None


__all__ = [
    "IV_MARKER",
    "MAX_HOPS",
    "BoundedWalker",
    "BudgetReached",
    "Product",
    "Recurrence",
    "SymExpr",
    "advance_of",
    "fold_constant",
    "fold_scalar",
    "public",
    "public_list",
    "substitute_iter_arg",
]
