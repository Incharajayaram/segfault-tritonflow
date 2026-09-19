"""Equality-saturation instruction selection: e-graph, rewrite rules, min-cost extraction.

The greedy selector in `isa/select.py` maps one TTIR operation to one schema
instruction by exact op-name match. That is optimal when the hardware names its
operations the way the IR does, and it cannot see a lowering at all when the two
disagree -- a chip with `fma` but no separate multiply-add pairing, a chip with
`relu` where the IR wrote `maxnumf(x, 0)`, a chip with add and negate but no
subtract. Committing to a rewrite order to bridge that gap reintroduces the
phase-ordering problem: rewriting `x*2` to a shift destroys the information that
a later `/2` would have cancelled.

An e-graph avoids the order by never discarding a form. Rewrites add equalities
to a congruence-closed set of equivalence classes, so every reachable form is
represented at once, and extraction then picks the cheapest whole lowering under
the schema's own cost model.

Fail-closed discipline follows `isa/select.py`: an operation with no admissible
instruction is priced `None`, never zero, and extraction refuses to return a term
containing one. Saturation is bounded and always terminates; when a bound stops
it rather than a fixed point, the result says so instead of implying optimality.

Rules live in `RULES` as data, each carrying the dtype class it is valid for and
whether it preserves IEEE-754 results. Rules that change floating-point rounding
(reassociation, fma contraction, reciprocal multiplication) are opt-in and are
never applied to float expressions unless the caller asks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "ENode",
    "EGraph",
    "Pattern",
    "PatVar",
    "PatNode",
    "Rule",
    "RULES",
    "SaturationResult",
    "Extraction",
    "Expr",
    "ExtractedExpr",
    "build_exprs_from_module",
    "greedy_cost",
    "leaf",
    "iter_operations",
    "saturate",
    "extract",
    "INFEASIBLE",
    "SchemaCostModel",
    "egraph_select",
    "EGraphSelection",
    "rules_for",
]

#: Cost assigned to an e-node that no schema instruction implements. Extraction
#: treats any term at or above this as "no lowering exists", so an unimplementable
#: node can never be chosen while a feasible alternative exists, and a root whose
#: whole class is infeasible is reported as a refusal rather than priced at zero.
INFEASIBLE = float("inf")


# --------------------------------------------------------------------------- #
# Core e-graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ENode:
    """One operator applied to e-class arguments.

    `op` is a TTIR operation name (`arith.addf`), or a leaf symbol for a value
    the expression does not define (`%x_5`, `const:2.0`). Children are e-class
    ids, so a single e-node stands for every term its classes can produce.
    """

    op: str
    children: tuple[int, ...] = ()

    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class EClass:
    """An equivalence class: the e-nodes known to compute the same value."""

    nodes: set[ENode] = field(default_factory=set)


class EGraph:
    """Congruence-closed equivalence classes over operator terms.

    Invariants, restored by `rebuild()` and asserted by `check_invariants()`:
      1. every e-node stored in `hashcons` has canonical children;
      2. two e-nodes equal after canonicalisation are in the same class;
      3. `find` is the identity on class ids that appear as `hashcons` values.
    """

    def __init__(self) -> None:
        self._parent: list[int] = []
        self.classes: dict[int, EClass] = {}
        self.hashcons: dict[ENode, int] = {}
        #: True when a merge has happened without a following rebuild.
        self._dirty = False

    # -- union-find ---------------------------------------------------------- #

    def find(self, eid: int) -> int:
        """Canonical id of `eid`, with path compression."""
        root = eid
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[eid] != root:
            self._parent[eid], eid = root, self._parent[eid]
        return root

    def _new_class(self) -> int:
        eid = len(self._parent)
        self._parent.append(eid)
        self.classes[eid] = EClass()
        return eid

    # -- construction -------------------------------------------------------- #

    def canonicalize(self, node: ENode) -> ENode:
        """`node` with every child replaced by its canonical class id."""
        if not node.children:
            return node
        return ENode(node.op, tuple(self.find(c) for c in node.children))

    def add(self, node: ENode) -> int:
        """Add `node`, returning its class. Identical nodes share a class (hashcons)."""
        node = self.canonicalize(node)
        existing = self.hashcons.get(node)
        if existing is not None:
            return self.find(existing)
        eid = self._new_class()
        self.classes[eid].nodes.add(node)
        self.hashcons[node] = eid
        return eid

    def add_expr(self, expr: Expr) -> int:
        """Add a whole expression tree, bottom up."""
        children = tuple(self.add_expr(c) for c in expr.children)
        return self.add(ENode(expr.op, children))

    # -- merging ------------------------------------------------------------- #

    def merge(self, a: int, b: int) -> int:
        """Assert that classes `a` and `b` compute the same value."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        # Union by size keeps the parent chains shallow.
        if len(self.classes[a].nodes) < len(self.classes[b].nodes):
            a, b = b, a
        self._parent[b] = a
        merged = self.classes.pop(b)
        self.classes[a].nodes |= merged.nodes
        self._dirty = True
        return a

    def rebuild(self) -> None:
        """Restore congruence: equal-after-canonicalisation nodes share a class.

        Runs a global fixed point rather than an incremental repair. Merging two
        classes can make their users congruent, and those users' users in turn,
        so each pass re-canonicalises every node and merges any pair that collide
        in the hashcons. A pass that finds a collision performs a merge, which
        strictly reduces the class count, so the loop terminates.

        The global form is chosen over egg's parent-pointer repair because it is
        checkable by inspection: after it returns, `check_invariants` holds by
        construction. Sizes here are bounded by the saturation node limit, so the
        extra sweep is not the cost that matters.
        """
        self._dirty = False
        while True:
            self.hashcons.clear()
            collisions: list[tuple[int, int]] = []
            for eid in [e for e in self.classes if self.find(e) == e]:
                nodes = {self.canonicalize(n) for n in self.classes[eid].nodes}
                self.classes[eid].nodes = nodes
                for node in nodes:
                    prev = self.hashcons.get(node)
                    if prev is None:
                        self.hashcons[node] = eid
                    elif self.find(prev) != eid:
                        collisions.append((prev, eid))
            if not collisions:
                return
            for a, b in collisions:
                self.merge(a, b)

    # -- inspection ---------------------------------------------------------- #

    def eclasses(self) -> dict[int, set[ENode]]:
        """Canonical class id -> its e-nodes."""
        out: dict[int, set[ENode]] = {}
        for eid in list(self.classes):
            root = self.find(eid)
            out.setdefault(root, set()).update(self.classes[eid].nodes)
        return out

    def total_nodes(self) -> int:
        return sum(len(n) for n in self.eclasses().values())

    def total_classes(self) -> int:
        return len(self.eclasses())

    def check_invariants(self) -> None:
        """Raise AssertionError if congruence or canonicality is broken."""
        for eid, nodes in self.eclasses().items():
            assert self.find(eid) == eid, f"non-canonical class id {eid}"
            for node in nodes:
                assert self.canonicalize(node) == node, f"non-canonical node {node}"
        seen: dict[ENode, int] = {}
        for eid, nodes in self.eclasses().items():
            for node in nodes:
                if node in seen:
                    assert seen[node] == eid, f"congruence broken for {node}"
                seen[node] = eid


# --------------------------------------------------------------------------- #
# Expressions (the input and output term language)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Expr:
    """A term: an operator over sub-terms, or a leaf when `children` is empty."""

    op: str
    children: tuple[Expr, ...] = ()

    def __str__(self) -> str:
        if not self.children:
            return self.op
        return f"{self.op}({', '.join(str(c) for c in self.children)})"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def ops(self) -> Iterator[str]:
        """Every non-leaf operator in the term, pre-order."""
        if self.children:
            yield self.op
        for c in self.children:
            yield from c.ops()


def leaf(name: str) -> Expr:
    return Expr(name, ())


# --------------------------------------------------------------------------- #
# Patterns and rules (data, not code branches)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatVar:
    """Matches any e-class, binding it to `name`."""

    name: str


@dataclass(frozen=True)
class PatNode:
    """Matches an e-node with this operator and matching arguments."""

    op: str
    args: tuple[Pattern, ...] = ()


Pattern = PatVar | PatNode

#: Which dtypes a rule is sound for. Selection passes the expression's dtype
#: class; a rule whose class does not match is never applied.
DTypeClass = Literal["any", "int", "float"]


@dataclass(frozen=True)
class Rule:
    """One directed rewrite, with the soundness conditions it depends on.

    `fp_exact` records whether applying the rule to IEEE-754 floating point
    yields a bit-identical result. A rule that is not `fp_exact` is excluded from
    float expressions unless the caller opts in, because a compiler that silently
    reassociates float arithmetic changes the numbers the program computes.
    """

    name: str
    lhs: Pattern
    rhs: Pattern
    dtype_class: DTypeClass = "any"
    fp_exact: bool = True
    note: str = ""


def _v(n: str) -> PatVar:
    return PatVar(n)


def _n(op: str, *args: Pattern) -> PatNode:
    return PatNode(op, tuple(args))


#: The default rewrite set.
#:
#: Soundness notes, per rule:
#:   commute-add/mul   IEEE-754 addition and multiplication are commutative on
#:                     values; NaN payload propagation may differ, which no
#:                     conforming program may depend on. Exact.
#:   assoc-add/mul     Reassociation changes the rounding points, so a+(b+c) and
#:                     (a+b)+c can differ in the last bit. Exact for integers,
#:                     not for floats: opt-in only.
#:   sub-to-add-neg    IEEE-754 defines x-y as x+(-y), including for signed zero
#:                     and infinities. Exact.
#:   mul2-to-add       Multiplication by 2 is exact (an exponent increment) and
#:                     x+x rounds identically. Exact.
#:   max0-to-relu      True by the definition of relu as max(x, 0), for a target
#:                     whose relu agrees on the sign of zero and on NaN.
#:   fma-contract      An fma rounds once where mul-then-add rounds twice, so the
#:                     results differ. Exact for integers, not for floats.
#:   div-to-recip-mul  1/c is generally inexact, so x*(1/c) differs from x/c.
#:                     Opt-in. (Exact only when c is a power of two, which the
#:                     narrower rule below covers.)
#:   mul1-identity     Multiplying by exactly 1 is the identity on every IEEE
#:                     value including NaN and signed zero. Exact.
#:   add0-identity     NOT exact: (-0.0) + 0.0 is +0.0, not -0.0. Opt-in.
#:   mul0-annihilate   NOT exact: NaN*0 and Inf*0 are NaN, not 0. Opt-in.
RULES: tuple[Rule, ...] = (
    Rule(
        "commute-add",
        _n("add", _v("a"), _v("b")),
        _n("add", _v("b"), _v("a")),
        note="IEEE addition is commutative on values.",
    ),
    Rule(
        "commute-mul",
        _n("mul", _v("a"), _v("b")),
        _n("mul", _v("b"), _v("a")),
        note="IEEE multiplication is commutative on values.",
    ),
    Rule(
        "assoc-add",
        _n("add", _n("add", _v("a"), _v("b")), _v("c")),
        _n("add", _v("a"), _n("add", _v("b"), _v("c"))),
        fp_exact=False,
        note="Reassociation moves the rounding point; integer-exact only.",
    ),
    Rule(
        "assoc-mul",
        _n("mul", _n("mul", _v("a"), _v("b")), _v("c")),
        _n("mul", _v("a"), _n("mul", _v("b"), _v("c"))),
        fp_exact=False,
        note="Reassociation moves the rounding point; integer-exact only.",
    ),
    Rule(
        "sub-to-add-neg",
        _n("sub", _v("a"), _v("b")),
        _n("add", _v("a"), _n("neg", _v("b"))),
        note="IEEE-754 defines subtraction as addition of the negation.",
    ),
    Rule(
        "add-neg-to-sub",
        _n("add", _v("a"), _n("neg", _v("b"))),
        _n("sub", _v("a"), _v("b")),
        note="Converse of sub-to-add-neg; lets a chip with SUB reclaim it.",
    ),
    Rule(
        "mul2-to-add",
        _n("mul", _v("a"), _n("const:2")),
        _n("add", _v("a"), _v("a")),
        note="Doubling is exact in binary floating point; x+x rounds identically.",
    ),
    Rule(
        "add-to-mul2",
        _n("add", _v("a"), _v("a")),
        _n("mul", _v("a"), _n("const:2")),
        note="Converse of mul2-to-add.",
    ),
    Rule(
        "max0-to-relu",
        _n("max", _v("a"), _n("const:0")),
        _n("relu", _v("a")),
        note="relu is defined as max(x, 0).",
    ),
    Rule(
        "relu-to-max0",
        _n("relu", _v("a")),
        _n("max", _v("a"), _n("const:0")),
        note="Converse of max0-to-relu; lets a chip with only MAX lower relu.",
    ),
    Rule(
        "fma-contract",
        _n("add", _n("mul", _v("a"), _v("b")), _v("c")),
        _n("fma", _v("a"), _v("b"), _v("c")),
        fp_exact=False,
        note="fma rounds once, mul-then-add rounds twice; integer-exact only.",
    ),
    Rule(
        "div-to-recip-mul",
        _n("div", _v("a"), _v("c")),
        _n("mul", _v("a"), _n("recip", _v("c"))),
        fp_exact=False,
        note="1/c is generally inexact, so the product differs from the quotient.",
    ),
    Rule(
        "mul1-identity",
        _n("mul", _v("a"), _n("const:1")),
        _v("a"),
        note="Multiplication by exactly 1 is the identity on every IEEE value.",
    ),
    Rule(
        "add0-identity",
        _n("add", _v("a"), _n("const:0")),
        _v("a"),
        fp_exact=False,
        note="(-0.0) + 0.0 is +0.0, so this loses the sign of zero.",
    ),
    Rule(
        "mul0-annihilate",
        _n("mul", _v("a"), _n("const:0")),
        _n("const:0"),
        fp_exact=False,
        note="NaN*0 and Inf*0 are NaN, not zero.",
    ),
)


def rules_for(dtype_class: DTypeClass, allow_inexact: bool = False) -> tuple[Rule, ...]:
    """The subset of `RULES` sound for `dtype_class`.

    Floats exclude every rule that is not bit-exact unless `allow_inexact`.
    Integers admit the reassociation and contraction rules, which are exact on
    two's-complement arithmetic.
    """
    out = []
    for rule in RULES:
        if rule.dtype_class not in ("any", dtype_class):
            continue
        if dtype_class == "float" and not rule.fp_exact and not allow_inexact:
            continue
        out.append(rule)
    return tuple(out)


# --------------------------------------------------------------------------- #
# E-matching
# --------------------------------------------------------------------------- #

Subst = dict[str, int]


def _match_in_class(eg: EGraph, pattern: Pattern, eid: int, subst: Subst) -> list[Subst]:
    """Every way `pattern` matches a term in class `eid`, extending `subst`."""
    eid = eg.find(eid)
    if isinstance(pattern, PatVar):
        bound = subst.get(pattern.name)
        if bound is None:
            out = dict(subst)
            out[pattern.name] = eid
            return [out]
        return [dict(subst)] if eg.find(bound) == eid else []

    results: list[Subst] = []
    for node in eg.classes[eid].nodes:
        if node.op != pattern.op or len(node.children) != len(pattern.args):
            continue
        partials = [dict(subst)]
        for arg_pat, child in zip(pattern.args, node.children, strict=True):
            nxt: list[Subst] = []
            for p in partials:
                nxt.extend(_match_in_class(eg, arg_pat, child, p))
            partials = nxt
            if not partials:
                break
        results.extend(partials)
    return results


def ematch(eg: EGraph, pattern: Pattern) -> list[tuple[int, Subst]]:
    """Every (class, binding) pair where `pattern` matches a term in that class."""
    out: list[tuple[int, Subst]] = []
    for eid in eg.eclasses():
        for subst in _match_in_class(eg, pattern, eid, {}):
            out.append((eid, subst))
    return out


def _instantiate(eg: EGraph, pattern: Pattern, subst: Subst) -> int:
    """Add `pattern` with its variables replaced by their bound classes."""
    if isinstance(pattern, PatVar):
        return eg.find(subst[pattern.name])
    children = tuple(_instantiate(eg, a, subst) for a in pattern.args)
    return eg.add(ENode(pattern.op, children))


# --------------------------------------------------------------------------- #
# Saturation
# --------------------------------------------------------------------------- #

SaturationStatus = Literal["saturated", "node_limit", "iter_limit"]


@dataclass(frozen=True)
class SaturationResult:
    """What saturation did, including why it stopped.

    `status == "saturated"` means a fixed point: no rule can add anything new, so
    the e-graph holds every form the rules can reach. Any other status means a
    bound stopped the search, so the extraction that follows is the best found,
    not a proven optimum -- reports must not present it as one.
    """

    status: SaturationStatus
    iterations: int
    nodes: int
    classes: int
    applications: int
    #: Rule name -> how many times it fired.
    rule_firings: dict[str, int] = field(default_factory=dict)

    @property
    def reached_fixpoint(self) -> bool:
        return self.status == "saturated"


def saturate(
    eg: EGraph,
    rules: Sequence[Rule],
    node_limit: int = 10_000,
    iter_limit: int = 30,
) -> SaturationResult:
    """Apply `rules` until a fixed point or a bound. Always terminates.

    Each iteration reads all matches before applying any of them, so a rule
    cannot cascade on its own output inside one pass. Growth is checked against
    `node_limit` after every iteration, which bounds the exponential blow-up that
    associativity and commutativity would otherwise produce.
    """
    firings: dict[str, int] = {}
    applications = 0

    for iteration in range(1, iter_limit + 1):
        # Read phase: collect every match against the current e-graph.
        found: list[tuple[Rule, int, Subst]] = []
        for rule in rules:
            for eid, subst in ematch(eg, rule.lhs):
                found.append((rule, eid, subst))

        # Write phase: instantiate right-hand sides and merge.
        changed = False
        for rule, eid, subst in found:
            try:
                new_eid = _instantiate(eg, rule.rhs, subst)
            except KeyError:
                # An rhs variable not bound by the lhs: the rule is malformed.
                continue
            if eg.find(new_eid) != eg.find(eid):
                eg.merge(new_eid, eid)
                changed = True
                applications += 1
                firings[rule.name] = firings.get(rule.name, 0) + 1
        eg.rebuild()

        if eg.total_nodes() > node_limit:
            return SaturationResult(
                "node_limit", iteration, eg.total_nodes(), eg.total_classes(),
                applications, firings,
            )
        if not changed:
            return SaturationResult(
                "saturated", iteration, eg.total_nodes(), eg.total_classes(),
                applications, firings,
            )

    return SaturationResult(
        "iter_limit", iter_limit, eg.total_nodes(), eg.total_classes(),
        applications, firings,
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Extraction:
    """The cheapest term extraction found for a root class."""

    cost: float
    term: Expr | None
    #: Per-class chosen cost, for auditing why a term won.
    class_costs: dict[int, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """False when no term in the root class avoids an unimplementable node."""
        return self.term is not None and self.cost < INFEASIBLE


def extract(
    eg: EGraph,
    root: int,
    cost_of: Callable[[ENode], float],
) -> Extraction:
    """Minimum-cost term in `root`'s class under an additive node cost.

    Iterates class costs to a fixed point rather than recursing, because an
    e-graph is cyclic in general: `x` and `x*1` inhabit one class, so a naive
    recursive walk would not terminate. Each pass can only lower a class cost, and
    costs are bounded below, so the loop converges; the iteration cap is a guard
    against a pathological cost function, not an expected exit.
    """
    classes = eg.eclasses()
    best_cost: dict[int, float] = dict.fromkeys(classes, INFEASIBLE)
    best_node: dict[int, ENode | None] = dict.fromkeys(classes, None)

    for _ in range(len(classes) + 2):
        changed = False
        for eid, nodes in classes.items():
            for node in nodes:
                own = cost_of(node)
                if own >= INFEASIBLE:
                    continue
                total = own
                for child in node.children:
                    child_cost = best_cost.get(eg.find(child), INFEASIBLE)
                    if child_cost >= INFEASIBLE:
                        total = INFEASIBLE
                        break
                    total += child_cost
                if total < best_cost[eid]:
                    best_cost[eid] = total
                    best_node[eid] = node
                    changed = True
        if not changed:
            break

    root = eg.find(root)
    if best_node.get(root) is None:
        return Extraction(INFEASIBLE, None, best_cost)

    def build(eid: int, depth: int = 0) -> Expr:
        eid = eg.find(eid)
        node = best_node[eid]
        assert node is not None, "feasible class must have a chosen node"
        # Depth guard: best_cost is a strict decrease down the chosen tree, so
        # this cannot fire for a well-formed cost function, but a zero-cost cycle
        # would otherwise recurse forever.
        if depth > len(classes) + 2:
            raise RecursionError(f"extraction cycle at class {eid}")
        return Expr(node.op, tuple(build(c, depth + 1) for c in node.children))

    return Extraction(best_cost[root], build(root), best_cost)


# --------------------------------------------------------------------------- #
# Schema bridge
# --------------------------------------------------------------------------- #

#: Canonical rewrite operators -> the TTIR op names the schemas actually declare.
#: The e-graph reasons in the canonical names so one rule set covers int and
#: float; pricing maps back to the concrete op the schema knows.
CANON_TO_TTIR: dict[str, tuple[str, ...]] = {
    "add": ("arith.addf", "arith.addi"),
    "sub": ("arith.subf", "arith.subi"),
    "mul": ("arith.mulf", "arith.muli"),
    "div": ("arith.divf", "arith.divsi"),
    "neg": ("arith.negf",),
    "abs": ("math.absf", "arith.absf"),
    "max": ("arith.maxnumf", "arith.maxsi"),
    "min": ("arith.minnumf", "arith.minsi"),
    "relu": ("relu",),
    "fma": ("fma", "math.fma"),
    "recip": ("recip", "math.recip"),
}

#: TTIR op name -> canonical operator, derived from CANON_TO_TTIR.
TTIR_TO_CANON: dict[str, str] = {
    ttir: canon for canon, names in CANON_TO_TTIR.items() for ttir in names
}


class SchemaCostModel:
    """Prices an e-node with the same cost model and admissibility as greedy selection.

    Delegates to `isa.select.select`, so a node is priced exactly when the greedy
    selector would accept it and at exactly the cost greedy would use. That is
    what makes the comparison meaningful: any difference in the chosen lowering
    comes from the search, never from a second opinion about price.
    """

    def __init__(
        self,
        schema: object,
        env: dict[str, int] | None = None,
        dtype_class: DTypeClass = "float",
        sizes: tuple[int, ...] = (1024,),
        dtype: str | None = None,
    ) -> None:
        self.schema = schema
        self.env = dict(env or {})
        self.dtype_class = dtype_class
        self.sizes = sizes
        self.dtype = dtype or ("f32" if dtype_class == "float" else "i32")
        self._cache: dict[str, float] = {}
        #: op -> the schema instruction chosen for it, for reporting.
        self.chosen: dict[str, str] = {}
        #: op -> why no instruction was admissible, for reporting refusals.
        self.refusals: dict[str, str] = {}

    def _descriptor_for(self, ttir_op: str) -> object:
        """The operand shape selection sees, with the op name in `base`.

        The recogniser hands `select` an `AccessDescriptor` whose `base` is the
        TTIR op name; predicates such as `in_bounds(base, length)` read its
        fields. Passing a bare op-name string instead makes every such predicate
        evaluate to *unknown*, and fail-closed selection then rejects every
        candidate -- so the e-graph would appear to find no lowering where the
        real pipeline finds one. Building the same descriptor keeps this cost
        model's verdict identical to the greedy selector's.
        """
        from tritonflow.recognize.descriptor import AccessDescriptor

        rank = len(self.sizes)
        return AccessDescriptor(
            base=ttir_op,
            sizes=self.sizes,
            strides=tuple([1] * rank),
            offsets=tuple([0] * rank),
            shape=(),
            order=tuple(range(rank)),
            dtype=self.dtype,
            loop_carried=False,
            increment=None,
        )

    def _price_ttir(self, ttir_op: str) -> float | None:
        from .select import select

        try:
            report = select(
                self.schema, "elementwise", self._descriptor_for(ttir_op), env=self.env
            )
        except Exception as exc:
            self.refusals[ttir_op] = f"selection raised {type(exc).__name__}: {exc}"
            return None
        if report.chosen is None or report.chosen_cost is None:
            reasons = [
                f"{c.instruction.name}: {c.rejected_by}"
                for c in report.candidates
                if not c.admissible and "op mismatch" not in (c.rejected_by or "")
            ]
            self.refusals[ttir_op] = "; ".join(reasons) or "no candidate claimed this op"
            return None
        self.chosen[ttir_op] = report.chosen.name
        return float(report.chosen_cost)

    def cost_of(self, node: ENode) -> float:
        """Additive cost of one e-node; INFEASIBLE when no instruction implements it."""
        if node.is_leaf():
            return 0.0
        cached = self._cache.get(node.op)
        if cached is not None:
            return cached

        candidates = CANON_TO_TTIR.get(node.op, (node.op,))
        # Prefer the dtype-appropriate spelling, but accept any the schema prices.
        best: float | None = None
        for ttir_op in candidates:
            if self.dtype_class == "float" and ttir_op.endswith(("i", "si", "ui")):
                continue
            if self.dtype_class == "int" and ttir_op.endswith("f"):
                continue
            price = self._price_ttir(ttir_op)
            if price is not None and (best is None or price < best):
                best = price
        if best is None:
            for ttir_op in candidates:
                price = self._price_ttir(ttir_op)
                if price is not None and (best is None or price < best):
                    best = price

        result = INFEASIBLE if best is None else best
        self._cache[node.op] = result
        return result


@dataclass(frozen=True)
class EGraphSelection:
    """The e-graph selector's verdict for one expression on one ISA."""

    original: Expr
    extracted: Expr | None
    cost: float
    saturation: SaturationResult
    rules_used: tuple[str, ...]
    #: Instruction chosen per TTIR op, for the audit trail.
    instructions: dict[str, str] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.extracted is not None and self.cost < INFEASIBLE

    @property
    def rewritten(self) -> bool:
        """True when extraction chose a different term than the input."""
        return self.extracted is not None and self.extracted != self.original


def egraph_select(
    schema: object,
    expr: Expr,
    env: dict[str, int] | None = None,
    dtype_class: DTypeClass = "float",
    allow_inexact: bool = False,
    node_limit: int = 10_000,
    iter_limit: int = 30,
    sizes: tuple[int, ...] = (1024,),
    dtype: str | None = None,
) -> EGraphSelection:
    """Saturate `expr` under the sound rules, then extract its cheapest lowering."""
    eg = EGraph()
    root = eg.add_expr(expr)
    eg.rebuild()

    rules = rules_for(dtype_class, allow_inexact=allow_inexact)
    sat = saturate(eg, rules, node_limit=node_limit, iter_limit=iter_limit)

    model = SchemaCostModel(
        schema, env=env, dtype_class=dtype_class, sizes=sizes, dtype=dtype
    )
    ext = extract(eg, root, model.cost_of)

    return EGraphSelection(
        original=expr,
        extracted=ext.term,
        cost=ext.cost,
        saturation=sat,
        rules_used=tuple(r.name for r in rules),
        instructions=dict(model.chosen),
    )


def greedy_cost(
    schema: object,
    expr: Expr,
    env: dict[str, int] | None = None,
    dtype_class: DTypeClass = "float",
    sizes: tuple[int, ...] = (1024,),
    dtype: str | None = None,
) -> float:
    """Cost of lowering `expr` as written, one op to one instruction.

    This is what `isa/select.py` produces for the same term: no rewriting, so an
    operator the schema does not name is simply unimplementable.
    """
    model = SchemaCostModel(
        schema, env=env, dtype_class=dtype_class, sizes=sizes, dtype=dtype
    )

    def walk(e: Expr) -> float:
        if not e.children:
            return 0.0
        own = model.cost_of(ENode(e.op, tuple(range(len(e.children)))))
        if own >= INFEASIBLE:
            return INFEASIBLE
        total = own
        for c in e.children:
            sub = walk(c)
            if sub >= INFEASIBLE:
                return INFEASIBLE
            total += sub
        return total

    return walk(expr)


# --------------------------------------------------------------------------- #
# Building expressions from parsed TTIR
# --------------------------------------------------------------------------- #

#: Operations that carry a value but are not arithmetic the rules reason about.
#: They become leaves, so an expression is an arithmetic island rooted at a
#: value the rest of the program defines.
_BOUNDARY_OPS = frozenset(
    {
        "tt.load", "tt.store", "tt.addptr", "tt.splat", "tt.make_range",
        "tt.get_program_id", "tt.broadcast", "tt.expand_dims", "tt.dot",
        "tt.reduce", "tt.func", "tt.return", "scf.for", "scf.yield",
    }
)


def iter_operations(module: object) -> Iterator[object]:
    """Every operation in `module`, descending into regions."""

    def walk(op: object) -> Iterator[object]:
        yield op
        for region in getattr(op, "regions", ()):
            for block in region.blocks:
                for inner in block.operations:
                    yield from walk(inner)

    for block in module.body.blocks:
        for op in block.operations:
            yield from walk(op)


@dataclass(frozen=True)
class ExtractedExpr:
    """One arithmetic island from a module, with the context needed to price it."""

    root: str
    expr: Expr
    dtype_class: DTypeClass
    dtype: str
    sizes: tuple[int, ...]


def _dtype_class_of(raw: str) -> DTypeClass:
    low = raw.lower().lstrip("!")
    if low.startswith(("f", "bf")):
        return "float"
    if low.startswith("i"):
        return "int"
    return "any"


def build_exprs_from_module(module: object, max_depth: int = 6) -> list[ExtractedExpr]:
    """Arithmetic expression trees rooted at each value the module computes.

    A value produced by a boundary operation (a load, a dot, an address
    computation) becomes a leaf: those are not arithmetic the rewrite rules
    reason about, and treating them as opaque keeps each expression to the
    arithmetic island the selector actually gets to choose within.

    Each island carries the dtype and element count of its root, because both the
    sound rule set and the cost model depend on them: integer islands admit
    reassociation that float islands must not, and the schemas price per element.
    """
    defs: dict[str, object] = {}
    order: list[str] = []
    for op in iter_operations(module):
        for res in getattr(op, "results", ()):
            rname = getattr(res, "name", None)
            if rname:
                defs[rname] = op
                order.append(rname)

    def type_of(value: str) -> tuple[DTypeClass, str, tuple[int, ...]]:
        op = defs.get(value)
        for res in getattr(op, "results", ()) if op is not None else ():
            if getattr(res, "name", None) != value:
                continue
            ty = getattr(res, "type", None)
            raw = getattr(ty, "raw", "") or ""
            shape = tuple(getattr(ty, "shape", ()) or ())
            elem = raw.split("x")[-1].strip("<> ") if "x" in raw else raw.strip("<> ")
            sizes = tuple(s for s in shape if isinstance(s, int)) or (1,)
            return _dtype_class_of(elem), elem or "f32", sizes
        return "any", "f32", (1,)

    def build(value: str, depth: int) -> Expr:
        op = defs.get(value)
        if op is None or depth >= max_depth:
            return leaf(value)
        if getattr(op, "name", "") in _BOUNDARY_OPS:
            return leaf(value)
        canon = TTIR_TO_CANON.get(getattr(op, "name", ""))
        if canon is None:
            return leaf(value)
        children = []
        for operand in getattr(op, "operands", ()):
            oname = getattr(operand, "name", None)
            children.append(leaf(str(operand)) if oname is None else build(oname, depth + 1))
        return Expr(canon, tuple(children))

    out: list[ExtractedExpr] = []
    for value in order:
        expr = build(value, 0)
        if not expr.children:
            continue
        cls, dtype, sizes = type_of(value)
        out.append(ExtractedExpr(value, expr, cls, dtype, sizes))
    return out
