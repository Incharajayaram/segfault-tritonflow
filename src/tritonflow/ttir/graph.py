"""Traversal over the parsed IR: def-use, region-aware walking, loop structure.

Three questions that look similar and must not be confused, because conflating
them is the failure mode names — the one that produces right-looking numbers
from a flattened module:

| Question | Function | Nested operations? |
|---|---|---|
| "every operation in this region, including inside nested regions" | :func:`walk_region` | **yes**, depth-first |
| "the operations *belonging to* this region" | :func:`region_ops` | no |
| "this region's operations in dependency order" | :func:`topo_within_region` | no, and never hoisted |

`topo_within_region` answers only within one region. There is deliberately no
function that returns a topological order for a whole module: an operation
inside `scf.for` must stay inside it, and the cheapest way to guarantee that is
to never build the list that would lose it.

A `tt.load` inside the loop body is *reachable* from the module (so it appears
in `walk_region(module.body)`, which is what def-use needs) and is *not* a member
of the module's region (so `topo_within_region` cannot move it out). Both
statements are asserted by tests, in both directions.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field

from .ssa import Module, Operation, Region, SsaValue

FOR_OP = "scf.for"
YIELD_OP = "scf.yield"

#: `scf.for`'s header is `%iv = lower to upper step step [iter_args(…)]`, so the
#: first three operands are the bounds and the rest are the iter-argument
#: initialisers, matched positionally against the block arguments after the
#: induction variable. Named because it is a convention, not a discovery.
FOR_BOUND_OPERANDS = 3


def walk_region(region: Region) -> Iterator[Operation]:
    """Every operation in `region`, depth-first, descending into nested regions.

    Use this to *find* things (def-use, idiom search, reporting). Do not use it
    to decide where something is emitted: it deliberately forgets the boundary,
    because a search that stops at regions misses the loop body, which is where
    every interesting operation lives.
    """
    for block in region.blocks:
        for op in block.operations:
            yield op
            for nested in op.regions:
                yield from walk_region(nested)


def region_ops(region: Region) -> tuple[Operation, ...]:
    """The operations belonging to this region, in source order. No descending."""
    return tuple(op for block in region.blocks for op in block.operations)


def topo_within_region(region: Region) -> list[Operation]:
    """This region's operations, ordered so a definition precedes its uses.

    Region-local by construction: the candidate set is :func:`region_ops`, so an
    operation inside a nested region cannot appear here at all. Ties are broken
    by source position through a heap, so the order is total and identical on
    every run and every `PYTHONHASHSEED` — a set iteration here would
    be invisible until two machines disagreed.

    SSA guarantees a definition dominates its uses, so for a well-formed module
    this returns source order. It is computed rather than assumed because
    "assumed topological" is how a later rewrite silently changes the answer.
    """
    local = region_ops(region)
    position = {id(op): index for index, op in enumerate(local)}
    blocked_by: dict[int, set[int]] = {index: set() for index in range(len(local))}
    blocks: dict[int, set[int]] = {index: set() for index in range(len(local))}

    for index, op in enumerate(local):
        for operand in op.operands:
            producer = operand.def_op
            producer_index = position.get(id(producer)) if producer is not None else None
            if producer_index is not None and producer_index != index:
                blocked_by[index].add(producer_index)
                blocks[producer_index].add(index)

    ready = [index for index in range(len(local)) if not blocked_by[index]]
    heapq.heapify(ready)
    ordered: list[Operation] = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(local[index])
        for dependent in sorted(blocks[index]):
            blocked_by[dependent].discard(index)
            if not blocked_by[dependent]:
                heapq.heappush(ready, dependent)

    if len(ordered) != len(local):
        # Unreachable for valid SSA (a cycle would be invalid IR). Kept total
        # rather than raising: the remaining operations are appended in source
        # order so a caller sees every operation exactly once.
        emitted = {id(op) for op in ordered}
        ordered.extend(op for op in local if id(op) not in emitted)
    return ordered


@dataclass(frozen=True)
class LoopInfo:
    """A structured reduction loop: `scf.for` and the values it threads.

    `iter_args[i]` is the block argument that `inits[i]` feeds on entry; pairing
 them is what makes the loop-carried operand of decidable at all.
    """

    op: Operation
    iv: SsaValue | None = None
    lower: SsaValue | None = None
    upper: SsaValue | None = None
    step: SsaValue | None = None
    iter_args: tuple[SsaValue, ...] = ()
    inits: tuple[SsaValue, ...] = ()
    body: Region | None = None

    @property
    def results(self) -> tuple[SsaValue, ...]:
        """The loop's **own result names** — what a post-loop reader consumes.

        Counted, not chosen: a consumer outside the loop (`tt.store` on the
        accumulator) addresses these names, so they have to survive into the
        program. They are *not* the same values as :attr:`yields`, and the
        difference is the whole reason both attributes exist.
        """
        return self.op.results

    @property
    def yields(self) -> tuple[SsaValue, ...]:
        """The values `scf.yield` re-threads into `iter_args`, positionally.

        `yields[i]` is the value the **next iteration's** `iter_args[i]` takes,
        so this is the loop's update function. It is what an executor needs and
        cannot infer: the yielded names are ordinary body definitions
        (`%a_ptrs_40`), and nothing in the program otherwise pairs them with the
        block arguments they replace. Without it a loop can be run zero or one
        times and no more (audit finding F7).

        Empty for a malformed loop with no `scf.yield` terminator — an absent
        terminator is a parse-time concern, not a claim that the loop yields
        nothing.
        """
        op = self.yield_op
        return () if op is None else tuple(op.operands)

    @property
    def yield_op(self) -> Operation | None:
        return self.body.entry.terminator if self.body is not None and self.body.entry else None

    def iter_arg_named(self, name: str) -> SsaValue | None:
        for value in self.iter_args:
            if value.name == name:
                return value
        return None


def iter_loops(module: Module) -> Iterator[LoopInfo]:
    """Every `scf.for` in the module, in source order."""
    for op in walk_region(module.body):
        if op.name != FOR_OP:
            continue
        entry = op.regions[0].entry if op.regions else None
        args = entry.args if entry is not None else ()
        bounds = op.operands[:FOR_BOUND_OPERANDS]
        yield LoopInfo(
            op=op,
            iv=args[0] if args else None,
            lower=bounds[0] if len(bounds) > 0 else None,
            upper=bounds[1] if len(bounds) > 1 else None,
            step=bounds[2] if len(bounds) > 2 else None,
            iter_args=tuple(args[1:]),
            inits=tuple(op.operands[FOR_BOUND_OPERANDS:]),
            body=op.regions[0] if op.regions else None,
        )


@dataclass(frozen=True)
class DefUseGraph:
    """One definition per value, plus the operations that read it.

    Keyed by SSA *name*, not by `Operation`: operations hold tuples and dicts, so
    they are not hashable, and a graph that needed a custom hash would be a
    hidden place for two equal operations to collide. Names are unique per
 module by construction (`to_ir` enforces it,).
    """

    defs: dict[str, SsaValue] = field(default_factory=dict)
    uses: dict[str, tuple[Operation, ...]] = field(default_factory=dict)
    operations: tuple[Operation, ...] = ()
    module: Module | None = field(default=None, compare=False, repr=False)
    """The module this graph was built from, kept as a back-reference.

    Added because recognition needs it and cannot honestly do without it: a
    Tier-1 `tt.load` operand is an `scf.for` iter_arg, and resolving the
    recurrence means finding the *loop* the block argument belongs to. A graph
    that carried only defs and uses could be built over any IR and would then
    quietly resolve every block argument as "not carried" — the exact omission
    the recognizer's descriptor walk must not make.

    `compare=False, repr=False` for the same reason `Region.parent` is: the
    module owns the operations the graph indexes, so including it in `==` or
    `repr` would make a graph comparison walk the whole tree (and `repr` would
    not terminate through the cycle)."""

    def is_defined(self, name: str) -> bool:
        return name in self.defs

    def value(self, name: str) -> SsaValue | None:
        return self.defs.get(name)

    def def_op(self, name: str) -> Operation | None:
        value = self.defs.get(name)
        return value.def_op if value is not None else None

    def users(self, name: str) -> tuple[Operation, ...]:
        return self.uses.get(name, ())

    def is_block_arg(self, name: str) -> bool:
        value = self.defs.get(name)
        return value is not None and value.def_op is None

    def operands_of(self, op: Operation) -> tuple[SsaValue, ...]:
        return op.operands

    def unused_values(self) -> tuple[str, ...]:
        """Defined but never read — deterministic, in definition order."""
        return tuple(name for name in self.defs if name not in self.uses)


def build_def_use(module: Module) -> DefUseGraph:
    """Build the graph over every operation, nested regions included.

    Block arguments are defined before the operations of the region they open,
    so a value is never used before it is defined. A name defined twice raises
    `ValueError`: `to_ir` already refuses such a module, so reaching
    here with a duplicate means the graph is being built over something that did
    not come from `build_ir`, and silently keeping the last definition is how
    wrong answers get a clean bill of health.
    """
    operations = tuple(walk_region(module.body))

    defs: dict[str, SsaValue] = {}
    for op in operations:
        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    _define(defs, arg)
        for result in op.results:
            _define(defs, result)

    collected: dict[str, list[Operation]] = {}
    for op in operations:
        for operand in op.operands:
            collected.setdefault(operand.name, []).append(op)

    return DefUseGraph(
        defs=defs,
        uses={name: tuple(users) for name, users in collected.items()},
        operations=operations,
        module=module,
    )


def _define(defs: dict[str, SsaValue], value: SsaValue) -> None:
    previous = defs.get(value.name)
    if previous is not None:
        raise ValueError(
            f"SSA name {value.name} is defined twice "
            f"(line {previous.def_op.line if previous.def_op else '?'} and "
            f"line {value.def_op.line if value.def_op else '?'}); build_def_use "
            "requires a module validated by build_ir"
        )
    defs[value.name] = value


__all__ = [
    "FOR_BOUND_OPERANDS",
    "FOR_OP",
    "YIELD_OP",
    "DefUseGraph",
    "LoopInfo",
    "build_def_use",
    "iter_loops",
    "region_ops",
    "topo_within_region",
    "walk_region",
]
