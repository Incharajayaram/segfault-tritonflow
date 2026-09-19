"""Detection and annotation: the recogniser's decision, handed to the emitter.

`detect_all` answers "what idioms are here"; `annotate` turns that into one
:class:`~tritonflow.emit.ir.Binding` per operation, which is the form
`emit.assemble` consumes.

**`annotate` never mutates the module**. It returns a new
:class:`AnnotationSet` keyed by operation *identity* (`id(op)`), not by a field
written back onto the operation. `Operation` is frozen, so this is not politeness:
a pass that annotated in place could not be run twice, and the second run would
see the first run's decisions as if they were input.

**Three outcomes per operation, and the third is not a hole:**

| Case | Binding | Reaches the program as |
|---|---|---|
| a structured access, a MAC, or an elementwise op | one `Binding` | an `Instr` |
| a memory operand the descriptor walk refused | one `Binding` with a `reason` and **no instruction** | `UNSUPPORTED` carrying that reason |
| an operation this module refuses by name (`unsafe=`) | none (`()`) | `UNSUPPORTED("no annotation: no idiom matched …")` |

The middle row is the one worth insisting on. A refused descriptor could
plausibly be dropped (return `()`), and then the program's marker would read
"no idiom matched this operation" — which is **false**: a memory idiom matched it
and the *addressing* was the problem. The reason has to name the actual failure,
because the coverage report's whole job is to say which frontier we stopped at.

**Bindings carry `kind`, not an instruction name, by default.** The emitter is
the selector's consumer, and assembly selects the minimum-cost variant
consistent with the verified attributes, so the recogniser states what
it recognised and assembly decides the instruction. `instruction_for` exists for
callers that already have a schema and want to name instructions directly (the
frozen stand-in's mode, used by the emitter checks) — it is a convenience for
exercising the layer, not the default path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..emit.ir import Binding, Imm, MemRef, SsaRef
from ..isa.predicates import COMPARE_OPS, PredicateError, predicate_code
from ..recognize import op_shapes as shapes
from ..recognize.descriptor import (
    AccessDescriptor,
    BudgetExhausted,
    Ok,
    Unstructured,
    describe,
    elementwise_descriptor,
)
from ..ttir.graph import DefUseGraph, LoopInfo, build_def_use, region_ops, walk_region
from ..ttir.ssa import Module, Operation, SsaValue
from .patterns import (
    DOT_OPERAND_ROLES,
    EPILOGUE_ID,
    MAC_ID,
    NON_COMPUTE_EPILOGUE_OPS,
    MatchResult,
    classify,
    tile_of,
)

#: The rule names an annotation's `kind` takes. They are the keys a rule module
#: (`isa/rules/tritonflow1.py`) maps to candidate instruction sets — see
#: `recognize/op_shapes.py`'s note on why `mac` is not `compute`.
RULE_MEMORY = shapes.RULE_MEMORY
RULE_MAC = shapes.RULE_MAC
RULE_ELEMENTWISE = shapes.RULE_ELEMENTWISE

#: The memory space the ISA-1 model has (one `flat` space).
DEFAULT_SPACE = "global"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _yield_position(loop: LoopInfo, value: SsaValue) -> int | None:
    """Where `value` sits among the loop's `iter_args`. `None` when it is not one."""
    for index, iter_arg in enumerate(loop.iter_args):
        if iter_arg.name == value.name:
            return index
    return None


def detect_mac(module: Module, graph: DefUseGraph | None = None) -> tuple[MatchResult, ...]:
    """Every `scf.for` + accumulator + `tt.dot` + `scf.yield` occurrence.

    The four requirements are checked in the order they can fail, and each
    failure is a *non-match* rather than an error: a loop without a dot is not a
    malformed MAC, it is Tier 0's loop, which does not exist, and Tier 3's, which
    must come back with zero matches.

    The yield check is the one that makes this a *reduction* rather than "there is
    a dot in a loop": the dot's result must be what `scf.yield` re-threads at the
    accumulator's own position. A dot whose result is discarded is a different
    program, and accepting it would give the emitter an accumulator value to keep
    that the loop never returns.
    """
    found: list[MatchResult] = []
    for loop in _loops(module):
        if loop.body is None:
            continue
        for op in region_ops(loop.body):
            if not shapes.is_dot(op):
                continue
            if len(op.operands) < 3:
                continue
            accumulator = op.operands[2]
            position = _yield_position(loop, accumulator)
            if position is None:
                continue
            yield_op = loop.yield_op
            if yield_op is None or position >= len(yield_op.operands):
                continue
            if not op.results or yield_op.operands[position].name != op.results[0].name:
                continue
            tile = tile_of(op)
            found.append(
                MatchResult(
                    pattern_id=MAC_ID,
                    ops=(op,),
                    bindings={
                        "a": op.operands[0],
                        "b": op.operands[1],
                        "acc": accumulator,
                        "tile": tile,
                    },
                    tile_shape=tile,
                    dtype=shapes.scalar_dtype(op.results[0].type),
                    input_precision=shapes.input_precision(op),
                    multiplicity=1,
                )
            )
    return tuple(_with_multiplicity(found))


def _loops(module: Module) -> tuple[LoopInfo, ...]:
    from ..ttir.graph import iter_loops

    return tuple(iter_loops(module))


def _with_multiplicity(matches: list[MatchResult]) -> list[MatchResult]:
    """Stamp every match with how many matches of its pattern this pass found.

    `dataclasses.replace` rather than a second constructor call: the count is the
    only field that differs, and rebuilding the record would let one of them drift.
    """
    from dataclasses import replace

    if not matches:
        return []
    counts: dict[str, int] = {}
    for match in matches:
        counts[match.pattern_id] = counts.get(match.pattern_id, 0) + 1
    return [replace(match, multiplicity=counts[match.pattern_id]) for match in matches]


def detect_epilogue(module: Module, graph: DefUseGraph | None = None) -> tuple[MatchResult, ...]:
    """The elementwise chain applied to a loop's accumulator, after the loop.

    One `MatchResult` per loop, `ops` in dependency order, `classify` set from the
    *sink* of the chain (the last operation a store reads), because
    `EpiloguePattern.classify` is singular and the sink is the operation the ISA's
    `EPI` instruction has to express.

    The chain is followed through the accumulator's users, transitively, so
    `addf` then `maxnumf` is one chain rather than two unrelated operations. Tier 1
    has a store and no arithmetic epilogue, so it gets zero matches — the same
    distinction the canon's `expect_epilogue_idiom` records.
    """
    matches: list[MatchResult] = []
    for loop in _loops(module):
        if loop.body is None:
            continue
        accumulators = _accumulator_results(loop)
        if not accumulators:
            continue
        chain = _elementwise_chain(loop, accumulators, module)
        if not chain:
            continue
        matches.append(
            MatchResult(
                pattern_id=EPILOGUE_ID,
                ops=tuple(chain),
                bindings={
                    "acc": accumulators[0],
                    "sink": chain[-1].results[0] if chain[-1].results else None,
                },
                tile_shape=None,
                dtype=shapes.scalar_dtype(chain[-1].results[0].type) if chain[-1].results else "",
                input_precision="ieee",
                multiplicity=1,
                classify=classify(chain[-1]),
            )
        )
    return tuple(_with_multiplicity(matches))


def _accumulator_results(loop: LoopInfo) -> tuple[SsaValue, ...]:
    """The loop's **results** that a `tt.dot` feeds, in yield position order.

    Positional, and it has to be: the value the epilogue reads is the loop's
    *result*, not the value `scf.yield` names. The two are different values with
    similar names — `scf.yield %acc_43` re-threads into `%acc#2`, and a chain
    search seeded from `%acc_43` finds nothing, because `%acc_43` is the dot's
    output *inside* the loop while every post-loop consumer reads `%acc#2`.

    That mistake was made here first: the search seeded on the yielded value and
    Tier 2's epilogue came back as zero matches, which would have shipped as
    "Tier 2 has no epilogue" — a silently missing idiom, not a crash.
    """
    results: list[SsaValue] = []
    yields = loop.yield_op
    if yields is None:
        return ()
    for index, operand in enumerate(yields.operands):
        producer = operand.def_op
        if producer is None or not shapes.is_dot(producer):
            continue
        if index < len(loop.op.results):
            results.append(loop.op.results[index])
    return tuple(results)


def _elementwise_chain(
    loop: LoopInfo,
    seeds: tuple[SsaValue, ...],
    module: Module,
) -> list[Operation]:
    """The post-loop elementwise operations reachable from `seeds`, in order.

    "Reachable" is followed through *every* user, and the walk is over the module's
    ops in source order so the chain is deterministic. A user that is a memory
    operation is not part of the chain (it is the DMA's), a container is not part
    of it either (see `patterns.NON_COMPUTE_EPILOGUE_OPS`), and a user that is not
    classified is still included — an unclassified epilogue op is exactly what the
    coverage report needs to see.
    """
    loop_line = loop.op.line
    wanted = {value.name for value in seeds}
    chain: list[Operation] = []
    frontier = set(wanted)
    for op in walk_region(module.body):
        if op.line <= loop_line or op.name in NON_COMPUTE_EPILOGUE_OPS:
            continue
        if not any(operand.name in frontier for operand in op.operands):
            continue
        if not op.results:
            continue
        chain.append(op)
        frontier.update(value.name for value in op.results)
    return chain


def detect_all(module: Module, graph: DefUseGraph | None = None) -> tuple[MatchResult, ...]:
    """Both patterns, MAC first. Order is fixed so the tuple is comparable."""
    return (*detect_mac(module, graph), *detect_epilogue(module, graph))


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #


@dataclass
class AnnotationSet:
    """`emit.ir.AnnotationSetLike`: `bindings_for(op)` -> zero, one, or many.

    `refusals` and `unsupported` are kept for the same reason the emitter keeps
    its markers: a caller that only saw `bindings_for` could not tell "we chose not
    to lower this" from "this module had nothing here".
    """

    matches: tuple[MatchResult, ...] = ()
    refusals: tuple[tuple[str, str], ...] = ()
    unsupported: tuple[str, ...] = ()
    subsumed_loop_slots: frozenset[tuple[int, int]] = frozenset()
    _by_id: dict[int, tuple[Binding, ...]] = field(default_factory=dict, repr=False)

    def bindings_for(self, op: Operation) -> tuple[Binding, ...]:
        return self._by_id.get(id(op), ())

    def annotate(self, op: Operation, bindings: tuple[Binding, ...]) -> None:
        self._by_id[id(op)] = bindings

    def refusal_for(self, op: Operation) -> str | None:
        for name, reason in self.refusals:
            if name == op.name:
                return reason
        return None

    @property
    def annotated_count(self) -> int:
        return len(self._by_id)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"AnnotationSet({self.annotated_count} operation(s), "
            f"{len(self.matches)} match(es), {len(self.refusals)} refusal(s))"
        )



def _address_cone(
    module: Module, graph: DefUseGraph, space: str = DEFAULT_SPACE
) -> tuple[set[int], set[tuple[int, int]]]:
    """(ids of operations subsumed by memory descriptors, subsumed loop-carried slots).

    A structured memory access carries its own base, sizes, strides, offsets and
    loop increment, so the pointer arithmetic that produced its address is redundant
    once the descriptor exists. That arithmetic is elided when *every* consumer is
    itself elided or is a memory operation reading it as an address or mask.

    Loop-carried pointers are followed through the loop: a pointer that enters an
    `iter_arg`, is advanced in the body and yielded back is one *slot*. When the
    slot's argument and result are consumed only by elided code, the whole chain
    (initialiser, increment, yield) is elided together and the slot is dropped from
    the emitted loop. A slot whose value is used by anything live keeps its chain.
    """
    from ..ttir.graph import iter_loops

    loops = list(iter_loops(module))
    loop_by_op = {id(info.op): info for info in loops}
    loop_of_yield = {id(info.yield_op): info for info in loops if info.yield_op is not None}
    slot_of_arg = {
        arg.name: (info, j) for info in loops for j, arg in enumerate(info.iter_args)
    }

    structured_mem_ops = []
    for op in graph.operations:
        if shapes.is_memory_op(op):
            ptr = shapes.pointer_operand(op)
            if ptr is not None and isinstance(describe(ptr, (space,), graph), Ok):
                structured_mem_ops.append(op)

    addr_producer_ids: set[int] = set()
    traced_slots: set[tuple[int, int]] = set()

    def trace(val) -> None:
        if val is None:
            return
        op = val.def_op
        if op is None:
            found = slot_of_arg.get(val.name)
            if found is not None:
                info, j = found
                key = (id(info.op), j)
                if key in traced_slots:
                    return
                traced_slots.add(key)
                trace(info.inits[j])
                if j < len(info.yields):
                    trace(info.yields[j])
            return
        if id(op) in addr_producer_ids:
            return
        if shapes.is_memory_op(op) or shapes.is_dot(op):
            return
        addr_producer_ids.add(id(op))
        for operand in op.operands:
            trace(operand)

    for mem_op in structured_mem_ops:
        trace(shapes.pointer_operand(mem_op))
        if mem_op.name == shapes.LOAD and len(mem_op.operands) > 1:
            trace(mem_op.operands[1])
        elif mem_op.name == shapes.STORE and len(mem_op.operands) > 2:
            trace(mem_op.operands[2])

    subsumed_ids = set(addr_producer_ids)
    slots = set(traced_slots)

    def user_is_live(user: Operation, name: str) -> bool:
        """Does `user` need `name` as a value, given the current subsumed set?"""
        if id(user) in subsumed_ids:
            return False
        if shapes.is_memory_op(user):
            val_op = shapes.value_operand(user)
            return val_op is not None and val_op.name == name
        loop = loop_by_op.get(id(user))
        if loop is not None:
            if any(v is not None and v.name == name for v in (loop.lower, loop.upper, loop.step)):
                return True
            return any(
                init.name == name and (id(user), j) not in slots for j, init in enumerate(loop.inits)
            )
        loop = loop_of_yield.get(id(user))
        if loop is not None:
            return any(
                y.name == name and (id(loop.op), j) not in slots for j, y in enumerate(loop.yields)
            )
        return True

    changed = True
    while changed:
        changed = False
        for op in graph.operations:
            if id(op) not in subsumed_ids:
                continue
            if any(user_is_live(u, r.name) for r in op.results for u in graph.uses.get(r.name, ())):
                subsumed_ids.remove(id(op))
                changed = True
        for key in list(slots):
            info = next(i for i in loops if id(i.op) == key[0])
            j = key[1]
            arg = info.iter_args[j]
            results = info.results
            arg_live = any(user_is_live(u, arg.name) for u in graph.uses.get(arg.name, ()))
            result_live = j < len(results) and any(
                user_is_live(u, results[j].name) for u in graph.uses.get(results[j].name, ())
            )
            if arg_live or result_live:
                slots.discard(key)
                changed = True

    return subsumed_ids, slots


def find_subsumed_address_ops(module: Module, graph: DefUseGraph, space: str = DEFAULT_SPACE) -> set[int]:
    """Ids of operations whose computation is fully subsumed into structured memory descriptors."""
    return _address_cone(module, graph, space)[0]


def find_subsumed_loop_slots(
    module: Module, graph: DefUseGraph, space: str = DEFAULT_SPACE
) -> set[tuple[int, int]]:
    """`(id(scf.for op), slot)` pairs whose loop-carried value is a subsumed pointer chain."""
    return _address_cone(module, graph, space)[1]


def annotate(
    module: Module,
    graph: DefUseGraph | None = None,
    *,
    instruction_for: Mapping[str, str] | None = None,
    select_here: bool = False,
    unsafe: tuple[str, ...] = (),
    space: str = DEFAULT_SPACE,
    elide_address_math: bool = True,
) -> AnnotationSet:
    """Every operation of `module` → its binding, or its explicit refusal.

    `instruction_for` maps a rule name to an instruction name for callers that
    already hold a schema (`{"memory": "DMA1D", …}`); when it is `None` the binding
    names no instruction and assembly selects one. `unsafe` lists operation names
    to leave unannotated, which is the contract's "an annotation covers an
    operation no instruction can express" failure mode, driven by name rather than
    by hoping a fixture contains one.
    """
    if graph is None:
        graph = build_def_use(module)

    annotations = AnnotationSet(matches=detect_all(module, graph))
    refusals: list[tuple[str, str]] = []
    unsupported: list[str] = []
    rules = dict(instruction_for or {})
    unsafe_set = frozenset(unsafe)

    subsumed_ids, loop_slots = _address_cone(module, graph, space) if elide_address_math else (set(), set())
    annotations.subsumed_loop_slots = frozenset(loop_slots)

    for op in walk_region(module.body):
        if op.name in unsafe_set:
            unsupported.append(op.name)
            continue
        binding = _binding_for(op, graph, rules, select_here, space, subsumed_ids)
        if binding is None:
            unsupported.append(op.name)
            continue
        if id(op) in subsumed_ids:
            binding = Binding(
                instruction=None,
                kind=binding.kind,
                operands=binding.operands,
                defs=binding.defs,
                descriptor=binding.descriptor,
                tile=binding.tile,
                subsumed=True,
            )
        if binding.reason:
            refusals.append((op.name, binding.reason))
        annotations.annotate(op, (binding,))

    annotations.refusals = tuple(refusals)
    annotations.unsupported = tuple(unsupported)
    return annotations


def _named(rules: Mapping[str, str], rule: str, select_here: bool) -> str | None:
    """The instruction name for `rule`, or `None` to let assembly select."""
    if select_here:
        return None
    return rules.get(rule)


def _binding_for(
    op: Operation,
    graph: DefUseGraph,
    rules: Mapping[str, str],
    select_here: bool,
    space: str,
    subsumed_ids: set[int] | frozenset[int] = frozenset(),
) -> Binding | None:
    defs = tuple(value.name for value in op.results)

    if shapes.is_memory_op(op):
        pointer = shapes.pointer_operand(op)
        if pointer is None:  # pragma: no cover - shapes.MEMORY_OPS implies operand 0
            return None
        result = describe(pointer, (space,), graph)
        if isinstance(result, Unstructured):
            return Binding(
                instruction=None,
                kind=RULE_MEMORY,
                reason=f"unstructured access at {pointer.name}: {result.reason}",
                defs=defs,
            )
        if isinstance(result, BudgetExhausted):
            return Binding(
                instruction=None,
                kind=RULE_MEMORY,
                reason=(
                    f"unstructured access at {pointer.name}: the walk exhausted its "
                    f"{result.limit}-hop budget"
                ),
                defs=defs,
            )
        descriptor: AccessDescriptor | None = result.descriptor
        if descriptor is None:  # pragma: no cover - Ok always carries one
            return None
        base_name = descriptor.base if (subsumed_ids and descriptor.base) else pointer.name
        operands: dict[str, object] = {
            "src" if op.name == shapes.LOAD else "dst": MemRef.of(space, base_name, descriptor),
        }
        value = shapes.value_operand(op)
        if value is not None:
            operands["value"] = _reference(value, graph)
        mask = _mask_operand(op)
        if mask is not None:
            if not (subsumed_ids and mask.def_op and id(mask.def_op) in subsumed_ids):
                operands["mask"] = _reference(mask, graph)
        return Binding(
            instruction=_named(rules, RULE_MEMORY, select_here),
            kind=RULE_MEMORY,
            operands=operands,
            defs=defs,
            descriptor=descriptor,
        )

    if shapes.is_dot(op):
        tile = tile_of(op)
        operands = {
            role: _reference(value, graph)
            for role, value in zip(DOT_OPERAND_ROLES, op.operands, strict=False)
        }
        return Binding(
            instruction=_named(rules, RULE_MAC, select_here),
            kind=RULE_MAC,
            operands=operands,
            defs=defs,
            descriptor=_mac_descriptor(op, graph, space),
            tile=tile,
        )

    if not op.operands:
        # A value-producing operation with nothing to read: `arith.constant`,
        # `tt.get_program_id`, `tt.make_range`. Leaving it unannotated would make
        # *every* consumer unavailable and the coverage report would blame the
        # consumers instead of this operation, which is a real lowering the ISA can
        # express (the constant is materialised).
        result = elementwise_descriptor(op, graph)
        if not isinstance(result, Ok):
            return Binding(
                instruction=None,
                kind=RULE_ELEMENTWISE,
                reason=result.reason,
                defs=defs,
            )
        materialised = _materialised_value(op)
        if not materialised.ok:
            return Binding(
                instruction=None,
                kind=RULE_ELEMENTWISE,
                reason=materialised.reason,
                defs=defs,
            )
        return Binding(
            instruction=_named(rules, RULE_ELEMENTWISE, select_here),
            kind=RULE_ELEMENTWISE,
            operands={"value": materialised.imm},
            defs=defs,
            descriptor=result.descriptor,
        )

    result = elementwise_descriptor(op, graph)
    if not isinstance(result, Ok):
        return Binding(
            instruction=None,
            kind=RULE_ELEMENTWISE,
            reason=result.reason,
            defs=defs,
        )
    operands: dict[str, object] = {
        f"in{index}": _reference(value, graph) for index, value in enumerate(op.operands)
    }
    if op.name in COMPARE_OPS:
        # The predicate is what makes a compare a compare: without it `sge` and `ne` are the
        # same instruction. It travels as the MLIR enum value in an Imm operand.
        literals = op.literal_tokens
        try:
            operands["predicate"] = Imm(value=predicate_code(op.name, literals[0] if literals else None))
        except PredicateError as error:
            return Binding(
                instruction=None,
                kind=RULE_ELEMENTWISE,
                reason=str(error),
                defs=defs,
            )
    return Binding(
        instruction=_named(rules, RULE_ELEMENTWISE, select_here),
        kind=RULE_ELEMENTWISE,
        operands=operands,
        defs=defs,
        descriptor=result.descriptor,
    )


#: The axis names `tt.get_program_id` accepts, in the order the emulator indexes
#: them. Triton spells the axis as the operation's operand token
#: (`tt.get_program_id x`).
PROGRAM_ID_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class _Materialised:
    """The immediate a no-operand operation materialises, or why there is none."""

    imm: Imm | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.imm is not None


def _literal_int(text: str) -> int | None:
    """The integer a `"<n> : i32"` attribute text starts with, or `None`."""
    head = text.strip().split(":", 1)[0].strip()
    try:
        return int(head, 0)
    except ValueError:
        return None


def _materialised_value(op: Operation) -> _Materialised:
    """The immediate a no-operand operation produces, as an `Imm`.

    Three operations reach here, they do not materialise the same kind of thing,
    and the program format has one operand slot for it — so the meaning is a
    function of the operation name, which the emitted program does record
    (`SourceRef.op_name`) and which this function is the only producer of:

    * `arith.constant` → the literal itself, `int` or `float`;
    * `tt.make_range` → the range's **length** (`end - start`);
    * `tt.get_program_id` → the **axis index** (`x`=0, `y`=1, `z`=2).

    Every one of the three used to record `Imm(0)`. `%c64_i32` said 0; a program
    id lost its axis. That made the emitted program *unexecutable as written*:
    the emulator runs the instruction stream and explicitly does not
    re-derive the kernel from the source module, and a stream claiming every
    constant is zero cannot be run that way. Nothing caught it because the
    emulator is the first consumer that has to read the value; every earlier
    consumer — the schema, the selector, the serialiser — only had to carry it.

    A non-zero range `start` is refused rather than dropped. `Imm(end - start)`
    cannot express an offset range, and encoding the length anyway would be a
    silent wrong answer of exactly the kind the placeholder already was.
    """
    if op.name == shapes.CONSTANT:
        constant = shapes.decode_constant(op)
        if not constant.ok:
            return _Materialised(reason=f"undecodable constant: {constant.reason}")
        return _Materialised(imm=Imm(constant.value))
    if op.name == "tt.make_range":
        start = _literal_int(op.attr.get("start", ""))
        end = _literal_int(op.attr.get("end", ""))
        if start is None or end is None:
            return _Materialised(reason="tt.make_range without decodable start/end bounds")
        if start != 0:
            return _Materialised(
                reason=(
                    f"tt.make_range with a non-zero start ({start}); the materialised "
                    "operand records a length and cannot express an offset range"
                )
            )
        return _Materialised(imm=Imm(end - start))
    if op.name == "tt.get_program_id":
        axis = op.tokens[0].strip() if op.tokens else ""
        if axis not in PROGRAM_ID_AXES:
            return _Materialised(reason=f"tt.get_program_id with an unknown axis {axis!r}")
        return _Materialised(imm=Imm(PROGRAM_ID_AXES.index(axis)))
    return _Materialised(
        reason=(
            f"{op.name} produces a value but materialises nothing this program can "
            "express; a fabricated operand would be executed as if it were real"
        )
    )


def _mask_operand(op: Operation) -> SsaValue | None:
    """A masked load's or store's mask: operand 1 for a load, operand 2 for a store.

    Positional `tt.load %p, %mask, %other` / `tt.store %p, %value, %mask`, which
    is the printer's convention rather than a discovery — named here so a change to
    it shows up as a missing mask instead of as a silently unmasked access.
    """
    if op.name == shapes.LOAD and len(op.operands) > 1:
        return op.operands[1]
    if op.name == shapes.STORE and len(op.operands) > 2:
        return op.operands[2]
    return None


def _reference(value: SsaValue, graph: DefUseGraph) -> object:
    """A register reference, unless the value is the result of a memory read.

    A `tt.load`'s result lives in whatever the load put it in, so a consumer of it
    referencing the *value* (rather than the access) is the honest model: an
    instruction stream that re-addressed the pointer would be loading a second
    time. `A` in a `tt.dot` is exactly this case — it is fed by a `tt.load`, and
    the emitter's dangling-operand rule needs the load's result name, not a fresh
    memory reference.
    """
    del graph
    return SsaRef(value.name)


def _mac_descriptor(op: Operation, graph: DefUseGraph, space: str) -> AccessDescriptor | None:
    """The A-tile's access descriptor, so a MAC's constraint has something to check.

    Walking `a` back to the `tt.load` that produced it and describing *its* pointer
    is what makes `aligned(a_base, 4)`, `m % 16 == 0` and `stride[1] == 1`
    decidable rather than `unknown` — and `unknown` is inadmissible under
    `isa/schema.py:evaluate`'s fail-closed rule. `None` when `a` is not
    the result of a describable load, which the emitter handles as any other
    absent descriptor.
    """
    if not op.operands:
        return None
    producer = op.operands[0].def_op
    if producer is None or not shapes.is_memory_op(producer):
        return None
    pointer = shapes.pointer_operand(producer)
    if pointer is None:
        return None
    result = describe(pointer, (space,), graph)
    if isinstance(result, Ok):
        return result.descriptor
    return None


__all__ = [
    "DEFAULT_SPACE",
    "RULE_ELEMENTWISE",
    "RULE_MAC",
    "RULE_MEMORY",
    "AnnotationSet",
    "annotate",
    "detect_all",
    "detect_epilogue",
    "detect_mac",
    "find_subsumed_address_ops",
    "find_subsumed_loop_slots",
]
