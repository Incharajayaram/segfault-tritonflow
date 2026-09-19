"""`assemble` — region-aware emission: Module + annotations + schema -> Program.

The one thing this module must not do is flatten. Both `tt.load`s of Tier 1 live
inside `scf.for`, `scf.yield` terminates the loop body, and the accumulator is
loop-carried: a flat topological sort over the def-use graph loses all three
facts and produces a program that looks plausible and computes the wrong thing.
So the unit of ordering here is the **region**:
:func:`order_regions` walks the region tree and reports, for every operation,
which region it belongs to; nothing downstream ever sees an operation without
one.

Placement follows from that, and it is mechanical rather than clever:

| Where the operation is | Where the instruction goes |
|---|---|
| in the kernel body, before any loop | `Program.instrs` |
| in a loop body | that `Loop.body`, with `Instr.loop == Loop.id` |
| in the kernel body, after a loop, reading a value the loop yields | `Program.epilogue` |
| in a nested `scf.for` | `UnsupportedMarker`, never a flattened instruction |

`scf.yield` is not an instruction. It is how the loop re-threads its
`iter_args`, so it is recorded as the loop's `results` and skipped.
`tt.return` produces no value and is skipped for the same reason.
Every other value-producing operation is either an `Instr` or a marker — the
union is exact, and `Program`'s construction-time validation is what checks it.

**Who decides what.** `assemble` does not choose instructions. The
recogniser/selector side does, and hands the decision over as a `Binding`
(`emit.ir.Binding`) through the annotation set. Assembly costs it, places it, and
**re-checks its constraint against the operand it was chosen for**
(:func:`check_constraint`) — re-validating rather than trusting the selector
transitively, because a selector bug that silently picks an inadmissible
instruction is exactly the failure this layer exists to catch. Where the schema
exposes an evaluator (`isa.schema.evaluate`), the predicate is re-evaluated too;
otherwise the check is the structural half — the constraint must be
present, and must have been chosen for *this* operand.

**The two seams are duck-typed on purpose.** `SchemaLike` and
`AnnotationSetLike` (in `emit.ir`) name the smallest sets of members assembly
needs from `isa/schema.py` and `idioms/detect.py`, so that every layer stays
exercisable against hand-built input, and a `Protocol` is the cheapest way to
say "this is the shape I consume" without inventing someone else's module.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import replace

from ..ttir.graph import DefUseGraph, iter_loops, topo_within_region
from ..ttir.ssa import Block, Module, Operation, Region
from .ir import (
    AssemblyError,
    Binding,
    EmissionRecord,
    Imm,
    Instr,
    InstructionLike,
    Loop,
    MemRef,
    OrderedOp,
    Program,
    SourceRef,
    SsaRef,
    UnsupportedMarker,
    operand_key,
)

FOR_OP = "scf.for"

#: Operations that produce no value and end a block. `scf.yield` re-threads the
#: loop's `iter_args`; `tt.return` ends the kernel. Neither is an instruction.
TERMINATOR_OPS = ("scf.yield", "tt.return")

#: Region-owning operations that are *not* lowerable operations: `tt.func` is the
#: kernel definition itself. Skipping it is not a dropped operation — it produces
#: no value, and its arguments are already `Program.inputs` via `_inputs()`. Its
#: body still belongs to the kernel region, which is why `order_regions` descends
#: into it instead of treating the module's top-level block as the whole module.
CONTAINER_OPS = ("tt.func",)

#: Operand roles, in the order a constraint is checked against. A schema's
#: constraint terms (`base`, `length`, `m`, `n`, `k`) belong to the memory or
#: accumulator operand, so the check follows the same preference the recogniser
#: uses when it fits an operand to a role.
PRIMARY_ROLES = ("acc", "dst", "src", "a", "b")


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def order_regions(module: Module, graph: DefUseGraph) -> list[OrderedOp]:
    """Every operation in emission order, each with the region it belongs to.

    Region-aware, source order within each region, and never a flat topological
    sort: a loop's own operation is reported *before* its body, and its body is
    reported with `loop_id` set, so one pass over this list can create the
    `Loop` and then fill it.

    `graph` checks that every ordered operation is one the graph knows about —
    passing a graph built from a different module is a bug that otherwise
    surfaces much later as a missing value.
    """
    known = set(map(id, graph.operations))
    loop_ids = {id(info.op): index for index, info in enumerate(iter_loops(module))}
    ordered: list[OrderedOp] = []
    _walk(module.body, None, False, loop_ids, known, ordered)
    return ordered


def _walk(
    region: Region,
    loop_id: int | None,
    nested: bool,
    loop_ids: dict[int, int],
    known: set[int],
    out: list[OrderedOp],
) -> None:
    for index, op in enumerate(topo_within_region(region)):
        if id(op) not in known:
            raise AssemblyError(
                f"{op.name} at line {op.line} is in the module but not in the def-use "
                "graph; was the graph built from a different module?"
            )
        out.append(
            OrderedOp(
                op=op,
                region=region,
                loop_id=loop_id,
                index=index,
                terminator=op.name in TERMINATOR_OPS,
                nested=nested,
            )
        )
        for child in op.regions:
            if op.name == FOR_OP:
                # A loop is *created* only when it sits in the kernel body. A
                # loop already inside a loop hands its own body down as nested:
                # the program format has no nested loop, so those operations are
                # marked rather than hoisted out of the region they belong to.
                owns_loop = loop_id is None and id(op) in loop_ids
                _walk(
                    child,
                    loop_ids[id(op)] if owns_loop else loop_id,
                    nested or not owns_loop,
                    loop_ids,
                    known,
                    out,
                )
                continue
            # Any other region-owning operation is a *container* — `tt.func`
            # holds the kernel body, and descending into it is what makes this a
            # traversal of the module rather than of its top-level block. A
            # container passes its own context down unchanged: it neither
            # creates a loop nor makes what is inside it nested.
            _walk(child, loop_id, nested, loop_ids, known, out)


# --------------------------------------------------------------------------- #
# One instruction
# --------------------------------------------------------------------------- #


def emit_instr(
    op: Operation,
    binding: Binding | None,
    schema: object | None = None,
    report: list[EmissionRecord] | None = None,
    *,
    loop: int | None = None,
    rejected: tuple[str, ...] = (),
    oracle_min_cost: float | None = None,
    gap: float | None = None,
    env: dict[str, int] | None = None,
) -> Instr | UnsupportedMarker:
    """Turn one decision into one instruction, or one marker. Never a guess.

    `loop` is placement, and placement is the caller's decision — but it is
    passed *in* rather than patched on afterwards, so an `Instr` never exists in
    a state where it has forgotten which region it came from.

    `rejected`, `oracle_min_cost` and `gap` are the *selection* evidence.
They are keyword-only and default to empty so a caller that
    chose the instruction itself passes nothing. They live here rather than on a
    separate record because a rejection with no instruction attached to it is not
    evidence of anything: the first version of this recorded the rejections on
    one record and the instruction on another, and a reader could not tell which
    choice the rejections belonged to.
    """
    source = SourceRef.of(op)

    if binding is None or binding.instruction is None:
        reason = (binding.reason if binding is not None else None) or (
            "no annotation: no idiom matched this operation and no rule lowered it"
        )
        marker = UnsupportedMarker.from_op(op, reason)
        _record(report, op, None, reason, loop, rejected=rejected, oracle_min_cost=oracle_min_cost)
        return marker

    instruction = _lookup_instruction(schema, binding.instruction)
    if instruction is None:
        reason = (
            f"instruction {binding.instruction!r} is not in the loaded schema; "
            "refusing to invent a lowering for it"
        )
        marker = UnsupportedMarker.from_op(op, reason)
        _record(report, op, None, reason, loop, rejected=rejected, oracle_min_cost=oracle_min_cost)
        return marker

    if not binding.operands:
        reason = (
            f"{binding.instruction} was chosen for {op.name} with no operands; an "
            "instruction with nothing to operate on is a decision, not a lowering"
        )
        marker = UnsupportedMarker.from_op(op, reason)
        _record(report, op, None, reason, loop, rejected=rejected, oracle_min_cost=oracle_min_cost)
        return marker

    cost = binding.cost if binding.cost is not None else _cost_of(instruction, binding, env)
    constrained = _constraint_operand(binding)
    cost_res = getattr(binding, "cost_result", None)
    if cost_res is None and hasattr(instruction, "name"):
        from tritonflow.isa.cost import CostQuery, CostResultError, CostUnknown, evaluate_cost
        try:
            cost_res = evaluate_cost(
                CostQuery(
                    instruction=instruction,
                    access=getattr(binding, "descriptor", None),
                    tile=getattr(binding, "tile", None),
                    env=env or {},
                )
            )
        except (CostUnknown, CostResultError) as exc:
            cost_res = None
            rejected = (*rejected, f"cost model: no structured cost for {instruction.name} ({exc})")
    instr = Instr(
        name=instruction.name,
        operands=dict(binding.operands),
        loop=loop,
        cost=float(cost),
        source_ops=(source,),
        defs=tuple(binding.defs),
        constraint=_constraint_text(getattr(instruction, "constraint", None)),
        constrained_on=None if constrained is None else _constraint_key(constrained),
        cost_result=cost_res,
        select_cost=float(cost),
    )
    _record(
        report, op, instr, None, loop, rejected=rejected, oracle_min_cost=oracle_min_cost, gap=gap
    )
    return instr


def check_constraint(
    instr: Instr,
    operand: object | None,
    *,
    evaluate: object | None = None,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
) -> None:
    """Re-validate `instr`'s own constraint against the operand it was chosen for.

    Three refusals, in increasing strength:

    1. the instruction records no constraint — nothing to re-check, and "nothing
       to check" must not read as "checked and fine";
    2. the operand is not the one the instruction was chosen for (the key
       differs) — the transitive-trust failure: a constraint validated against
       operand A says nothing about operand B;
    3. an evaluator is available and does not return exactly `True` — the
       predicate language is fail-closed (`evaluate` returns `False` or
 `"unknown"` for anything it cannot decide,), and `"unknown"` here
       is a violation rather than a pass.
    """
    if not instr.constraint:
        raise AssemblyError(
            f"{instr.name} records no constraint, so its admissibility cannot be "
            "re-checked independently; the emitter must carry the constraint it was "
            "selected under"
        )
    key = _constraint_key(operand)
    if instr.constrained_on is not None and key != instr.constrained_on:
        raise AssemblyError(
            f"{instr.name} was chosen for operand {instr.constrained_on!r} but is being "
            f"checked against {key!r}; a constraint that held for one operand does not "
            "transfer to another"
        )
    if callable(evaluate):
        verdict = _call_evaluator(evaluate, instr.constraint, operand, tile, env)
        if verdict is not True:
            raise AssemblyError(
                f"{instr.name} violates its own constraint {instr.constraint!r} on "
                f"{key}: evaluate() returned {verdict!r} (fail-closed: only True passes)"
            )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble(
    module: Module,
    graph: DefUseGraph,
    annotations: object,
    schema: object,
    *,
    report: list[EmissionRecord] | None = None,
    selector: object | None = None,
    env: dict[str, int] | None = None,
) -> Program:
    """`Module` + annotations + schema -> `Program`.

    One pass over `order_regions`, each instruction or marker appended to the
    container its region dictates. Three contract failure modes are refused
    loudly rather than absorbed:

    * **two annotations claim one operation** — an internal error naming both,
      because silently keeping one of them is a decision nobody made;
    * **an annotation names an instruction the schema does not have** — an
      `UNSUPPORTED` marker, because that is a schema gap, not a kernel we cannot
      express;
    * **a selection is needed and the selector is missing** — an `AssemblyError`,
      not an `UNSUPPORTED` marker: a missing stage must not be reportable as a
      limitation of the target ISA.

    `selector` defaults to `isa.select.select` when that module is importable
    (`select(schema, kind, descriptor, tile, env) -> SelectionReport`), so the
    real pipeline passes nothing and a test passes a stub.

    `env` is the launch environment the schema's symbolic terms resolve against
    (declared in `fixtures/launch_env.json`). It is
    threaded into selection and costing because without it every symbolic
    predicate is `unknown` — selection fail-closes to the conservative variant,
    or to `UNSUPPORTED`, which reads as an ISA limitation that is really a
    missing fact. Found by the production-path check: with the stand-in selector
    (satisfied by construction) the drop was invisible; with the real selector it
    turned every matmul tile into DMA1D and costed a 2048-word access at 1.0.
    """
    loop_infos = list(iter_loops(module))
    by_op = {id(info.op): info for info in loop_infos}
    loop_ids = {id(info.op): index for index, info in enumerate(loop_infos)}
    loop_lines = [info.op.line for info in loop_infos]
    evaluator = getattr(schema, "evaluate", None)
    available = set(_inputs(module))
    producers = {
        name: (value.def_op.name if value.def_op is not None else None)
        for name, value in graph.defs.items()
    }

    loops: list[Loop] = []
    loop_bodies: dict[int, list[Instr | UnsupportedMarker]] = {}
    instrs: list[Instr] = []
    epilogue: list[Instr] = []
    unsupported: list[UnsupportedMarker] = []

    resolve = selector if selector is not None else _default_selector()

    for item in order_regions(module, graph):
        op = item.op

        if op.name in TERMINATOR_OPS or op.name in CONTAINER_OPS:
            continue  # re-threading / return / kernel definition: not an instruction

        if op.name == FOR_OP and item.loop_id is None and not item.nested:
            dropped = frozenset(
                j for (loop_op_id, j) in getattr(annotations, "subsumed_loop_slots", frozenset()) if loop_op_id == id(op)
            )
            loop = _make_loop(by_op[id(op)], loop_ids[id(op)], dropped)
            loops.append(loop)
            loop_bodies[loop.id] = []
            # The loop's induction variable, its `iter_args` and the values it
            # yields are all available from here on: `order_regions` guarantees
            # the loop is reported before its body and before the post-loop
            # operations, and SSA guarantees nothing earlier can name them.
            available.update(_loop_values(loop))
            _record(report, op, None, None, item.loop_id)
            continue

        if item.nested or op.name == FOR_OP:
            reason = (
                "nested scf.for: the program format has no nested loop, and flattening "
                "it would hoist its body out of the region it belongs to"
                if op.name == FOR_OP
                else "inside a nested scf.for; not representable, and not hoisted"
            )
            marker = UnsupportedMarker.from_op(op, reason)
            _place(marker, item.loop_id, loop_bodies, unsupported)
            _record(report, op, None, reason, item.loop_id)
            continue

        emitted = _emit_checked(
            op,
            annotations,
            schema,
            resolve,
            item.loop_id,
            evaluator,
            report,
            available=frozenset(available),
            producers=producers,
            env=env,
        )
        if emitted is None:
            available.update(value.name for value in op.results)
            continue
        if isinstance(emitted, UnsupportedMarker):
            _place(emitted, item.loop_id, loop_bodies, unsupported)
            continue
        available.update(emitted.defs)
        if item.loop_id is not None:
            loop_bodies[item.loop_id].append(emitted)
        elif _is_epilogue(op, loop_lines):
            epilogue.append(emitted)
        else:
            instrs.append(emitted)

    finished = [replace(loop, body=tuple(loop_bodies[loop.id])) for loop in loops]
    return Program(
        isa_name=str(getattr(schema, "name", "<hand-built>")),
        schema_version=int(getattr(schema, "schema_version", 0)),
        kernel_name=_kernel_name(module),
        loops=tuple(finished),
        instrs=tuple(instrs),
        epilogue=tuple(epilogue),
        unsupported=tuple(unsupported),
        total_cost=total_cost(finished, instrs, epilogue),
        inputs=_inputs(module),
    )


def total_cost(loops: list[Loop], instrs: list[Instr], epilogue: list[Instr]) -> float:
    """`Program.total_cost` for a set of containers not yet in a `Program`.

    `math.fsum`, matching `Program.cost_sum`, so assembly and validation cannot
    disagree in the last bit.
    """
    return math.fsum(
        [instr.cost for loop in loops for instr in loop.instrs]
        + [instr.cost for instr in instrs]
        + [instr.cost for instr in epilogue]
    )


def _make_loop(info, loop_id: int, dropped: frozenset[int] = frozenset()) -> Loop:
    """The emitted `Loop`, minus loop-carried slots whose value is a subsumed pointer chain.

    A dropped slot carried an address that the memory descriptors already encode
    (base, strides, loop increment), so its initialiser, increment and yield were
    elided upstream; keeping the slot would name values no instruction defines.
    """
    op = info.op

    def keep(values):
        return tuple(v for j, v in enumerate(values) if j not in dropped)

    return Loop(
        id=loop_id,
        induction_var=info.iv.name if info.iv is not None else None,
        lower=None if info.lower is None else SsaRef(info.lower.name),
        upper=None if info.upper is None else SsaRef(info.upper.name),
        step=None if info.step is None else SsaRef(info.step.name),
        iter_args=tuple(value.name for value in keep(info.iter_args)),
        inits=tuple(value.name for value in keep(info.inits)),
        results=tuple(value.name for value in keep(info.results)),
        yields=tuple(value.name for value in keep(info.yields)),
        body=(),
        source=SourceRef.of(op),
    )


def _emit_checked(
    op: Operation,
    annotations: object,
    schema: object,
    selector: object | None,
    loop_id: int | None,
    evaluator: object,
    report: list[EmissionRecord] | None,
    env: dict[str, int] | None = None,
    *,
    available: frozenset[str] = frozenset(),
    producers: dict[str, str | None] | None = None,
) -> Instr | UnsupportedMarker:
    bindings = _bindings_for(annotations, op)
    if len(bindings) > 1:
        raise AssemblyError(
            f"{len(bindings)} annotations claim {op.name} at line {op.line} "
            f"({', '.join(sorted(str(b.instruction) for b in bindings))}); keep one — "
            "silently dropping the others is a decision nobody made"
        )
    binding = bindings[0] if bindings else None
    if binding is not None and getattr(binding, "subsumed", False):
        _record(report, op, None, "subsumed into structured memory access descriptor", loop_id)
        return None
    binding, selection = _select(selector, schema, op, binding, env)
    if binding is not None and binding.instruction is not None:
        unavailable = _unavailable(binding, available, producers or {})
        if unavailable:
            reason = _unavailable_reason(unavailable)
            marker = UnsupportedMarker.from_op(op, reason)
            _record(
                report,
                op,
                None,
                reason,
                loop_id,
                rejected=selection["rejected"],
                oracle_min_cost=selection["oracle_min_cost"],
            )
            return marker
    emitted = emit_instr(
        op,
        binding,
        schema,
        report,
        loop=loop_id,
        rejected=selection["rejected"],
        oracle_min_cost=selection["oracle_min_cost"],
        gap=selection["gap"],
        env=env,
    )
    if isinstance(emitted, UnsupportedMarker):
        return emitted
    check_constraint(
        emitted,
        _constraint_operand(binding),
        evaluate=evaluator,
        tile=binding.tile,
        env=env,
    )
    return emitted


def _loop_values(loop: Loop) -> set[str]:
    names = set(loop.iter_args) | set(loop.results)
    if loop.induction_var:
        names.add(loop.induction_var)
    return names


def _unavailable(
    binding: Binding,
    available: frozenset[str],
    producers: dict[str, str | None],
) -> list[str]:
    """Operand values this instruction would read that the program never produces.

    A value whose producer exists but was not lowered is a *propagating*
    limitation: the consumer becomes `UNSUPPORTED` too, so the frontier is
    reported instead of being papered over. A value the module does not
    define at all is an annotation bug and is raised, because no amount of
    marking makes it right — that is an operand reference nothing can ever
    satisfy, one layer up.
    """
    missing: list[str] = []
    for role, operand in sorted(binding.operands.items()):
        for name in _operand_names(operand):
            if name in available:
                continue
            if name not in producers:
                raise AssemblyError(
                    f"the annotation for this operation reads {name} (role {role!r}), which "
                    "no operation in the module defines and which is not a program input; "
                    "refusing to emit an instruction with an operand nothing can produce"
                )
            missing.append(name)
    return missing


def _operand_names(operand: object) -> tuple[str, ...]:
    if isinstance(operand, SsaRef):
        return (operand.name,)
    if isinstance(operand, MemRef):
        return (operand.base,)
    return ()


def _unavailable_reason(names: list[str]) -> str:
    producers = sorted({name for name in names})
    return (
        "operand(s) unavailable because the operation that produces them was not "
        f"lowered: {', '.join(producers)}; this instruction is marked UNSUPPORTED "
        "rather than emitted against a value the program never computes"
    )


#: The selection evidence for an operation that was not selected here.
_NO_EVIDENCE: dict = {"rejected": (), "oracle_min_cost": None, "gap": None}


def _select(
    selector: object | None,
    schema: object,
    op: Operation,
    binding: Binding | None,
    env: dict[str, int] | None = None,
) -> tuple[Binding | None, dict]:
    """Fill in `binding.instruction` when the recogniser did not choose one.

    Assembly drives selection, so this is where a rejected candidate's
    rejection reasons and the oracle gap arrive. An unannotated operation is
    *not* selected for: it is the method's explicit `UNSUPPORTED(<op name>)`
    case, and running a selector over "nothing was recognised" would invent a
    lowering for it.

    Returns the binding **and the evidence**, rather than recording the evidence
    itself. Recording here was the first version, and it produced two records per
    selected operation — one carrying the rejected candidates with no
    instruction, one carrying the instruction with no rejections — so a reader
    could not tell which choice the rejections belonged to. The caller
    now attaches the evidence to the instruction it belongs to.
    """
    if binding is None or binding.instruction is not None:
        return binding, _NO_EVIDENCE
    if binding.reason:
        # A *refusal*, not a request for selection. The recogniser produced this
        # binding to say "a memory idiom matched and the addressing is what I
        # could not describe", and running a selector over a descriptor that is
        # `None` would ask it to cost an access nobody described — with a schema
        # that registers its constraints, the selector answers, and an
        # instruction is emitted for an operand the recogniser explicitly could
        # not read. Found by writing the recognizer: every `Unstructured` operand
        # turned into a lowered instruction.
        return binding, _NO_EVIDENCE
    if not binding.kind:
        raise AssemblyError(
            f"the annotation for {op.name} at line {op.line} names no instruction and no "
            "kind, so nothing can be enumerated for it; a rejected annotation has to say "
            "which instruction set it belongs to (memory or compute)"
        )
    if not callable(selector):
        raise AssemblyError(
            f"{op.name} at line {op.line} is annotated but carries no chosen instruction, "
            "and no selector is available (isa.select.select is not importable). Pass "
            "selector=... or have the annotation name its instruction — a missing stage "
            "is not the target ISA's limitation, so it must not be reported as one"
        )
    direction = "store" if op.name == "tt.store" else ("load" if op.name == "tt.load" else None)
    selection = _call_selector(
        selector, schema, binding.kind, binding.descriptor, binding.tile, env, direction=direction
    )
    # `aligned(X, k)` decides against the ACTIVE machine's allocator promise:
    # selection is per-schema, and a second machine may promise differently.
    try:
        from ..isa.schema import set_active_schema

        set_active_schema(getattr(schema, "name", "tritonflow1"))
    except (ImportError, AttributeError, KeyError):  # stand-in schemas have no name hook
        pass
    chosen = getattr(selection, "chosen", None)
    chosen_name = _instruction_name(chosen)
    chosen_cost = getattr(selection, "chosen_cost", None)
    # Every candidate *except the chosen one* is evidence, whether it was refused
    # by its own predicate or merely cost more: e.g. `DMA1D` is "rejected with
    # `cost(1.00*words) > 0.60*words`", and the rejection is recorded rather
    # than the alternative being dropped.
    rejected = tuple(
        f"{_candidate_name(c)}: {_candidate_reason(c, chosen_cost)}"
        for c in getattr(selection, "candidates", ())
        if _candidate_name(c) != chosen_name
    )
    oracle_min_cost = getattr(selection, "oracle_min_cost", None)
    gap = getattr(selection, "gap", None)
    if getattr(selection, "no_admissible_lowering", False):
        reason = (
            f"no {binding.kind} instruction is admissible for this operand "
            f"({len(rejected)} candidate(s) rejected: {'; '.join(rejected) or 'none enumerated'})"
        )
        return (
            Binding(kind=binding.kind, descriptor=binding.descriptor, reason=reason),
            {"rejected": rejected, "oracle_min_cost": oracle_min_cost, "gap": gap},
        )

    if chosen_name is None:
        raise AssemblyError(
            f"the selector returned a selection for {op.name} that names no instruction; "
            "a SelectionReport with no admissible lowering must set no_admissible_lowering"
        )
    return (
        replace(binding, instruction=chosen_name, cost=chosen_cost),
        {"rejected": rejected, "oracle_min_cost": oracle_min_cost, "gap": gap},
    )


def _instruction_name(chosen: object | None) -> str | None:
    if chosen is None:
        return None
    name = getattr(chosen, "name", None)
    return str(name) if name is not None else str(chosen)


def _candidate_name(candidate: object) -> str:
    return _instruction_name(getattr(candidate, "instruction", None)) or "?"


def _candidate_reason(candidate: object, chosen_cost: float | None) -> str:
    """Why this candidate was not chosen, as text.

    Three cases, and the third is the one worth naming: `rejected_by` set is a
    predicate that failed (`selector.md` postcondition 2 forbids it being empty
    for a rejected candidate); admissible-but-costlier is a *cost* comparison,
    which the contract's expected-results table also calls a rejection; and a
    candidate with neither is reported as unreasoned rather than rendered as an
    empty column someone will read as "fine".
    """
    reason = getattr(candidate, "rejected_by", None)
    if reason:
        return str(reason)
    if getattr(candidate, "admissible", False):
        cost = getattr(candidate, "cost", None)
        return f"cost({cost}) > {chosen_cost}"
    return "rejected with no predicate recorded (selector.md postcondition 2)"


def _call_selector(
    selector: object,
    schema: object,
    kind: str,
    descriptor: object,
    tile: tuple[int, ...] | None,
    env: dict[str, int] | None,
    direction: str | None = None,
) -> object:
    """Call the selector with the arguments its signature accepts."""
    args = (schema, kind, descriptor, tile, env)
    limit = _positional_arity(selector)
    if direction is not None and limit is not None and limit >= 6:
        return selector(schema, kind, descriptor, tile, env, direction)
    if limit is None or limit >= len(args):
        try:
            return selector(*args, direction=direction)
        except TypeError:
            return selector(*args)
    return selector(*args[:limit])


def _call_evaluator(
    evaluate: object,
    constraint: str,
    operand: object,
    tile: tuple[int, ...] | None,
    env: dict[str, int] | None,
) -> object:
    """Same arity adaptation for the schema's constraint evaluator.

    The real schema evaluates `(constraint, descriptor, tile, env)`; the stand-in
    predates tile and env. Dropping them silently would fail-closed every real
    re-validation to `unknown` — `check_constraint` only passes on exactly
    `True` — so the arity probe is load-bearing, not cosmetic.
    """
    args = (constraint, operand, tile, env)
    limit = _positional_arity(evaluate)
    if limit is None or limit >= len(args):
        return evaluate(*args)
    return evaluate(*args[:2])


def _positional_arity(callable_: object) -> int | None:
    """Number of positional parameters the callable accepts *as seen by a caller*.

    `inspect.signature` on a bound method already hides `self`, so the stand-in's
    `(self, schema, kind, descriptor, tile)` and the real module function's
    `(schema, kind, descriptor, tile, env)` measure correctly against the same
    probe. `None` means "unknown signature — pass everything and let it raise".
    """
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return None
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        else:
            break
    return count


def _default_selector() -> object | None:
    """`isa.select.select` if the `isa` package is importable, else `None`.

    Imported lazily and tolerated missing on purpose: every layer must be
    exercisable before its neighbours land, so a test that supplies its own
    selector must not need `isa/` to exist.
    """
    try:
        from ..isa.select import select
    except ImportError:  # pragma: no cover - state before isa/select.py exists
        return None
    return select


def _bindings_for(annotations: object, op: Operation) -> tuple[Binding, ...]:
    getter = getattr(annotations, "bindings_for", None)
    if not callable(getter):
        raise AssemblyError(
            "the annotation set passed to assemble() has no bindings_for(op); it must "
            f"satisfy emit.ir.AnnotationSetLike (got {type(annotations).__name__})"
        )
    return tuple(getter(op))


def _lookup_instruction(schema: object | None, name: str) -> InstructionLike | None:
    if schema is None:
        return None
    getter = getattr(schema, "instruction", None)
    if callable(getter):
        return getter(name)
    if isinstance(schema, Mapping):
        return schema.get(name)
    raise AssemblyError(
        f"schema of type {type(schema).__name__} provides no instruction lookup; it must "
        "satisfy emit.ir.SchemaLike (an `instruction(name)` method) or be a mapping"
    )


def _cost_of(
    instruction: InstructionLike,
    binding: Binding,
    env: dict[str, int] | None,
) -> float:
    cost_for = getattr(instruction, "cost_for", None)
    if callable(cost_for):
        # The real schema grounds symbolic terms (`words`, strides) against the
        # launch environment. Without it a per-word cost degenerates to its
        # constant factor — 1.00 on a 2048-word access: still a number, so
        # nothing downstream notices the program was mispriced.
        return float(cost_for(binding.descriptor, binding.tile, env))
    cost_of = getattr(instruction, "cost_of", None)
    if callable(cost_of):
        return float(cost_of(binding.descriptor, binding.tile))
    cost = getattr(instruction, "cost", None)
    if isinstance(cost, int | float):
        return float(cost)
    raise AssemblyError(
        f"instruction {instruction.name!r} carries neither a cost_for()/cost_of() method "
        "nor a numeric cost; refusing to emit an instruction whose cost we cannot account for"
    )


def _constraint_operand(binding: Binding | None) -> object | None:
    if binding is None:
        return None
    if binding.descriptor is not None:
        return binding.descriptor
    for role in PRIMARY_ROLES:
        if role in binding.operands:
            return binding.operands[role]
    return next(iter(binding.operands.values()), None)


def _constraint_text(value: object | None) -> str | None:
    """The instruction's constraint as *text*.

    The schema types a constraint as a `Predicate`,
    which carries its own authoritative text; the frozen stand-in stores a plain
    string. `Instr.constraint` is `str | None` — that is what the serialiser
    writes and the deserialiser parses back — so the coercion belongs here, at
    the boundary between the two representations.

    It is here rather than in the serialiser on purpose. Widening `_q` would
    have made the crash disappear while leaving a `Predicate` object in a field
    both `ir.py` and the file format declare to be text; the next consumer to
    read `Instr.constraint` would have met the same surprise. Anything that is
    neither text nor carries text is an `AssemblyError` naming the type, not a
    `str()` conversion that would serialise an unparseable constraint the
    deserialiser could not read back.
    """
    if value is None or isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    raise AssemblyError(
        f"a schema constraint must be text or carry it (a parsed predicate), got "
        f"{type(value).__name__}; carrying a non-text constraint would serialise to a "
        "program the deserialiser cannot read back"
    )


def _constraint_key(operand: object | None) -> str:
    """The identity a constraint is checked against.

    An `Operand` by `operand_key`, a recognizer descriptor by its canonical
    `descriptor_key()`. Both are text, so the two paths cannot silently compare
    different kinds of thing.
    """
    if operand is None:
        return "none"
    if isinstance(operand, SsaRef | Imm | MemRef):
        return operand_key(operand)
    key = getattr(operand, "descriptor_key", None)
    if callable(key):
        return str(key())
    return str(operand)


def _place(
    marker: UnsupportedMarker,
    loop_id: int | None,
    loop_bodies: dict[int, list[Instr | UnsupportedMarker]],
    unsupported: list[UnsupportedMarker],
) -> None:
    if loop_id is None:
        unsupported.append(marker)
    else:
        loop_bodies[loop_id].append(marker)


def _is_epilogue(op: Operation, loop_lines: list[int]) -> bool:
    """Post-loop.

    "Post-loop" is a *position*, not a dataflow relation: every operation after
    the last `scf.for` is epilogue, whether or not it reads a loop result
    directly. Classifying only the direct consumers split one contiguous source
    region across two containers, and `Program.execution_order()` walks `body`
    before `epilogue` — so `%out_28 = addf %acc#2, ...` (line 74, epilogue) was
    scheduled *after* the `tt.store` at line 85 that consumes it (finding F9,
    caught by executing the kernel). A container split that contradicts source
    order is not a labelling preference; it is a wrong program.
    """
    if not loop_lines:
        return False
    return op.line > max(loop_lines)


def _kernel_name(module: Module) -> str:
    functions = module.functions()
    if functions:
        return functions[0].name
    return module.source_path.rsplit("/", 1)[-1].removesuffix(".ttir")


def _inputs(module: Module) -> tuple[str, ...]:
    """Values that arrive from outside the program.

    Kernel arguments plus any block arguments opening the module region. This is
    what makes "dangling operand" decidable instead of indistinguishable from
    "came from the caller" (`emit.ir.Program.inputs`).
    """
    names: list[str] = []
    entry: Block | None = module.body.entry
    if entry is not None:
        names.extend(arg.name for arg in entry.args)
    for function in module.functions():
        names.extend(arg.name for arg in function.args)
    return tuple(dict.fromkeys(names))


def _record(
    report: list[EmissionRecord] | None,
    op: Operation,
    instr: Instr | None,
    reason: str | None,
    loop_id: int | None,
    *,
    rejected: tuple[str, ...] = (),
    oracle_min_cost: float | None = None,
    gap: float | None = None,
) -> None:
    if report is None:
        return
    report.append(
        EmissionRecord(
            op_name=op.name,
            source=SourceRef.of(op),
            instruction=None if instr is None else instr.name,
            cost=None if instr is None else instr.cost,
            reason=reason,
            loop_id=loop_id,
            placement="kernel body" if loop_id is None else f"loop {loop_id}",
            rejected=rejected,
            oracle_min_cost=oracle_min_cost,
            gap=gap,
        )
    )


__all__ = [
    "CONTAINER_OPS",
    "FOR_OP",
    "PRIMARY_ROLES",
    "TERMINATOR_OPS",
    "assemble",
    "check_constraint",
    "emit_instr",
    "order_regions",
    "total_cost",
]
