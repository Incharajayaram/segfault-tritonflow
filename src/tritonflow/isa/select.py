"""Instruction selection: enumerate, filter fail-closed, min cost, report the gap.

The whole module is five small functions because the schema is small by
design: exhaustive enumeration *is* the oracle, which is what makes "the
generator chose" auditable rather than asserted.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from tritonflow.isa.semantics import is_parseable

from .schema import Instruction, IsaSchema, evaluate, set_active_schema

__all__ = ["Candidate", "SelectionReport", "enumerate_candidates", "oracle_min", "select"]


@dataclass(frozen=True)
class Candidate:
    """One instruction's verdict against one operand."""

    instruction: Instruction
    admissible: bool
    rejected_by: str | None  # the predicate text that failed, or "unknown"
    cost: float | None
    cost_result: object | None = None


@dataclass(frozen=True)
class SelectionReport:
    """The full decision: everything enumerated, rejections attributed, gap visible."""

    chosen: Instruction | None
    chosen_cost: float | None
    candidates: tuple[Candidate, ...] = ()
    oracle_min_cost: float | None = None
    oracle_chosen: Instruction | None = None
    gap: float = 0.0
    no_admissible_lowering: bool = False
    chosen_cost_result: object | None = None


def _matches_op(entry: str, base_op: str) -> bool:
    """Check if an opcode entry declared in schema matches the descriptor operation."""
    entry = entry.lower()
    base = base_op.lower()
    if entry == base:
        return True
    if "." in base:
        suffix = base.split(".", 1)[1]
        if entry == suffix:
            return True
    if "." in entry:
        entry_suffix = entry.split(".", 1)[1]
        if entry_suffix == base:
            return True
    return False


def enumerate_candidates(
    schema: IsaSchema,
    kind: str,
    descriptor: object,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
    direction: str | None = None,
    machine: object | None = None,
) -> tuple[Candidate, ...]:
    set_active_schema(schema.name)
    """Every instruction of `kind` against this operand, including rejected ones.

    Postcondition 1: nothing is pre-filtered — the rejected rows are the audit
    trail. Postcondition 2: a rejection names the failing predicate's *text*
    (never `"unknown"` as a reason string; an undecidable predicate is recorded
    with the literal verdict so the reason is visible).
    """

    def reason_of(instruction: object) -> str:
        """The predicate's author text, whatever shape the schema stores it in.

        The real schema stores parsed `Predicate` objects; a frozen stand-in
        stores the raw string. Both must yield the *text*,
        because the rejection reason is the predicate the author
        wrote, and a schema shape difference must not change what is reported.
        """
        constraint = getattr(instruction, "constraint", "")
        return constraint if isinstance(constraint, str) else constraint.text

    out: list[Candidate] = []
    for instruction in schema.of_kind(kind):
        if not getattr(instruction, 'semantics', None) or not is_parseable(instruction.semantics):
            out.append(
                Candidate(
                    instruction,
                    False,
                    f"instruction has missing or unparseable semantics: {getattr(instruction, 'semantics', None)!r}",
                    None,
                )
            )
            continue
        serves = getattr(instruction, "serves", None)
        if direction is not None and callable(serves) and not serves(direction):
            out.append(
                Candidate(
                    instruction,
                    False,
                    f"space mismatch: declared direction {getattr(instruction, 'direction', None)!r}, "
                    f"transfers {list(getattr(instruction, 'transfers', None) or ())} "
                    f"cannot lower a {direction} of a global kernel buffer",
                    None,
                )
            )
            continue
        verdict = evaluate(reason_of(instruction), descriptor, tile, env)
        reason: str | None = None

        if kind == "elementwise":
            # The source op name is passed as the descriptor for compute ops, or as descriptor.base
            op_name = getattr(descriptor, "base", descriptor)
            if op_name and isinstance(op_name, str) and not op_name.startswith("%"):
                instr_ops = getattr(instruction, "ops", ())
                if not instr_ops or not any(_matches_op(entry, op_name) for entry in instr_ops):
                    out.append(
                        Candidate(
                            instruction,
                            False,
                            f"op mismatch: {op_name} not in {list(instr_ops)}",
                            None,
                        )
                    )
                    continue

        if verdict is True:

            # Enforce sparse data precondition: if the instruction requires sparse execution,
            # input descriptor or env must declare sparse format/metadata.
            if verdict is True and getattr(instruction, "sparse", False):
                desc_sparse = bool(getattr(descriptor, "sparse", False))
                env_sparse = bool(env and env.get("sparse", False))
                if not (desc_sparse or env_sparse):
                    verdict = False
                    reason = "sparse data precondition not met: input tensor is dense"

            # Enforce format precondition (e.g. microscaling MXFP8/NVFP4):
            if verdict is True and getattr(instruction, "format", None):
                fmt = instruction.format.lower()
                desc_dtype = str(getattr(descriptor, "dtype", "")).lower().replace("!", "")
                desc_fmt = str(getattr(descriptor, "format", "")).lower()
                env_fmt = str(env.get("format", "")).lower() if env else ""
                if fmt not in (desc_dtype, desc_fmt, env_fmt):
                    verdict = False
                    reason = f"format precondition not met: requires {instruction.format}"

        if verdict is True:
            from .cost import (
                CostQuery,
                CostResultError,
                CostUnknown,
                evaluate_cost,
                load_machine_by_name,
            )
            mach = machine
            if mach is None:
                with contextlib.suppress(Exception):
                    mach = load_machine_by_name(schema.name)
            dir_val = direction.lower() if (direction and direction.lower() in ("load", "store")) else None
            query = CostQuery(
                instruction=instruction,
                access=descriptor,
                tile=tile,
                direction=dir_val,
                env=env or {},
                machine=mach,
            )
            try:
                res = evaluate_cost(query, mach)
                out.append(Candidate(instruction, True, None, res.select_cost, cost_result=res))
            except (CostUnknown, CostResultError) as error:
                out.append(Candidate(instruction, False, f"cost undecidable: {error}", None, None))
        else:
            text = reason_of(instruction)
            if reason is None:
                reason = text if verdict is False else f"unknown: {text}"
            out.append(Candidate(instruction, False, reason, None))
    return tuple(out)


def select(
    schema: IsaSchema,
    kind: str,
    descriptor: object,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
    direction: str | None = None,
    machine: object | None = None,
) -> SelectionReport:
    """Minimum cost among the admissible; no default fallback.

    Postcondition 3: ties break by schema declaration order — `min` over
    `(cost, declaration_index)` is deterministic because `of_kind` is ordered.
    Postcondition 4: nothing admissible means `chosen is None` and
    `no_admissible_lowering=True`; the emitter turns that into `UNSUPPORTED`.
    Postcondition 5: `oracle_min` runs the same enumeration the other way, and
    the gap is reported, not hidden.
    """
    candidates = enumerate_candidates(schema, kind, descriptor, tile, env, direction, machine=machine)
    admissible = [c for c in candidates if c.admissible and c.cost is not None]
    if not admissible:
        oracle = oracle_min(schema, kind, descriptor, tile, env, direction, machine=machine)
        return SelectionReport(
            chosen=None,
            chosen_cost=None,
            candidates=candidates,
            oracle_min_cost=oracle.oracle_min_cost,
            oracle_chosen=None,
            gap=0.0,
            no_admissible_lowering=True,
        )
    order = {id(c.instruction): i for i, c in enumerate(candidates)}
    best = min(admissible, key=lambda c: (c.cost, order[id(c.instruction)]))
    oracle = oracle_min(schema, kind, descriptor, tile, env, direction, machine=machine)
    gap = max(0.0, (best.cost or 0.0) - (oracle.oracle_min_cost or 0.0))
    return SelectionReport(
        chosen=best.instruction,
        chosen_cost=best.cost,
        candidates=candidates,
        oracle_min_cost=oracle.oracle_min_cost,
        oracle_chosen=oracle.oracle_chosen,
        gap=gap,
        no_admissible_lowering=False,
    )


def oracle_min(
    schema: IsaSchema,
    kind: str,
    descriptor: object,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
    direction: str | None = None,
    machine: object | None = None,
) -> SelectionReport:
    """The exhaustive oracle: same enumeration, optimum reported separately.

    On this schema the greedy rule (min cost among admissible) and the oracle
    agree by construction — there is no coupling between instructions, because
    one operand maps to exactly one instruction. The function exists so the
    *report* carries the oracle column and a future ISA with coupled costs
    (ISA-2's bank moves) cannot silently drop the comparison.
    """
    candidates = enumerate_candidates(schema, kind, descriptor, tile, env, direction, machine=machine)
    admissible = [c for c in candidates if c.admissible and c.cost is not None]
    if not admissible:
        return SelectionReport(
            chosen=None,
            chosen_cost=None,
            candidates=candidates,
            oracle_min_cost=None,
            oracle_chosen=None,
            gap=0.0,
            no_admissible_lowering=True,
        )
    order = {id(c.instruction): i for i, c in enumerate(candidates)}
    best = min(admissible, key=lambda c: (c.cost, order[id(c.instruction)]))
    return SelectionReport(
        chosen=best.instruction,
        chosen_cost=best.cost,
        candidates=candidates,
        oracle_min_cost=best.cost,
        oracle_chosen=best.instruction,
        gap=0.0,
        no_admissible_lowering=False,
    )
