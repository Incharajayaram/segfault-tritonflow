"""The cost model: CostQuery, CostResult, the term registry, and the evaluator.

The model now sees the *access
shape* (bytes, transactions, bank conflicts, contiguity, alignment, strides),
not just an element count. Three jobs that one scalar used to conflate are now
separate results of one evaluation:

  select_cost — dimensionless ranking scalar consumed by isa/select.py
  time        — cycles estimate (Layer B params), carries a basis label
  resources   — exact, additive bytes / transactions / conflicts / macs

Fail-closed discipline, extended to costing: a term that cannot resolve is
`None` (unknown), never 0.0. Selection rejects a candidate whose cost is
unknown with a named reason, exactly as it rejects an undecidable predicate.

Determinism: evaluation is pure; `math.fsum` for the ranking scalar; caching
keyed on the access shape so CoalescingUnit/BankConflictUnit results repeat
within one kernel for free .
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .schema import CostExpr, Instruction

__all__ = [
    "CostQuery",
    "MachineParams",
    "MachineParam",
    "CostResult",
    "TimeEstimate",
    "Resources",
    "CostUnknown",
    "CostResultError",
    "evaluate_cost",
    "term_registry",
    "register_term",
    "load_machine_params",
    "load_machine_by_name",
    "UNKNOWN_MACHINE",
    "reset_cache",
]

#: The Layer B parameter set handed to every cost expression as `machine.*`.
#: Frozen for one compilation . An absent machine file means the
#: compiler refuses to price that target — no silent defaults .
UNKNOWN_MACHINE = "__unknown__"


class CostResultError(Exception):
    """A cost could not be computed for a reason that is a caller bug."""


class CostUnknown(Exception):
    """A term could not resolve for this operand: the candidate is not priced.

    This is the fail-closed hook, made loud. `select.py` catches it and records
    a rejection reason; nothing in the cost path catches and defaults.
    """


# --------------------------------------------------------------------------- #
# Layer B: machine parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MachineParam:
    value: float
    ci95: tuple[float, float] | None = None
    basis: str = "fitted"  # fitted | declared


@dataclass(frozen=True)
class MachineParams:
    """Layer B: fitted + declared parameters for one target."""

    machine: str
    fit_id: str
    source_kind: str  # simx | rtlsim | fpga | declared
    schema_version: int
    params: dict[str, MachineParam] = field(default_factory=dict)
    residuals: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        """A parameter value, or `CostUnknown`. A missing machine parameter is
        never defaulted ."""
        found = self.params.get(name)
        if found is None:
            raise CostUnknown(f"machine parameter {name!r} is not defined for {self.machine!r}")
        return found.value

    def declared(self) -> dict[str, float]:
        return {k: p.value for k, p in self.params.items() if p.basis == "declared"}


#: A conservative, labelled default machine for the *declared-only* toy ISAs
#: (tritonflow1/2). Their cycles are `basis: "declared"`, never "measured" —
#: the spec forbids pretending otherwise (§5.1).
_DECLARED_FALLBACK = MachineParams(
    machine=UNKNOWN_MACHINE,
    fit_id="declared/no-ground-truth",
    source_kind="declared",
    schema_version=1,
    params={},
    residuals={},
)


def load_machine_params(path: str) -> MachineParams:
    """Load `isa/machines/<name>.json`. Hard error on unknown schema version."""
    import json

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    version = int(raw.get("schema", 0))
    if version != 1:
        raise CostResultError(
            f"machine file {path}: schema version {version} is not supported (want 1)"
        )
    params: dict[str, MachineParam] = {}
    for name, entry in (raw.get("params") or {}).items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise CostResultError(f"machine file {path}: param {name!r} has no value")
        ci = entry.get("ci95")
        params[name] = MachineParam(
            value=float(entry["value"]),
            ci95=(float(ci[0]), float(ci[1])) if ci else None,
            basis=str(entry.get("basis", "fitted")),
        )
    return MachineParams(
        machine=str(raw.get("machine", "?")),
        fit_id=str(raw.get("fit_id", "")),
        source_kind=str((raw.get("source") or {}).get("kind", "declared")),
        schema_version=version,
        params=params,
        residuals=dict(raw.get("residuals") or {}),
    )


def load_machine_by_name(name: str) -> MachineParams:
    """Load machine parameters by ISA name (e.g. 'vortex_rvgpu', 'tritonflow1')."""
    machines_dir = Path(__file__).parent / "machines"
    candidate = machines_dir / f"{name}.json"
    if candidate.is_file():
        return load_machine_params(str(candidate))
    return _DECLARED_FALLBACK


# --------------------------------------------------------------------------- #
# The cost query
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostQuery:
    """Everything the model may read. Nothing else is in scope ."""

    instruction: Instruction
    access: object | None = None
    tile: tuple[int, ...] | None = None
    dtype: object = None
    space: str = "global"
    direction: Literal["load", "store"] | None = None
    env: Mapping[str, int] = field(default_factory=dict)
    machine: MachineParams | None = None


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TimeEstimate:
    cycles_p50: float
    cycles_lo: float | None = None  # from a fitted param's CI, not a guess
    cycles_hi: float | None = None
    basis: Literal["measured", "fitted", "analytic", "declared"] = "analytic"


@dataclass(frozen=True)
class Resources:
    dram_bytes: int = 0
    scratch_bytes: int = 0
    transactions: int = 0
    bank_conflicts: int = 0
    mac_ops: int = 0
    registers: int = 0


@dataclass(frozen=True)
class CostResult:
    select_cost: float
    time: TimeEstimate | None
    resources: Resources
    terms: dict[str, float]
    confidence: Literal["exact", "modelled", "extrapolated", "unknown"]
    provenance: str

    def __post_init__(self) -> None:
        if self.select_cost < 0.0:
            raise CostResultError("select_cost evaluated negative")

    @property
    def cost(self) -> float:
        """Compatibility property for code reading .cost."""
        return self.select_cost


# --------------------------------------------------------------------------- #
# Term registry
# --------------------------------------------------------------------------- #


#: term name -> provider(descriptor, tile, env, machine) -> float
#: A provider raises CostUnknown rather than returning 0 for an unknown fact.
TermProvider = Callable[[object, tuple[int, ...] | None, dict[str, int] | None, MachineParams], float]

_registry: dict[str, TermProvider] = {}
_analysis_cache: dict[tuple, tuple[float, float]] = {}


def register_term(name: str) -> Callable[[TermProvider], TermProvider]:
    def deco(fn: TermProvider) -> TermProvider:
        _registry[name] = fn
        return fn

    return deco


def term_registry() -> dict[str, str]:
    """The registry with provider sources — terms.md is generated from this."""
    return {name: f"{fn.__module__}.{fn.__qualname__}" for name, fn in sorted(_registry.items())}


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


def _descriptor_int(value: object, env: dict[str, int] | None) -> int | None:
    from .schema import _as_int  # reuse the maybe-int grounding

    resolved = _as_int(value, env)
    return None if resolved is None else resolved


def _descriptor_length(descriptor: object, env: dict[str, int] | None) -> int | None:
    from .schema import _length

    resolved = _length(descriptor, env)
    return None if resolved is None else resolved


def _resolved_seq(descriptor: object, attr: str, env: dict[str, int] | None) -> list[int | None]:
    out: list[int | None] = []
    for element in tuple(getattr(descriptor, attr, ()) or ()):
        out.append(element if isinstance(element, int) else _descriptor_int(element, env))
    return out


def _access_geometry(descriptor: object, env: dict[str, int] | None) -> tuple[int, int, int]:
    """(rows, inner_extent, inner_stride) of the tile as the memory system sees it.

    The inner dimension is the one with the smallest |stride| — the contiguous run
    the hardware actually streams — not merely the last listed. A stride or extent
    that cannot be resolved is `CostUnknown`, never a guess.
    """
    sizes = _resolved_seq(descriptor, "sizes", env)
    strides = _resolved_seq(descriptor, "strides", env)
    if not sizes:
        length = _descriptor_length(descriptor, env)
        if length is None:
            raise CostUnknown("access extent is unresolvable")
        return 1, int(length), 1
    if any(s is None for s in sizes):
        raise CostUnknown("an access extent is unresolvable for this operand")
    if not strides:
        strides = [1] * len(sizes)
    if len(strides) != len(sizes) or any(s is None for s in strides):
        raise CostUnknown("an access stride is unresolvable for this operand")
    inner = min(range(len(sizes)), key=lambda i: abs(int(strides[i])) if sizes[i] != 1 else 10**18)
    rows = 1
    for i, extent in enumerate(sizes):
        if i != inner:
            rows *= int(extent)
    return rows, int(sizes[inner]), abs(int(strides[inner])) or 1


def _coalesce(descriptor: object, env: dict[str, int] | None, machine: MachineParams) -> tuple[float, float]:
    """(transactions, efficiency) for the *whole* access, not one warp.

    Uses the existing `CoalescingUnit` with the inner row as the lane set, so the
    line count for one row is exactly what that unit reports; rows multiply it.
    A gather/scatter descriptor has no contiguity to exploit and is priced at one
    transaction per element — the conservative bound.
    """
    from ..emu.hardware import CoalescingUnit

    line = int(machine.get("cache_line_bytes"))
    element_bytes = _element_bytes(descriptor, env) or 4
    if getattr(descriptor, "is_gather_scatter", False):
        elements = _descriptor_length(descriptor, env)
        if elements is None:
            raise CostUnknown("gather extent is unresolvable")
        transactions = float(elements)
        return transactions, min(1.0, elements * element_bytes / max(1.0, transactions * line))
    rows, extent, stride = _access_geometry(descriptor, env)
    if extent <= 0 or rows <= 0:
        return 0.0, 1.0
    key = (extent, stride, element_bytes, line)
    if key not in _analysis_cache:
        report = CoalescingUnit(cache_line_bytes=line, warp_size=extent).analyze(
            base_address=0, stride_elements=stride, element_bytes=element_bytes
        )
        _analysis_cache[key] = (float(report.num_transactions), float(report.coalescing_efficiency))
    per_row, efficiency = _analysis_cache[key]
    return per_row * rows, efficiency


def _element_bytes(descriptor: object, env: dict[str, int] | None) -> int | None:
    from .schema import _scalar_dtype_bits

    bits = _scalar_dtype_bits(descriptor)
    if bits is None:
        return 4
    return max(1, bits // 8)


def _bank_conflicts(descriptor: object, env: dict[str, int] | None, machine: MachineParams) -> float:
    """Extra serialised bank cycles for one warp-wide access of this tile's inner row.

    Lane `i` touches word `i * inner_stride`. One bank-cycle can serve at most
    `banks` lanes, so lanes beyond that wait for the next cycle by capacity, not by
    conflict: the group considered is `min(extent, banks)`. A contiguous access then
    has zero conflicts, and only strides sharing a factor with the bank count add
    any. Computed for any descriptor; only banked-scratchpad instructions
    reference the term in their cost expression.
    """
    from ..emu.hardware import BankConflictUnit

    _, extent, stride = _access_geometry(descriptor, env)
    banks = int(machine.get("lmem_banks"))
    lanes = max(1, min(extent, banks))
    unit = BankConflictUnit(num_banks=banks)
    element_bytes = _element_bytes(descriptor, env) or 4
    report = unit.analyze([lane * stride * element_bytes for lane in range(lanes)])
    return float(report.total_conflicts)


@register_term("elements")
def _t_elements(descriptor, tile, env, machine):
    """Number of elements the access moves."""
    v = _descriptor_length(descriptor, env)
    if v is None:
        # Check if tile can provide elements
        if tile and all(t is not None for t in tile):
            return float(math.prod(tile))
        raise CostUnknown("element count is unresolvable for this operand")
    return float(v)


@register_term("words")
def _t_words(descriptor, tile, env, machine):
    """Alias of `elements`."""
    return _t_elements(descriptor, tile, env, machine)


@register_term("bytes")
def _t_bytes(descriptor, tile, env, machine):
    """`elements` times the element width in bytes."""
    elements = _t_elements(descriptor, tile, env, machine)
    eb = _element_bytes(descriptor, env)
    if eb is None:
        return elements * 4.0
    return elements * float(eb)


@register_term("line_words")
def _t_line_words(descriptor, tile, env, machine):
    """Elements per cache line for this operand's dtype (from the machine's declared line size)."""
    return float(machine.get("cache_line_bytes")) / float(_element_bytes(descriptor, env) or 4)


@register_term("transactions")
def _t_transactions(descriptor, tile, env, machine):
    """Cache-line transactions for the whole access (one per element for a gather)."""
    return _coalesce(descriptor, env, machine)[0]


@register_term("coalescing_efficiency")
def _t_coalesce_eff(descriptor, tile, env, machine):
    """Requested bytes divided by transacted bytes, at most 1."""
    return _coalesce(descriptor, env, machine)[1]


@register_term("contiguous")
def _t_contiguous(descriptor, tile, env, machine):
    """1 when the innermost stride is 1, otherwise 0."""
    strides = tuple(getattr(descriptor, "strides", ()) or ())
    if not strides:
        return 1.0
    inner = strides[-1]
    v = inner if isinstance(inner, int) else _descriptor_int(inner, env)
    if v is None:
        return 1.0
    return 1.0 if v == 1 else 0.0


@register_term("bank_conflicts")
def _t_bank_conflicts(descriptor, tile, env, machine):
    """Extra serialised bank cycles for one bank-cycle worth of lanes."""
    return _bank_conflicts(descriptor, env, machine)


@register_term("alignment")
def _t_alignment(descriptor, tile, env, machine):
    """Base offset modulo the machine alignment, in words."""
    base = getattr(descriptor, "base_num", None)
    if base is None:
        return 0.0
    v = base if isinstance(base, int) else _descriptor_int(base, env)
    if v is None:
        return 0.0
    align_words = int(machine.get("alignment_words")) if "alignment_words" in machine.params else 4
    return float(v % max(1, align_words))


@register_term("m")
def _t_m(descriptor, tile, env, machine):
    """Tile dimension m."""
    if tile and len(tile) >= 1 and tile[0] is not None:
        return float(tile[0])
    if env and "m" in env and env["m"] is not None:
        return float(env["m"])
    raise CostUnknown("tile dimension m is required and absent")


@register_term("n")
def _t_n(descriptor, tile, env, machine):
    """Tile dimension n."""
    if tile and len(tile) >= 2 and tile[1] is not None:
        return float(tile[1])
    if env and "n" in env and env["n"] is not None:
        return float(env["n"])
    raise CostUnknown("tile dimension n is required and absent")


@register_term("k")
def _t_k(descriptor, tile, env, machine):
    """Tile dimension k."""
    if tile and len(tile) >= 3 and tile[2] is not None:
        return float(tile[2])
    if env and "k" in env and env["k"] is not None:
        return float(env["k"])
    raise CostUnknown("tile dimension k is required and absent")


@register_term("mac_ops")
def _t_mac_ops(descriptor, tile, env, machine):
    """`m * n * k` multiply-accumulates."""
    return _t_m(descriptor, tile, env, machine) * _t_n(descriptor, tile, env, machine) * _t_k(descriptor, tile, env, machine)


@register_term("trip_count")
def _t_trip_count(descriptor, tile, env, machine):
    """Loop trip count from the launch environment, else 1."""
    if not env:
        return 1.0
    return float(env.get("trip_count", 1))


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #


def _resolve(
    node: object,
    terms: dict[str, float],
    machine: MachineParams,
    instr_name: str,
    text: str,
    descriptor: object = None,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
) -> float:
    from .schema import BinOp, Call, Index, Name, Num, Unary

    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Unary):
        return -_resolve(node.operand, terms, machine, instr_name, text, descriptor, tile, env)
    if isinstance(node, BinOp):
        left = _resolve(node.left, terms, machine, instr_name, text, descriptor, tile, env)
        right = _resolve(node.right, terms, machine, instr_name, text, descriptor, tile, env)
        if node.operator == "+":
            return left + right
        if node.operator == "-":
            return left - right
        if node.operator == "*":
            return left * right
        if node.operator == "/":
            if right == 0.0:
                raise CostResultError(f"{instr_name}: cost {text!r} divides by zero")
            return left / right
        raise CostResultError(f"{instr_name}: operator {node.operator!r} is not arithmetic in {text!r}")
    if isinstance(node, Index):
        name = f"{node.name}[{node.index}]"
        if name not in terms:
            raise CostUnknown(f"{instr_name}: term {name!r} is unresolvable in {text!r}")
        return terms[name]
    if isinstance(node, Name):
        if node.text.startswith("machine."):
            param = node.text[len("machine."):]
            try:
                return machine.get(param)
            except CostUnknown as error:
                raise CostUnknown(f"{instr_name}: {error} (in {text!r})") from error
        if node.text in terms:
            return terms[node.text]
        raise CostUnknown(f"{instr_name}: cost {text!r} references unknown term {node.text!r}")
    if isinstance(node, Call):
        fname = node.function
        if fname in ("min", "max"):
            values = [_resolve(a, terms, machine, instr_name, text, descriptor, tile, env) for a in node.args]
            if not values:
                raise CostResultError(f"{instr_name}: {fname}() with no arguments")
            return min(values) if fname == "min" else max(values)
        if fname == "select":
            if len(node.args) != 3:
                raise CostResultError(f"{instr_name}: select() takes three arguments")
            pred = node.args[0]
            verdict = _eval_cost_predicate(pred, terms, machine, instr_name, text, descriptor, tile, env)
            if verdict == "unknown" or verdict is None:
                raise CostUnknown(f"{instr_name}: select() predicate is undecidable in {text!r}")
            chosen = node.args[1] if verdict is True else node.args[2]
            return _resolve(chosen, terms, machine, instr_name, text, descriptor, tile, env)
        raise CostResultError(f"{instr_name}: function {fname!r} is not available in cost expressions")
    raise CostResultError(f"{instr_name}: cannot evaluate {node!r} in {text!r}")


def _eval_cost_predicate(
    pred: object,
    terms: dict[str, float],
    machine: MachineParams,
    instr_name: str,
    text: str,
    descriptor: object = None,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
) -> bool | str:
    from .schema import UNKNOWN, BinOp, Predicate
    from .schema import evaluate as eval_predicate

    if isinstance(pred, BinOp) and pred.operator in ("==", "!=", "<", "<=", ">", ">="):
        try:
            left_val = _resolve(pred.left, terms, machine, instr_name, text, descriptor, tile, env)
            right_val = _resolve(pred.right, terms, machine, instr_name, text, descriptor, tile, env)
            if pred.operator == "==":
                return left_val == right_val
            if pred.operator == "!=":
                return left_val != right_val
            if pred.operator == "<":
                return left_val < right_val
            if pred.operator == "<=":
                return left_val <= right_val
            if pred.operator == ">":
                return left_val > right_val
            if pred.operator == ">=":
                return left_val >= right_val
        except (CostUnknown, CostResultError):
            pass

    rendered = _render(pred)
    if not rendered or rendered == "?":
        return UNKNOWN
    try:
        return eval_predicate(Predicate(rendered), descriptor, tile, env)
    except Exception:
        return UNKNOWN


def _render(node: object) -> str:
    from .schema import BinOp, Call, Index, Name, Num, Unary

    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Name):
        return node.text
    if isinstance(node, Index):
        return f"{node.name}[{node.index}]"
    if isinstance(node, Unary):
        return f"-{_render(node.operand)}"
    if isinstance(node, BinOp):
        return f"{_render(node.left)} {node.operator} {_render(node.right)}"
    if isinstance(node, Call):
        return f"{node.function}({', '.join(_render(a) for a in node.args)})"
    return "?"


def _prepare_terms(
    descriptor: object,
    tile: tuple[int, ...] | None,
    env: dict[str, int] | None,
    machine: MachineParams,
) -> dict[str, float]:
    """Resolve every registered term, plus stride[i]/size[i] index forms."""
    terms: dict[str, float] = {}
    for name, provider in _registry.items():
        try:
            terms[name] = provider(descriptor, tile, env, machine)
        except CostUnknown:
            continue
    # Indexed sequence terms: stride[i], size[i], offsets[i]
    for seq_name, attr in (
        ("stride", "strides"),
        ("size", "sizes"),
        ("offset", "offsets"),
        ("shape", "shape"),
    ):
        sequence = tuple(getattr(descriptor, attr, ()) or ())
        for i, element in enumerate(sequence):
            v = element if isinstance(element, int) else _descriptor_int(element, env)
            if v is not None:
                terms[f"{seq_name}[{i}]"] = float(v)
    return terms


def evaluate_cost(
    instr_or_query: Instruction | CostQuery,
    descriptor: object = None,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
    machine: MachineParams | None = None,
) -> CostResult:
    """The full cost of an instruction or query. Raises CostUnknown when any referenced
    term is unresolvable."""
    if isinstance(instr_or_query, CostQuery):
        instr = instr_or_query.instruction
        descriptor = instr_or_query.access
        tile = instr_or_query.tile
        env = dict(instr_or_query.env) if instr_or_query.env else None
        machine = instr_or_query.machine
    else:
        instr = instr_or_query

    machine = machine or _DECLARED_FALLBACK
    terms = _prepare_terms(descriptor, tile, env, machine)

    def evaluate_expr(expr: CostExpr) -> float:
        return _resolve(expr.ast, terms, machine, instr.name, expr.text, descriptor, tile, env)

    cost_expr = getattr(instr, "select_cost", None) or instr.cost
    if cost_expr is None:
        raise CostUnknown(f"{instr.name}: has no select_cost/cost expression")

    if isinstance(cost_expr, (int, float)) and not isinstance(cost_expr, bool):
        select_cost = float(cost_expr)
    else:
        select_cost = evaluate_expr(cost_expr)

    if select_cost < 0.0:
        raise CostResultError(f"{instr.name}: select_cost is negative ({select_cost})")

    time_est: TimeEstimate | None = None
    if instr.time is not None:
        if isinstance(instr.time, (int, float)) and not isinstance(instr.time, bool):
            cycles = float(instr.time)
        else:
            cycles = evaluate_expr(instr.time)
        basis: Literal["measured", "fitted", "analytic", "declared"] = (
            "declared" if machine.source_kind == "declared" else "fitted"
        )
        lo = hi = None
        if machine.source_kind != "declared":
            span = 0.0
            for p in machine.params.values():
                if p.ci95:
                    width = (p.ci95[1] - p.ci95[0]) / 2
                    span = max(span, width / max(1e-9, abs(p.value)))
            lo, hi = cycles * (1 - span), cycles * (1 + span)
        time_est = TimeEstimate(cycles_p50=cycles, cycles_lo=lo, cycles_hi=hi, basis=basis)

    resources = Resources(
        dram_bytes=int(terms.get("bytes", 0.0)),
        transactions=int(terms.get("transactions", 0.0)),
        bank_conflicts=int(terms.get("bank_conflicts", 0.0)),
        mac_ops=int(terms.get("mac_ops", 0.0)),
    )

    confidence: Literal["exact", "modelled", "extrapolated", "unknown"] = (
        "extrapolated" if machine.source_kind == "declared" else "modelled"
    )
    return CostResult(
        select_cost=select_cost,
        time=time_est,
        resources=resources,
        terms=dict(sorted(terms.items())),
        confidence=confidence,
        provenance=f"{machine.machine}/{machine.fit_id}",
    )


def reset_cache() -> None:
    _analysis_cache.clear()
