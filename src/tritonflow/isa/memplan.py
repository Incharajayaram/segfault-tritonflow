"""Scratchpad allocation: liveness, constraint encoding, and two solvers.

A tile loaded into on-chip scratchpad needs a concrete byte offset. The offset
has to respect four things at once: the space's capacity, its alignment, the
fact that two simultaneously-live tiles may not overlap, and -- optionally --
bank separation. This module derives those facts, encodes them, and solves them
three ways: exactly with OR-Tools CP-SAT, exactly with z3, and with a
deterministic interval-colouring fallback that always terminates.

The two exact solvers are different families on purpose. CP-SAT is SAT-based
constraint programming, the family ACT uses (Jain et al., OOPSLA 2026,
arXiv:2510.09932), and it states the problem as two-dimensional packing over
(live range x address range). z3 is SMT, and states disjointness as one
disjunction per simultaneously-live pair. Keeping both makes the comparison
measurable rather than asserted, and means neither is load-bearing: both are
optional, and greedy alone is always enough to produce a valid allocation.

Three properties are deliberate.

**Capacity is sourced, never guessed.** A scratchpad with no capacity the
compiler can point at is refused with `CapacityUndeclared` naming the space and
what would fix it. `vortex_rvgpu` is the one target whose capacity has an
upstream source (`third_party/vortex/VX_config.toml`), and the number is read
from that file at call time rather than pasted here, so it cannot drift from the
checksummed snapshot.

**Scratch residency is read from the schema.** Which values land in scratch is
decided by the destination space in each instruction's own `semantics` string
(`"scratch[0:length] = global[0:length]"` puts its result in scratch), not by a
table in this file. An ISA that renames or re-points an instruction changes this
analysis by changing its schema, which is the whole point of a declarative ISA.

**The checker does not trust the solver.** `check_allocation` re-derives every
constraint from the problem and validates an allocation from scratch. Both
solvers' outputs go through it, and a solver whose answer fails it raises rather
than falling back quietly -- a wrong offset that got silently replaced by a
correct one is a bug that would never be found.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Allocation",
    "AllocationProblem",
    "CapacityUndeclared",
    "MemPlanError",
    "ScratchSpace",
    "Tile",
    "Violation",
    "bank_of",
    "check_allocation",
    "have_cpsat",
    "have_z3",
    "plan",
    "scratch_space",
    "scratch_tiles",
    "solve_cpsat",
    "solve_exact",
    "solve_greedy",
    "solve_z3",
]


class MemPlanError(Exception):
    """Base for allocation failures."""


class CapacityUndeclared(MemPlanError):
    """No capacity for this scratchpad can be sourced, so allocation fails closed.

    Raised rather than defaulted: a guessed capacity turns "this kernel does not
    fit" into "this kernel fits", which is the one answer an allocator must never
    invent.
    """


# --------------------------------------------------------------------------- #
# Sourced machine parameters
# --------------------------------------------------------------------------- #

_VORTEX_CONFIG = Path(__file__).resolve().parents[3] / "third_party" / "vortex" / "VX_config.toml"

_DTYPE_BITS = {
    "f8": 8, "f16": 16, "bf16": 16, "f32": 32, "f64": 64,
    "i1": 8, "i8": 8, "i16": 16, "i32": 32, "i64": 64,
    "u8": 8, "u16": 16, "u32": 32, "u64": 64,
}

#: Memory-space kinds that denote on-chip scratchpad storage.
SCRATCH_KINDS = frozenset({"scratchpad", "banked"})


def _dtype_bytes(dtype: str | None) -> int:
    bits = _DTYPE_BITS.get((dtype or "").replace("!", "").strip())
    return max(1, bits // 8) if bits else 4


def _vortex_lmem_capacity_bytes(config_path: Path | None = None) -> int | None:
    """Vortex scratchpad capacity in bytes, read from the vendored upstream config.

    Upstream declares `VX_CFG_LMEM_LOG_SIZE` and derives the byte capacity as
    `1 << VX_CFG_LMEM_LOG_SIZE`. That the quantity is bytes rather than words is
    upstream's own reading of it: `sim/simx/cta_dispatcher.cpp` initialises
    `lmem_capacity_(1u << VX_CFG_LMEM_LOG_SIZE)`, and `sim/simx/types.h` compares
    a byte address against it directly.
    """
    path = config_path or _VORTEX_CONFIG
    if not path.exists():
        return None
    match = re.search(r"^\s*VX_CFG_LMEM_LOG_SIZE\s*=\s*(\d+)\s*$", path.read_text(), re.M)
    return 1 << int(match.group(1)) if match else None


def _sourced_capacity(isa_name: str, space_name: str) -> tuple[int | None, str]:
    """(capacity_bytes, provenance). `None` means nothing can be sourced."""
    if isa_name == "vortex_rvgpu" and space_name == "scratch":
        cap = _vortex_lmem_capacity_bytes()
        if cap is not None:
            return cap, "third_party/vortex/VX_config.toml:VX_CFG_LMEM_LOG_SIZE (1 << 14)"
    return None, "no upstream source for this space"


# --------------------------------------------------------------------------- #
# Problem structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScratchSpace:
    """A scratchpad's allocation-relevant parameters, with where they came from."""

    isa: str
    name: str
    kind: str
    capacity_bytes: int
    banks: int
    interleave_bytes: int
    alignment_bytes: int
    capacity_source: str


@dataclass(frozen=True)
class Tile:
    """A value resident in scratchpad over an inclusive index range.

    `first` and `last` are inclusive indices into the program's flattened
    execution order. A tile whose live range touches a loop is widened to the
    whole loop span, because a value crossing the back-edge is live on every
    iteration, not only between its textual definition and use.
    """

    name: str
    size_bytes: int
    first: int
    last: int
    alignment_bytes: int
    producer: str = ""
    in_loop: bool = False

    def overlaps(self, other: Tile) -> bool:
        """True when both are live at some common index."""
        return self.first <= other.last and other.first <= self.last


@dataclass(frozen=True)
class AllocationProblem:
    space: ScratchSpace
    tiles: tuple[Tile, ...]
    bank_spread: bool = False
    """Require simultaneously-live tiles to start in different banks.

    Off by default and **not** a sourced hardware requirement. Upstream declares
    a bank count and an interleave granularity but no per-instruction access
    schedule, and without a schedule there is no sound way to say which two
    accesses are issued in the same cycle. What this models is the weaker,
    stateable property that distinct live tiles begin in distinct banks, the
    usual padding heuristic. It is exposed so the cost of the heuristic can be
    measured, not because the hardware demands it.
    """

    @property
    def conflict_pairs(self) -> tuple[tuple[int, int], ...]:
        """Index pairs that are simultaneously live, and so may not overlap."""
        return tuple(
            (i, j)
            for i in range(len(self.tiles))
            for j in range(i + 1, len(self.tiles))
            if self.tiles[i].overlaps(self.tiles[j])
        )

    @property
    def total_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tiles)


@dataclass(frozen=True)
class Allocation:
    offsets: dict[str, int] = field(default_factory=dict)
    solver: str = ""
    feasible: bool = False
    timed_out: bool = False
    peak_bytes: int = 0
    solve_seconds: float = 0.0
    reason: str | None = None

    @property
    def placed(self) -> int:
        return len(self.offsets)


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def bank_of(byte_offset: int, space: ScratchSpace) -> int:
    """Which bank a byte offset lands in.

    Upstream maps the low-order bits of the *word* address to the bank index
    (`VX_local_mem.sv`: `req_bank_idx = addr[0 +: BANK_SEL_BITS]`), i.e. words are
    interleaved round-robin across banks.
    """
    if space.banks <= 1:
        return 0
    return (byte_offset // space.interleave_bytes) % space.banks


# --------------------------------------------------------------------------- #
# Space + liveness derivation
# --------------------------------------------------------------------------- #


def scratch_space(schema: object, space_name: str | None = None) -> ScratchSpace:
    """The scratchpad space of `schema`, with a sourced capacity.

    Raises `CapacityUndeclared` when no capacity can be sourced, naming the ISA,
    the space and the two ways to fix it.
    """
    isa_name = str(getattr(schema, "name", "") or "")
    spaces = tuple(getattr(getattr(schema, "data_model", None), "memory_spaces", ()) or ())
    candidates = [s for s in spaces if getattr(s, "kind", "") in SCRATCH_KINDS]
    if space_name is not None:
        candidates = [s for s in candidates if getattr(s, "name", "") == space_name]
    if not candidates:
        raise CapacityUndeclared(
            f"{isa_name}: no scratchpad memory space is declared"
            + (f" under the name {space_name!r}" if space_name else "")
            + f"; declared spaces: {[getattr(s, 'name', '?') for s in spaces]}"
        )

    space = candidates[0]
    name = str(getattr(space, "name", ""))
    capacity, provenance = _sourced_capacity(isa_name, name)
    if capacity is None:
        raise CapacityUndeclared(
            f"{isa_name}.{name}: scratchpad capacity is not declared and cannot be "
            f"sourced ({provenance}). Allocation refuses rather than assume a size. "
            f"Fix by adding a capacity to the memory space in the schema and plumbing "
            f"it through isa.schema.MemorySpace, or by vendoring an upstream "
            f"configuration this module can read."
        )

    words = int(getattr(space, "alignment_words", 1) or 1)
    interleave = int(getattr(space, "interleave_bytes", 4) or 4)
    elem = _dtype_bytes(getattr(space, "dtype", ""))
    return ScratchSpace(
        isa=isa_name,
        name=name,
        kind=str(getattr(space, "kind", "")),
        capacity_bytes=capacity,
        banks=max(1, int(getattr(space, "banks", 1) or 1)),
        interleave_bytes=max(1, interleave),
        alignment_bytes=max(1, words * elem),
        capacity_source=provenance,
    )


def _destination_space(semantics: str | None) -> str | None:
    """The memory space an instruction writes, from the left side of its semantics.

    `"scratch[0:length] = global[0:length]"` -> `"scratch"`. A semantics string
    with no assignment (a barrier, say) writes no addressable space and yields
    `None`.
    """
    if not semantics or "=" not in semantics:
        return None
    lhs = semantics.split("=", 1)[0].strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", lhs)
    return match.group(1) if match else None


def _reads_space(semantics: str | None, space_name: str) -> bool:
    """Whether the right-hand side of `semantics` reads `space_name`.

    `"dst[0:length] = scratch[base:base+length]"` reads `scratch`, so the tile it
    names is still live at that instruction.
    """
    if not semantics or "=" not in semantics:
        return False
    rhs = semantics.split("=", 1)[1]
    return re.search(rf"\b{re.escape(space_name)}\s*\[", rhs) is not None


def _tile_bytes(instr: object) -> int:
    """Byte footprint of the tile an instruction deposits in scratch.

    Taken from whichever operand descriptor states the tile shape; the element
    width comes from the descriptor's own dtype.
    """
    from tritonflow.emit.ir import MemRef

    best = 0
    for operand in getattr(instr, "operands", {}).values():
        access = getattr(operand, "access", None) if isinstance(operand, MemRef) else None
        sizes = getattr(access, "sizes", None)
        if not sizes:
            continue
        elems = 1
        for dim in sizes:
            if not isinstance(dim, int):
                elems = 0
                break
            elems *= dim
        if elems:
            best = max(best, elems * _dtype_bytes(getattr(access, "dtype", None)))
    return best


def _loop_spans(program: object) -> tuple[dict[int, tuple[int, int]], dict[int, object]]:
    """Flattened-index span of every loop, and the loop each index belongs to."""
    spans: dict[int, tuple[int, int]] = {}
    owner: dict[int, object] = {}
    index = 0
    for item in program.execution_order():
        members = getattr(item, "instrs", None)
        if members is not None and hasattr(item, "id"):
            start = index
            for _ in members:
                owner[index] = item
                index += 1
            if index > start:
                spans[id(item)] = (start, index - 1)
        else:
            index += 1
    return spans, owner


def scratch_tiles(program: object, schema: object, space: ScratchSpace) -> tuple[Tile, ...]:
    """Live ranges of every value the program parks in `space`.

    Residency is decided per instruction by the destination space in its schema
    semantics. Two shapes occur: an instruction that *loads into* scratch defines
    a value, and the value is the tile; an instruction that *stores into* scratch
    defines nothing, and the tile is named by its destination operand's base. A
    tile defined or used anywhere inside a loop is widened to that loop's whole
    span.
    """
    from tritonflow.emit.ir import MemRef, SsaRef

    instrs = program.instructions()
    spans, owner = _loop_spans(program)
    declared_all = getattr(schema, "instructions", {}) or {}

    resident: dict[str, int] = {}
    sizes: dict[str, int] = {}
    producers: dict[str, str] = {}
    last_use: dict[str, int] = {}

    def _touch(name: str, index: int) -> None:
        if name in resident:
            last_use[name] = max(last_use.get(name, index), index)

    for index, instr in enumerate(instrs):
        declared = declared_all.get(getattr(instr, "name", ""), None)
        semantics = getattr(declared, "semantics", None) if declared is not None else None
        operands = getattr(instr, "operands", {}) or {}

        if _destination_space(semantics) == space.name:
            nbytes = _tile_bytes(instr)
            if nbytes > 0:
                defs = tuple(getattr(instr, "defs", ()) or ())
                # A store into scratch defines no SSA value; the region it writes
                # is named by the destination operand instead.
                keys = defs or tuple(
                    op.base for op in operands.values() if isinstance(op, MemRef) and op.base
                )[:1]
                for key in keys:
                    resident.setdefault(key, index)
                    sizes[key] = max(sizes.get(key, 0), nbytes)
                    producers.setdefault(key, getattr(instr, "name", ""))
                    last_use.setdefault(key, index)

        for operand in operands.values():
            if isinstance(operand, SsaRef):
                _touch(operand.name, index)
            elif isinstance(operand, MemRef) and _reads_space(semantics, space.name):
                _touch(operand.base, index)

    carried: set[str] = set()
    for loop in getattr(program, "loops", ()) or ():
        carried.update(getattr(loop, "iter_args", ()) or ())
        carried.update(getattr(loop, "yields", ()) or ())
        carried.update(getattr(loop, "inits", ()) or ())

    tiles: list[Tile] = []
    for name, defined_at in sorted(resident.items(), key=lambda kv: (kv[1], kv[0])):
        first, last = defined_at, last_use.get(name, defined_at)
        in_loop = False
        for index in (defined_at, last):
            loop = owner.get(index)
            span = spans.get(id(loop)) if loop is not None else None
            if span:
                first, last, in_loop = min(first, span[0]), max(last, span[1]), True
        if name in carried:
            for span in spans.values():
                if span[0] <= first <= span[1] or span[0] <= last <= span[1]:
                    first, last, in_loop = min(first, span[0]), max(last, span[1]), True
        tiles.append(
            Tile(
                name=name,
                size_bytes=sizes[name],
                first=first,
                last=last,
                alignment_bytes=space.alignment_bytes,
                producer=producers.get(name, ""),
                in_loop=in_loop,
            )
        )
    return tuple(tiles)


# --------------------------------------------------------------------------- #
# Independent checker
# --------------------------------------------------------------------------- #


def check_allocation(problem: AllocationProblem, allocation: Allocation) -> tuple[Violation, ...]:
    """Re-validate an allocation from the problem alone.

    Deliberately shares no code with either solver: it re-derives the overlap
    pairs, re-checks capacity and alignment, and re-computes banks. An allocation
    that passes here is correct no matter which solver produced it.
    """
    space = problem.space
    offsets = allocation.offsets
    out: list[Violation] = []

    if not allocation.feasible:
        return ()

    for tile in problem.tiles:
        if tile.name not in offsets:
            out.append(Violation("unplaced", f"{tile.name} has no offset"))
    if out:
        return tuple(out)

    for tile in problem.tiles:
        offset = offsets[tile.name]
        if offset < 0:
            out.append(Violation("negative", f"{tile.name} at {offset}"))
        if offset + tile.size_bytes > space.capacity_bytes:
            out.append(
                Violation(
                    "capacity",
                    f"{tile.name} occupies [{offset}, {offset + tile.size_bytes}) "
                    f"beyond capacity {space.capacity_bytes}",
                )
            )
        if tile.alignment_bytes > 1 and offset % tile.alignment_bytes:
            out.append(
                Violation("alignment", f"{tile.name} at {offset} is not a multiple of {tile.alignment_bytes}")
            )

    for i, j in problem.conflict_pairs:
        a, b = problem.tiles[i], problem.tiles[j]
        oa, ob = offsets[a.name], offsets[b.name]
        if oa < ob + b.size_bytes and ob < oa + a.size_bytes:
            out.append(
                Violation(
                    "overlap",
                    f"{a.name} [{oa}, {oa + a.size_bytes}) and {b.name} "
                    f"[{ob}, {ob + b.size_bytes}) are live together and overlap",
                )
            )
        if problem.bank_spread and bank_of(oa, space) == bank_of(ob, space):
            out.append(
                Violation(
                    "bank",
                    f"{a.name} and {b.name} are live together and both start in "
                    f"bank {bank_of(oa, space)}",
                )
            )

    return tuple(out)


def _peak(problem: AllocationProblem, offsets: dict[str, int]) -> int:
    return max((offsets[t.name] + t.size_bytes for t in problem.tiles if t.name in offsets), default=0)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #


def _blockers(problem: AllocationProblem, index: int, offsets: dict[str, int]) -> list[tuple[int, int]]:
    """`(offset, size)` of every already-placed tile live at the same time as `index`."""
    tile = problem.tiles[index]
    return [
        (offsets[other.name], other.size_bytes)
        for j, other in enumerate(problem.tiles)
        if j != index and other.name in offsets and tile.overlaps(other)
    ]


def solve_greedy(problem: AllocationProblem) -> Allocation:
    """Deterministic interval colouring. Always terminates.

    Tiles are placed longest-live-first, then largest-first, then by name, each at
    the lowest aligned offset that clears every already-placed tile it is live
    with. The ordering is total, so the result is reproducible across runs.

    Termination: each probe either succeeds or advances `candidate` past the end
    of a blocking tile, and there are finitely many blockers, so the inner loop
    ends; the capacity check bounds it regardless.
    """
    started = time.perf_counter()
    space = problem.space
    order = sorted(
        range(len(problem.tiles)),
        key=lambda i: (
            -(problem.tiles[i].last - problem.tiles[i].first),
            -problem.tiles[i].size_bytes,
            problem.tiles[i].name,
        ),
    )

    offsets: dict[str, int] = {}
    for i in order:
        tile = problem.tiles[i]
        align = max(1, tile.alignment_bytes)
        blockers = _blockers(problem, i, offsets)
        taken_banks = {bank_of(start, space) for start, _ in blockers} if problem.bank_spread else set()

        candidate = 0
        while candidate + tile.size_bytes <= space.capacity_bytes:
            candidate += (-candidate) % align
            clash = next(
                (
                    (start, size)
                    for start, size in blockers
                    if candidate < start + size and start < candidate + tile.size_bytes
                ),
                None,
            )
            if clash is not None:
                candidate = clash[0] + clash[1]
                continue
            if problem.bank_spread and bank_of(candidate, space) in taken_banks:
                candidate += space.interleave_bytes
                continue
            break

        if candidate + tile.size_bytes > space.capacity_bytes:
            return Allocation(
                solver="greedy",
                feasible=False,
                solve_seconds=time.perf_counter() - started,
                reason=(
                    f"{tile.name} ({tile.size_bytes} B) does not fit: lowest free aligned "
                    f"offset {candidate} exceeds capacity {space.capacity_bytes}"
                ),
            )
        offsets[tile.name] = candidate

    return Allocation(
        offsets=offsets,
        solver="greedy",
        feasible=True,
        peak_bytes=_peak(problem, offsets),
        solve_seconds=time.perf_counter() - started,
    )


def have_cpsat() -> bool:
    """Whether OR-Tools CP-SAT is available."""
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        return False
    return True


def have_z3() -> bool:
    """Whether z3 is available."""
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def solve_cpsat(problem: AllocationProblem, timeout_ms: int = 5000) -> Allocation:
    """Minimise peak usage with OR-Tools CP-SAT. Always returns within the timeout.

    Encoded as two-dimensional packing, which is the natural shape of the problem:
    each tile is a rectangle whose x-extent is its (fixed) live range and whose
    y-extent is its (variable) address range, and `AddNoOverlap2D` forbids two
    rectangles from overlapping in both dimensions at once. That is exactly "two
    tiles live at the same moment may not share an address", stated once as a
    global constraint instead of as a disjunction per pair -- which is what lets
    the solver's interval propagators prune, rather than enumerating pairs.

    This is the solver family ACT uses (SAT-based constraint programming via
    OR-Tools); `solve_z3` is a different family (SMT) kept alongside it so the two
    can be measured against each other.
    """
    started = time.perf_counter()
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return Allocation(
            solver="cpsat",
            feasible=False,
            solve_seconds=time.perf_counter() - started,
            reason="ortools is not installed (pip install ortools)",
        )

    space = problem.space
    cap = space.capacity_bytes
    model = cp_model.CpModel()

    starts, x_ivs, y_ivs = [], [], []
    for index, tile in enumerate(problem.tiles):
        align = max(1, tile.alignment_bytes)
        # Alignment by construction: offset = align * k, which propagates far
        # better than a modulo constraint over a free integer.
        k = model.NewIntVar(0, max(0, (cap - tile.size_bytes)) // align, f"k_{index}")
        start = model.NewIntVar(0, max(0, cap - tile.size_bytes), f"o_{index}")
        model.Add(start == align * k)
        end = model.NewIntVar(tile.size_bytes, cap, f"e_{index}")
        model.Add(end == start + tile.size_bytes)

        starts.append(start)
        y_ivs.append(model.NewIntervalVar(start, tile.size_bytes, end, f"y_{index}"))
        # Live range is a constant interval; half-open so that touching ranges
        # (last == other.first) still count as simultaneously live.
        x_ivs.append(
            model.NewIntervalVar(
                tile.first, (tile.last - tile.first) + 1, tile.last + 1, f"x_{index}"
            )
        )

    if problem.tiles:
        model.AddNoOverlap2D(x_ivs, y_ivs)

    if problem.bank_spread and space.banks > 1:
        banks = []
        for index, start in enumerate(starts):
            word = model.NewIntVar(0, cap // space.interleave_bytes, f"w_{index}")
            model.AddDivisionEquality(word, start, space.interleave_bytes)
            bank = model.NewIntVar(0, space.banks - 1, f"b_{index}")
            model.AddModuloEquality(bank, word, space.banks)
            banks.append(bank)
        for i, j in problem.conflict_pairs:
            model.Add(banks[i] != banks[j])

    peak = model.NewIntVar(0, cap, "peak")
    if problem.tiles:
        model.AddMaxEquality(peak, [s + t.size_bytes for s, t in zip(starts, problem.tiles, strict=True)])
    model.Minimize(peak)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0
    solver.parameters.num_search_workers = 1  # deterministic across runs
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        offsets = {t.name: solver.Value(s) for s, t in zip(starts, problem.tiles, strict=True)}
        return Allocation(
            offsets=offsets,
            solver="cpsat",
            feasible=True,
            timed_out=(status == cp_model.FEASIBLE),
            peak_bytes=_peak(problem, offsets),
            solve_seconds=elapsed,
            reason=None if status == cp_model.OPTIMAL else "feasible but not proven optimal within the timeout",
        )
    if status == cp_model.INFEASIBLE:
        return Allocation(
            solver="cpsat",
            feasible=False,
            solve_seconds=elapsed,
            reason="no allocation satisfies capacity, alignment and non-overlap",
        )
    return Allocation(
        solver="cpsat",
        feasible=False,
        timed_out=True,
        solve_seconds=elapsed,
        reason=f"CP-SAT returned {solver.StatusName(status)} within {timeout_ms} ms",
    )


def solve_z3(problem: AllocationProblem, timeout_ms: int = 5000) -> Allocation:
    """Minimise peak usage with z3 (SMT). Always returns within the timeout.

    Kept as a second, independent exact solver from a different family than
    `solve_cpsat`, encoded the way SMT expresses disjointness naturally: one
    `Or(a before b, b before a)` per simultaneously-live pair.
    """
    started = time.perf_counter()
    try:
        import z3
    except ImportError:
        return Allocation(
            solver="z3",
            feasible=False,
            solve_seconds=time.perf_counter() - started,
            reason="z3 is not installed (pip install z3-solver)",
        )

    space = problem.space
    opt = z3.Optimize()
    opt.set("timeout", int(timeout_ms))

    vars_ = [z3.Int(f"o_{i}") for i in range(len(problem.tiles))]
    peak = z3.Int("peak")

    for var, tile in zip(vars_, problem.tiles, strict=True):
        opt.add(var >= 0)
        opt.add(var + tile.size_bytes <= space.capacity_bytes)
        if tile.alignment_bytes > 1:
            opt.add(var % tile.alignment_bytes == 0)
        opt.add(peak >= var + tile.size_bytes)

    for i, j in problem.conflict_pairs:
        a, b = problem.tiles[i], problem.tiles[j]
        opt.add(z3.Or(vars_[i] + a.size_bytes <= vars_[j], vars_[j] + b.size_bytes <= vars_[i]))
        if problem.bank_spread and space.banks > 1:
            opt.add(
                (vars_[i] / space.interleave_bytes) % space.banks
                != (vars_[j] / space.interleave_bytes) % space.banks
            )

    opt.minimize(peak)
    status = opt.check()
    elapsed = time.perf_counter() - started

    if status == z3.sat:
        model = opt.model()
        offsets = {t.name: model.evaluate(v).as_long() for v, t in zip(vars_, problem.tiles, strict=True)}
        return Allocation(
            offsets=offsets,
            solver="z3",
            feasible=True,
            peak_bytes=_peak(problem, offsets),
            solve_seconds=elapsed,
        )
    if status == z3.unsat:
        return Allocation(
            solver="z3",
            feasible=False,
            solve_seconds=elapsed,
            reason="no allocation satisfies capacity, alignment and non-overlap",
        )
    return Allocation(
        solver="z3",
        feasible=False,
        timed_out=True,
        solve_seconds=elapsed,
        reason=f"z3 returned {status} within {timeout_ms} ms",
    )


def solve_exact(problem: AllocationProblem, timeout_ms: int = 5000, backend: str = "auto") -> Allocation:
    """Exact allocation via `backend`: `"cpsat"`, `"z3"`, or `"auto"`.

    `"auto"` prefers CP-SAT, which handles packing of this shape better, and
    falls back to z3 when OR-Tools is absent.
    """
    if backend == "cpsat":
        return solve_cpsat(problem, timeout_ms=timeout_ms)
    if backend == "z3":
        return solve_z3(problem, timeout_ms=timeout_ms)
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; expected cpsat, z3 or auto")
    if have_cpsat():
        return solve_cpsat(problem, timeout_ms=timeout_ms)
    return solve_z3(problem, timeout_ms=timeout_ms)


def plan(
    problem: AllocationProblem,
    *,
    prefer_exact: bool = True,
    timeout_ms: int = 5000,
    backend: str = "auto",
) -> Allocation:
    """Allocate, checking whatever the solver returns.

    Tries an exact solver when asked for and available, falls back to greedy on
    timeout or absence. Every feasible result is validated by `check_allocation`;
    a solver whose answer fails validation raises `MemPlanError` rather than being
    silently replaced, because that is a solver bug and hiding it loses it.
    """
    exact: Allocation | None = None
    if prefer_exact and (have_cpsat() or have_z3()):
        exact = solve_exact(problem, timeout_ms=timeout_ms, backend=backend)
        if exact.feasible:
            _validated(problem, exact)
            return exact
        if exact.reason and "no allocation satisfies" in exact.reason:
            return exact

    result = solve_greedy(problem)
    if result.feasible:
        _validated(problem, result)
    if exact is not None and exact.timed_out:
        return Allocation(
            offsets=result.offsets,
            solver="greedy",
            feasible=result.feasible,
            timed_out=True,
            peak_bytes=result.peak_bytes,
            solve_seconds=exact.solve_seconds + result.solve_seconds,
            reason=result.reason or f"{exact.solver} timed out; greedy result used",
        )
    return result


def _validated(problem: AllocationProblem, allocation: Allocation) -> None:
    violations = check_allocation(problem, allocation)
    if violations:
        raise MemPlanError(
            f"{allocation.solver} solver produced an invalid allocation: "
            + "; ".join(str(v) for v in violations[:5])
        )
