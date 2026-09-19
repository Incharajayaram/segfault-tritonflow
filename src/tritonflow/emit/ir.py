"""The instruction stream as executable structures.

This is the type that crosses the pipeline's last real boundary — the
recognizer decides, the emulator and the PyTorch seam consume — so it is
frozen first and edited only by its owner.

**One representation rule, stated once, because it is visible everywhere: the
program model is *name-based*.** `source_ops` holds `SourceRef`s, not
`Operation`s; `iter_args`/`results`/`defs`/`inputs` hold SSA names, not
`SsaValue`s. Using `list[Operation]` and `list[SsaValue]` directly would be
unserialisable: `Operation` carries dicts and regions, and `Operation`/
`SsaValue` are not hashable, so a program built from them could neither
round-trip through text nor be compared. A `SourceRef` carries the same
identity an emulator or a coverage report needs — op name, position, `loc`
name — in a form that survives `serialize`. The deviation is this one sentence
wide, and it is what makes the round-trip assertion mean something rather than
compare two references to the same object.

**What is enforced here, rather than trusted.** The placement invariant says
an `Instr` whose source lies in a region is emitted *in* that region's
`Loop`, and `scf.yield` becomes `iter_args` re-threading rather than an
instruction. A data structure cannot enforce that by itself, so `Program`
validates on construction: unique loop ids, every instruction and marker placed
in exactly one container, `loop == <owning loop id>` on every loop body entry,
one yielded value per `iter_args` entry, `total_cost` equal to the sum of the
selected instructions' costs, and **no operand that nothing defines**. That last
one is the machine-checkable half of Principle II: a value we cannot produce is
a refusal, never a zero, and never a guess. `Program.inputs` exists to make it
decidable, so that without a declared entry set "came from outside the kernel"
and "dangles" are the same string.

`Instr.cost` is *recorded*, and `Program.total_cost` is `math.fsum` of the
recorded costs, so a consumer needs no schema to interpret a serialised
program. `fsum` rather than `sum` because the sum is compared for equality
across processes and left-to-right float addition is the kind of order
dependence that shows up as a one-ulp diff on someone else's machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from ..ttir.ssa import Operation, Region

#: A loop's identity inside one program: small, dense, assigned in source order.
LoopId = int

#: The program's two marker kinds. A syntax-layer refusal reaches the
#: program as a marker too — it is still a *reason*, not a dropped operation.
UNSUPPORTED = "UNSUPPORTED"
PARSE_UNSUPPORTED = "PARSE_UNSUPPORTED"
MARKER_KINDS = (UNSUPPORTED, PARSE_UNSUPPORTED)


class AssemblyError(ValueError):
    """A program that cannot be assembled, validated or serialised.

    The distinction this class exists to draw: `PARSE_UNSUPPORTED` is *user
    input we refuse to guess about* — an expected outcome with a diagnostic.
    `AssemblyError` is *our own bug* — a dangling operand, an instruction that
    violated its own constraint, a program claiming a cost it did not spend.
    Mixing them is how a pipeline bug gets reported as a limitation of the
    target ISA.
    """


# --------------------------------------------------------------------------- #
# Source identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceRef:
    """Where an instruction came from: the operation, its position, its `loc`.

    `line`/`col` are `0` when the operation was built by hand rather than
    parsed. `loc_name` is the original Python identifier (`a_ptrs`, `pid_m`)
    that Triton's printer preserved — the one piece of provenance a reader of
    the emitted program actually recognises.
    """

    op_name: str
    line: int = 0
    col: int = 0
    loc_name: str | None = None

    @classmethod
    def of(cls, op: Operation) -> SourceRef:
        return cls(
            op_name=op.name,
            line=op.line,
            col=op.col,
            loc_name=op.loc.name if op.loc is not None else None,
        )

    def __str__(self) -> str:
        where = f"{self.line}:{self.col}" if self.line else "?"
        loc = f" loc({self.loc_name})" if self.loc_name else ""
        return f"{self.op_name}@{where}{loc}"


# --------------------------------------------------------------------------- #
# Operands
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SsaRef:
    """A value produced inside the program (or declared in `Program.inputs`)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Imm:
    """A literal. `int` and `float` are distinguished on purpose: an address
    advance of `32` and a tolerance of `32.0` are not the same value, and the
    serialised form has to keep the difference."""

    value: int | float

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MemRef:
    """A memory reference: a space, a base, and the addressing decision.

    `access` is the recognizer's `AccessDescriptor`, kept **opaque** — this
    module does not import `recognize`, so the recogniser can change its
    descriptor without emitting a new revision of the program format. What is
    compared and serialised is `access_key`, the descriptor's canonical text
    key (`recognize.descriptor.descriptor_key`). Storing the key rather than
    the descriptor object is what lets the round-trip hold for addressing,
    because a descriptor object has no textual inverse.
    """

    space: str
    base: str
    access_key: str | None = None
    """The descriptor's canonical text key — what is compared and serialised."""
    access: object | None = field(default=None, compare=False, repr=False)
    """The descriptor itself (`recognize.descriptor.AccessDescriptor`), kept
    opaque. `compare=False` because an object with no textual inverse cannot
    participate in `deserialize(serialize(p)) == p`; `access_key` is the
    comparison, and a reader can re-attach the descriptor from the key."""

    @classmethod
    def of(cls, space: str, base: str, access: object | None = None) -> MemRef:
        """Build from a descriptor when one is available.

        Key extraction prefers a `descriptor_key()` method, falls back to
        `str(access)`. Documented rather than clever: when the recognizer's
        `descriptor_key` lands, `MemRef.of` is the one place to point at it.
        """
        return cls(space=space, base=base, access_key=_access_key(access), access=access)

    def __str__(self) -> str:
        key = self.access_key or "no-access"
        return f"{self.space}:{self.base}[{key}]"


def _access_key(access: object | None) -> str | None:
    if access is None:
        return None
    key = getattr(access, "descriptor_key", None)
    if callable(key):
        return str(key())
    return str(access)


#: `Operand = SsaRef(name) | Imm(int | float) | MemRef(...)`.
Operand = SsaRef | Imm | MemRef


def operand_key(operand: Operand) -> str:
    """A stable identity for an operand, used where a descriptor would be.

    Also the value `Instr.constrained_on` holds, so a constraint can be
    re-checked against *the operand it was chosen for* rather than whichever
    operand happens to be in hand — the transitive-trust failure the brief
    names.
    """
    if isinstance(operand, MemRef):
        return f"mem:{operand.space}:{operand.base}:{operand.access_key or 'none'}"
    if isinstance(operand, Imm):
        return f"imm:{operand.value!r}"
    return f"ssa:{operand.name}"


# --------------------------------------------------------------------------- #
# Decisions, instructions, loops, programs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Binding:
    """The **decision** for one operation, up to the point the emitter takes over.

    Produced by the recogniser (`idioms.detect.annotate` → `AnnotationSet`, the
    method of record's `tritonflow.*` annotation), consumed by
    `assemble.emit_instr`. It carries what recognition established — the idiom's
    `kind`, the `descriptor`, the `tile`, and the operands already resolved to
    roles — and *optionally* a chosen `instruction` for callers that have
    already selected (hand-built inputs, or a recogniser that had to decide for
    itself). When `instruction` is `None`, assembly selects it from the schema via
    `isa/select.py`, which selects the minimum-cost variant consistent with the
    verified attributes — the emitter is the selector's consumer, not the
    recogniser.
    """

    instruction: str | None = None
    """Schema instruction name if already chosen, else `None` to select here."""

    kind: str | None = None
    """`memory` or `compute` — which instruction set the selector enumerates.

    TAIDL's real entry point is
    `add_instruction(instruction, computation_attr, addressing_attr, cost,
    update, constraints)` (`repos/taidl/taidl/accelerator.py:40`): the α/β
    attribute split is a declared property of an instruction, so an annotation
    has to name which side of the split it lands on before anything can be
    enumerated for it.
    """

    operands: dict[str, Operand] = field(default_factory=dict)
    """Role -> operand, e.g. `{"acc": ..., "a": ..., "b": ...}` for a MAC."""

    defs: tuple[str, ...] = ()
    """Names this instruction *produces*. Needed for the dangling-operand
    check: a value an instruction materialises is defined by it."""

    descriptor: object | None = field(default=None, compare=False, repr=False)
    """The memory operand a constraint is checked against (the recognizer's type)."""

    tile: tuple[int, int, int] | None = None
    """`(m, n, k)` when the decision carries a tile shape."""

    cost: float | None = None
    """Cost from the selector. `None` means "ask the schema"."""

    reason: str | None = None
    """Why this is unsupported, when it is. Never empty by accident: an
    `UNSUPPORTED` marker with no reason is a dropped operation wearing a hat."""
    subsumed: bool = False
    """True if this operation is subsumed by a structured memory descriptor."""



@dataclass(frozen=True)
class UnsupportedMarker:
    """An operation we chose not to, or could not, lower.

    First-class and placed exactly like an instruction: inside the `Loop` it
    came from when it came from one, in `Program.unsupported`
    otherwise. It carries the originating `loc` name, because the whole point
    is that a reader can find the kernel line we gave up on.
    """

    op_name: str
    reason: str = ""
    loc_name: str | None = None
    kind: str = UNSUPPORTED
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        if self.kind not in MARKER_KINDS:
            raise AssemblyError(
                f"marker kind {self.kind!r} is not one of {MARKER_KINDS}; "
                "an unknown kind would serialise as something a consumer cannot route"
            )

    @classmethod
    def from_op(
        cls,
        op: Operation,
        reason: str,
        *,
        kind: str = UNSUPPORTED,
    ) -> UnsupportedMarker:
        source = SourceRef.of(op)
        return cls(
            op_name=op.name,
            reason=reason,
            loc_name=source.loc_name,
            kind=kind,
            source=source,
        )

    def __str__(self) -> str:
        where = self.source or SourceRef(self.op_name, loc_name=self.loc_name)
        return f"{self.kind}({self.op_name} at {where}): {self.reason}"


@dataclass(frozen=True)
class Instr:
    """One emitted instruction. Name, roles, cost, provenance.

    `loop` is the owning loop's id — `None` for a top-level instruction. It is
    redundant with placement on purpose: the region-preservation property is
    asserted *from both directions* (an `Instr` in `Loop.body` must say
    `loop == id`), and a redundancy that is checked is a redundancy that cannot
    drift.

    `constraint` and `constrained_on` are the text of the instruction's own
    constraint and the key of the operand it was chosen for. They exist because
    `check_constraint(instr, operand)` has to re-validate independently — and
    it cannot do that from the schema alone, or it would be trusting the same
    lookup the selector used.
    """

    name: str
    operands: dict[str, Operand] = field(default_factory=dict)
    loop: LoopId | None = None
    cost: float = 0.0
    source_ops: tuple[SourceRef, ...] = ()
    defs: tuple[str, ...] = ()
    constraint: str | None = None
    constrained_on: str | None = None
    cost_result: object | None = None
    select_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.select_cost == 0.0 and self.cost != 0.0:
            object.__setattr__(self, "select_cost", self.cost)
        elif self.cost == 0.0 and self.select_cost != 0.0:
            object.__setattr__(self, "cost", self.select_cost)

    @property
    def roles(self) -> tuple[str, ...]:
        """Operand roles, sorted — the order everything textual uses, so a
        dict built in a different order serialises to the same bytes."""
        return tuple(sorted(self.operands))

    def operand(self, role: str) -> Operand | None:
        return self.operands.get(role)

    @property
    def source(self) -> SourceRef | None:
        return self.source_ops[0] if self.source_ops else None

    def __str__(self) -> str:
        roles = " ".join(f"{r}={self.operands[r]}" for r in self.roles)
        return f"{self.name} {roles}".rstrip()


@dataclass(frozen=True)
class AsyncOp:
    """An asynchronous accelerator operation handle (e.g. Vortex DXA async copy, WGMMA launch)."""

    handle: str
    launch_instr: Instr
    wait_instr: Instr | None = None
    completion_barrier: int | None = None


@dataclass(frozen=True)
class Loop:
    """A recovered `scf.for`, with the values it threads.

    `iter_args` are the block arguments the loop carries; `results` are the
    values `scf.yield` re-threads out of it, one per iter_arg. `scf.yield`
    itself is *not* an instruction and has no representation here beyond that
    pairing — emitting it, or hoisting a body `tt.load` out of `body`, are the
    two bugs this structure exists to make impossible.
    """

    id: LoopId
    induction_var: str | None = None
    lower: Operand | None = None
    upper: Operand | None = None
    step: Operand | None = None
    iter_args: tuple[str, ...] = ()
    inits: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    yields: tuple[str, ...] = ()
    """The values `scf.yield` re-threads, positionally: `yields[i]` becomes the
    next iteration's `iter_args[i]`. `results[i]` is the *post-loop* name for the
    same position. The two are different names for the same slot — the body
    defines `%a_ptrs_40` and the loop's result is `%acc_25` — and a consumer
    needs both: `yields` to run the loop again, `results` to read its output.
    Deriving one from the other is impossible, which is why it is recorded.
    """
    body: tuple[Instr | UnsupportedMarker, ...] = ()
    source: SourceRef | None = None

    @property
    def instrs(self) -> tuple[Instr, ...]:
        return tuple(item for item in self.body if isinstance(item, Instr))

    @property
    def markers(self) -> tuple[UnsupportedMarker, ...]:
        return tuple(item for item in self.body if isinstance(item, UnsupportedMarker))

    @property
    def iter_arg_pairs(self) -> tuple[tuple[str, str], ...]:
        """`(init, iter_arg)` per entry — the re-threading, spelled out."""
        return tuple(zip(self.inits, self.iter_args, strict=False))

    def __str__(self) -> str:
        return (
            f"loop {self.id} {self.induction_var} iter_args({', '.join(self.iter_args)}) "
            f"-> ({', '.join(self.results)}) [{len(self.body)} item(s)]"
        )


@dataclass(frozen=True)
class Program:
    """The emitted program. Validated on construction, not on use.

    Containers are exclusive: each instruction or marker lives in exactly one of
    `Loop.body`, `instrs` or `epilogue`, and `unsupported` holds the ones with no
    loop to belong to. `epilogue` is not a sub-list of `instrs` — sharing would
    mean `total_cost` either double-counts or silently depends on which list a
    reader walked.
    """

    isa_name: str
    schema_version: int
    kernel_name: str = ""
    loops: tuple[Loop, ...] = ()
    instrs: tuple[Instr, ...] = ()
    epilogue: tuple[Instr, ...] = ()
    unsupported: tuple[UnsupportedMarker, ...] = ()
    total_cost: float = 0.0
    inputs: tuple[str, ...] = ()
    async_ops: tuple[AsyncOp, ...] = ()
    total_time_cycles: float | None = None
    resources: object | None = None

    def __post_init__(self) -> None:
        validate_program(self)

    # -- derived views ------------------------------------------------------ #

    @property
    def aggregate_resources(self) -> object:
        from tritonflow.isa.cost import Resources
        dram_b = scratch_b = tx = bc = mac = regs = 0
        for i in self.instructions():
            res = getattr(i, "cost_result", None)
            if res is not None and getattr(res, "resources", None) is not None:
                r = res.resources
                dram_b += getattr(r, "dram_bytes", 0)
                scratch_b += getattr(r, "scratch_bytes", 0)
                tx += getattr(r, "transactions", 0)
                bc += getattr(r, "bank_conflicts", 0)
                mac += getattr(r, "mac_ops", 0)
                regs += getattr(r, "registers", 0)
        return Resources(
            dram_bytes=dram_b,
            scratch_bytes=scratch_b,
            transactions=tx,
            bank_conflicts=bc,
            mac_ops=mac,
            registers=regs,
        )

    @property
    def aggregate_time_cycles(self) -> float:
        total = 0.0
        for i in self.instructions():
            res = getattr(i, "cost_result", None)
            if res is not None and getattr(res, "time", None) is not None:
                total += getattr(res.time, "cycles", 0.0)
        return total

    @property
    def cost_sum(self) -> float:
        """The cost the selected instructions add up to."""
        return math.fsum(
            [instr.cost for loop in self.loops for instr in loop.instrs]
            + [instr.cost for instr in self.instrs]
            + [instr.cost for instr in self.epilogue]
        )

    @property
    def body(self) -> tuple[Loop | Instr, ...]:
        """Top-level items in source order, loops and instructions interleaved.

        Derived from source position rather than stored, because a stored order
        is a second thing that can disagree with the first. Ties (hand-built
        sources with no line) keep construction order, so the result is total
        and reproducible.
        """
        items: list[Loop | Instr] = [*self.loops, *self.instrs]
        return tuple(sorted(items, key=_source_position))

    def execution_order(self) -> tuple[Loop | Instr, ...]:
        """`body` followed by `epilogue`: the order an emulator walks."""
        return (*self.body, *self.epilogue)

    def instructions(self) -> tuple[Instr, ...]:
        """Every instruction, in execution order, loops expanded."""
        out: list[Instr] = []
        for item in self.execution_order():
            out.extend(item.instrs if isinstance(item, Loop) else (item,))
        return tuple(out)

    def markers(self) -> tuple[UnsupportedMarker, ...]:
        """Every marker, top-level and in-loop, in execution order."""
        out: list[UnsupportedMarker] = []
        for item in self.execution_order():
            out.extend(item.markers if isinstance(item, Loop) else ())
        return (*out, *self.unsupported)

    def loop(self, loop_id: LoopId) -> Loop | None:
        for candidate in self.loops:
            if candidate.id == loop_id:
                return candidate
        return None

    def defined_names(self) -> tuple[str, ...]:
        """Every name the program produces or receives, in a stable order."""
        names = list(self.inputs)
        for loop in self.loops:
            if loop.induction_var:
                names.append(loop.induction_var)
            names.extend(loop.iter_args)
            names.extend(loop.results)
        for instr in self.instructions():
            names.extend(instr.defs)
        return tuple(names)

    def __str__(self) -> str:
        return (
            f"Program({self.isa_name} v{self.schema_version}, kernel={self.kernel_name or '?'}, "
            f"{len(self.loops)} loop(s), {len(self.instructions())} instr(s), "
            f"{len(self.markers())} unsupported, cost={self.total_cost})"
        )


# --------------------------------------------------------------------------- #
# Assembly-time structures (never serialised)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrderedOp:
    """One operation, with the region it belongs to. What `order_regions` yields.

    `region` is the point: a flat topological sort over the def-use graph loses
    which region an operation came from, and an operation inside `scf.for` that
    has lost its region cannot be placed back inside the loop.
    `loop_id` is `None` for the kernel body and for anything not in a loop.
    `terminator` records that this operation ends its block — for a loop body
    that is `scf.yield`, which re-threads `iter_args` and is never an
    instruction. `nested` means the operation sits inside a nested
    `scf.for`, i.e. inside a loop that is itself inside a loop: the program
    format has no nested loop, so such an operation is marked `UNSUPPORTED`
    rather than hoisted up into the enclosing loop.
    """

    op: Operation
    region: Region
    loop_id: LoopId | None = None
    index: int = 0
    terminator: bool = False
    nested: bool = False

    @property
    def name(self) -> str:
        return self.op.name

    def __str__(self) -> str:
        where = f"loop {self.loop_id}" if self.loop_id is not None else "kernel body"
        return f"{self.name} ({where}#{self.index})"


@dataclass(frozen=True)
class EmissionRecord:
    """One line of the selection evidence: what was decided, and where it went.

    The fields after `cost` are `SelectionReport`'s: a rejected candidate
    carries the predicate that failed it, and the gap against the oracle is
    reported rather than hidden. `rejected` holds `name: reason` per rejected
    candidate, so the completeness report can show *why* the cheaper-looking
    instruction was not used.
    """

    op_name: str
    source: SourceRef | None
    instruction: str | None
    cost: float | None = None
    reason: str | None = None
    loop_id: LoopId | None = None
    placement: str = "instrs"
    rejected: tuple[str, ...] = ()
    oracle_min_cost: float | None = None
    gap: float | None = None

    def __str__(self) -> str:
        chosen = self.instruction or "UNSUPPORTED"
        cost = "" if self.cost is None else f" cost={self.cost}"
        why = f" ({self.reason})" if self.reason else ""
        gap = "" if not self.gap else f" gap={self.gap}"
        rejected = f" rejected[{', '.join(self.rejected)}]" if self.rejected else ""
        return f"{self.op_name} -> {chosen}{cost}{gap} in {self.placement}{why}{rejected}"


def _source_position(item: Loop | Instr) -> tuple[int, int]:
    if isinstance(item, Instr):
        source = item.source_ops[0] if item.source_ops else None
    else:
        source = item.source
    if source is None:
        return (0, 0)
    return (source.line, source.col)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_program(program: Program) -> None:
    """Raise :class:`AssemblyError` unless `program` is internally consistent.

    Cheap enough to run on construction *and* again in `serialize`, which is
    what makes "a dangling operand reference raises `AssemblyError`" true of
    the serialiser as written rather than true by accident of the constructor
    having run first.
    """
    seen_loop_ids: set[int] = set()
    for loop in program.loops:
        if loop.id in seen_loop_ids:
            raise AssemblyError(f"two loops share id {loop.id}")
        seen_loop_ids.add(loop.id)
        if loop.results and len(loop.results) != len(loop.iter_args):
            raise AssemblyError(
                f"loop {loop.id} yields {len(loop.results)} value(s) for "
                f"{len(loop.iter_args)} iter_args; scf.yield must re-thread exactly "
                "the values the loop carries"
            )
        if loop.inits and len(loop.inits) != len(loop.iter_args):
            raise AssemblyError(
                f"loop {loop.id} has {len(loop.inits)} initialiser(s) for "
                f"{len(loop.iter_args)} iter_args"
            )
        if loop.yields and len(loop.yields) != len(loop.iter_args):
            raise AssemblyError(
                f"loop {loop.id} yields {len(loop.yields)} value(s) for "
                f"{len(loop.iter_args)} iter_args; scf.yield must re-thread exactly "
                "the values the loop carries"
            )
        for item in loop.body:
            if isinstance(item, Instr) and item.loop != loop.id:
                raise AssemblyError(
                    f"{item.name} is emitted inside loop {loop.id} but records "
                    f"loop={item.loop}; an instruction carries where it was placed"
                )

    placements: dict[tuple[str, int, int], str] = {}
    for where, instrs in _placements(program).items():
        for instr in instrs:
            if where != "loop" and instr.loop is not None:
                raise AssemblyError(
                    f"{instr.name} is placed in {where} but records loop={instr.loop}; "
                    "an instruction at the top level belongs to no loop"
                )
            _record_placement(placements, instr.source, f"{where}:{instr.name}")
    for where, markers in _marker_placements(program).items():
        for marker in markers:
            source = marker.source or SourceRef(marker.op_name)
            _record_placement(placements, source, f"{where}:{marker.op_name}")

    defined: dict[str, str] = {}

    def define(name: str, who: str) -> None:
        previous = defined.get(name)
        if previous is not None and previous != who:
            raise AssemblyError(f"name {name} is defined twice ({previous} and {who})")
        defined[name] = who

    for name in program.inputs:
        define(name, "Program.inputs")
    for loop in program.loops:
        if loop.induction_var:
            define(loop.induction_var, f"loop {loop.id} induction variable")
        for name in loop.iter_args:
            define(name, f"loop {loop.id} iter_args")
        for name in loop.results:
            define(name, f"loop {loop.id} results")

    for instr in program.instructions():
        for name in instr.defs:
            define(name, f"{instr.name}")

    for instr in program.instructions():
        for role in instr.roles:
            operand = instr.operands[role]
            if isinstance(operand, SsaRef) and operand.name not in defined:
                raise AssemblyError(
                    f"dangling operand: {instr.name} reads {operand.name} for role "
                    f"{role!r}, which nothing defines (make it a Program.inputs entry "
                    "if it arrives from outside the kernel)"
                )

    total = program.cost_sum
    if program.total_cost != total:
        raise AssemblyError(
            f"total_cost={program.total_cost!r} but the selected instructions sum to "
            f"{total!r}; the header must be summable without the schema (postcondition 5)"
        )


def _placements(program: Program) -> dict[str, tuple[Instr, ...]]:
    return {
        "loop": tuple(instr for loop in program.loops for instr in loop.instrs),
        "instrs": program.instrs,
        "epilogue": program.epilogue,
    }


def _marker_placements(program: Program) -> dict[str, tuple[UnsupportedMarker, ...]]:
    return {
        "loop": tuple(m for loop in program.loops for m in loop.markers),
        "unsupported": program.unsupported,
    }


def _record_placement(
    placements: dict[tuple[str, int, int], str], source: SourceRef | None, where: str
) -> None:
    key = (
        source.op_name if source else "?",
        source.line if source else 0,
        source.col if source else 0,
    )
    previous = placements.get(key)
    if previous is not None:
        raise AssemblyError(
            f"{key[0]} at {key[1]}:{key[2]} is emitted twice ({previous} and {where}); "
            "one operation produces one instruction or one marker, never both"
        )
    placements[key] = where


# --------------------------------------------------------------------------- #
# Duck-typed seam to the ISA schema module
# --------------------------------------------------------------------------- #


class InstructionLike(Protocol):
    """What `emit_instr` needs from a schema instruction.

    A `Protocol`, not an import, so that a layer stays testable without its
    upstream: the emitter must be exercisable against a hand-built schema
    without depending on `isa/schema.py` directly. This is the smallest set
    of members that makes costing and constraint re-checking possible.
    """

    name: str
    kind: str
    constraint: str | None

    def cost_of(self, descriptor: object | None, tile: tuple[int, int, int] | None) -> float: ...


class SchemaLike(Protocol):
    """What `assemble` needs from an ISA schema: name, version, lookup."""

    name: str
    schema_version: int

    def instruction(self, name: str) -> InstructionLike | None: ...


class AnnotationSetLike(Protocol):
    """What `assemble` needs from the recogniser's annotation set.

    One method, deliberately. `bindings_for(op)` returns `()` for an operation
    no idiom matched, one `Binding` for the normal case, and two or more for
    the "two annotations claim the same operation" failure the contract
    requires to be an error rather than a silent drop. The operation object is
    the argument rather than a name because `Operation` is unhashable and
    cannot be a dict key.
    """

    def bindings_for(self, op: Operation) -> tuple[Binding, ...]: ...


__all__ = [
    "MARKER_KINDS",
    "PARSE_UNSUPPORTED",
    "UNSUPPORTED",
    "AnnotationSetLike",
    "AssemblyError",
    "Binding",
    "EmissionRecord",
    "Imm",
    "Instr",
    "InstructionLike",
    "Loop",
    "LoopId",
    "MemRef",
    "Operand",
    "OrderedOp",
    "Program",
    "SchemaLike",
    "SourceRef",
    "SsaRef",
    "UnsupportedMarker",
    "operand_key",
    "validate_program",
]
