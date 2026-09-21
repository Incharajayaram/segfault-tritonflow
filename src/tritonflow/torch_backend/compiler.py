"""The Dynamo backend: the door `torch.compile` comes in through.

    torch.compile(fn, backend="tritonflow")

registers here. `tritonflow_backend` is handed an FX graph and the example inputs,
and must return a callable that reproduces the graph's result. Everything else in
this project exists to make that callable's answer *correct and auditable* rather
than merely present.

**Two paths, and the seam never blurs them.**

*The lowering path.* Every kernel the seam can lower is a **recorded lowering**:
a frozen TTIR text plus the launch environment it was compiled under
(`kernels/`). The seam runs it through the *real* pipeline — parse, build
def-use, annotate, assemble against the real ISA-1 schema with the real selector,
then execute on `emu/exec.py`. Nothing about that chain is stubbed, and the
program it executes is an artifact a reader can print with `disassemble`.

*The fallback path.* A graph the seam cannot lower runs on eager PyTorch and
produces a `FallbackRecord` naming why. This is the hardest rule this module
follows: the eager floor is acknowledged in writing, per graph, and never
disguised. What the seam must not do is run eager code *and report success*,
which is the custom-backend sin this discipline exists to avoid.

**Where the TTIR comes from, and in what order.** Read off the graph, Inductor
produces *Python source for a Triton kernel*; the TTIR this pipeline consumes
exists only after `triton.compile` has run. So the seam tries five sources in a
stated order and records which one answered (`CompiledKernel.provenance`):

1. **`attached`** — TTIR a caller put on the graph (`extract_ttir`). The general
   hand-off, and the only source that can lower a kernel nobody here has heard of.
2. **`inductor`** — Path 1a. Inductor is run on the FX graph (or a one-op
   synthetic graph) under `CompileSpy`; every `triton.compiler.compile` TTIR is
   captured and lowered by the identical pipeline. Scoped to matmul/linear/relu
   in v1. When Inductor's CPU path never calls Triton, the spy returns nothing
   and this stage records a note — it does not invent TTIR.
3. **`dynamic`** — Path 1. Triton's own compiler is invoked at plan time for the
   graph's actual op and shape, GPU-free, and the TTIR it produces is lowered by
   the identical pipeline. This is extraction, not a lookup: it works for shapes
   and tile sizes nobody recorded. It needs Triton installed (the `extract`
   extra), and its absence is a `FallbackRecord` naming *why*, never a guess.
4. **`flaggems`** — Path 2. If `flag_gems` is importable, an op it implements is
   compiled from its Triton source through the same extractor.
5. **`recorded`** — Path 3. A frozen TTIR text in `kernels/`, the day-1 path.

Anything else is eager PyTorch with a `FallbackRecord`, which is Path 4 and the
contract's hardest rule. `NOT_EXTRACTED` survives as the reason text for a graph
whose *op* has no extraction kernel, and its wording changed with the code:
it used to say extraction never happens, which is no longer true.

**An all-or-nothing decision per graph.** Lowering *part* of an FX graph would
mean reimplementing partition and stitch semantics, which is Inductor's job and
not this project's. So a graph is lowered only when it is exactly one matchable
kernel and nothing else; any extra node sends the whole graph to fallback. That is
a conservative rule stated out loud, and the alternative — guessing at partial
coverage — is how a backend reports numbers it did not compute.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch._dynamo.backends.registry import register_backend

from ..emit.assemble import assemble
from ..emit.ir import Program
from ..emu.exec import ProgramNotExecutable, UnsupportedInstruction, emulate
from ..emu.precision import PrecisionPolicy, tf32_truncate
from ..extract import (
    Extracted,
    ExtractionError,
    ExtractionUnavailable,
    extract_for_op,
    extract_via_inductor,
    flaggems_bridge,
    is_inductor_op,
    is_supported,
)
from ..extract import (
    capability as extraction_capability,
)
from ..idioms.detect import annotate
from ..isa.schema import load_builtin
from ..recognize.op_shapes import input_precision, shape_of
from ..ttir.graph import DefUseGraph, build_def_use, walk_region
from ..ttir.ssa import Module, Operation
from ..ttir.to_ir import parse_module
from . import device_interface
from .device_interface import DEVICE_NAME

__all__ = [
    "EXTRACTED_PATHS",
    "NOT_EXTRACTED",
    "CompiledKernel",
    "FallbackRecord",
    "LoweringError",
    "LoweringPlan",
    "extraction_provenance",
    "lower_and_run",
    "plan_graph",
    "prepare",
    "recorded_kernels",
    "tritonflow_backend",
    "verify_device",
]

#: The gap that remains after Path 1, stated rather than papered over: extraction
#: covers a named set of ops, and a graph outside that set has no TTIR source at
#: all. It is no longer true that nothing is extracted — say the true thing.
NOT_EXTRACTED = (
    "the graph is outside the extractable op set: no kernel source exists for it, "
    "so there is nothing to compile with triton.compile. Matmul, linear/addmm and "
    "elementwise (add, sub, mul, div, relu, neg) are extractable; see "
    "extract.SUPPORTED_OPS. A caller may also attach TTIR directly with "
    "compiler.extract_ttir."
)

#: Which source produced a lowering, in the order the seam tries them. Recorded
#: on every `CompiledKernel` so a report can say *how* something was lowered
#: instead of only that it was.
EXTRACTED_PATHS: tuple[str, ...] = (
    "attached",
    "inductor",
    "dynamic",
    "flaggems",
    "recorded",
)

#: Where the recorded lowerings live. Inside the package, because a backend that
#: reads a path in the *checkout* cannot be imported from anywhere else.
KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: The launch environment each recorded lowering was compiled under.
#:
#: Keys are exactly as they appear in the TTIR — scalars under their bare name and
#: symbols under their SSA spelling — because the same dictionary serves two
#: consumers with different naming: selection resolves `%sam` from the schema's
#: predicate symbols, and the emulator looks up the *parameter* `%M`. Defining it
#: once, in the IR's own names, is what keeps those two from drifting.
#:
#: The values are the physical strides of *contiguous* tensors at this shape:
#: A is `(128, 64)` so its k-stride is 64; B is `(64, 128)` so its k-stride is
#: `N = 128`; C is `(128, 128)` so its row stride is 128. The previous value for
#: `%sbk` in the ledger was 64, which addresses the wrong rows — found by executing
#: against fp64, and corrected here to the same value `fixtures/launch_env.json`
#: carries for Tier 1.
_LAUNCH_ENV: dict[str, int] = {
    "M": 128,
    "N": 128,
    "K": 64,
    "%M": 128,
    "%N": 128,
    "%K": 64,
    "%sam": 64,
    "%sak": 1,
    "%sbk": 128,
    "%sbn": 1,
    "%scm": 128,
    "%scn": 1,
}

#: Which recorded lowering covers which kernel entry point.
_RECORDED_ENV: dict[str, dict[str, int]] = {"matmul_128x128x64": dict(_LAUNCH_ENV)}


class LoweringError(RuntimeError):
    """The seam refused a graph, with a reason a `FallbackRecord` can carry."""


class DtypeNotSupported(LoweringError):
    """The device does not declare this input's dtype.

    A subclass of `LoweringError` because the message and the mechanism are the
    same (refuse, never upcast), and separate from it because it is the one
    refusal whose cause is the *caller's data* rather than a mistake in this
    file. That distinction decides what happens next: a binding mistake means
    this code is wrong and must be loud, while an undeclared dtype means the
    graph simply cannot run here — and the contract says such a graph runs in
    PyTorch with a record. Measured before this existed: every fp16 and bf16
    matmul, i.e. essentially every real model, raised out of the compiled
    callable instead of falling back.
    """


@dataclass(frozen=True)
class CompiledKernel:
    """A recorded lowering, prepared: the program and how to feed it.

    `pointers` and `scalars` come from the parsed module's *function signature*
    (`Function.args`), so the mapping between a caller's tensors and the kernel's
    parameters is read off the IR rather than assumed from a name convention.
    `load_roots`/`store_roots` come from walking the def-use graph back from the
    `tt.load`/`tt.store` operands, which is how the seam knows which pointer
    arguments are inputs and which are outputs — the same information the
    recogniser derives, obtained the same way (by following the pointers).
    """

    name: str
    entry: str
    program: Program
    pointers: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: Mapping[str, int]
    grid: tuple[int, int, int]
    extents: Mapping[str, tuple[int, ...]]
    input_precision: str
    provenance: str = "recorded"
    """Which route produced this program: see :data:`EXTRACTED_PATHS`.

    Read by the plan and by any coverage report, because "lowered" and "lowered
    by extraction" are different claims and a number that cannot tell them apart
    is the kind of number this project exists not to publish.
    """

    tile: tuple[int, int, int] = (64, 64, 32)
    """`(BM, BN, BK)`, read off the module's own `tt.dot` operand types.

    The runner pads to *this*, so a kernel compiled for a 128-wide tile is no
    longer padded as if it were 64 — the hard-coded 64/64/32 that used to live in
    the runner was a second, silent statement of the same fact.
    """

    pads: tuple[int, int, int] = (0, 0, 0)
    """`(rows, inner, cols)` added to the caller's logical extents.

    Carried so the runner can *check* the padding it derives against what the
    program was compiled for: a mismatch is a `LoweringError`, because padding by
    the wrong amount addresses the wrong elements and returns a wrong answer
    rather than failing.
    """

    flat_width: int | None = None
    """Padded 1-D buffer extent, for the elementwise kernels. `None` when 2-D."""

    bias_lowered: bool = False
    """Whether the bias addition is *inside* the kernel that just ran.

    The recorded matmul lowerings add the bias in PyTorch after the kernel (`the
    bias addition is NOT lowered to the ISA`), while the extracted linear kernel
    carries it in the program. Adding it again would double it, so which of the
    two happened is recorded here instead of being inferred from the op name.
    """

    def default_policy(self) -> PrecisionPolicy:
        """The precision contract this kernel declares, not a default chosen here.

        `tt.dot`'s `inputPrecision` attribute (read by `op_shapes.input_precision`)
        says tf32 for the frozen Tier-1 kernel, and `M`/`N`/`K` give the reduction
        length the tolerance is derived from. Executing under `ieee` instead would
        make the emulator *more* accurate than the program it is executing and
        would let the seam's tolerance be `0` — a bound that passes by comparing
        the emulator with itself rather than with torch.
        """
        return PrecisionPolicy.for_tile(
            input_precision=self.input_precision,  # type: ignore[arg-type]
            reduction_length=int(self.scalars.get("%K") or 0),
        )

    @property
    def problem_shape(self) -> tuple[int | None, int | None, int | None]:
        """`(M, N, K)` of the *problem*, from the signature scalars.

        Read from the launch environment the lowering was compiled under, so a
        caller asking "does this kernel apply to my tensors?" compares two
        statements of the same fact rather than trusting a comment.
        """
        return (self.scalars.get("%M"), self.scalars.get("%N"), self.scalars.get("%K"))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"CompiledKernel({self.name}, entry={self.entry}, "
            f"instrs={len(self.program.instructions())}, "
            f"markers={len(self.program.markers())})"
        )


@dataclass(frozen=True)
class FallbackRecord:
    """One graph the seam did not lower, and the reason a reader can act on."""

    reason: str
    stage: str
    nodes: tuple[str, ...] = ()
    loc: str | None = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - presentation only
        where = f" at {self.loc}" if self.loc else ""
        return f"fallback[{self.stage}]{where}: {self.reason}"


@dataclass
class LoweringPlan:
    """What the seam decided about one graph, and what it observed while deciding."""

    lowered: list[CompiledKernel] = field(default_factory=list)
    fallbacks: list[FallbackRecord] = field(default_factory=list)
    nodes: tuple[str, ...] = ()
    notes: list[FallbackRecord] = field(default_factory=list)
    """Reasons a *higher-priority* source was skipped, when a lower one answered.

    Kept apart from `fallbacks` because they mean different things: a fallback is
    work that ran in PyTorch, while a note is the answer to "why did a recorded
    fixture lower this graph rather than a fresh extraction". Without notes the two
    most different causes — Triton is not installed, and Triton has no kernel for
    this op — are indistinguishable from the outside, since both leave the same
    `provenance == "recorded"` behind.
    """

    multi_kernel: bool = False
    """Whether the interpreter path ran, lowering node by node."""

    node_lowerings: int = 0
    """Nodes the interpreter lowered to the ISA (0 until it has run once)."""

    node_fallbacks: int = 0
    """Nodes the interpreter ran in PyTorch, each with its own record above."""

    @property
    def fully_lowered(self) -> bool:
        """Every operation in this graph ran on the ISA.

        Cannot be true while any node fell back to eager.
        """
        has_lowering = bool(self.lowered) or self.node_lowerings > 0
        no_fallbacks = self.node_fallbacks == 0 and len(self.fallbacks) == 0
        return has_lowering and no_fallbacks

    def compilation_summary(self) -> dict[str, Any]:
        """A per-compilation summary: nodes lowered, nodes fallen back, with causes."""
        n_lowered = len(self.lowered) if self.lowered else self.node_lowerings
        n_fallen_back = self.node_fallbacks
        total = n_lowered + n_fallen_back
        fraction = (n_lowered / total) if total > 0 else 0.0
        return {
            "fully_lowered": self.fully_lowered,
            "nodes_lowered": n_lowered,
            "nodes_fallen_back": n_fallen_back,
            "total_nodes": total,
            "lowered_fraction": fraction,
            "fallback_causes": [
                {"stage": f.stage, "nodes": list(f.nodes), "reason": f.reason}
                for f in self.fallbacks
            ],
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "fully_lowered": self.fully_lowered,
            "lowered": [k.name for k in self.lowered],
            "provenance": [k.provenance for k in self.lowered],
            "multi_kernel": self.multi_kernel,
            "node_lowerings": self.node_lowerings,
            "node_fallbacks": self.node_fallbacks,
            "notes": [
                {"reason": n.reason, "stage": n.stage, "nodes": list(n.nodes)}
                for n in self.notes
            ],
            "fallbacks": [
                {
                    "reason": f.reason,
                    "stage": f.stage,
                    "loc": f.loc,
                    "nodes": list(f.nodes),
                }
                for f in self.fallbacks
            ],
            "nodes": list(self.nodes),
        }


# --------------------------------------------------------------------------- #
# The recorded lowerings
# --------------------------------------------------------------------------- #


def recorded_kernels() -> dict[str, str]:
    """`{stem: ttir text}` for every recorded lowering in the package."""
    if not KERNEL_DIR.is_dir():  # pragma: no cover - packaging failure
        return {}
    return {path.stem: path.read_text() for path in sorted(KERNEL_DIR.glob("*.ttir"))}


def _declared_precision(module: Module) -> str:
    """`tt.dot`'s declared input precision, `"ieee"` when it declares none.

    Delegated to `recognize.op_shapes.input_precision`, which is the function the
    recogniser uses — so the emulator's precision and the recogniser's opinion of
    it cannot disagree about the same attribute.
    """
    for op in walk_region(module.body):
        if op.name == "tt.dot":
            return input_precision(op)
    return "ieee"


def _module_of(name: str) -> Module:
    """The parsed module of a recorded lowering.

    Public to the checks rather than private, because "the grid is derived from the
    kernel's own tile width" is a claim about the *IR*, and a check that wants to
    verify it has to read the same IR the derivation reads.
    """
    parsed = parse_module(recorded_kernels()[name])
    if not parsed.ok:
        raise LoweringError(f"{name}: PARSE_UNSUPPORTED — {parsed.diagnostic}")
    return parsed.module


def _problem_extents(
    inputs: Sequence[str], outputs: Sequence[str], env: Mapping[str, int]
) -> dict[str, tuple[int, ...]]:
    """The **buffer** extent of each pointer argument, `(M,K)`/`(K,N)`/`(M,N)`.

    The one derivation in this module that is a *convention* rather than a
    reading, so it is checked rather than trusted: Triton passes a matmul's
    pointers in the order `(a, b, c)`, and a caller's tensors must equal these
    extents — mismatch is a `LoweringError` naming both shapes, never a reshape.

    Tile descriptors cannot supply this. `describe(op)` returns the shape of the
    *tile* the kernel loads (`(64, 32)` for Tier 1's A tile), while the kernel's
    index arithmetic addresses the *whole* tensor: `pid_m * 64 * sam` reaches row
    127 of a 128-row buffer. Allocating storage at tile size — which an earlier
    version of this function did — puts the second program out of bounds, and the
    emulator then refuses correctly (`StorageError`), which is how the bug was
    found rather than shipped.
    """
    rows, columns, inner = env.get("M"), env.get("N"), env.get("K")
    extents: dict[str, tuple[int, ...]] = {}
    if rows and columns and inner:
        if len(inputs) >= 1:
            extents[inputs[0]] = (rows, inner)
        if len(inputs) >= 2:
            extents[inputs[1]] = (inner, columns)
        for name in outputs:
            extents[name] = (rows, columns)
        return extents

    # Outside the matmul convention, the buffer extent is the kernel's own flat
    # address range: the largest `end` any `tt.make_range` in the kernel declares,
    # per pointer. This is the 1-D case (vecadd): the kernel addresses n elements
    # of each buffer and the emulator refuses anything smaller, so under-allocating
    # is caught — but over-allocating to the kernel's declared range is exactly
    # what the hardware's allocation would be. Found by executing t0 on the GPU
    # path: the output buffer was previously allocated at `(1,)` and the emulator
    # correctly refused with `StorageError`.
    #
    # The kernel text is not available here (this function receives only the env
    # and the input/output name lists), so the caller — `prepare`, which holds the
    # parsed module — stashes the derived flat width under a reserved env key.
    flat = int(env.get("%__flat_width__") or 0)
    if flat:
        for name in (*inputs, *outputs):
            extents.setdefault(name, (flat,))
    return extents


def _pointer_roots(graph: DefUseGraph, module: Module, start: str) -> frozenset[str]:
    """Function-argument pointer names reachable from `start` by following operands.

    A breadth-first walk with a visited set, not an unguarded recursion: the
    corpus's chains are a handful of hops, but nothing in the IR bounds them, and
    an unbounded walk over a cyclic graph is a hang rather than a wrong answer.

    Block arguments of an `scf.for` body are aliases — the body's `i`-th iter_arg
    argument is fed by `inits[i]` — so the walk crosses the loop boundary and
    reaches the pointer the *init* came from. Without that, every pointer in Tier 1
    would look like it came from nowhere, because Tier 1's loads address block
    arguments.
    """
    function = module.functions()[0]
    pointer_args = {value.name for value in function.args if value.type.kind == "ptr"}
    definitions: dict[str, Operation] = {}
    aliases: dict[str, str] = {}
    for op in walk_region(module.body):
        for result in op.results:
            definitions[result.name] = op
    for loop in _loops(module):
        body_args = _block_args(loop)
        inits = list(loop.operands[3:])
        for index, init in enumerate(inits):
            position = index + 1
            if position < len(body_args):
                aliases[body_args[position].name] = init.name

    found: set[str] = set()
    seen: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in pointer_args:
            found.add(name)
            continue
        if name in aliases:
            pending.append(aliases[name])
            continue
        op = definitions.get(name)
        if op is None:
            continue
        pending.extend(operand.name for operand in op.operands)
    return frozenset(found)


def _loops(module: Module) -> list[Operation]:
    return [op for op in walk_region(module.body) if op.name == "scf.for"]


def _block_args(loop: Operation) -> tuple[Any, ...]:
    if not loop.regions or not loop.regions[0].blocks:
        return ()
    return loop.regions[0].blocks[0].args


_BLOCK_RE = re.compile(r"^\s*(\d+)")


def _block_width(module: Module) -> int:
    """The kernel's tile width: the largest `tt.make_range` extent it builds.

    The attribute *text* is `"64 : i32"`, so the parse must take the **leading**
    integer. Concatenating every digit in the string gives `6432` — it eats the
    `32` of the type name — which yields a block wider than the problem, a grid of
    `(1, 1, 1)`, and a program that computes one tile while the seam reports the
    whole result. That was this function's first version.
    """
    extents: list[int] = []
    for op in walk_region(module.body):
        if op.name != "tt.make_range":
            continue
        raw = op.attributes.get("end")
        match = _BLOCK_RE.match(str(getattr(raw, "value", raw) or ""))
        if match:
            extents.append(int(match.group(1)))
    return max(extents) if extents else 0


def _tile_from_module(module: Module) -> tuple[int, int, int]:
    """`(BM, BN, BK)`, read off the module's own `tt.dot` types.

    A reading, not a convention: the result type is the `(BM, BN)` tile and the
    operand whose column extent differs from the result's is the `(BM, BK)` one,
    so `BK` comes from the IR too. Deriving the tile here rather than hard-coding
    it in the runner is what lets a kernel compiled for a different tile be run
    without editing the runner — and what stops the two from disagreeing.
    """
    for op in walk_region(module.body):
        if op.name != "tt.dot" or not op.results:
            continue
        result_shape = shape_of(op.results[0].type)
        if len(result_shape) != 2 or not all(isinstance(d, int) for d in result_shape):
            continue
        inner: int | None = None
        for operand in op.operands:
            shape = shape_of(operand.type)
            if len(shape) == 2 and isinstance(shape[1], int) and shape[1] != result_shape[1]:
                inner = int(shape[1])
                break
        return (
            int(result_shape[0]),  # type: ignore[arg-type]
            int(result_shape[1]),  # type: ignore[arg-type]
            inner if inner is not None else (_block_width(module) or 32),
        )
    block = _block_width(module) or 64
    return (block, block, 32)


def _flat_width(module: Module, env: Mapping[str, int]) -> int:
    """The 1-D buffer extent: the padded width the caller declared, if any.

    `env["%__flat_width__"]` is the extractor's statement of the buffer it
    compiled for, and it wins. Falling back to the kernel's `tt.make_range` width
    is the old behaviour and is right only for a kernel whose *problem* is one
    block wide.
    """
    return int(env.get("%__flat_width__") or 0) or _block_width(module)


def _derive_grid(module: Module, env: Mapping[str, int]) -> tuple[int, int, int]:
    """The launch grid: the padded problem size divided by the kernel's tile.

    Two shapes of kernel, decided by what the env declares, and they are not the
    same division. A 2-D matmul divides the tile it read off its own `tt.dot`, so
    grid and tile cannot drift apart. A 1-D elementwise kernel divides the
    *padded flat width* by its block: its `tt.make_range` width is the block, not
    the problem, and using the block as the grid basis computes exactly one block
    while the seam reports the whole buffer — a wrong answer, not an error.
    """
    rows, columns = env.get("M"), env.get("N")
    flat = int(env.get("%__flat_width__") or 0)
    if flat and not env.get("K"):
        block = int(env.get("%__block__") or 0) or _block_width(module) or 1
        return (max(1, math.ceil(flat / block)), 1, 1)
    if not rows or not columns:
        return (1, 1, 1)
    bm, bn, _ = _tile_from_module(module)
    return (max(1, rows // bm), max(1, columns // bn), 1)


def prepare(
    name: str,
    ttir: str,
    env: Mapping[str, int],
    grid: tuple[int, int, int] | None = None,
    provenance: str = "recorded",
) -> CompiledKernel:
    """Run one recorded lowering through the real pipeline. Raises `LoweringError`.

    Every stage is the shipped one: `parse_module` → `build_def_use` → `annotate`
    (the real recogniser) → `assemble` with the real schema and the *default*
    selector → the program the emulator executes. The only thing this function
    supplies that the pipeline would otherwise derive is the launch environment,
    which is a property of the launch and not of the kernel text.
    """
    parsed = parse_module(ttir)
    if not parsed.ok:
        raise LoweringError(f"{name}: PARSE_UNSUPPORTED — {parsed.diagnostic}")
    module = parsed.module
    graph = build_def_use(module)
    try:
        schema = load_builtin("tritonflow1")
    except Exception as exc:  # pragma: no cover - packaging failure
        raise LoweringError(f"{name}: the ISA-1 schema did not load: {exc}") from exc

    program = assemble(
        module,
        graph,
        annotate(module, graph),
        schema,
        env=dict(env),
    )
    if program.markers():
        marker = program.markers()[0]
        raise LoweringError(
            f"{name}: the pipeline refused this kernel — {marker.kind} at "
            f"{marker.loc_name or '?'} ({marker.op_name}): {marker.reason}"
        )

    function = module.functions()[0]
    pointers = tuple(value.name for value in function.args if value.type.kind == "ptr")
    scalars = {
        value.name: int(env[value.name])
        for value in function.args
        if value.type.kind != "ptr" and value.name in env
    }

    inputs: list[str] = []
    outputs: list[str] = []
    for op in walk_region(module.body):
        if op.name not in ("tt.load", "tt.store") or not op.operands:
            continue
        # `tt.load %ptr` reads through its first operand; `tt.store %ptr, %value`
        # writes through its first operand too (`op_shapes.pointer_operand`). Both
        # are operand 0, and the *operation name* is what says read or write.
        roots = _pointer_roots(graph, module, op.operands[0].name)
        target = inputs if op.name == "tt.load" else outputs
        for root in sorted(roots):
            if root not in target:
                target.append(root)

    # A pointer the function takes that nothing loads or stores is still addressed
    # by the program (the emulator requires every pointer input), so it is an input
    # by default rather than silently dropped.
    for pointer in pointers:
        if pointer not in inputs and pointer not in outputs:
            inputs.append(pointer)

    return CompiledKernel(
        name=name,
        entry=function.name,
        program=program,
        pointers=pointers,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        scalars=scalars,
        grid=grid if grid is not None else _derive_grid(module, env),
        extents=_problem_extents(
            inputs, outputs, {**env, "%__flat_width__": _flat_width(module, env)}
        ),
        input_precision=_declared_precision(module),
        provenance=provenance,
        tile=_tile_from_module(module),
        flat_width=int(env.get("%__flat_width__") or 0) or None,
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def lower_and_run(
    kernel: CompiledKernel,
    tensors: Sequence[torch.Tensor],
    *,
    policy: PrecisionPolicy | None = None,
) -> torch.Tensor:
    """Execute a prepared kernel on `tensors` and return the written buffer.

    Argument mapping is positional against the kernel's *pointer* arguments, which
    is how Triton calls a kernel: the pointer parameters come first and in order.
    Tensors are moved to fp32 because ISA-1's data model declares one dtype and a
    silent upcast is exactly what the contract refuses ("clear error naming the
    dtype; no silent upcast") — so a non-fp32 input is refused here.
    """
    if len(tensors) > len(kernel.inputs):
        raise LoweringError(
            f"{kernel.name} takes {len(kernel.inputs)} pointer argument(s) "
            f"({', '.join(kernel.inputs)}) but {len(tensors)} were supplied"
        )
    storage: dict[str, Any] = {}
    for name, tensor in zip(kernel.inputs, tensors, strict=False):
        if tensor.dtype != torch.float32:
            raise DtypeNotSupported(
                f"{kernel.name}: tensor for {name} has dtype {tensor.dtype}; the toy "
                "device declares f32 only and will not silently upcast"
            )
        expected = kernel.extents.get(name)
        if expected and tuple(tensor.shape) != tuple(expected):
            raise LoweringError(
                f"{kernel.name}: tensor for {name} has shape {tuple(tensor.shape)} but "
                f"this lowering addresses {tuple(expected)}; the mapping is positional "
                "and a mismatch is refused rather than reshaped"
            )
        storage[name] = tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    # Every buffer the kernel addresses must exist at its **problem** extent, not at
    # tile extent: storage is the whole tensor, and the index arithmetic reaches all
    # of it across the grid.
    for name in kernel.inputs:
        storage.setdefault(name, np.zeros(kernel.extents.get(name) or (1,), dtype=np.float32))
    for name in kernel.outputs:
        storage.setdefault(name, np.zeros(kernel.extents.get(name) or (1,), dtype=np.float32))
    for name, value in kernel.scalars.items():
        storage[name] = value

    grid_m, grid_n, _ = kernel.grid
    produced: str | None = None
    counter = policy or kernel.default_policy()
    # One program per tile, in launch order: this is what makes `pid_m`/`pid_n`
    # meaningful. A single `emulate` call executes one program, so a 128x128 problem
    # at a 64-wide tile is four programs — and each one accumulates into the same
    # buffer, exactly as the hardware would.
    for pid_m in range(max(1, grid_m)):
        for pid_n in range(max(1, grid_n)):
            per_program = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in storage.items()
            }
            written = emulate(kernel.program, per_program, counter, grid=(pid_m, pid_n, 0))
            for name in kernel.outputs:
                if name in written:
                    storage[name] = written[name]
                    produced = name

    if produced is None:
        raise LoweringError(
            f"{kernel.name}: the program wrote no output buffer; expected one of "
            f"{', '.join(kernel.outputs) or '(none declared)'}"
        )
    array = storage[produced]
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Graph reading
# --------------------------------------------------------------------------- #

#: The two ways execution says "this program cannot run on this machine". Both are
#: fallbacks with a record; `StorageError` (our bug) and a `LoweringError` that
#: means a binding was built wrong (also our bug) are deliberately not in this
#: tuple and propagate.
_EXECUTION_REFUSALS = (ProgramNotExecutable, UnsupportedInstruction)

#: Refusals that are about the *data this graph was handed*, not about the
#: machine or this code: today, a dtype the device does not declare. These are
#: fallbacks too — the alternative is that `torch.compile(..., backend="tritonflow")`
#: raises on an fp16 model, which fails the caller's program rather than running
#: it, and "a lowering failure never fails the run" is the whole contract.
_INPUT_REFUSALS = (DtypeNotSupported,)

#: Everything caught as a fallback with a record, at every call site.
_REFUSALS = _EXECUTION_REFUSALS + _INPUT_REFUSALS


def _refusal_record(exc: Exception, nodes: tuple[str, ...]) -> FallbackRecord:
    """A record that distinguishes the two refusals by what they actually mean.

    `ProgramNotExecutable` is the assembler having said no in advance, with a
    marker naming the operation. An `UnsupportedInstruction` is the emulator
    refusing an instruction the assembler *did* emit — found by bridging a
    `sigmoid`, whose `math.exp` the ISA's elementwise instruction accepts while
    the machine's interpreter has no case for it. Collapsing them into one reason
    would hide which side of the pipeline is missing the coverage.
    """
    marker = getattr(exc, "marker", None)
    if isinstance(exc, ProgramNotExecutable):
        reason = "the emitted program carries an UNSUPPORTED marker"
        stage = "execute"
    elif isinstance(exc, DtypeNotSupported):
        reason = "an input dtype the device does not declare (f32 only)"
        stage = "inputs"
    else:
        reason = "the emulator cannot execute an instruction this program uses"
        stage = "execute"
    return FallbackRecord(
        reason=reason,
        stage=stage,
        nodes=nodes,
        loc=getattr(marker, "loc_name", None),
        detail=str(exc),
    )


#: The FX targets that mean "matrix multiply". `linear` and `addmm` decompose
#: into matmul + bias add; the seam handles them by lowering the matmul part and
#: adding the bias back in PyTorch, which is honest: the bias addition is NOT
#: lowered to the ISA, and that fact is visible in the plan.
MATMUL_TARGETS = frozenset({"mm", "matmul", "linear", "addmm"})

#: Targets that only rearrange what a kernel already computed. A graph of
#: `mm -> getitem` is one kernel plus plumbing; leaving these unfiltered would
#: make every real Dynamo graph look like it does more than one thing.
PLUMBING_TARGETS = frozenset(
    {"getitem", "view", "reshape", "t", "transpose", "expand", "clone", "contiguous", "detach"}
)

#: Elementwise targets that can follow a matmul and are handled post-lowering.
ELEMENTWISE_TARGETS = frozenset({"relu", "relu_"})


#: Elementwise ops the *extractor* has a Triton kernel for. Deliberately not the
#: same set as `ELEMENTWISE_TARGETS`: that one means "applied in PyTorch after a
#: matmul lowering", and folding `add` into it would let a `mm -> add` graph
#: claim the add was lowered when the runner only replays relu. Two sets, two
#: meanings, and the split is what keeps the second one from becoming a lie.
EXTRACTABLE_ELEMENTWISE = frozenset({"relu", "relu_", "neg", "add", "sub", "mul", "div", "abs", "clamp"})


def _elementwise_kwargs_ok(kwargs: Mapping[str, Any]) -> bool:
    """Whether a node's keyword arguments leave the elementwise semantics alone.

    `relu(t, inplace=False)` is the default spelling and means exactly `relu(t)`, so
    refusing it on principle kept every `nn.ReLU` node in PyTorch — measured, not
    assumed: the exported graph carries `{'inplace': False}` and a plain
    `not kwargs` test sent the node to the eager floor. `inplace=True` is a
    different matter (it aliases its input, which a kernel returning a new buffer
    cannot honour), and `alpha` changes the arithmetic of `add`/`sub`, so both are
    refused.
    """
    for key, value in kwargs.items():
        if key == "inplace" and value is False:
            continue
        return False
    return True

#: The unary subset of the above, which binds one operand instead of two.
_UNARY_ELEMENTWISE = frozenset({"relu", "relu_", "neg", "abs"})


def extraction_candidate(
    targets: Sequence[str], example_inputs: Sequence[torch.Tensor]
) -> tuple[str, list[tuple[int, ...]], bool] | None:
    """`(op_name, operand shapes, has_bias)` for a graph one kernel can cover.

    A *candidate*, not a decision: the op name is read off the FX graph and the
    shapes off the example inputs. Whether the TTIR that comes back from it
    actually compiles, parses and lowers is settled by `prepare`, which is the
    only thing entitled to say so.
    """
    meaningful = [name for name in targets if name not in PLUMBING_TARGETS]
    if not meaningful:
        return None
    tensor_shapes = [
        tuple(t.shape) for t in example_inputs if isinstance(t, torch.Tensor)
    ]
    if len(meaningful) == 1 and len(meaningful) == len(targets) and meaningful[0] not in MATMUL_TARGETS:
        # One computational node with tensor/scalar operands.
        op = "relu" if meaningful[0] == "relu_" else meaningful[0]
        converted = [
            t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
            for t in example_inputs
            if isinstance(t, (torch.Tensor, int, float))
        ]
        max_ops = 3 if op == "clamp" else 2
        if not (1 <= len(converted) <= max_ops):
            return None
        max_t = max(converted, key=lambda x: x.dim())
        dominant_shape = tuple(max_t.shape)
        for t in converted:
            if tuple(t.shape) != dominant_shape and t.numel() != 1:
                return None
        return op, [dominant_shape for _ in converted], False
    matmuls = [name for name in meaningful if name in MATMUL_TARGETS]
    others = [
        name
        for name in meaningful
        if name not in MATMUL_TARGETS and name not in ELEMENTWISE_TARGETS
    ]
    if len(matmuls) != 1 or others:
        return None
    two_d = [shape for shape in tensor_shapes if len(shape) == 2]
    if len(two_d) < 2:
        return None
    has_bias = any(len(shape) == 1 for shape in tensor_shapes) or len(two_d) > 2
    # Canonical order, decided here because this is the only place that knows the
    # op: `extract_for_op` is handed `[(M, K), (K, N)]` and never has to guess
    # which of two 2-D operands is the activation. Dynamo hands a `linear` over as
    # `(weight, [bias,] x)` and an `addmm` as `(bias, mat1, mat2)`, so reading the
    # activation off `two_d[0]` for both would swap M and N on every `linear` —
    # producing a plausibly-shaped, wrong answer.
    # Call order — `F.linear(x, w)`, `torch.mm(a, b)`, `torch.addmm(bias, m1, m2)` —
    # which is also Dynamo's placeholder order. A *module*'s weight is a
    # `get_attr` and never appears here, so `nn.Linear` has no weight in
    # `example_inputs` and returns None: those graphs are handled by the
    # interpreter, which can see the fetched parameter. The previous version read
    # `two_d[-1]` as the activation and `two_d[0]` as the weight, which for a
    # one-element list is the same tensor and would have computed `x @ x.T`.
    name = matmuls[0]
    a_shape, b_shape = two_d[0], two_d[1]
    M, K = a_shape
    if name == "linear":
        if b_shape[1] == K:
            N = b_shape[0]  # torch layout (out, in): the kernel transposes it
        elif b_shape[0] == K:
            N = b_shape[1]  # already (in, out)
        else:
            return None
    else:
        if b_shape[0] != K:
            return None
        N = b_shape[1]
    return name, [(M, K), (K, N)], has_bias


def _lower_extracted(
    extracted: Extracted, provenance: str, nodes: tuple[str, ...]
) -> tuple[CompiledKernel | None, FallbackRecord | None]:
    """Run extracted TTIR through the real pipeline and say what happened.

    One place, so Path 1 and Path 2 cannot disagree about what "lowered" means:
    parse, def-use, recogniser, selector, assembler — and a refusal, never a
    partial acceptance, when the assembler emits an `UNSUPPORTED` marker.
    """
    try:
        kernel = prepare(
            extracted.name, extracted.ttir, extracted.env, provenance=provenance
        )
    except LoweringError as exc:
        return None, FallbackRecord(
            reason=f"the {provenance} TTIR did not lower",
            stage="lower",
            nodes=nodes,
            detail=str(exc),
        )
    return (
        replace(
            kernel,
            tile=extracted.tile,
            pads=extracted.pads,
            flat_width=(extracted.padded[0] if extracted.is_elementwise else None),
            bias_lowered=bool(extracted.has_bias and extracted.kind == "linear"),
            extents=_extents_with_bias(kernel, extracted),
        ),
        None,
    )


def _extents_with_bias(
    kernel: CompiledKernel, extracted: Extracted
) -> dict[str, tuple[int, ...]]:
    """The kernel's extents, plus the bias pointer's when the program reads one.

    `_problem_extents` derives `A`, `B` and `C` from the matmul convention and a
    `linear` kernel has a fourth pointer it knows nothing about. Without this the
    bias buffer is allocated at length 1 while the kernel indexes it up to the
    tile width, and the emulator refuses with a `StorageError` — which is the
    correct behaviour for an undersized buffer and the wrong buffer size.
    """
    extents = dict(kernel.extents)
    if extracted.kind == "linear" and extracted.has_bias and len(kernel.inputs) >= 3:
        extents[kernel.inputs[2]] = (extracted.padded[2],)
    return extents


def _extract_via_inductor(
    op_name: str,
    shapes: Sequence[Sequence[int]],
    has_bias: bool,
    nodes: tuple[str, ...],
    *,
    gm: Any | None = None,
    example_inputs: Sequence[Any] | None = None,
) -> tuple[list[CompiledKernel], FallbackRecord | None]:
    """Path 1a: run Inductor under CompileSpy and lower captured TTIR."""
    if not is_inductor_op(op_name):
        return [], FallbackRecord(
            reason=f"{op_name!r} is outside the Inductor capture op set (matmul/linear/relu)",
            stage="inductor",
            nodes=nodes,
        )
    try:
        extracted, reason = extract_via_inductor(
            gm, example_inputs, op_name, shapes, has_bias=has_bias
        )
    except Exception as exc:
        return [], FallbackRecord(
            reason="Inductor TTIR capture raised",
            stage="inductor",
            nodes=nodes,
            detail=f"{type(exc).__name__}: {exc}",
        )
    if extracted is None:
        return [], FallbackRecord(
            reason="Inductor did not supply lowerable TTIR",
            stage="inductor",
            nodes=nodes,
            detail=reason,
        )
    kernel, record = _lower_extracted(extracted, "inductor", nodes)
    if kernel is None:
        return [], record
    return [kernel], None


def _extract_dynamic(
    op_name: str, shapes: Sequence[Sequence[int]], has_bias: bool, nodes: tuple[str, ...]
) -> tuple[list[CompiledKernel], FallbackRecord | None]:
    """Path 1: compile the op at runtime with Triton and lower what comes back."""
    cap = extraction_capability()
    if not cap.available:
        return [], FallbackRecord(
            reason="dynamic TTIR extraction is unavailable",
            stage="extract",
            nodes=nodes,
            detail=cap.reason,
        )
    try:
        extracted = extract_for_op(op_name, shapes, has_bias=has_bias)
    except ExtractionUnavailable as exc:
        return [], FallbackRecord(
            reason="dynamic TTIR extraction is unavailable",
            stage="extract",
            nodes=nodes,
            detail=str(exc),
        )
    except ExtractionError as exc:
        return [], FallbackRecord(
            reason=f"triton refused to compile the {op_name} kernel",
            stage="extract",
            nodes=nodes,
            detail=str(exc),
        )
    if extracted is None:
        return [], FallbackRecord(
            reason=f"no extraction kernel exists for {op_name} at these shapes",
            stage="extract",
            nodes=nodes,
            detail=NOT_EXTRACTED,
        )
    kernel, record = _lower_extracted(extracted, "dynamic", nodes)
    if kernel is None:
        return [], record
    return [kernel], None


def _extract_via_flaggems(
    op_name: str, shapes: Sequence[Sequence[int]], has_bias: bool, nodes: tuple[str, ...]
) -> tuple[list[CompiledKernel], FallbackRecord | None]:
    """Path 2: compile FlagGems' own kernel for this op, when FlagGems is here."""
    cap = flaggems_bridge.capability()
    if not cap.available:
        return [], FallbackRecord(
            reason="the FlagGems bridge is unavailable",
            stage="flaggems",
            nodes=nodes,
            detail=cap.reason,
        )
    try:
        extracted = flaggems_bridge.extract_for_op(op_name, shapes, has_bias=has_bias)
    except flaggems_bridge.FlagGemsUnsupported as exc:
        return [], FallbackRecord(
            reason=f"flag_gems implements {op_name} but it could not be signed",
            stage="flaggems",
            nodes=nodes,
            detail=str(exc),
        )
    except ExtractionUnavailable as exc:
        return [], FallbackRecord(
            reason="TTIR extraction is unavailable", stage="flaggems", nodes=nodes, detail=str(exc)
        )
    except ExtractionError as exc:
        return [], FallbackRecord(
            reason=f"triton refused to compile the flag_gems {op_name} kernel",
            stage="flaggems",
            nodes=nodes,
            detail=str(exc),
        )
    if extracted is None:
        return [], FallbackRecord(
            reason=f"flag_gems does not implement {op_name}",
            stage="flaggems",
            nodes=nodes,
            detail=(
                "the bridge looks in flag_gems.ops and then the package root; see "
                "extract.flaggems_bridge.OP_KERNELS"
            ),
        )
    kernel, record = _lower_extracted(extracted, "flaggems", nodes)
    if kernel is None:
        return [], record
    return [kernel], None


def extraction_provenance(kernel: CompiledKernel) -> str:
    """Which route lowered a kernel, for a report that has to say *how*."""
    return kernel.provenance


def extraction_refusal(
    targets: Sequence[str], example_inputs: Sequence[torch.Tensor]
) -> str:
    """Why `extraction_candidate` declined, in terms a reader can act on.

    "Not a single extractable op" is true of a softmax and of a 40-node model
    alike, so the message names the case: an op with no kernel, a kernel whose
    operand count does not match, or a graph that genuinely is more than one op.
    """
    meaningful = [name for name in targets if name not in PLUMBING_TARGETS]
    tensors = sum(1 for t in example_inputs if isinstance(t, torch.Tensor))
    if len(meaningful) == 1:
        op = meaningful[0]
        if not is_supported(op) and op not in flaggems_bridge.OP_KERNELS:
            return (
                f"neither the extractor nor the FlagGems bridge has a kernel for {op!r}; "
                "extract.SUPPORTED_OPS and extract.flaggems_bridge.OP_KERNELS list what they know"
            )
        return (
            f"the {op!r} node supplied {tensors} tensor operand(s), which is not a "
            "count any kernel here binds (one or two are expected)"
        )
    unsupported = [
        name
        for name in meaningful
        if name not in MATMUL_TARGETS and name not in ELEMENTWISE_TARGETS
    ]
    if unsupported:
        # Name them. "This graph is not a single kernel" is true of a model with an
        # unsupported softmax in it and of a model with forty unknown ops, and only
        # one of those tells a reader what to do next.
        return (
            "the graph mixes a matmul with op(s) that no kernel covers "
            f"({', '.join(sorted(set(unsupported))[:4])}); nothing here lowers part of a graph"
        )
    if len([name for name in meaningful if name in MATMUL_TARGETS]) == 1:
        return (
            "the matmul operands are not a bindable pair: the extractor needs the "
            "activation and the weight as separate tensors, and a module's weight "
            "arrives as a get_attr, not an input (the per-node interpreter handles those)"
        )
    return (
        f"the graph is not a single extractable op: it has {len(targets)} "
        f"operation(s) ({', '.join(targets[:6])})"
    )


@dataclass(frozen=True)
class GraphMatch:
    """One candidate lowering for a graph, before execution."""

    kernel_name: str
    tensors: tuple[str, ...]
    outputs: tuple[str, ...]


def graph_tensors(graph: Any) -> tuple[str, ...]:
    """The graph's `placeholder` node names, in signature order."""
    placeholders = [
        node.name for node in graph.graph.nodes if node.op == "placeholder"
    ]
    return tuple(placeholders)


def graph_targets(graph: Any) -> tuple[str, ...]:
    """The `target` names of the graph's *computational* nodes.

    `placeholder`, `output` and `get_attr` are skipped. `get_attr` is the one that
    matters, and skipping it is a measurement rather than a preference: an
    `nn.Linear`'s weight and bias arrive as attribute fetches, so counting them as
    operations made every module-based graph look like it did three things when it
    did one. That single extra node kept `nn.Linear` — and therefore every MLP — on
    the eager path, and nothing anywhere said why.
    """
    names: list[str] = []
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        target = node.target
        names.append(getattr(target, "__name__", str(target)))
    return tuple(names)


def extract_ttir(graph: Any) -> dict[str, str]:
    """TTIR texts attached to a graph, keyed by entry name.

    The one supported hand-off: a caller (or a future Inductor pass) attaches
    `graph.meta["tritonflow"] = {"ttir": {name: text}, "env": {...}, "grid": {...}}`
    and the seam consumes it through the identical pipeline. An empty dict means
    "nothing declared", which is the case the recorded lowerings cover.
    """
    meta = getattr(graph, "meta", None) or {}
    declared = meta.get(DEVICE_NAME) or {}
    ttir = declared.get("ttir") or {}
    if not isinstance(ttir, Mapping):
        return {}
    return {str(key): str(value) for key, value in ttir.items()}


def plan_graph(graph: Any, example_inputs: Sequence[torch.Tensor]) -> LoweringPlan:
    """Decide what to do with one graph. The seam's whole judgement lives here.

    The decision is all-or-nothing and the reason strings are the point: a reader
    of a coverage report gets "this graph has 5 nodes, one kernel applies to 2 of
    them" rather than "partially supported".
    """
    nodes = tuple(node.name for node in graph.graph.nodes)
    targets = graph_targets(graph)
    tensors = graph_tensors(graph)
    plan = LoweringPlan(nodes=nodes)

    if not tensors:
        plan.fallbacks.append(
            FallbackRecord(
                reason="the graph takes no tensor arguments, so no kernel can apply",
                stage="match",
                nodes=nodes,
            )
        )
        return plan

    declared = extract_ttir(graph)

    # 1. TTIR the caller attached wins: it is the general path, and it is the only
    #    one in which the seam lowers a kernel it was not told about in advance.
    for name, text in declared.items():
        try:
            plan.lowered.append(prepare(name, text, _env_for(graph)))
        except LoweringError as exc:
            plan.fallbacks.append(
                FallbackRecord(
                    reason="the declared TTIR did not lower",
                    stage="lower",
                    nodes=nodes,
                    detail=str(exc),
                )
            )
    if plan.lowered:
        plan.fallbacks.clear()
        return plan

    # Check for unsupported kwargs on computational nodes
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        if "alpha" in node.kwargs and node.kwargs["alpha"] != 1:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=f"{node.name}: unsupported kwarg 'alpha'={node.kwargs['alpha']!r}",
                    stage="extract",
                    nodes=nodes,
                    detail=NOT_EXTRACTED,
                )
            )
            return plan
        if "rounding_mode" in node.kwargs and node.kwargs["rounding_mode"] is not None:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=f"{node.name}: unsupported kwarg 'rounding_mode'={node.kwargs['rounding_mode']!r}",
                    stage="extract",
                    nodes=nodes,
                    detail=NOT_EXTRACTED,
                )
            )
            return plan
        if "out" in node.kwargs and node.kwargs["out"] is not None:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=f"{node.name}: unsupported kwarg 'out'",
                    stage="extract",
                    nodes=nodes,
                    detail=NOT_EXTRACTED,
                )
            )
            return plan
        if "dtype" in node.kwargs:
            dt = node.kwargs["dtype"]
            if dt is not None and str(dt) not in ("torch.float32", "float32", "<class 'float'>"):
                plan.fallbacks.append(
                    FallbackRecord(
                        reason=f"{node.name}: unsupported non-default dtype {dt!r}",
                        stage="extract",
                        nodes=nodes,
                        detail=NOT_EXTRACTED,
                    )
                )
                return plan
        t_raw = getattr(node.target, "__name__", str(node.target))
        if t_raw.startswith("aten."):
            t_raw = t_raw[len("aten."):]
        t_name = t_raw.split(".")[0]
        if t_name in ("add", "add_", "sub", "sub_") and len(node.args) > 2 and node.args[2] != 1:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=f"{node.name}: unsupported positional alpha={node.args[2]!r}",
                    stage="extract",
                    nodes=nodes,
                    detail=NOT_EXTRACTED,
                )
            )
            return plan
        if t_name in ("div", "div_") and len(node.args) > 2 and node.args[2] is not None:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=f"{node.name}: unsupported positional rounding_mode={node.args[2]!r}",
                    stage="extract",
                    nodes=nodes,
                    detail=NOT_EXTRACTED,
                )
            )
            return plan

    # 2. Inductor capture, then template extraction, then FlagGems. Inductor is
    #    first among generated sources because that is the pitch hand-off
    #    (Inductor → triton.compile → TTIR). When Inductor emits no Triton (common
    #    on CPU-only hosts), the note explains why and the next sources run.
    attempts: list[FallbackRecord] = []
    candidate = extraction_candidate(targets, example_inputs)
    if candidate is None:
        attempts.append(
            FallbackRecord(
                reason=extraction_refusal(targets, example_inputs),
                stage="extract",
                nodes=nodes,
                detail=NOT_EXTRACTED,
            )
        )
    else:
        op_name, shapes, has_bias = candidate
        stages = (
            (
                lambda o, s, b, n: _extract_via_inductor(
                    o, s, b, n, gm=graph, example_inputs=example_inputs
                )
            ),
            _extract_dynamic,
            _extract_via_flaggems,
        )
        for stage in stages:
            kernels, record = stage(op_name, shapes, has_bias, nodes)
            if kernels:
                # The reasons the earlier stages declined are *notes*, not
                # fallbacks: the graph lowered, and a reader still needs to know
                # that the lowering came from the second source because the first
                # was unavailable rather than because it had no kernel.
                plan.lowered.extend(kernels)
                plan.notes.extend(attempts)
                return plan
            if record is not None:
                attempts.append(record)

    # 3. A recorded lowering, matched on the graph's own tensor shapes. Kept
    #    rather than deleted now that extraction exists, because it is the only
    #    path that keeps working when Triton is absent — which is what "graceful
    #    degradation" means here, and it would not mean anything if the fallback
    #    it degrades to had been removed.
    if _is_single_kernel_graph(targets):
        matches = _match_recorded(graph, example_inputs)
        if matches:
            plan.lowered.append(matches[0])
            plan.notes.extend(attempts)
            return plan
        attempts.append(
            FallbackRecord(
                reason=(
                    f"no recorded lowering applies to shapes "
                    f"{', '.join(str(tuple(t.shape)) for t in example_inputs[:2])}"
                ),
                stage="match",
                nodes=nodes,
                detail=NOT_EXTRACTED,
            )
        )

    # 4. Anything else is eager. The reason is the shape of the graph, not a
    #    hedge: "partially supported" is not a thing this seam can report.
    plan.fallbacks.extend(attempts)
    return plan


def _is_single_kernel_graph(targets: Sequence[str]) -> bool:
    """Whether the graph is exactly one matmul op plus plumbing and optional
    elementwise ops (relu). `linear` counts as a matmul with bias handled
    outside the kernel.
    """
    meaningful = [name for name in targets if name not in PLUMBING_TARGETS]
    # Exactly one matmul, optionally followed by elementwise ops
    matmuls = [t for t in meaningful if t in MATMUL_TARGETS]
    others = [t for t in meaningful if t not in MATMUL_TARGETS and t not in ELEMENTWISE_TARGETS]
    return len(matmuls) == 1 and len(others) == 0


def _is_matmul_only_graph(targets: Sequence[str]) -> bool:
    """Whether the graph consists entirely of matmul/linear + elementwise ops.

    Multi-layer networks (linear -> relu -> linear) are matmul-only: every
    computational op is either a matmul variant or an elementwise op that we
    handle in PyTorch. This enables the interpreter path that lowers each
    matmul individually.
    """
    meaningful = [name for name in targets if name not in PLUMBING_TARGETS]
    if not meaningful:
        return False
    matmuls = [t for t in meaningful if t in MATMUL_TARGETS]
    [t for t in meaningful if t in ELEMENTWISE_TARGETS]
    others = [t for t in meaningful if t not in MATMUL_TARGETS and t not in ELEMENTWISE_TARGETS]
    # Need at least one matmul and no unsupported ops
    return len(matmuls) >= 1 and len(others) == 0


def _matmul_reference(target: Any, args: Sequence[Any], kwargs: dict[str, Any], k: int, precision: str):
    """fp64 value of a matmul-family node and the absolute bound on an fp32 machine's error.

    Operands are rounded to tf32 first when `precision` is tf32 (that rounding is
    what the machine does, so it belongs to the reference). The remaining error is
    fp32 accumulation, `k*u/(1-k*u)` of the sum of absolute products, and one
    rounding of the (bias-added) result.
    """
    name = getattr(target, "__name__", str(target))
    if kwargs and any(v not in (None, 1) for key, v in kwargs.items() if key in ("beta", "alpha")):
        raise ValueError("beta/alpha scaling is not modelled")

    def operand(x: torch.Tensor) -> np.ndarray:
        array = x.detach().cpu().numpy().astype(np.float32)
        return (tf32_truncate(array) if precision == "tf32" else array).astype(np.float64)

    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    bias = None
    if name == "linear":
        x, w = tensors[0], tensors[1]
        bias = tensors[2].detach().cpu().numpy().astype(np.float64) if len(tensors) > 2 else None
        lhs, rhs = operand(x), operand(w).T
    elif name == "addmm":
        bias = tensors[0].detach().cpu().numpy().astype(np.float64)
        lhs, rhs = operand(tensors[1]), operand(tensors[2])
    elif name in ("mm", "matmul"):
        lhs, rhs = operand(tensors[0]), operand(tensors[1])
    else:
        raise ValueError(f"unknown matmul-family target {name!r}")
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("operands are not a conforming 2-D pair")
    u = 2.0**-24
    gamma = k * u / (1.0 - k * u)
    product = lhs @ rhs
    reach = np.abs(lhs) @ np.abs(rhs)
    value = product if bias is None else product + bias
    bound = gamma * reach + u * (np.abs(value) + gamma * reach)
    if bias is not None:
        bound = bound + u * np.abs(bias)
    return value, bound


def _multi_kernel_interpret(
    graph: Any,
    example_inputs: Sequence[torch.Tensor],
    plan: LoweringPlan | None = None,
) -> Callable[..., Any]:
    """Interpret a multi-kernel FX graph, lowering each matmul/linear it can.

    Every matmul-family node goes through the same sources the single-kernel path
    uses — extraction, then the FlagGems bridge, then a recorded lowering — and
    every node that could not be lowered lands on the plan as a `FallbackRecord`
    naming the node and the reason. Reporting is the point: the counters are
    exposed as `tritonflow_counters` and the reasons as plan fallbacks, so a graph
    whose matmuls all failed cannot look like one that lowered all of them.
    """
    from torch.fx.interpreter import Interpreter

    # This list starts empty on purpose. Seeding it with `plan.fallbacks` made the
    # assignment at the end a no-op, so the *stage* reasons from `plan_graph`
    # survived a run that contradicted them — the plan then said "not a single
    # extractable op" about a graph whose every node had just been lowered.
    records: list[FallbackRecord] = []
    eager_nodes: list[str] = []
    counters = {"lowered": 0, "eager": 0}

    def _kernel_for(x_input, weight, target_name, has_bias):
        """A lowering for one node, trying the same sources the seam tries.

        Returns `None` when no source could lower it, having appended the reason
        to `records` — the whole point of this function is that the *reason* is
        kept. The previous version wrapped the attempt in `except Exception: pass`
        and incremented a counter nobody read, so a graph whose every matmul
        failed to lower produced a plan indistinguishable from one that lowered
        all of them.
        """
        node = (target_name,)
        M, K = int(x_input.shape[0]), int(x_input.shape[1])
        N = int(weight.shape[0]) if target_name == "linear" else int(weight.shape[1])
        op = "linear" if target_name in ("linear", "addmm") else "mm"
        shapes = [(M, K), (K, N)]
        attempts: list[FallbackRecord] = []
        kernels, record = _extract_via_inductor(op, shapes, has_bias, node)
        if kernels:
            return kernels[0]
        if record is not None:
            attempts.append(record)
        for stage in (_extract_dynamic, _extract_via_flaggems):
            kernels, record = stage(op, shapes, has_bias, node)
            if kernels:
                # Earlier-stage misses are notes on the plan, not node fallbacks:
                # the node lowered, and a reader still needs to know inductor
                # was tried first.
                if plan is not None and attempts:
                    plan.notes.extend(attempts)
                return kernels[0]
            if record is not None:
                attempts.append(record)
        # The recorded lowering last, so this path still works with Triton absent.
        matches = _match_recorded_shapes(M, N, K, node)
        if matches:
            if plan is not None and attempts:
                plan.notes.extend(attempts)
            return matches[0]
        records.extend(attempts)
        records.append(
            FallbackRecord(
                reason=f"no lowering applies to the {target_name} node at {(M, K)} @ {(K, N)}",
                stage="match",
                nodes=node,
                detail=NOT_EXTRACTED,
            )
        )
        return None

    def _demote(target_name: str, mismatch: str) -> None:
        """A node whose lowered result disagreed with eager ran in PyTorch after all."""
        counters["eager"] += 1
        eager_nodes.append(target_name)
        records.append(FallbackRecord(reason=mismatch, stage="shadow", nodes=(target_name,)))

    class TritonFlowInterpreter(Interpreter):
        def _shadow(self, result, target, args, kwargs, matmul):
            """`(value to return, mismatch reason or None)` for one lowered node.

            The eager reference runs on clones (an in-place target must not change the
            caller's tensors twice). A single elementwise op is exact or one correctly
            rounded fp32 operation on each side, so lowered and eager must agree within
            `2 * 2**-24` relative. A matmul is compared with an fp64 product of the
            operands *as the machine sees them* (rounded to tf32 when the kernel
            declares tf32), within the fp32 accumulation bound `gamma_k * sum|a*b|`
            plus one rounding of the result: the format's own error is part of the
            reference, so it does not widen the tolerance. On a mismatch the eager
            value is what the graph returns.
            """
            eager_args = tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args)
            eager_kwargs = {
                key: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for key, v in kwargs.items()
            }
            eager = Interpreter.call_function(self, target, eager_args, eager_kwargs)
            got = result.detach().cpu().double().numpy()
            want = eager.detach().cpu().double().numpy()
            if got.shape != want.shape:
                return eager, f"lowered result shape {got.shape} != eager shape {want.shape}"
            if not (np.isfinite(got).all() and np.isfinite(want).all()):
                return eager, "non-finite values: the lowered result cannot be certified against eager"
            if matmul is None:
                diff = np.abs(got - want)
                if bool(np.all(diff <= 2.0 * 2.0**-24 * np.abs(want))):
                    return result, None
                return eager, f"lowered result disagrees with eager: max error {float(diff.max()):.3g} exceeds 2*2^-24 relative"
            try:
                reference, bound = _matmul_reference(target, args, kwargs, *matmul)
            except ValueError as exc:
                return eager, f"no derived reference for this matmul: {exc}"
            diff = np.abs(got - reference)
            if bool(np.all(diff <= bound)):
                return result, None
            return eager, (
                f"lowered result disagrees with the {matmul[1]} reference: max error "
                f"{float(diff.max()):.3g} exceeds the derived bound {float(np.max(bound)):.3g}"
            )

        def call_function(self, target, args, kwargs):
            target_name = getattr(target, "__name__", str(target))
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]

            # Elementwise nodes lower too, and they have to be tried *before* the
            # matmul branch: a `relu` between two linears is a kernel like any
            # other, and leaving it to PyTorch is what made a three-layer MLP
            # report two thirds lowered. Broadcasting is where this stops: the
            # elementwise kernels walk a flat buffer, so operands of different
            # shapes are refused and run in PyTorch with a record.
            if target_name in EXTRACTABLE_ELEMENTWISE and _elementwise_kwargs_ok(kwargs):
                op = "relu" if target_name == "relu_" else target_name
                unary = target_name in _UNARY_ELEMENTWISE
                converted_args = [
                    a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
                    for a in args
                    if isinstance(a, (torch.Tensor, int, float))
                ]
                expected_arity = 1 if unary else (3 if op == "clamp" else 2)
                arity_ok = len(converted_args) == expected_arity
                if arity_ok and converted_args:
                    max_t = max(converted_args, key=lambda x: x.dim())
                    dominant_shape = tuple(max_t.shape)
                    can_shape = all(
                        tuple(t.shape) == dominant_shape or t.numel() == 1
                        for t in converted_args
                    )
                    if can_shape:
                        shapes = [dominant_shape for _ in converted_args]
                        if op == "relu":
                            kernels, record = _extract_via_inductor(
                                op, shapes, False, (target_name,)
                            )
                            if kernels:
                                try:
                                    result = _run_with_padding(
                                        kernels[0], converted_args, op, False
                                    )
                                    value, mismatch = self._shadow(
                                        result, target, args, kwargs, None
                                    )
                                    if mismatch is None:
                                        counters["lowered"] += 1
                                        return value
                                    _demote(target_name, mismatch)
                                    return value
                                except _REFUSALS as exc:
                                    records.append(_refusal_record(exc, (target_name,)))
                            elif record is not None and plan is not None:
                                plan.notes.append(record)
                        kernels, record = _extract_dynamic(op, shapes, False, (target_name,))
                        if kernels:
                            try:
                                result = _run_with_padding(kernels[0], converted_args, op, False)
                                value, mismatch = self._shadow(result, target, args, kwargs, None)
                                if mismatch is None:
                                    counters["lowered"] += 1
                                    return value
                                _demote(target_name, mismatch)
                                return value
                            except _REFUSALS as exc:
                                records.append(_refusal_record(exc, (target_name,)))
                        elif record is not None:
                            records.append(record)

            if target_name in MATMUL_TARGETS:
                two_d = [t for t in tensor_args if t.dim() == 2]
                bias = next((t for t in tensor_args if t.dim() == 1), None)
                if len(two_d) >= 2:
                    # **Call order**, which is what `Interpreter.call_function`
                    # hands over: `linear(input, weight, bias)`, `mm(self, mat2)`,
                    # `addmm(input, mat1, mat2)`. The activation is first in every
                    # one of them. This used to read `two_d[-1]` for `linear`,
                    # copying the placeholder-order assumption from the
                    # `example_inputs` path — where the weight is a `get_attr` and
                    # never appears at all — so M and N came out swapped, as
                    # `dyn_linear_128x8x64_bias` for an `8x128x64` layer.
                    x_input, weight = two_d[0], two_d[1]
                    kernel = _kernel_for(x_input, weight, target_name, bias is not None)
                    if kernel is not None:
                        try:
                            result = _run_with_padding(
                                kernel, list(tensor_args), target_name, has_relu=False
                            )
                            value, mismatch = self._shadow(result, target, args, kwargs, (int(x_input.shape[1]), kernel.input_precision))
                            if mismatch is None:
                                counters["lowered"] += 1
                                return value
                            _demote(target_name, mismatch)
                            return value
                        except _REFUSALS as exc:
                            # A node the machine refused *after* selection accepted it
                            # is a fact about that node, and it is recorded as one —
                            # as is a node whose *input dtype* the device does not
                            # declare, which is a caller's data, not a defect here.
                            # A plain `LoweringError` is not caught: it means the
                            # binding was wrong, which is ours, and absorbing it
                            # would be the silent fallback this seam exists to avoid.
                            records.append(_refusal_record(exc, (target_name,)))
            counters["eager"] += 1
            eager_nodes.append(target_name)
            records.append(
                FallbackRecord(
                    reason=f"operation {target_name} is unsupported for device lowering",
                    stage="node",
                    nodes=(target_name,),
                )
            )
            return super().call_function(target, args, kwargs)

        def call_method(self, target, args, kwargs):
            if target in PLUMBING_TARGETS:
                return super().call_method(target, args, kwargs)
            if hasattr(torch, target):
                return self.call_function(getattr(torch, target), args, kwargs)
            counters["eager"] += 1
            eager_nodes.append(target)
            records.append(
                FallbackRecord(
                    reason=f"method {target} is unsupported for device lowering",
                    stage="node",
                    nodes=(target,),
                )
            )
            return super().call_method(target, args, kwargs)

    interp = TritonFlowInterpreter(graph)

    def run(*args):
        result = interp.run(*args)
        if plan is not None:
            # Assigned *after* the run, because these numbers only exist once
            # something has been interpreted. The stage reasons collected on the
            # way in are *replaced* by the node-level truth: what this graph did is
            # now known, and keeping both would double-count.
            plan.multi_kernel = True
            plan.node_lowerings = counters["lowered"]
            plan.node_fallbacks = counters["eager"]
            plan.fallbacks = list(records)
            if eager_nodes:
                # One record for the nodes that ran in PyTorch, naming them: a
                # counter with no names is not something a reader can act on, and
                # an empty `fallbacks` list would make `fully_lowered` true for a
                # graph that had an eager node in it.
                plan.fallbacks.append(
                    FallbackRecord(
                        reason=(
                            f"{len(eager_nodes)} node(s) ran in PyTorch: "
                            f"{', '.join(sorted(set(eager_nodes))[:8])}"
                        ),
                        stage="node",
                        nodes=tuple(sorted(set(eager_nodes))),
                    )
                )
        return result

    run.tritonflow_counters = counters  # type: ignore[attr-defined]
    if plan is not None:
        run.compilation_summary = plan.compilation_summary  # type: ignore[attr-defined]
    return run


#: The tile the recorded matmul was compiled at. Only used to build the launch
#: env; the runner pads by the extents `prepare` derives, and `prepare` reads the
#: tile off the module itself.
_RECORDED_TILE = (64, 64, 32)


def _recorded_env(M: int, N: int, K: int) -> dict[str, int]:
    """The launch env for the recorded matmul at `(M, N, K)`, padded to its tile."""
    bm, bn, bk = _RECORDED_TILE
    rows = max(bm, math.ceil(M / bm) * bm)
    inner = max(bk, math.ceil(K / bk) * bk)
    cols = max(bn, math.ceil(N / bn) * bn)
    return {
        "M": rows, "N": cols, "K": inner,
        "%M": rows, "%N": cols, "%K": inner,
        "%sam": inner, "%sak": 1, "%sbk": cols, "%sbn": 1, "%scm": cols, "%scn": 1,
    }


def _match_recorded_shapes(
    M: int, N: int, K: int, nodes: tuple[str, ...]
) -> list[CompiledKernel]:
    """Every recorded lowering that prepares for this explicit shape.

    Padded to the *recorded* tile, because the artifact's index arithmetic was
    compiled for a 64-wide block and addresses the whole padded buffer.
    """
    env = _recorded_env(M, N, K)
    matches: list[CompiledKernel] = []
    for _name, ttir_text in recorded_kernels().items():
        try:
            matches.append(prepare(f"matmul_{M}x{N}x{K}", ttir_text, env))
        except LoweringError:
            continue
    return matches


def _match_recorded(graph: Any, example_inputs: Sequence[torch.Tensor]) -> list[CompiledKernel]:
    """Match a single-matmul graph against the recorded lowerings. Path 3.

    The operand *roles* are resolved by `extraction_candidate`, which is the one
    place that knows what each op's operand order means; this function only pads
    and prepares. The old version untangled `linear` and `addmm` on its own and
    got `addmm` backwards — `(bias, mat1, mat2)` read as though `mat1` were the
    weight — which produced a transposed M and N for every `addmm` graph.
    """
    targets = graph_targets(graph)
    if not _is_single_kernel_graph(targets):
        return []
    candidate = extraction_candidate(targets, example_inputs)
    if candidate is None:
        return []
    shapes = candidate[1]
    if len(shapes) < 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2:
        return []
    M, K = shapes[0]
    K2, N = shapes[1]
    if K != K2:
        return []
    return _match_recorded_shapes(int(M), int(N), int(K), tuple(targets))


def _env_for(graph: Any) -> dict[str, int]:
    meta = getattr(graph, "meta", None) or {}
    declared = meta.get(DEVICE_NAME) or {}
    env = declared.get("env") or {}
    return {str(key): int(value) for key, value in env.items()}


# --------------------------------------------------------------------------- #
# The registered backend
# --------------------------------------------------------------------------- #


def _backend_return(value: Any) -> Any:
    """Wrap a result the way Dynamo's backend contract expects: as a sequence.

    A backend's returned callable must yield the graph's outputs *as a sequence*,
    because Dynamo addresses them positionally. Returning a bare `Tensor` for a
    single-output graph is not a harmless simplification: measured on torch
    2.12.1+cpu, Dynamo then takes element `[0]` of it, so a `(128, 128)` matmul
    result arrives at the caller as its first row — a 128-element tensor whose
    values are all individually *correct* and whose shape is silently wrong. The
    numeric check that compares against `torch.matmul` catches it immediately;
    nothing else would.
    """
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _eager_fallback(graph: Any) -> Callable[..., Any]:
    """An eager callable for the graph, used when the seam refuses to lower it."""
    compiled = graph

    def run(*args: Any) -> Any:
        return _backend_return(compiled(*args))

    return run


@dataclass(frozen=True)
class _Bound:
    """The caller's tensors placed against the kernel's pointer parameters."""

    operands: list[torch.Tensor]
    bias: torch.Tensor | None
    out_shape: tuple[int, ...]


def _bind_operands(
    kernel: CompiledKernel, tensors: Sequence[torch.Tensor], target_name: str
) -> _Bound:
    """Place the caller's tensors against `kernel.inputs`, and check the placement.

    Position, not shape, is what a Triton launch binds by, and Dynamo's placeholder
    order is not the call order — a `linear` graph arrives as `(weight, [bias,] x)`
    while the kernel's parameters are `(x, w, bias)`. So the placement follows the
    documented order per op and then *validates itself* against the extents the
    program actually addresses: the activation's inner extent must equal the
    kernel's `K`, and the weight must be transposable into `(K, N)`. A placement
    that does not validate raises, naming both shapes, because a silently
    transposed operand is a wrong answer with no error anywhere.
    """
    tensor_list = [t for t in tensors if isinstance(t, torch.Tensor) and t.dim() >= 1]
    bias_tensor = next((t for t in tensor_list if t.dim() == 1), None)
    if kernel.flat_width is not None:
        converted: list[torch.Tensor] = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                converted.append(t)
            elif isinstance(t, (int, float)):
                converted.append(torch.tensor(t, dtype=torch.float32))
        if not converted:
            raise LoweringError(f"{kernel.name}: no tensor operand was supplied")
        max_t = max(converted, key=lambda x: x.dim())
        out_shape = tuple(max_t.shape)
        bound_ops: list[torch.Tensor] = []
        for t in converted[: len(kernel.inputs)]:
            if tuple(t.shape) != out_shape:
                t = t.expand(out_shape).contiguous() if out_shape else t.reshape(())
            bound_ops.append(t)
        return _Bound(
            operands=bound_ops,
            bias=None,
            out_shape=out_shape,
        )

    two_d = [t for t in tensor_list if t.dim() == 2]
    if len(two_d) < 2:
        raise LoweringError(
            f"{kernel.name}: two 2-D operands are required to bind it, got "
            f"{len(two_d)} (shapes: {[tuple(t.shape) for t in tensor_list]})"
        )
    want_a = tuple(kernel.extents.get(kernel.inputs[0], ()))
    want_b = tuple(kernel.extents.get(kernel.inputs[1], ())) if len(kernel.inputs) > 1 else ()

    activation, weight = two_d[0], two_d[1]
    # The *logical* inner extent comes from the caller's tensor, never from the
    # program's extents: those are padded to the tile, so reading K from them
    # makes a (37, 53) activation look like it should have 64 columns and every
    # unaligned matmul fails to bind. The extents are the padding *budget*, and
    # they are checked as one below.
    inner = int(activation.shape[1])
    if target_name == "linear":
        # torch's `linear` is `x @ weight.T` with weight stored `(out, in)`; a
        # graph that already transposed it hands over `(in, out)`. Where the
        # inner extent sits distinguishes them, and neither case is guessed.
        if weight.shape[1] == inner:
            weight_bound = weight.t()
        elif weight.shape[0] == inner:
            weight_bound = weight
        else:
            raise LoweringError(
                f"{kernel.name}: the weight operand (shape {tuple(weight.shape)}) has no "
                f"inner extent {inner} on either axis, so it is not a linear weight "
                f"for an activation of shape {tuple(activation.shape)}"
            )
    else:
        if weight.shape[0] != inner:
            raise LoweringError(
                f"{kernel.name}: the second matmul operand has shape "
                f"{tuple(weight.shape)} but the activation's inner extent is {inner}"
            )
        weight_bound = weight

    for tensor, wanted, role in (
        (activation, want_a, "activation"),
        (weight_bound, want_b, "weight"),
    ):
        if len(wanted) != 2:
            continue
        if tensor.shape[0] > wanted[0] or tensor.shape[1] > wanted[1]:
            raise LoweringError(
                f"{kernel.name}: the {role} has shape {tuple(tensor.shape)} but the "
                f"program addresses {wanted}; padding cannot shrink an operand"
            )
    return _Bound(
        operands=[activation, weight_bound],
        bias=bias_tensor,
        out_shape=(int(activation.shape[0]), int(weight_bound.shape[1])),
    )


def _pad_to_extent(kernel: CompiledKernel, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Pad a tensor up to the extent the program addresses. Never shrinks."""
    import torch.nn.functional as F

    wanted = tuple(kernel.extents.get(name, ()))
    if not wanted or tuple(tensor.shape) == wanted:
        return tensor
    if len(wanted) == 1:
        flat = tensor.reshape(-1)
        if flat.numel() > wanted[0]:
            raise LoweringError(
                f"{kernel.name}: operand for {name} has {flat.numel()} elements but the "
                f"program addresses {wanted[0]}; padding cannot shrink it"
            )
        return F.pad(flat, (0, wanted[0] - flat.numel()))
    if len(wanted) != len(tensor.shape):
        raise LoweringError(
            f"{kernel.name}: operand for {name} has rank {tensor.dim()} but the program "
            f"addresses rank {len(wanted)} ({wanted})"
        )
    pads: list[int] = []
    for got, want in zip(reversed(tuple(tensor.shape)), reversed(wanted), strict=False):
        if want < got:
            raise LoweringError(
                f"{kernel.name}: operand for {name} has shape {tuple(tensor.shape)} which "
                f"exceeds the addressed extent {wanted}; padding cannot shrink it"
            )
        pads.extend([0, want - got])
    return F.pad(tensor, tuple(pads))


def _run_with_padding(
    kernel: CompiledKernel,
    tensors: Sequence[torch.Tensor],
    target_name: str,
    has_relu: bool,
) -> torch.Tensor:
    """Run a lowering on the caller's tensors, padding by the kernel's own tile.

    Padding comes from the extents the program addresses — which `prepare`
    derived from the launch env and, for an extracted kernel, from the tile it was
    compiled against — instead of the 64/64/32 that used to be written here for
    every kernel. A kernel compiled for a 128-wide tile is now padded as one.

    Elementwise kernels are cropped back to the caller's shape. A bias is added
    here **only** when the kernel did not already add it, which `bias_lowered`
    records: the extracted `linear` kernel carries the bias inside the program
    while the recorded matmul does not, and adding it in both places doubles it
    silently.
    """
    bound = _bind_operands(kernel, tensors, target_name)
    # One operand per pointer parameter, in order. A parameter the caller has no
    # tensor for is filled with zeros — except the last one, which is where a
    # `linear` kernel reads its bias: appending the bias *after* the filler would
    # pass one tensor too many and `lower_and_run` refuses that (correctly).
    padded: list[torch.Tensor] = []
    names = list(kernel.inputs)
    for index, name in enumerate(names):
        if index < len(bound.operands):
            padded.append(_pad_to_extent(kernel, name, bound.operands[index]))
        elif bound.bias is not None and index == len(names) - 1:
            bias = bound.bias
            padded.append(
                _pad_to_extent(kernel, name, bias.float()) if bias.dim() == 1 else bias
            )
        else:
            padded.append(torch.zeros(kernel.extents.get(name, (1,)), dtype=torch.float32))

    result = lower_and_run(kernel, padded)

    if kernel.flat_width is not None:
        lanes = int(bound.operands[0].numel())
        return result.reshape(-1)[:lanes].reshape(bound.out_shape)

    rows, cols = bound.out_shape
    result = result[:rows, :cols]
    if bound.bias is not None and not kernel.bias_lowered:
        result = result + bound.bias.float()
    if has_relu:
        result = torch.relu(result)
    return result


#: The ISA the FX proof-of-concept path lowers to. A module-level name rather
#: than a literal at the call site, so a demo can retarget the live seam without
#: editing the backend.
FX_ISA = "tritonflow1"


@register_backend(name="tritonflow")
def tritonflow_backend(graph: Any, example_inputs: Sequence[torch.Tensor]) -> Callable[..., Any]:
    """The registered Dynamo backend.

    Returns a callable whose *results* are the graph's results either way; what
    differs is who computed them, and that difference is recorded on the callable
    as `tritonflow_plan` so a caller (or the coverage report) can read it instead of
    taking a promise.

    A lowering failure is not raised at compile time: `torch.compile` is allowed
    to succeed while individual operations fall back (the contract's first failure
    mode), and the honest place to say so is a record, with the numbers still
    correct.
    """
    plan = plan_graph(graph, example_inputs)
    kernels = list(plan.lowered)
    targets = graph_targets(graph)
    meaningful = [t for t in targets if t not in PLUMBING_TARGETS]

    if not kernels:
        # No recorded TTIR lowering matched. Before falling back to eager, try
        # lowering the FX graph itself (`fx_lower`): that route needs no frozen
        # TTIR, which is the whole point of it — it is the only path here that
        # can lower a graph nobody prepared in advance. It is deliberately narrow
        # and returns None for anything outside its op set, so this is an
        # addition to the fallback chain and never a replacement for it.
        from .fx_lower import lower_fx_graph

        reasons: list[str] = []
        fx = lower_fx_graph(graph, example_inputs, isa_name=FX_ISA, report=reasons)
        if fx is not None and fx.fully_lowered:

            def run_fx(*args: Any) -> Any:
                tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
                if not tensors:
                    return graph(*args)
                try:
                    produced = fx.run(tensors)
                except ProgramNotExecutable:
                    plan.fallbacks.append(
                        FallbackRecord(
                            reason="the FX-lowered program could not execute",
                            stage="execute",
                            nodes=plan.nodes,
                        )
                    )
                    return graph(*args)
                return _backend_return(torch.from_numpy(produced))

            run_fx.tritonflow_plan = plan  # type: ignore[attr-defined]
            run_fx.tritonflow_fx = fx  # type: ignore[attr-defined]
            return run_fx

        for reason in reasons:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=reason, stage="fx-lower", nodes=plan.nodes, detail=f"isa={FX_ISA}"
                )
            )

        # If fx_lower did not lower the whole graph, node-by-node lowering interpreter
        # is the primary path (Task E3).
        if len(meaningful) > 1:
            run = _multi_kernel_interpret(graph, example_inputs, plan)
            run.tritonflow_plan = plan  # type: ignore[attr-defined]
            run.tritonflow_multi_kernel = True  # type: ignore[attr-defined]
            return run

        run = _eager_fallback(graph)
        run.tritonflow_plan = plan  # type: ignore[attr-defined]
        return run

    kernel = kernels[0]
    _targets = graph_targets(graph)
    _meaningful = [t for t in _targets if t not in PLUMBING_TARGETS and t not in ELEMENTWISE_TARGETS]
    _target_name = _meaningful[0] if _meaningful else ""
    _has_relu = any(t in ELEMENTWISE_TARGETS for t in _targets)

    # Shadow-verify lowered program on example_inputs
    try:
        shadow_tensors = [arg for arg in example_inputs if isinstance(arg, torch.Tensor)]
        if shadow_tensors:
            shadow_result = _run_with_padding(kernel, shadow_tensors, _target_name, _has_relu)
            eager_res = graph(*example_inputs)
            if isinstance(eager_res, (list, tuple)):
                eager_res = eager_res[0]
            if _target_name:
                k_val = int(shadow_tensors[0].shape[-1]) if shadow_tensors[0].dim() >= 2 else 1
                derived_tol = float(k_val * (2.0 * 2.0**-11 + 2.0**-24))
            else:
                derived_tol = float(2.0**-24)
            peak = max(1e-9, float(eager_res.detach().abs().max()))
            rel_err = float((shadow_result.detach() - eager_res.detach()).abs().max()) / peak
            if rel_err > derived_tol:
                plan.fallbacks.append(
                    FallbackRecord(
                        reason=f"shadow verification mismatch: relative error {rel_err:g} exceeds derived tolerance {derived_tol:g}",
                        stage="shadow-verify",
                        nodes=plan.nodes,
                    )
                )
                plan.lowered.clear()
    except _REFUSALS as exc:
        plan.fallbacks.append(_refusal_record(exc, plan.nodes))
        plan.lowered.clear()
    except Exception as exc:
        plan.fallbacks.append(
            FallbackRecord(
                reason=f"shadow verification failed: {exc}",
                stage="shadow-verify",
                nodes=plan.nodes,
            )
        )
        plan.lowered.clear()

    if not plan.lowered:
        run = _eager_fallback(graph)
        run.tritonflow_plan = plan
        return run

    def run(*args: Any) -> Any:
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if not tensors:
            return graph(*args)
        try:
            result = _run_with_padding(kernel, tensors, _target_name, _has_relu)
        except _REFUSALS as exc:
            # The refusals that ARE fallbacks: the program carries an `UNSUPPORTED`
            # marker, the emulator has no case for an instruction the assembler
            # emitted, or an input is a dtype this device does not declare. The
            # contract routes all three to eager with a record. The dtype case used
            # to propagate, which meant an fp16 or bf16 model did not run *at all*.
            plan.fallbacks.append(_refusal_record(exc, plan.nodes))
            plan.lowered.clear()
            return graph(*args)
        # Anything else propagates. A plain `LoweringError` (a binding built wrong)
        # and a `StorageError` are *our* bugs; absorbing either into an eager answer
        # would be the silent-fallback sin this seam exists to avoid. That
        # distinction is the same one `emit.ir.AssemblyError` draws.
        return _backend_return(result)

    run.tritonflow_plan = plan  # type: ignore[attr-defined]
    run.tritonflow_kernel = kernel  # type: ignore[attr-defined]
    return run


def verify_device() -> dict[str, Any]:
    """Install the device and report what PyTorch now believes about it.

    The assertions that matter are PyTorch's own lookups, not this function's
    return value: `get_interface_for_device("tritonflow")` must resolve to this class,
    the backend must appear in `list_backends()`, and the device-state round trip
 ('s `current_device`/`set_device`) must actually move. Everything the
    return value reports is read back out of PyTorch, so a wrong claim here is
    visible in the check that consumes it.
    """
    from torch._dynamo.backends.registry import list_backends
    from torch._dynamo.device_interface import get_interface_for_device

    installed = device_interface.install()
    interface = get_interface_for_device(DEVICE_NAME)
    previous = interface.current_device()
    interface.set_device(0)
    round_tripped = interface.current_device()
    if previous != 0:
        interface.set_device(previous)
    return {
        **installed,
        "interface": interface.__name__,
        "is_available": interface.is_available(),
        "device_count": interface.device_count(),
        "current_device": round_tripped,
        "device_round_trip": previous == round_tripped or round_tripped == 0,
        "registered_backend": "tritonflow" in list_backends(),
        "interface_is_ours": interface is device_interface.TritonFlowInterface,
        "slot_inventory": device_interface.measure_slot_inventory(),
    }


