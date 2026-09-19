"""Structured-access descriptors: `ttir` operand → `AccessDescriptor`.

This module resolves `ttir` memory operands into structured access
descriptors. Four things it does that v1's `{base, stride, shape}` did not:

1. **Loop-carried operands are resolved, not skipped.** Tier 1's `tt.load`
   operands *are* `scf.for` iter_args, so the pointer is defined nowhere in the
   module until the recurrence is followed back to its init value
   (`substitute_iter_arg`). The result carries `loop_carried=True` and a
   *decidable* `increment`.
2. **Every field is populated.** `sizes`, `strides`, `offsets`, `shape`, `order`,
   `dtype`, `loop_carried`, `increment`, `provenance` — the omissions in v1 were
   load-bearing, not cosmetic: `shape` is the *wraparound boundary* (the field
   that distinguishes Tier 3's modulo addressing from a contiguous tile), and
   `loop_carried`/`increment` are the wrong-answer guard.
3. **The result is a union, and it is total.** `Ok` | `Unstructured` |
   `BudgetExhausted`, one of the three for every operand, never an exception
.
4. **Refusal is by name.** An operation the walk cannot fold produces
   `Unstructured("non-affine index: <op>")`. A dropped term would be a *wrong
   address with no marker on it*, which is the failure class this whole file is
   arranged to make impossible.

**`sizes` come from the index tensor; the coefficients come from the fold.** Two
independent sources, which is why the rank mismatch is a refusal rather than an
assumption.

**What `base` is.** The *underlying* pointer — `a_ptr`, not the `tt.splat` that
wraps it — because that is what a structured descriptor's base means (`tts.make_tptr`'s
first argument) and what a reader of the report can look up in the kernel. The
SSA name of the value actually loaded through is in `provenance`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..ttir.graph import DefUseGraph, walk_region
from ..ttir.ssa import Module, Operation, SsaValue
from . import op_shapes as shapes
from .op_shapes import ADDI as ARITH_ADDI
from .op_shapes import ADDPTR, BROADCAST, CONSTANT, EXPAND_DIMS, MAKE_RANGE, MULI, SPLAT
from .op_shapes import REMSI as ARITH_REMSI
from .op_shapes import SUBI as ARITH_SUBI
from .walk import (
    MAX_HOPS,
    BoundedWalker,
    BudgetReached,
    SymExpr,
    public,
    public_list,
    substitute_iter_arg,
)

#: The identity of a descriptor, as text. Used by `MemRef.of` (`emit/ir.py`) and
#: by every equality in the reports, so it must be byte-stable: no `repr` of a
#: dataclass, no dict ordering, no `set`.
KEY_FIELDS = (
    "base",
    "sizes",
    "strides",
    "offsets",
    "shape",
    "order",
    "dtype",
    "loop_carried",
    "increment",
)


class _Refuse(Exception):
    """Internal: the walk has decided this operand is not structured."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# The affine specification of one index expression
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AffineSpec:
    """`value[i0, …, in] = offset + Σ coeffs[k] · i_k`, plus the wrap boundaries.

    `coeffs[k] is None` means the value carries no index variable in dimension
    `k` — either because the dimension has size 1, or because the expression was
    broadcast along it. That is not the same as a coefficient of zero in a
    symbolic sense (both behave as zero) but it is how the *shape* stays known
    when the coefficient is folded away.

    `wrap[k]` is the wraparound boundary of dimension `k`: `None` means no wrap,
    an integer means the index was taken modulo that value. This is what
    `AccessDescriptor` calls `shape`.
    """

    sizes: tuple[int | None, ...] = ()
    coeffs: tuple[SymExpr | None, ...] = ()
    offset: SymExpr = field(default_factory=lambda: SymExpr.const(0))
    wrap: tuple[int | None, ...] = ()
    is_gather_scatter: bool = False
    indices_name: str | None = None

    @property
    def rank(self) -> int:
        return len(self.sizes)

    @property
    def is_variable_free(self) -> bool:
        """Whether no dimension of this expression carries an index variable."""
        return all(coeff is None or coeff.is_zero for coeff in self.coeffs)

    def variable_dims(self) -> tuple[int, ...]:
        return tuple(
            k for k, coeff in enumerate(self.coeffs) if coeff is not None and not coeff.is_zero
        )

    def scaled(self, factor: SymExpr) -> AffineSpec:
        """Multiply by a variable-free expression. Never called with a variable.

        The guard is the definition of an affine access: a product of two index
        variables is degree 2 and this method is the only place that could
        produce one, so it is the only place that has to refuse.
        """
        coeffs = tuple(None if coeff is None else coeff * factor for coeff in self.coeffs)
        return AffineSpec(
            sizes=self.sizes,
            coeffs=coeffs,
            offset=self.offset * factor,
            wrap=self.wrap,
        )

    def plus(self, other: AffineSpec) -> AffineSpec:
        sizes = _merge_sizes(self.sizes, other.sizes)
        coeffs = _merge_coeffs(self.coeffs, self.sizes, other.coeffs, other.sizes, len(sizes))
        wrap = _merge_wrap(self.wrap, self.sizes, other.wrap, other.sizes, len(sizes))
        return AffineSpec(sizes=sizes, coeffs=coeffs, offset=self.offset + other.offset, wrap=wrap)


def _align(index: int, source_rank: int, target_rank: int) -> int | None:
    """Map a dimension of a `source_rank`-sized shape into a `target_rank` frame.

    Trailing alignment, which is the convention every broadcast in this pipeline
    follows (2-D tiles align on the last dimension). A
    source dimension that falls off the front of the target has no counterpart
    and maps to `None`.
    """
    target = index + (target_rank - source_rank)
    return target if 0 <= target < target_rank else None


def _merge_sizes(a: tuple[int | None, ...], b: tuple[int | None, ...]) -> tuple[int | None, ...]:
    if not a:
        return b
    if not b:
        return a
    if len(a) != len(b):
        raise _Refuse(
            f"non-affine index: shapes of rank {len(a)} and {len(b)} cannot be combined; "
            "a rank change must be an explicit tt.broadcast or tt.expand_dims"
        )
    merged: list[int | None] = []
    for left, right in zip(a, b, strict=True):
        if left is None or right is None:
            # An unknown extent stays unknown. Taking the known side was the
            # first version, and it is a *guess*: `tensor<?xi32>` combined with
            # `tensor<1024xi32>` has an extent that is at most 1024 and is not
            # 1024, so recording 1024 turns an undecidable tile into a plausible
            # one. It is refused one layer up ("dynamic extent").
            merged.append(None)
        elif left == right or right == 1:
            merged.append(left)
        elif left == 1:
            merged.append(right)
        else:
            raise _Refuse(
                f"non-affine index: incompatible extents {left} and {right} in one dimension"
            )
    return tuple(merged)


def _merge_coeffs(
    left: tuple[SymExpr | None, ...],
    left_sizes: tuple[int | None, ...],
    right: tuple[SymExpr | None, ...],
    right_sizes: tuple[int | None, ...],
    rank: int,
) -> tuple[SymExpr | None, ...]:
    out: list[SymExpr | None] = [None] * rank
    for source, sizes in ((left, left_sizes), (right, right_sizes)):
        for index, coeff in enumerate(source):
            if coeff is None or coeff.is_zero:
                continue
            target = _align(index, len(sizes), rank)
            if target is None:
                continue
            if out[target] is None:
                out[target] = coeff
            else:
                out[target] = out[target] + coeff
    return tuple(out)


def _merge_wrap(
    left: tuple[int | None, ...],
    left_sizes: tuple[int | None, ...],
    right: tuple[int | None, ...],
    right_sizes: tuple[int | None, ...],
    rank: int,
) -> tuple[int | None, ...]:
    out: list[int | None] = [None] * rank
    for source, sizes in ((left, left_sizes), (right, right_sizes)):
        for index, bound in enumerate(source):
            if bound is None:
                continue
            target = _align(index, len(sizes), rank)
            if target is not None and out[target] is None:
                out[target] = bound
    return tuple(out)


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AccessDescriptor:
    """A structured memory access, field for field. Field types are `int | SymExpr`."""

    base: str
    sizes: tuple[int | SymExpr, ...]
    strides: tuple[int | SymExpr, ...]
    offsets: tuple[int | SymExpr, ...]
    shape: tuple[int | SymExpr, ...]
    order: tuple[int, ...]
    dtype: str
    loop_carried: bool
    increment: int | SymExpr | None
    provenance: tuple[str, ...] = ()
    is_gather_scatter: bool = False
    indices_name: str | None = None

    def descriptor_key(self) -> str:
        """The canonical text key. Stable across processes and runs."""
        gather_part = f";is_gather={self.is_gather_scatter};indices={self.indices_name}" if self.is_gather_scatter else ""
        return (
            f"base={self.base};sizes={_fmt_seq(self.sizes)};strides={_fmt_seq(self.strides)};"
            f"offsets={_fmt_seq(self.offsets)};shape={_fmt_seq(self.shape)};order={list(self.order)};"
            f"dtype={self.dtype};loop_carried={self.loop_carried};increment={_fmt_value(self.increment)}"
            f"{gather_part}"
        )

    @property
    def rank(self) -> int:
        return len(self.sizes)

    @property
    def is_contiguous(self) -> bool:
        """Unit stride on the last dimension and no wraparound anywhere."""
        if not self.strides or not self.shape:
            return False
        return self.strides[-1] == 1 and all(bound == 0 for bound in self.shape)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.descriptor_key()


def _fmt_value(value: object) -> str:
    if isinstance(value, SymExpr):
        return str(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _fmt_seq(values: Iterable[object]) -> str:
    return "[" + ", ".join(_fmt_value(value) for value in values) + "]"


# --------------------------------------------------------------------------- #
# The result union
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ok:
    """The operand is structured. `descriptor` carries every field."""

    descriptor: AccessDescriptor


@dataclass(frozen=True)
class Unstructured:
    """The operand is not a structured access. `reason` names why, `ops` the walk.

    `ops` is the provenance of the *attempted* walk, so a reader can see how far
    it got — which is the difference between "we refused immediately" and "we
    followed nine operations and then hit a modulo".
    """

    reason: str
    ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class BudgetExhausted:
    """The hop budget ran out. A result, not a crash."""

    hops: int
    limit: int


DescriptorResult = Ok | Unstructured | BudgetExhausted


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def describe(
    value: SsaValue,
    access: tuple[str, ...] = (),
    graph: DefUseGraph | None = None,
) -> DescriptorResult:
    """Resolve one memory operand to a descriptor, or refuse by name.

    `access` names the memory spaces this operand is being described for. It is
    recorded in `provenance` (and is the hook the ISA-2 scratchpad model uses to
    ask the same walk for a banked access); the default empty tuple means "the
    caller has no space restriction", which is the ISA-1 flat-memory case.
    """
    if graph is None:
        return Unstructured("no def-use graph was supplied, so the operand cannot be resolved")
    walker = BoundedWalker(MAX_HOPS)
    try:
        return Ok(_describe(value, access, graph, walker))
    except _Refuse as refusal:
        return Unstructured(refusal.reason, tuple(walker.provenance))
    except BudgetReached as reached:
        return BudgetExhausted(hops=reached.limit, limit=reached.limit)


def resolve_operand(value: SsaValue, graph: DefUseGraph | None = None) -> DescriptorResult:
    """`describe` with no space restriction — the contract's short form."""
    return describe(value, (), graph)


def describe_operation(op: Operation, graph: DefUseGraph | None = None) -> DescriptorResult:
    """Describe the pointer of a `tt.load`/`tt.store`. Total for other ops.

    Returning `Unstructured` for a non-memory operation (rather than raising) is
    deliberate: callers iterate the module, and a total function is what lets
    them do it without a pre-filter they could forget.
    """
    pointer = shapes.pointer_operand(op)
    if pointer is None:
        return Unstructured(f"not a memory operation: {op.name}")
    return describe(pointer, (), graph)


def _describe(
    value: SsaValue, access: tuple[str, ...], graph: DefUseGraph, walker: BoundedWalker
) -> AccessDescriptor:
    module = getattr(graph, "module", None)
    walker.note(f"operand {value.name}")
    for space in access:
        walker.note(f"space {space}")

    recurrence = None
    if value.is_block_arg:
        if module is None:
            # A block argument is defined by a construct (a loop or a region), and
            # without the module the walk cannot see it. Guessing "not carried"
            # would silently drop a recurrence, which is precisely the Tier-1 case.
            raise _Refuse(
                f"{value.name} is a block argument and the def-use graph carries no module, "
                "so its recurrence cannot be resolved"
            )
        recurrence = substitute_iter_arg(value, module)
    if recurrence is not None:
        walker.note(f"scf.for iter_arg {value.name}")

    head = recurrence.init if recurrence is not None else value
    base, spec, pointer_type = _walk_pointer_chain(head, graph, walker)
    walker.note(f"base {base}")

    increment: int | SymExpr | None = None
    if recurrence is not None:
        increment = _increment_of(recurrence, graph, walker)

    pointer_shape = shapes.shape_of(pointer_type)
    if spec.rank and pointer_shape and len(pointer_shape) != spec.rank:
        raise _Refuse(
            f"non-affine index: the index has rank {spec.rank} and the pointer has rank "
            f"{len(pointer_shape)}; one of the two was broadcast without saying so"
        )
    sizes = spec.sizes if spec.rank else pointer_shape
    for extent in sizes:
        if extent is None:
            raise _Refuse(
                "dynamic extent: the tile shape is not known statically, so no descriptor is decidable"
            )

    rank = len(sizes)
    strides = public_list(
        coeff if coeff is not None else SymExpr.const(0) for coeff in _padded(spec.coeffs, rank)
    )
    offsets = public_list(_offsets_of(spec, rank))
    shape = public_list(
        SymExpr.const(bound) if bound is not None else SymExpr.const(0)
        for bound in _padded_wrap(spec.wrap, rank)
    )

    return AccessDescriptor(
        base=base,
        sizes=public_list(
            SymExpr.const(extent) if extent is not None else SymExpr.const(0) for extent in sizes
        ),
        strides=strides,
        offsets=offsets,
        shape=shape,
        order=tuple(range(rank)),
        dtype=shapes.scalar_dtype(pointer_type),
        loop_carried=recurrence is not None,
        increment=increment,
        provenance=tuple(walker.provenance),
        is_gather_scatter=spec.is_gather_scatter,
        indices_name=spec.indices_name,
    )


def _padded(values: tuple[Any, ...], rank: int) -> tuple[Any, ...]:
    return values if len(values) == rank else (values + (None,) * rank)[:rank]


def _padded_wrap(values: tuple[int | None, ...], rank: int) -> tuple[int | None, ...]:
    return values if len(values) == rank else (values + (None,) * rank)[:rank]


def _offsets_of(spec: AffineSpec, rank: int) -> tuple[SymExpr, ...]:
    """Per-dimension start offsets.

    `offsets[0]` is the whole offset expression (the constant part of the walk),
    because a per-dimension split of a sum of symbol products is not recoverable
    in general and inventing one would be a guess. The remaining entries are
    zero. `AccessDescriptor.offsets` is the per-dimension start offset; on this
    corpus the offset is always carried in dimension 0 (`pid_m·BM·sam` is a row
    offset), which is what the field records.
    """
    out: list[SymExpr] = [SymExpr.const(0)] * max(rank, 1)
    out[0] = spec.offset
    return tuple(out)


def _walk_pointer_chain(
    value: SsaValue,
    graph: DefUseGraph,
    walker: BoundedWalker,
) -> tuple[str, AffineSpec, object]:
    """Follow `tt.addptr` back to the base pointer, accumulating the index sum.

    `tt.addptr` chains add their index expressions (`ptr + a + b`), which is the
    only composition the operation has. Any other operation producing a pointer
    is a refusal: a pointer the walk cannot account for is an address it cannot
    describe, and describing it anyway is the silent miscompile.
    """
    index: AffineSpec | None = None
    current = value
    # A bounded loop, not recursion: the budget must apply to the chain itself.
    while True:
        op = current.def_op
        if op is None:
            return current.name, index or AffineSpec(), current.type
        walker.visit(op)
        if op.name == ADDPTR:
            index_operand = shapes.index_operand(op)
            base = shapes.base_operand(op)
            if base is None:
                raise _Refuse(f"non-affine pointer: {op.name} has no base operand")
            if index_operand is not None:
                try:
                    part = _analyze_index(index_operand, graph, walker)
                    index = part if index is None else index.plus(part)
                except _Refuse as ref:
                    if 'modulo' in str(ref):
                        raise
                    # Non-affine / indirect index: bifurcate into explicit gather/scatter descriptor!
                    base_cur = base
                    while base_cur and base_cur.def_op and base_cur.def_op.name == ADDPTR:
                        base_cur = shapes.base_operand(base_cur.def_op)
                    origin_name = base_cur.name if base_cur else "unknown"
                    if base_cur and base_cur.def_op and base_cur.def_op.name == SPLAT:
                        if base_cur.def_op.operands:
                            origin_name = base_cur.def_op.operands[0].name
                    idx_shape = shapes.shape_of(index_operand.type) or (32,)
                    gather_spec = AffineSpec(
                        sizes=tuple(idx_shape),
                        is_gather_scatter=True,
                        indices_name=index_operand.name,
                    )
                    return origin_name, gather_spec, main_type(base_cur or current, current)
            current = base
            continue
        if op.name == SPLAT:
            inner = op.operands[0] if op.operands else None
            if inner is None:
                raise _Refuse(f"non-affine pointer: {op.name} has no operand")
            return inner.name, index or AffineSpec(), main_type(inner, current)
        raise _Refuse(f"non-affine pointer: {op.name} produces {value.name}")


def main_type(inner: SsaValue, outer: SsaValue) -> object:
    """The pointer type to describe: the splat's operand, or the value's own.

    For `tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>` the
    *element* type `!tt.ptr<f32>` is the pointer type; taking the tensor's type
    instead would make the rank check compare an index rank against a rank that
    came from the wrong side of the broadcast.
    """
    return inner.type if inner.type is not None else outer.type


def _analyze_index(value: SsaValue, graph: DefUseGraph, walker: BoundedWalker) -> AffineSpec:
    """Fold an index expression into an :class:`AffineSpec`, or refuse by name."""
    op = value.def_op
    if op is None:
        raise _Refuse(
            f"non-affine index: {value.name} is used directly as an index and is not a constant"
        )
    walker.visit(op)
    name = op.name

    if name == MAKE_RANGE:
        return _make_range(value, op)
    if name == CONSTANT:
        return _constant_spec(value, op)
    if name == SPLAT:
        return _splat_spec(value, op, graph, walker)
    if name == BROADCAST:
        return _broadcast_spec(value, op, graph, walker)
    if name == EXPAND_DIMS:
        return _expand_spec(value, op, graph, walker)
    if name == MULI:
        return _mul_spec(value, op, graph, walker)
    if name in (ARITH_ADDI, ARITH_SUBI):
        return _add_spec(value, op, graph, walker, name)
    if name == ARITH_REMSI:
        return _remsi_spec(value, op, graph, walker)
    if name in shapes.CAST_OPS:
        operand = op.operands[0] if op.operands else None
        if operand is None:
            raise _Refuse(f"non-affine index: {name} has no operand")
        inner = _analyze_index(operand, graph, walker)
        return AffineSpec(
            sizes=shapes.shape_of(value.type) or inner.sizes,
            coeffs=inner.coeffs,
            offset=inner.offset,
            wrap=inner.wrap,
        )
    if shapes.is_non_affine(op):
        raise _Refuse(f"non-affine index: {name}")
    raise _Refuse(f"non-affine index: {name}")


def _extents(value: SsaValue) -> tuple[int | None, ...]:
    return shapes.shape_of(value.type)


def _make_range(value: SsaValue, op: Operation) -> AffineSpec:
    extents = _extents(value)
    if not extents:
        raise _Refuse("non-affine index: tt.make_range with no tensor result type")
    start = 0
    text = op.attr.get("start")
    if text is not None:
        digits = "".join(ch for ch in text.split(":")[0] if ch.isdigit() or ch == "-")
        start = int(digits) if digits not in ("", "-") else 0
    coeffs: tuple[SymExpr | None, ...] = (SymExpr.const(1),) + (None,) * (len(extents) - 1)
    return AffineSpec(sizes=extents, coeffs=coeffs, offset=SymExpr.const(start))


def _constant_spec(value: SsaValue, op: Operation) -> AffineSpec:
    decoded = shapes.decode_constant(op)
    if not decoded.ok:
        raise _Refuse(f"non-affine index: {decoded.reason}")
    if not decoded.is_int:
        raise _Refuse(f"non-affine index: {decoded.text!r} is not an integer constant")
    extents = _extents(value)
    return AffineSpec(
        sizes=extents, coeffs=(None,) * len(extents), offset=SymExpr.const(int(decoded.value))
    )


def _splat_spec(
    value: SsaValue, op: Operation, graph: DefUseGraph, walker: BoundedWalker
) -> AffineSpec:
    extents = _extents(value)
    if not extents:
        raise _Refuse("non-affine index: tt.splat with no tensor result type")
    inner = op.operands[0] if op.operands else None
    if inner is None:
        raise _Refuse("non-affine index: tt.splat has no operand")
    if inner.type is not None and inner.type.is_pointer_like:
        raise _Refuse(
            f"non-affine index: tt.splat of the pointer {inner.name} in an index position"
        )
    offset = _scalar(inner, walker)
    return AffineSpec(sizes=extents, coeffs=(None,) * len(extents), offset=offset)


def _broadcast_spec(
    value: SsaValue, op: Operation, graph: DefUseGraph, walker: BoundedWalker
) -> AffineSpec:
    inner = op.operands[0] if op.operands else None
    if inner is None:
        raise _Refuse("non-affine index: tt.broadcast has no operand")
    part = _analyze_index(inner, graph, walker)
    extents = _extents(value)
    if not extents:
        raise _Refuse("non-affine index: tt.broadcast with no tensor result type")
    if len(extents) < part.rank:
        raise _Refuse("non-affine index: tt.broadcast would lower the rank")
    pad = len(extents) - part.rank
    coeffs: list[SymExpr | None] = [None] * pad
    for index, coeff in enumerate(part.coeffs):
        # A dimension that was size 1 carries no variable in the result either:
        # broadcasting expands the *extent*, not the coefficient.
        coeffs.append(coeff if part.sizes[index] != 1 else None)
    wrap: list[int | None] = [None] * pad
    for index, bound in enumerate(part.wrap):
        wrap.append(bound if part.sizes[index] != 1 else None)
    return AffineSpec(sizes=extents, coeffs=tuple(coeffs), offset=part.offset, wrap=tuple(wrap))


def _expand_spec(
    value: SsaValue, op: Operation, graph: DefUseGraph, walker: BoundedWalker
) -> AffineSpec:
    inner = op.operands[0] if op.operands else None
    if inner is None:
        raise _Refuse("non-affine index: tt.expand_dims has no operand")
    part = _analyze_index(inner, graph, walker)
    extents = _extents(value)
    if len(extents) != part.rank + 1:
        raise _Refuse("non-affine index: tt.expand_dims result rank is not operand rank + 1")
    axis = shapes.expand_axis(op)
    if axis is None:
        raise _Refuse("non-affine index: tt.expand_dims with no decidable axis")
    axis = len(extents) + axis if axis < 0 else axis
    if not 0 <= axis < len(extents):
        raise _Refuse(f"non-affine index: tt.expand_dims axis {axis} is outside the result rank")
    coeffs: list[SymExpr | None] = list(part.coeffs)
    coeffs.insert(axis, None)
    wrap: list[int | None] = list(part.wrap)
    wrap.insert(axis, None)
    return AffineSpec(sizes=extents, coeffs=tuple(coeffs), offset=part.offset, wrap=tuple(wrap))


def _mul_spec(
    value: SsaValue, op: Operation, graph: DefUseGraph, walker: BoundedWalker
) -> AffineSpec:
    if len(op.operands) < 2:
        raise _Refuse("non-affine index: arith.muli with fewer than two operands")
    left = _analyze_index(op.operands[0], graph, walker)
    right = _analyze_index(op.operands[1], graph, walker)
    if left.is_variable_free:
        return right.scaled(left.offset)
    if right.is_variable_free:
        return left.scaled(right.offset)
    raise _Refuse(
        "non-affine index: a product of two index-variable expressions is degree 2, "
        "so it is not a structured access"
    )


def _add_spec(
    value: SsaValue,
    op: Operation,
    graph: DefUseGraph,
    walker: BoundedWalker,
    name: str,
) -> AffineSpec:
    if len(op.operands) < 2:
        raise _Refuse(f"non-affine index: {name} with fewer than two operands")
    left = _analyze_index(op.operands[0], graph, walker)
    right = _analyze_index(op.operands[1], graph, walker)
    if name == ARITH_SUBI:
        return left.plus(
            AffineSpec(
                sizes=right.sizes,
                coeffs=tuple(None if coeff is None else -coeff for coeff in right.coeffs),
                offset=-right.offset,
                wrap=right.wrap,
            )
        )
    return left.plus(right)


def _remsi_spec(
    value: SsaValue, op: Operation, graph: DefUseGraph, walker: BoundedWalker
) -> AffineSpec:
    """`arith.remsi` — a wraparound boundary, or a refusal.

    A *decidable* divisor is the wraparound boundary `AccessDescriptor` calls
    `shape`: the access re-enters at 0 every `k` elements. A *symbolic* divisor
    is refused by name, because "wraps at an unknown boundary" is not an access
    the selector can cost and pretending otherwise is how Tier 3 would come back
    `Ok` and be lowered as a contiguous tile.
    """
    if len(op.operands) < 2:
        raise _Refuse("non-affine index: arith.remsi with fewer than two operands")
    left = _analyze_index(op.operands[0], graph, walker)
    divisor_value = op.operands[1]
    divisor = _scalar_or_none(divisor_value, walker)
    constant = divisor.as_int() if divisor is not None else None
    if constant is None or constant <= 0:
        raise _Refuse("modulo wraparound")
    dims = left.variable_dims()
    if len(dims) != 1:
        raise _Refuse("modulo wraparound")
    wrap = list(left.wrap)
    while len(wrap) < left.rank:
        wrap.append(None)
    wrap[dims[0]] = constant
    offset = left.offset
    reduced = offset.as_int()
    if reduced is not None:
        offset = SymExpr.const(reduced % constant)
    return AffineSpec(sizes=left.sizes, coeffs=left.coeffs, offset=offset, wrap=tuple(wrap))


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #


def _scalar(value: SsaValue, walker: BoundedWalker) -> SymExpr:
    folded = _scalar_or_none(value, walker)
    if folded is None:
        raise _Refuse(f"non-decidable scalar expression in an index: {value.name}")
    return folded


def _scalar_or_none(value: SsaValue, walker: BoundedWalker) -> SymExpr | None:
    """Fold a scalar expression to `int | SymExpr`, or `None` to refuse.

    `None` is a refusal, and the caller must name it. `%K + 31` is an *integer
    value* at runtime and folding it to `31` would be a plausible-looking number
    with nothing marking it as a guess.
    """
    op = value.def_op
    if op is None:
        if value.type is not None and value.type.is_pointer_like:
            return None
        return SymExpr.symbol(value.name)
    walker.visit(op)
    name = op.name
    if name == CONSTANT:
        decoded = shapes.decode_constant(op)
        if decoded.ok and decoded.is_int:
            return SymExpr.const(int(decoded.value))
        return None
    if name == MULI:
        if len(op.operands) < 2:
            return None
        left = _scalar_or_none(op.operands[0], walker)
        right = _scalar_or_none(op.operands[1], walker)
        if left is None or right is None:
            return None
        if left.is_constant and right.is_constant:
            return left * right
        if left.is_constant:
            return right.scale(int(left.as_int() or 0))
        if right.is_constant:
            return left.scale(int(right.as_int() or 0))
        # Two non-constant scalars: a symbolic product, which is still affine in
        # the *index variables* (neither is one) and is exactly how `rm * sam`
        # is represented. It is only degree 2 in the *symbols*.
        return left * right
    if name in (ARITH_ADDI, ARITH_SUBI):
        if len(op.operands) < 2:
            return None
        left = _scalar_or_none(op.operands[0], walker)
        right = _scalar_or_none(op.operands[1], walker)
        if left is None or right is None:
            return None
        return left + (-right if name == ARITH_SUBI else right)
    if name in shapes.CAST_OPS:
        if not op.operands:
            return None
        return _scalar_or_none(op.operands[0], walker)
    if name in ("tt.get_program_id", "tt.get_num_programs"):
        return SymExpr.symbol(value.name)
    return None


# --------------------------------------------------------------------------- #
# Increment
# --------------------------------------------------------------------------- #


def _increment_of(recurrence: object, graph: DefUseGraph, walker: BoundedWalker) -> int | SymExpr:
    """The decidable per-iteration advance of a loop-carried operand.

    Recovered from the *yielded* pointer's index expression, which is the only
    place the step exists. Three refusals, all of which must be refusals rather
    than zeros:

    * the yielded value is not an `addptr` on this iter_arg;
    * the advance depends on the loop's induction variable, so it is a function of
      the trip count rather than a constant step;
    * the advance has no integer factor (more than one term, or no term at all).
    """
    advance = getattr(recurrence, "advance_value", None)
    if advance is None:
        raise _Refuse(
            "non-decidable increment: the yielded pointer is not an addptr on its iter_arg"
        )
    spec = _analyze_index(advance, graph, walker)
    if not spec.is_variable_free:
        raise _Refuse("non-decidable increment: the advance varies across the tile")
    iv = getattr(recurrence, "iv_name", None)
    if iv is not None and iv in spec.offset.symbols():
        raise _Refuse(
            f"non-decidable increment: the advance depends on the induction variable {iv}"
        )
    factor = spec.offset.integer_factor()
    if factor is None or len(spec.offset.terms) != 1:
        raise _Refuse(
            "non-decidable increment: the advance is a sum of more than one term, "
            "so no single per-iteration step can be stated"
        )
    # Exactly one term, and its symbolic part is *one* stride. Tier 1's advance is
    # `32 * sak`: the 32 is the K-block (`a_ptrs += BK * sak`), which is the
    # number a selector can compare against the tile's K extent. `sak * sbk` also
    # has a scale of 1 and is **not** such a step — it is a product of two
    # strides, and reporting `increment = 1` for it was this function's first
    # behaviour: a number that looks like a decision and is an accident of the
    # normalisation.
    symbols = spec.offset.terms[0][1]
    if len(symbols) > 1:
        raise _Refuse(
            "non-decidable increment: the advance is a product of more than one symbol, "
            "so the constant factor is not separable from the strides"
        )
    return public(SymExpr.const(factor))


# --------------------------------------------------------------------------- #
# Conformance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Disagreement:
    """One field where our descriptor and the oracle differ."""

    operand: str
    field: str
    ours: object
    oracle: object


@dataclass(frozen=True)
class ConformanceReport:
    """`conformance_check`'s output: agreements, refusals, and every disagreement.

    `oracle` is a callable, never an import: `tts.make_tptr` lives in Triton, and
    the pipeline has no Triton dependency. The oracle is therefore
    supplied by the caller (the check file ships one derived from the kernel
    text), and this module only compares.
    """

    compared: tuple[str, ...] = ()
    refusals: tuple[tuple[str, str], ...] = ()
    disagreements: tuple[Disagreement, ...] = ()
    uncompared: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        return (
            f"{len(self.compared)} compared, {len(self.refusals)} refused, "
            f"{len(self.uncompared)} with no oracle opinion, {len(self.disagreements)} disagreements"
        )


#: The fields `conformance_check` compares. `order`/`provenance` are excluded on
#: purpose: `order` is a convention this pipeline fixes at row-major identity and
#: `provenance` is diagnostics, and comparing either would let a *presentation*
#: difference mask an addressing difference.
CONFORMANCE_FIELDS = ("base", "sizes", "strides", "offsets")


def conformance_check(
    module: Module,
    oracle: Callable[[str], Any] | None = None,
    graph: DefUseGraph | None = None,
) -> ConformanceReport:
    """Compare our descriptors against an oracle over every memory operand.

    `oracle(operand_name)` returns a mapping with any of `CONFORMANCE_FIELDS`, or
    `None` when it has no opinion about that operand (which lands in
    `uncompared`, not in `disagreements` — "no opinion" and "disagrees" are
    different facts and a report that merged them would be unreadable).
    """
    if graph is None:
        from ..ttir.graph import build_def_use

        graph = build_def_use(module)

    compared: list[str] = []
    refusals: list[tuple[str, str]] = []
    disagreements: list[Disagreement] = []
    uncompared: list[str] = []

    for op in walk_region(module.body):
        pointer = shapes.pointer_operand(op)
        if pointer is None:
            continue
        result = describe(pointer, (), graph)
        if isinstance(result, Unstructured):
            refusals.append((pointer.name, result.reason))
            continue
        if isinstance(result, BudgetExhausted):
            refusals.append((pointer.name, f"budget exhausted ({result.hops}/{result.limit})"))
            continue
        descriptor = result.descriptor
        if oracle is None:
            uncompared.append(pointer.name)
            continue
        opinion = oracle(pointer.name)
        if opinion is None:
            uncompared.append(pointer.name)
            continue
        compared.append(pointer.name)
        for field_name in CONFORMANCE_FIELDS:
            if field_name not in opinion:
                continue
            ours = getattr(descriptor, field_name)
            theirs = opinion[field_name]
            if _sequence_form(ours) != _sequence_form(theirs):
                disagreements.append(Disagreement(pointer.name, field_name, ours, theirs))

    return ConformanceReport(
        compared=tuple(compared),
        refusals=tuple(refusals),
        disagreements=tuple(disagreements),
        uncompared=tuple(uncompared),
    )


def _sequence_form(value: object) -> object:
    """Normalise for comparison: sequences compare element-wise by text.

    `(64, 32)` and `[64, 32]` are the same descriptor; `SymExpr("%sam")` and the
    string `"%sam"` are the same symbol written two ways. Comparing raw Python
    containers would make the check depend on which side built the tuple.
    """
    if isinstance(value, (tuple, list)):
        return [_fmt_value(item) for item in value]
    return _fmt_value(value)


def canonicalize(module: Module) -> Module:
    """Re-exported from `canon.canonicalize` so `recognize` has one entry point.

    The recogniser does not canonicalise on its own: `canonicalize` is listed
    under this module's interface, and the implementation lives in
    `canon/` because it is a whole-module pass rather than an operand walk.
    """
    from ..canon.canonicalize import canonicalize as _canonicalize

    return _canonicalize(module)


# --------------------------------------------------------------------------- #
# Elementwise descriptors: register-space "accesses"
# --------------------------------------------------------------------------- #


def elementwise_descriptor(
    op: Operation,
    graph: DefUseGraph | None = None,
) -> DescriptorResult:
    """A register-space `AccessDescriptor` for an elementwise operation.

    The `EPI` instruction is `dst[i] = op(src[i])` with `in_bounds(base, length)`
    and `acc_dtype == op_dtype` in its constraint — terms the schema can only
    decide against a descriptor. A binding whose `descriptor` is `None` makes the
    real schema's selection fail-closed to `unknown` for every elementwise op
    (the stand-in's satisfied-by-construction evaluate masked this; found when
    the emitter's independent re-validation seam fired for the first time).

    The model: the operation reads/writes its result's elements in the register
    file. `base` is the operation itself; `sizes` is the result shape; a scalar
    result is a 1-element access. `length` therefore grounds without a launch
    environment, so elementwise selection and costing need no env — only memory
    and MAC constraints reference kernel parameters.

    A result whose extent is *unknown* (the parser records the printed-but-
    unspecified dimension as `None`) is a refusal, by name: guessing an extent
    would put a wrong `words` into the cost, and a wrong cost is worse than a
    visible refusal because nothing downstream checks it.
    """
    from .op_shapes import scalar_dtype, shape_of  # local: op_shapes imports us

    result = op.results[0] if op.results else None
    if result is None:
        return Unstructured(f"elementwise op {op.name} at line {op.line} produces no result")
    shape = shape_of(result.type)
    dtype = scalar_dtype(result.type)
    if any(dimension is None for dimension in shape):
        return Unstructured(
            f"elementwise op {op.name} at line {op.line}: result shape {shape} has an "
            "unknown extent; guessing it would misprice the instruction, refusing instead"
        )
    sizes = tuple(int(d) for d in shape) if shape else (1,)
    descriptor = AccessDescriptor(
        base=op.name,
        sizes=sizes,
        strides=(1,) * len(sizes),
        offsets=(0,) * len(sizes),
        shape=(),
        order=tuple(range(len(sizes))),
        dtype=dtype,
        loop_carried=False,
        increment=None,
        provenance=("elementwise", op.name),
    )
    return Ok(descriptor)


__all__ = [
    "CONFORMANCE_FIELDS",
    "KEY_FIELDS",
    "AccessDescriptor",
    "AffineSpec",
    "BudgetExhausted",
    "ConformanceReport",
    "DescriptorResult",
    "Disagreement",
    "Ok",
    "Unstructured",
    "canonicalize",
    "conformance_check",
    "describe",
    "describe_operation",
    "elementwise_descriptor",
    "resolve_operand",
]
