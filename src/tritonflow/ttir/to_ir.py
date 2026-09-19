"""`RawModule` → `Module`. Where text becomes semantics.

There is one line between syntax and semantics, and this module is the semantic
side of it: the syntax layer refuses what it cannot *read*, and this module
refuses what it can read but cannot *mean*. Every diagnostic raised here
carries ``layer="ir"`` so a failing test names the owning layer instead of
starting a negotiation.

What happens here, in the order the invariants need it:

1. **The syntax verdict is passed through, not re-derived.** If `RawModule.diagnostics`
   is non-empty the module is already unusable; returning the
   syntax layer's own diagnostic keeps the two failure routes from blurring.
2. **`loc` references are bound** against the `#loc` table. A reference with no
   entry is invalid IR, not a syntax error — the text was read fine, it just
   does not resolve.
3. **Operands resolve.** A `%name` that no definition provides is invalid IR,
   and a *bare* multi-result name is an arity mismatch:
   `%acc_25:3 = scf.for …` defines three values named `%acc_25`,
   `%acc_25#1`, `%acc_25#2`, so `arith.addi %acc_25, %acc_25` is arity 1 against
   a 3-result definition.
4. **Every result is bound** and every result name is defined exactly
   once — never last-def-wins.
5. **Every type is parsed**, and a type that is not a type is invalid IR
   (``tensor<64x64x???>``) while a dynamic shape (``tensor<?x64xf32>``) is a
   legal type with an unknown extent. The distinction is the whole reason
   :func:`~tritonflow.ttir.types.parse_type` never judges.

**The result-type rule, stated once.** The printer omits a result type when it
can be inferred, so for a single-result operation with no ``->`` type the single
type after the colon is recorded and marked
:attr:`~tritonflow.ttir.ssa.SsaValue.type_is_inferred`. For `tt.load` that
text is the *operand* pointer type, not the loaded element type; the element is
available as ``value.type.element`` and is deliberately not fabricated here,
because inventing the loaded element type is semantics that belongs to the
recogniser, next to the op that defines it.
"""

from __future__ import annotations

import os
from pathlib import Path

# `build_def_use`, `walk_region` and `topo_within_region` are re-exported so the
# graph-traversal interface is reachable from this module too. The definitions
# live in `graph.py`, next to the traversal they belong with — one
# implementation, one import surface.
from .graph import build_def_use, topo_within_region, walk_region
from .parser import ParseDiagnostic, RawModule, RawOp, parse_raw
from .ssa import Attr, Block, Loc, Module, Operation, ParseResult, Region, SsaValue
from .types import parse_type

BOM = "\ufeff"


def parse_module(text: str, *, source_path: str = "<string>") -> ParseResult:
    """The public entry point: arbitrary text in, exactly one half out.

    Never raises. Composes the two layers in the order the seam
    requires — syntax first, then semantics — because running the semantic pass
    over text that could not be read would produce diagnostics about the wrong
    thing.
    """
    return build_ir(parse_raw(text, source_path=source_path))


def parse_file(path: str | os.PathLike[str]) -> ParseResult:
    """`parse_module` for a file, handling BOM and CRLF.

    Decoding is explicit and tolerant: a fixture checked out on another platform
    must not become a parse failure.
    """
    target = Path(path)
    text = target.read_bytes().decode("utf-8", errors="replace")
    if text.startswith(BOM):
        text = text[len(BOM) :]
    return parse_module(text, source_path=str(target))


def build_ir(raw: RawModule, *, raise_on_invalid: bool = False) -> ParseResult:
    """Turn a `RawModule` into a `Module`, or into an `ir`-layer diagnostic.

    With `raise_on_invalid`, an invalid module raises `ValueError` carrying the
    diagnostic. It is off by default so the normal route is a returned result
    (`raise` is for tests and for a CLI that wants a traceback).
    """
    if raw.diagnostics:
        # The syntax layer already refused this. Report *its* diagnostic: a
        # semantic complaint about a module nobody could read would name the
        # wrong owner and hide the real problem.
        return ParseResult(diagnostic=raw.diagnostics[0])

    builder = _Builder(raw)
    module = builder.run()
    if module is not None:
        return ParseResult(module=module)

    diagnostic = builder.first_diagnostic()
    if raise_on_invalid:
        raise ValueError(
            f"invalid IR at {diagnostic.line}:{diagnostic.col}: "
            f"expected {diagnostic.expected}, found {diagnostic.found} "
            f"(layer={diagnostic.layer})"
        )
    return ParseResult(diagnostic=diagnostic)


def bind_results(op: RawOp) -> list[SsaValue]:
    """One `SsaValue` per result of `op` — **every** result, not the first.

    The Tier-1 loop binds three (`%acc_25`, `%acc_25#1`, `%acc_25#2`), the
    multi-result case this function exists for. A discarded result position
    (`None`, which is allowed) produces no value but still advances
    :attr:`SsaValue.index`, so the surviving values keep the slot they had in
    the printed form.
    """
    arity = len(op.results)
    values: list[SsaValue] = []
    for index, name in enumerate(op.results):
        if name is None:
            continue  # a discarded result has no name to bind
        text, inferred = _result_type_text(op, index)
        values.append(
            SsaValue(
                name=name,
                type=parse_type(text),
                def_op=None,
                loc=None,
                index=index,
                arity=arity,
                type_is_inferred=inferred,
            )
        )
    return values


def _result_type_text(op: RawOp, index: int) -> tuple[str, bool]:
    """The type text for result `index`, and whether it was inferred.

    See the module docstring: an omitted result type is recorded from the type
    list after the colon and flagged, never invented.

    The list after the colon is the *operand* types, in operand order, and the
    printer omits the result type only when it is the first operand's
    (`arith.muli %a, %b : i32`, `tt.addptr %p, %off : tensor<…>, tensor<…>`).
    So a single result with no printed type takes ``operand_types[0]``. This is
    the corpus's whole unprinted-result population: every result-bearing
    operation in all four fixtures has either a result type or at least one
    operand type, which is why an empty result here still means invalid IR.
    """
    if index < len(op.result_types):
        return op.result_types[index], False
    if len(op.results) == 1 and op.operand_types:
        return op.operand_types[0], True
    return "", False


class _Builder:
    """One pass over a `RawModule`. Collects diagnostics instead of raising."""

    def __init__(self, raw: RawModule) -> None:
        self.raw = raw
        self.loc_table: dict[str, Loc] = {
            ref: Loc(name=loc.name, line=loc.line, col=loc.col)
            for ref, loc in raw.loc_table.items()
        }
        self.values: dict[str, SsaValue] = {}
        self.diagnostics: list[ParseDiagnostic] = []

    # -- diagnostics ---------------------------------------------------------

    def fail(self, op: RawOp, expected: str, found: str) -> None:
        self.diagnostics.append(
            _diagnostic(line=op.line, col=op.col, expected=expected, found=found)
        )

    def first_diagnostic(self) -> ParseDiagnostic:
        """The earliest diagnostic, deterministically.

        `ParseResult` carries one diagnostic, so which one is a decision the
        layer has to make explicitly: earliest position, then insertion order.
        Picking "whatever the walk happened to report first" would make the
 message depend on traversal and break.
        """
        return min(
            self.diagnostics,
            key=lambda d: (d.line, d.col),
        )

    # -- the pass ------------------------------------------------------------

    def run(self) -> Module | None:
        ops: list[Operation] = []
        for raw_op in self.raw.ops:
            built = self.operation(raw_op)
            if built is not None:
                ops.append(built)

        if self.diagnostics:
            return None

        block = Block(args=(), operations=tuple(ops), terminator=ops[-1] if ops else None)
        body = Region(blocks=(block,))
        for op in ops:
            _attach_parents(op)
        module = Module(
            body=body,
            loc_table=self.loc_table,
            source_path=self.raw.source_path,
            triton_version=self.raw.triton_version,
        )
        # The module's own region is not owned by an operation; every region
        # inside it is.
        return module

    def operation(self, raw_op: RawOp) -> Operation | None:
        """Build one operation, or `None` when it was reported invalid."""
        loc = self.resolve_loc(raw_op)
        operands = self.resolve_operands(raw_op)
        results = bind_results(raw_op)
        attributes = {name: Attr(name=name, value=value) for name, value in raw_op.attrs.items()}

        for value in results:
            if not value.type.is_valid:
                self.fail(
                    raw_op,
                    expected="a type",
                    found=f"{value.type.raw or '<missing>'!r} for result {value.name}",
                )
            previous = self.values.get(value.name)
            if previous is not None:
                self.fail(
                    raw_op,
                    expected="exactly one definition per SSA name (never last-def-wins)",
                    found=f"{value.name} defined twice (first at line {previous.def_op.line if previous.def_op else '?'})",
                )
            self.values[value.name] = value

        regions = tuple(self.region(region, raw_op) for region in raw_op.regions)

        built = Operation(
            name=raw_op.name,
            operands=tuple(operands),
            results=tuple(results),
            attributes=attributes,
            regions=regions,
            loc=loc,
            line=raw_op.line,
            col=raw_op.col,
            tokens=tuple(t for t in raw_op.operands),
        )
        # Results point back at their definition: `def_op` is a back-reference,
        # so it is filled in after construction (see the ssa module docstring).
        for value in results:
            _set_def_op(value, built)
        return built

    def region(self, raw_region, owner: RawOp) -> Region:
        blocks: list[Block] = []
        for raw_block in raw_region.blocks:
            args = tuple(self.block_arg(name, text, owner) for name, text in raw_block.args)
            ops = [
                built for raw_op in raw_block.ops if (built := self.operation(raw_op)) is not None
            ]
            index = raw_block.terminator_index
            terminator = ops[index] if 0 <= index < len(ops) else (ops[-1] if ops else None)
            blocks.append(Block(args=args, operations=tuple(ops), terminator=terminator))
        return Region(blocks=tuple(blocks))

    def block_arg(self, name: str, type_text: str, owner: RawOp) -> SsaValue:
        value = SsaValue(
            name=name,
            type=parse_type(type_text),
            def_op=None,
            index=0,
            arity=1,
        )
        if not value.type.is_valid:
            self.fail(
                owner,
                expected="a type for a block argument",
                found=f"{type_text!r} for {name}",
            )
        if name in self.values:
            self.fail(
                owner,
                expected="exactly one definition per SSA name",
                found=f"block argument {name} shadows an existing value",
            )
        self.values[name] = value
        return value

    # -- loc -----------------------------------------------------------------

    def resolve_loc(self, raw_op: RawOp) -> Loc | None:
        raw_loc = raw_op.loc
        if raw_loc is None:
            return None
        if raw_loc.name.startswith("#"):
            found = self.loc_table.get(raw_loc.name)
            if found is None:
                self.fail(
                    raw_op,
                    expected="a `#loc` reference that the table defines",
                    found=raw_loc.name,
                )
            return found
        return Loc(name=raw_loc.name, line=raw_loc.line, col=raw_loc.col)

    # -- operands ------------------------------------------------------------

    def resolve_operands(self, raw_op: RawOp) -> list[SsaValue]:
        """Resolve the value references; literal tokens are not operands.

        `tt.get_program_id x` and `arith.cmpi slt, …` carry op spelling where an
        operand list would be. They stay in `Operation.tokens` and are not
        resolved, because there is nothing to resolve them against.
        """
        resolved: list[SsaValue] = []
        for token in raw_op.operands:
            if not token.startswith("%"):
                continue
            value = self.values.get(token)
            if value is None:
                self.fail(
                    raw_op,
                    expected="an operand that some operation defines",
                    found=token,
                )
                continue
            if value.arity > 1 and value.index == 0:
                self.fail(
                    raw_op,
                    expected=(
                        f"one of the {value.arity} names the definition printed "
                        f"({value.name}, {value.name}#1, …)"
                    ),
                    found=f"{token} used with arity 1",
                )
                continue
            resolved.append(value)
        return resolved


def _set_def_op(value: SsaValue, op: Operation) -> None:
    """Give a result its back-reference (see the ssa module docstring)."""
    object.__setattr__(value, "def_op", op)


def _attach_parents(op: Operation) -> None:
    """Point every region at the operation that owns it.

    Done after construction because `Operation` is frozen: the region must exist
    before the operation does, and the back-reference must name the *final*
    operation object rather than a discarded intermediate.
    """
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                _attach_parents(child)
        object.__setattr__(region, "parent", op)


def _diagnostic(*, line: int, col: int, expected: str, found: str) -> ParseDiagnostic:
    return ParseDiagnostic(
        kind="PARSE_UNSUPPORTED",
        line=line,
        col=col,
        expected=expected,
        found=found,
        snippet="",
        layer="ir",
    )


__all__ = [
    "bind_results",
    "build_def_use",
    "build_ir",
    "parse_file",
    "parse_module",
    "topo_within_region",
    "walk_region",
]
