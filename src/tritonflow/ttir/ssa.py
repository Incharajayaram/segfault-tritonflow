"""The parsed IR model — the semantic half of the parser/IR seam.

`RawModule` (the syntax layer) carries text: SSA names as written, types as strings,
attributes as strings, no terminator special-casing. `Module` carries
*semantics*: every operand is a resolved :class:`SsaValue`, every value has
exactly one defining operation, every type is parsed, every `loc` reference is
bound against the module's table.

The invariants of the semantic model, and where each is enforced:

| Invariant | Enforced by |
|---|---|
| every `SsaValue` has exactly one defining `Operation` | `to_ir.build_ir` |
| every operand reference resolves, or is a function argument | `to_ir.build_ir` |
| `def_op` is `None` for a block argument | construction here |
| `results` is a list — a multi-result op binds all of them | `to_ir.bind_results` |
| regions are never flattened | construction here + `graph.walk_region` |

**Two representation choices worth stating, because both are visible from
outside.**

*Parent links.* A `Region` knows the `Operation` that owns it, which means
`Module → Region → Block → Operation → Region` is a cycle. Two consequences
were handled rather than discovered later: `parent` is `compare=False,
repr=False`, so `repr` and `==` over a module terminate and stay deterministic
(the round-trip test depends on this), and it is filled in *after* construction with
`object.__setattr__` — the one sanctioned escape hatch for a back-reference in
an otherwise immutable tree. Nothing else in these dataclasses is mutable.

*Operand tokens.* `scf.for`'s operands are `%c0_i32, %1, %c1_i32, …` but
`tt.get_program_id`'s operand is `x` and `arith.cmpi`'s first operand is `slt` —
op spelling, not values. `Operation.operands` holds only resolved
:SsaValue:`s`, and :attr:`Operation.tokens` keeps the operand
list exactly as written so the `x` and the `slt` are not silently dropped:
`operands` is `tokens` filtered to the entries that name a value, in order. That
positional relationship is asserted by a test, because it is the kind of thing
that quietly stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import ParseDiagnostic
from .types import TypeExpr

#: Kinds that mean "a kernel definition" rather than "an operation".
FUNCTION_OP = "tt.func"


@dataclass(frozen=True)
class Loc:
    """One `loc` binding. `line`/`col` are `None` when the table entry did not
    carry them (`loc("a_ptrs")` has no position; `loc("file.py":31:0)` does)."""

    name: str
    line: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class Attr:
    """One attribute, value kept as text until a consumer needs more.

    Text is deliberate: `{inputPrecision = tf32}` and `{axis = 1 : i32}` are
    different value grammars, and turning both into Python numbers here would
    put a second, silent parser in the IR layer.
    """

    name: str
    value: str


@dataclass(frozen=True)
class SsaValue:
    """A value: a block argument (`def_op is None`) or an operation result."""

    name: str
    type: TypeExpr
    def_op: Operation | None = field(default=None, compare=False, repr=False)
    loc: Loc | None = None
    index: int = 0
    """Slot within the defining operation's results. `0` for a block argument.

 Together with:attr:`arity` this is what makes decidable: a
    multi-result value is referenced as `%acc_25#2`, and the *bare* `%acc_25`
    is only ever the first slot.
    """

    arity: int = 1
    """Number of results of the defining operation (`1` for a block argument).

    A use of a value with `arity > 1` and `index == 0` is an arity mismatch:
    the printed base name is not a name for all three results.
    """

    type_is_inferred: bool = False
    """Whether `type.raw` is the type the printer *omitted* for this result.

    The printer drops a result type it can infer, so for a single-result
    operation with no `->` type the one type after the colon is recorded here
    and flagged. For `tt.load` that text describes the operand pointer, not the
    loaded element: the element is `type.element`, and deriving it is the
    recogniser's job rather than a guess made in the IR layer.
    """

    @property
    def is_block_arg(self) -> bool:
        return self.def_op is None

    @property
    def is_multi_result(self) -> bool:
        return self.arity > 1


@dataclass(frozen=True)
class Block:
    """A basic block. `terminator` is the last operation, recorded as a fact
    about position — which operation *is* a terminator is not this layer's call."""

    args: tuple[SsaValue, ...] = ()
    operations: tuple[Operation, ...] = ()
    terminator: Operation | None = field(default=None, compare=False, repr=False)

    @property
    def name(self) -> str:
        return f"block[{len(self.args)} args, {len(self.operations)} ops]"


@dataclass(frozen=True)
class Region:
    """A region: one or more blocks, owned by an operation. Never flattened."""

    blocks: tuple[Block, ...] = ()
    parent: Operation | None = field(default=None, compare=False, repr=False)

    @property
    def entry(self) -> Block | None:
        return self.blocks[0] if self.blocks else None

    @property
    def owner_name(self) -> str:
        return self.parent.name if self.parent is not None else "<module>"


@dataclass(frozen=True)
class Operation:
    """A resolved operation. `results` is a list, not a value."""

    name: str
    operands: tuple[SsaValue, ...] = ()
    results: tuple[SsaValue, ...] = ()
    attributes: dict[str, Attr] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()
    loc: Loc | None = None
    line: int = 0
    col: int = 0
    tokens: tuple[str, ...] = ()
    """The operand list exactly as written — see the module docstring.

    `operands` are the `tokens` that name a value, in the same relative order,
    so `x`, `y` and `slt` survive here instead of vanishing.
    """

    # -- queries used by recognition and emission ----------------------------

    @property
    def literal_tokens(self) -> tuple[str, ...]:
        """Operand tokens that are not value references (`x`, `y`, `slt`)."""
        return tuple(t for t in self.tokens if not t.startswith("%"))

    @property
    def is_multi_result(self) -> bool:
        return len(self.results) > 1

    @property
    def attr(self) -> dict[str, str]:
        """Raw attribute values, for callers that only want the text."""
        return {name: a.value for name, a in self.attributes.items()}

    def operand_named(self, name: str) -> SsaValue | None:
        for value in self.operands:
            if value.name == name:
                return value
        return None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.name} @line {self.line}"


@dataclass(frozen=True)
class Function:
    """A view over one `tt.func` operation.

    A *view*, not a copy: `Module.body` already contains the operation, and
    storing the same kernel twice is how two representations of one thing start
    disagreeing. Everything here delegates to :attr:`op`.
    """

    op: Operation

    @property
    def symbol(self) -> str:
        """The symbol name exactly as written, `@` included."""
        return self.op.attributes.get("sym_name", Attr("sym_name", "")).value

    @property
    def name(self) -> str:
        return self.symbol.lstrip("@")

    @property
    def args(self) -> tuple[SsaValue, ...]:
        entry = self.body.entry if self.body is not None else None
        return entry.args if entry is not None else ()

    @property
    def result_types(self) -> tuple[TypeExpr, ...]:
        return tuple(v.type for v in self.op.results)

    @property
    def body(self) -> Region | None:
        return self.op.regions[0] if self.op.regions else None

    @property
    def return_type_text(self) -> tuple[str, ...]:
        """The `-> (…)` type list as written, if the printer emitted one."""
        return tuple(v.type.raw for v in self.op.results)

    @property
    def loc(self) -> Loc | None:
        return self.op.loc


@dataclass(frozen=True)
class Module:
    """A whole module: one region plus the `#loc` table, kept not discarded."""

    body: Region
    loc_table: dict[str, Loc] = field(default_factory=dict)
    source_path: str = "<string>"
    triton_version: str | None = None

    def find_function(self, name: str) -> Function | None:
        """The kernel named `name`, with or without its leading `@`."""
        wanted = name.lstrip("@")
        for op in self.body.blocks[0].operations if self.body.blocks else ():
            if op.name != FUNCTION_OP:
                continue
            if op.attributes.get("sym_name", Attr("sym_name", "")).value.lstrip("@") == wanted:
                return Function(op)
        return None

    def functions(self) -> tuple[Function, ...]:
        return tuple(
            Function(op)
            for op in (self.body.blocks[0].operations if self.body.blocks else ())
            if op.name == FUNCTION_OP
        )


@dataclass(frozen=True)
class ParseResult:
    """Exactly one of the two halves is set."""

    module: Module | None = None
    diagnostic: ParseDiagnostic | None = None

    def __post_init__(self) -> None:
        if (self.module is None) == (self.diagnostic is None):
            raise ValueError(
                "ParseResult must carry exactly one of module/diagnostic; "
                f"got module={'set' if self.module is not None else 'None'}, "
                f"diagnostic={'set' if self.diagnostic is not None else 'None'}"
            )

    @property
    def ok(self) -> bool:
        return self.module is not None

    def unwrap(self) -> Module:
        """The module, or raise the diagnostic as a ValueError.

        For callers that already established `ok` — reading it is the difference
        between a decision and a `None`-check that someone eventually forgets.
        """
        if self.module is None:  # pragma: no cover - guarded by unwrap()'s callers
            raise ValueError(f"no module: {self.diagnostic}")
        return self.module


__all__ = [
    "FUNCTION_OP",
    "Attr",
    "Block",
    "Function",
    "Loc",
    "Module",
    "Operation",
    "ParseDiagnostic",
    "ParseResult",
    "Region",
    "SsaValue",
]
