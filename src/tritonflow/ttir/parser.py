"""`ttir` text -> `RawModule`. The syntax layer's parser, and the parser/IR seam.

This module produces the raw, un-typechecked module (the producer half of the
parser/IR seam). It is **strictly syntactic**: types stay text, attribute
values stay text, operands stay SSA names as written, and nothing here decides
what a type means or whether a use is well-typed. Those are `to_ir.py`'s calls, and
the reason the two layers can work independently.

`parse_raw` is **total**: for any string it returns a `RawModule`, never raises,
never hangs. The only way it reports a problem is `diagnostics`.

Text-level decisions this file makes, in one place
--------------------------------------------------
These are the places the raw module's shape is not self-evident. Each one is
structural, not semantic; none of them requires knowing what an op *means*.

1. **`module { … }` is a container, not an operation.** `RawModule.ops` is the
   top level *inside* the wrapper, so the wrapper contributes no `RawOp` and no
   op count. A nested `module` would be an op; the pinned printer does not emit
   one.
2. **The trailing type list is ambiguous, so a rule is stated.** MLIR's
   unambiguous form is `op … : <operand types> -> <result types>`. When there is
   no `->`, a single `:` list binds to the **operands**, except for an op with
   **no SSA operands** (`arith.constant 0 : i32`, `tt.make_range {…} : T`),
   where it binds to the result. This reproduces every op in the corpus
   correctly and needs no per-op table.
3. **Non-SSA operands are recorded verbatim.** `arith.constant dense<2>`,
   `arith.cmpi slt, …` and `tt.get_program_id x` put a literal or a keyword in
   operand position; the text is kept as written rather than dropped, since
   dropping it would lose information the semantic layer (`to_ir.py`) needs and
   inventing a field would change the frozen seam.
4. **`key = value` in the operand clause is an attribute.** `tt.dot …,
   inputPrecision = tf32 : …` has no braces; it still lands in `attrs`, which is
   where the recogniser reads `input_precision` from.
5. **A `{` after an operand clause is a region or an attribute dictionary.**
   Decided by content, not by op name: `{end = 64 : i32, …}` starts with
   `name =` and is a dictionary; `{ %x = arith… }` starts with an SSA name or a
   dotted op and is a region; `{}` is an empty region.
6. **`tt.func`'s parameters are the first block's arguments.** The declaration
   form (`[visibility] @symbol(params)`) is recognised structurally — a symbol
   token followed by a parenthesis — and its `sym_name`/`visibility` land in
   `attrs` under those two reserved keys. This is the only way the symbol name
   can reach the semantic layer without adding a field to the frozen dataclass.
7. **`loc` is carried, not resolved.** `loc(#loc25)` is recorded as
   `RawLoc(name="#loc25")`, i.e. the reference as written; the table entry
   `#loc25 = loc("rm"(#loc1))` is recorded as `RawLoc(name="rm")`. Resolving a
   reference against the table is a semantic decision (`loc` lookup happens in
   `to_ir.py`, and a missing key is an invalid-IR error), not a syntactic one.
   Forms with no simple name (`loc(unknown)`, `loc(callsite(…))`) keep their
   inner text verbatim rather than being coerced into a name.
8. **The terminator is positional.** `RawBlock.terminator_index` is the index of
   the last operation, whatever it is: MLIR requires a terminator to be last, and
   naming it would mean claiming that `scf.yield` is special — which is a
   semantic judgement this module does not make.
9. **One diagnostic, then stop.** A syntax failure makes the module unusable
   (a non-empty `diagnostics` means the module is unusable), so the
   parser stops at the first one instead of emitting a cascade of guesses. Every
   diagnostic carries a line and, where cheap, a column, and `layer="syntax"`.
10. **Nesting is bounded.** Region nesting deeper than `MAX_REGION_DEPTH` is
    reported as a diagnostic rather than recursing until `RecursionError`.
    The limit is far above anything the pinned printer emits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import lexer as lx

# --------------------------------------------------------------------------
# Frozen seam. Field definitions only.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawLoc:
    """A source location: a name, and a position when the printer gave one.

    `name` is a Python variable name (`"a_ptrs"`), a path (`"k.mlir"`), a table
    reference as written (`"#loc25"`), `"unknown"`, or — for forms with no name
    at all — the inner text verbatim (`"callsite(#loc1 at #loc2)"`).
    """

    name: str
    line: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class ParseDiagnostic:
    """The one way this layer reports failure (`kind` is always the marker)."""

    kind: str = "PARSE_UNSUPPORTED"
    line: int = 0
    col: int = 0
    expected: str = ""
    found: str = ""
    snippet: str = ""
    layer: str = "syntax"


@dataclass(frozen=True)
class RawOp:
    """One operation, exactly as written. No interpretation."""

    name: str
    results: list[str | None]
    operands: list[str]
    attrs: dict[str, str]
    result_types: list[str]
    operand_types: list[str]
    regions: list[RawRegion]
    loc: RawLoc | None
    line: int
    col: int


@dataclass(frozen=True)
class RawBlock:
    """A region's basic block. `ops[-1]` is the terminator, by position."""

    args: list[tuple[str, str]]
    ops: list[RawOp]
    terminator_index: int


@dataclass(frozen=True)
class RawRegion:
    """A `{ … }` region. Nested regions stay nested."""

    blocks: list[RawBlock]


@dataclass(frozen=True)
class RawModule:
    """The whole parse. `diagnostics` non-empty means: unusable, refuse it."""

    ops: list[RawOp]
    loc_table: dict[str, RawLoc]
    source_path: str
    triton_version: str | None
    diagnostics: list[ParseDiagnostic]


#: Region nesting deeper than this is a diagnostic, not a `RecursionError`.
MAX_REGION_DEPTH = 96
#: `%name:1000000` must not allocate a million strings.
MAX_RESULT_ARITY = 256

_VISIBILITY = ("public", "private", "nested")
# Words that END an operand clause. Everything else in that position is an
# operand value (`slt`, `x`, …). `to`/`step` are deliberately absent: they are
# skipped *inside* the clause by `parse_operand_clause`, and listing them here
# would truncate the `scf.for` header at its lower bound.
_CLAUSE_KEYWORDS = ("loc", "attributes", "else", "module")
_OPERAND_STOP_PUNCT = ":{})]="


class _Parser:
    """Recursive descent over the token stream, with a guarded recursion depth."""

    def __init__(self, text: str, source_path: str) -> None:
        self.text = text
        self.source_path = source_path
        self.toks = lx.tokenize(text)
        self.i = 0
        self.diags: list[ParseDiagnostic] = []
        self.loc_table: dict[str, RawLoc] = {}

    # ---------------------------------------------------------------- tokens
    def peek(self, k: int = 0) -> lx.Token:
        j = self.i + k
        if j >= len(self.toks):
            return self.toks[-1]
        return self.toks[j]

    def at_end(self) -> bool:
        return self.peek().kind == lx.EOF

    def text_of(self, lo: int, hi: int) -> str:
        """The original source text covering token indices `[lo, hi)`."""
        if hi <= lo:
            return ""
        return self.text[self.toks[lo].offset : self.toks[hi - 1].end_offset].strip()

    def fail(self, expected: str, *, found: str | None = None, tok: lx.Token | None = None) -> None:
        t = tok or self.peek()
        if self.diags:
            return
        lines = self.text.splitlines()
        snippet = lines[t.line - 1].strip() if 1 <= t.line <= len(lines) else ""
        self.diags.append(
            ParseDiagnostic(
                kind="PARSE_UNSUPPORTED",
                line=t.line,
                col=t.col,
                expected=expected,
                found=found if found is not None else (t.text or t.kind),
                snippet=snippet,
                layer="syntax",
            )
        )

    # ------------------------------------------------------------- utilities
    def split_top_level(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """Split `[lo, hi)` on `,` and `*` at bracket depth 0.

        `*` is a type-list separator in the printer's `tt.dot` form
        (`A * B -> C`), not an operator, so it splits with the comma.
        """
        parts: list[tuple[int, int]] = []
        depth = 0
        start = lo
        for k in range(lo, hi):
            t = self.toks[k]
            if t.kind != lx.PUNCT:
                continue
            if t.text in "(<[":
                depth += 1
            elif t.text in ")>]":
                depth -= 1
            elif depth == 0 and t.text in (",", "*"):
                parts.append((start, k))
                start = k + 1
        parts.append((start, hi))
        return [(a, b) for a, b in parts if a < b]

    def find_matching(self, opening: int, open_ch: str, close_ch: str) -> int:
        """Index of the `close_ch` matching the bracket at `opening`, else -1."""
        depth = 0
        for k in range(opening, len(self.toks)):
            t = self.toks[k]
            if t.kind != lx.PUNCT:
                continue
            if t.text == open_ch:
                depth += 1
            elif t.text == close_ch:
                depth -= 1
                if depth == 0:
                    return k
        return -1

    def skip_bracket_group(self) -> int:
        """Advance past a balanced bracket group starting at the current token."""
        t = self.peek()
        pairs = {"(": ")", "[": "]", "{": "}"}
        close = pairs.get(t.text)
        if close is None:
            self.i += 1
            return self.i
        end = self.find_matching(self.i, t.text, close)
        if end < 0:
            self.i = len(self.toks) - 1
        else:
            self.i = end + 1
        return self.i

    # ----------------------------------------------------------------- entry
    def parse(self) -> RawModule:
        ops: list[RawOp] = []
        try:
            ops = self.parse_unit()
        except RecursionError:  # pragma: no cover - guarded by MAX_REGION_DEPTH
            self.fail("a shallower region nesting")
        except Exception as exc:
            # Catching everything is deliberate, not laziness: `parse_raw` is
            # total, so even a bug in here has to come out as a
            # diagnostic rather than as a traceback in a teammate's pipeline. The
            # message says it was internal, so it is not mistaken for a verdict
            # on the input.
            self.fail(f"a parseable module (internal {type(exc).__name__}: {exc})")
        return RawModule(
            ops=ops,
            loc_table=self.loc_table,
            source_path=self.source_path,
            triton_version=None,
            diagnostics=self.diags,
        )

    def parse_unit(self) -> list[RawOp]:
        """Top level: `#loc` entries, the `module` container, and its contents."""
        ops: list[RawOp] = []
        while not self.at_end() and not self.diags:
            before = self.i
            t = self.peek()
            if t.kind == lx.ERROR:
                self.fail("a lexable token", found=f"lexical error: {t.text}", tok=t)
                break
            if t.kind == lx.LOCREF and self.peek(1).text == "=":
                self.parse_loc_entry()
            elif t.kind == lx.IDENT and t.text == "module":
                self.i += 1
                self.parse_container(ops)
            elif t.kind == lx.PUNCT and t.text == "}":
                self.fail("an operation or a `#loc` entry", found="'}' that closes nothing")
                break
            else:
                op = self.parse_op(depth=0)
                if op is not None:
                    ops.append(op)
            if self.i == before and not self.diags:
                self.fail("the parser to make progress", found="no token consumed")
        return ops

    def parse_container(self, ops: list[RawOp]) -> None:
        """`module { … } [loc(…)]` — a container, contributing no `RawOp`."""
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "{"):
            self.fail("'{' after `module`")
            return
        self.i += 1
        ops.extend(self.parse_ops(depth=1))
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "}"):
            self.fail("'}' to close `module`")
            return
        self.i += 1
        self.maybe_loc()

    def parse_ops(self, depth: int) -> list[RawOp]:
        """Operations up to (not including) the closing `}` or EOF.

        Used for the `module` container, whose contents are the top level.
        """
        out: list[RawOp] = []
        while not self.at_end() and not self.diags:
            t = self.peek()
            if t.kind == lx.PUNCT and t.text == "}":
                break
            if t.kind == lx.ERROR:
                self.fail("a lexable token", found=f"lexical error: {t.text}", tok=t)
                break
            before = self.i
            op = self.parse_op(depth)
            if op is not None:
                out.append(op)
            if self.i == before and not self.diags:
                self.fail("the parser to make progress", found="no token consumed")
        return out

    # -------------------------------------------------------------- an op
    def parse_op(self, depth: int) -> RawOp | None:
        start = self.peek()
        line, col = start.line, start.col
        results = self.parse_lhs()
        name_tok = self.peek()
        if name_tok.kind not in (lx.OPNAME, lx.IDENT, lx.STRING):
            self.fail("an operation name", tok=name_tok)
            return None
        self.i += 1
        name = name_tok.text.strip('"')
        return self.parse_op_body(name, results, line, col, depth)

    @staticmethod
    def is_discard(t: lx.Token) -> bool:
        """The `_` name, i.e. a result position nobody wanted."""
        return t.kind == lx.IDENT and t.text == "_"

    def parse_lhs(self) -> list[str | None] | None:
        """`%a =` / `%a, %b =` / `%acc:3 =` / `%a, _ =`, or `None` if no definition.

        Multi-result arity is carried, not judged: `%acc_25:3` becomes three
        names, with the printer's own `#N` convention, which is what the corpus
        uses when it refers back to `%acc_25#2`. A `_` position becomes `None`,
        which is why `results` is typed `list[str | None]`.
        """
        if self.peek().kind != lx.SSA and not self.is_discard(self.peek()):
            return None
        j = self.i
        names: list[lx.Token] = []
        while True:
            tk = self.toks[j] if j < len(self.toks) else self.toks[-1]
            if tk.kind == lx.SSA or self.is_discard(tk):
                names.append(tk)
                j += 1
            else:
                return None
            nxt = self.toks[j] if j < len(self.toks) else self.toks[-1]
            if nxt.kind == lx.PUNCT and nxt.text == ",":
                j += 1
                continue
            break
        eq = self.toks[j] if j < len(self.toks) else self.toks[-1]
        if not (eq.kind == lx.PUNCT and eq.text == "="):
            return None

        out: list[str | None] = []
        for tk in names:
            if self.is_discard(tk):
                out.append(None)
                continue
            base, sep, arity = tk.text.partition(":")
            if sep:
                try:
                    n = int(arity)
                except ValueError:
                    self.fail("a numeric multi-result arity", found=tk.text, tok=tk)
                    return None
                if n > MAX_RESULT_ARITY:
                    self.fail(
                        f"a multi-result arity of at most {MAX_RESULT_ARITY}",
                        found=tk.text,
                        tok=tk,
                    )
                    return None
                out.append(base)
                out.extend(f"{base}#{k}" for k in range(1, max(n, 1)))
            else:
                out.append(None if base == "_" else base)
        self.i = j + 1
        return out

    def parse_op_body(
        self,
        name: str,
        results: list[str | None] | None,
        line: int,
        col: int,
        depth: int,
    ) -> RawOp:
        attrs: dict[str, str] = {}
        decl_args = self.parse_declarator(attrs)
        operands, clause_attrs, iv_name, iter_names, has_ssa = self.parse_operand_clause()
        attrs.update(clause_attrs)

        result_types: list[str] = []
        colon_types: list[str] = []
        saw_arrow = False
        regions: list[RawRegion] = []
        loc: RawLoc | None = None

        while not self.at_end() and not self.diags:
            t = self.peek()
            if t.kind == lx.ARROW:
                self.i += 1
                saw_arrow = True
                result_types.extend(self.parse_arrow_types())
                continue
            if t.kind == lx.PUNCT and t.text == ":":
                self.i += 1
                colon_types.extend(self.parse_type_list())
                continue
            if t.kind == lx.IDENT and t.text == "attributes":
                self.i += 1
                attrs.update(self.parse_brace_dict())
                continue
            if t.kind == lx.PUNCT and t.text == "<" and self.peek(1).text == "{":
                self.i += 1
                attrs.update(self.parse_brace_dict())
                if self.peek().kind == lx.PUNCT and self.peek().text == ">":
                    self.i += 1
                continue
            if t.kind == lx.PUNCT and t.text == "(" and self.peek(1).text == "{":
                self.i += 1
                if depth + 1 > MAX_REGION_DEPTH:
                    self.fail_region_too_deep(depth)
                    break
                first_args = (
                    decl_args
                    if decl_args is not None
                    else self.loop_args(iv_name, iter_names, colon_types, result_types)
                )
                decl_args = None
                regions.append(self.parse_region(depth + 1, first_args))
                if self.peek().kind == lx.PUNCT and self.peek().text == ")":
                    self.i += 1
                continue
            if t.kind == lx.PUNCT and t.text == "{":
                if self.looks_like_attr_dict():
                    attrs.update(self.parse_brace_dict())
                else:
                    if depth + 1 > MAX_REGION_DEPTH:
                        self.fail_region_too_deep(depth)
                        break
                    first_args = (
                        decl_args
                        if decl_args is not None
                        else self.loop_args(iv_name, iter_names, colon_types, result_types)
                    )
                    decl_args = None  # only the first region carries them
                    regions.append(self.parse_region(depth + 1, first_args))
                continue
            if t.kind == lx.IDENT and t.text == "loc" and self.peek(1).text == "(":
                loc = self.parse_loc()
                continue
            if t.kind == lx.IDENT and t.text == "else" and self.peek(1).text == "{":
                self.i += 1
                regions.append(self.parse_region(depth + 1, None))
                continue
            break

        operand_types = colon_types
        if not saw_arrow and colon_types and not has_ssa:
            # Decision 2: an op with no SSA operands prints its *result* type.
            result_types = colon_types
            operand_types = []
        elif not saw_arrow and len(colon_types) == 1 and " to " in colon_types[0]:
            # MLIR cast operations: 
            src_t, dst_t = colon_types[0].split(" to ", 1)
            operand_types = [src_t.strip()]
            result_types = [dst_t.strip()]

        return RawOp(
            name=name,
            results=list(results) if results else [],
            operands=operands,
            attrs=attrs,
            result_types=result_types,
            operand_types=operand_types,
            regions=regions,
            loc=loc,
            line=line,
            col=col,
        )

    @staticmethod
    def loop_args(
        iv_name: str | None,
        iter_names: list[str],
        colon_types: list[str],
        result_types: list[str],
    ) -> list[tuple[str, str]]:
        """The loop body's block arguments: the induction variable, then iter args."""
        args: list[tuple[str, str]] = []
        if iv_name is not None:
            args.append((iv_name, colon_types[0] if colon_types else ""))
        for k, nm in enumerate(iter_names):
            args.append((nm, result_types[k] if k < len(result_types) else ""))
        return args

    def fail_region_too_deep(self, depth: int) -> None:
        self.fail(
            f"region nesting of at most {MAX_REGION_DEPTH} levels",
            found=f"{depth + 1} levels",
        )
        self.skip_region()

    def skip_region(self) -> None:
        """Consume a balanced `{ … }` without building it (depth-limit bail-out)."""
        start = self.peek()
        end = self.find_matching(self.i, "{", "}")
        if end < 0:
            self.i = len(self.toks) - 1
            return
        self.i = end + 1
        if start.kind == lx.PUNCT and start.text != "{":  # pragma: no cover - defensive
            self.i = start.end_offset

    # ------------------------------------------------------- declarator form
    def parse_declarator(self, attrs: dict[str, str]) -> list[tuple[str, str]] | None:
        """`[visibility] @symbol(param, …)` -> typed block arguments.

        Recognised structurally (a symbol followed by a parenthesis), not by op
        name. `sym_name`/`visibility` are the two reserved keys in `attrs`.
        """
        start = self.i
        j = self.i
        visibility = None
        if self.toks[j].kind == lx.IDENT and self.toks[j].text in _VISIBILITY:
            visibility = self.toks[j].text
            j += 1
        if self.toks[j].kind != lx.SYMBOL:
            return None
        symbol = self.toks[j].text
        j += 1
        if not (self.toks[j].kind == lx.PUNCT and self.toks[j].text == "("):
            return None
        self.i = j
        args = self.parse_param_list()
        attrs["sym_name"] = symbol
        if visibility is not None:
            attrs["visibility"] = visibility
        _ = start
        return args

    def parse_param_list(self) -> list[tuple[str, str]]:
        """`(%name: type [loc(…)], …)` -> `[(name, type text)]`, verbatim types."""
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "("):
            self.fail("'(' to open the parameter list")
            return []
        end = self.find_matching(self.i, "(", ")")
        if end < 0:
            self.fail("')' to close the parameter list")
            return []
        args: list[tuple[str, str]] = []
        for lo, hi in self.split_top_level(self.i + 1, end):
            colon = None
            depth = 0
            for k in range(lo, hi):
                t = self.toks[k]
                if t.kind != lx.PUNCT:
                    continue
                if t.text in "(<[":
                    depth += 1
                elif t.text in ")>]":
                    depth -= 1
                elif t.text == ":" and depth == 0:
                    colon = k
                    break
            if colon is None:
                args.append(("", self.text_of(lo, hi)))
                continue
            name = self.text_of(lo, colon)
            args.append((name, self.strip_trailing_loc(colon + 1, hi)))
        self.i = end + 1
        return args

    def strip_trailing_loc(self, lo: int, hi: int) -> str:
        """Drop a trailing `loc(…)` from a type description (the printer's order)."""
        for k in range(hi - 1, lo - 1, -1):
            t = self.toks[k]
            if t.kind == lx.IDENT and t.text == "loc" and self.toks[k + 1].text == "(":
                close = self.find_matching(k + 1, "(", ")")
                if close == hi - 1:
                    return self.text_of(lo, k)
        return self.text_of(lo, hi)

    # --------------------------------------------------------- operand clause
    def parse_operand_clause(
        self,
    ) -> tuple[list[str], dict[str, str], str | None, list[str], bool]:
        """Operands, inline attributes, the induction variable, and `iter_args`.

        Returns `(operands, attrs, iv_name, iter_arg_names, has_ssa_operand)`.
        Keyword-driven, so it is syntax: `iter_args` names become block
        arguments, while their initial values become operands.
        """
        operands: list[str] = []
        attrs: dict[str, str] = {}
        iv_name: str | None = None
        iter_names: list[str] = []
        has_ssa = False
        first = True

        while not self.at_end() and not self.diags:
            t = self.peek()
            if self.stops_operands(t):
                break
            # `scf.for %iv = …` — only ever the first token after the op name.
            if t.kind == lx.SSA and self.peek(1).text == "=":
                if not (first and iv_name is None):
                    self.fail("an operand", found=f"unexpected definition {t.text}")
                    break
                iv_name = t.text
                self.i += 2
                first = False
                continue
            if t.kind == lx.IDENT and t.text in ("to", "step"):
                self.i += 1
                first = False
                continue
            if t.kind == lx.IDENT and t.text == "iter_args" and self.peek(1).text == "(":
                self.i += 1
                names, values = self.parse_iter_args()
                iter_names.extend(names)
                operands.extend(values)
                has_ssa = has_ssa or any(v.startswith("%") for v in values)
                first = False
                continue
            if t.kind == lx.PUNCT and t.text == "^":
                # `^` starts either a branch target (`cf.br ^bb1(%x : i32)`, an
                # operand) or the next block's label (`^bb1(%x: i32):`, which is
                # structure). Only the label is followed by `:`, so that decides.
                # Without this, a branch target eats the label after it and the
                # orphaned `:` is read as a type list.
                after_target = self.after_target()
                if self.is_label_at(after_target):
                    break
                lo = self.i
                self.i = after_target
                operands.append(self.text_of(lo, self.i))
                first = False
                if self.peek().kind == lx.PUNCT and self.peek().text == ",":
                    self.i += 1
                continue
            if t.kind == lx.IDENT and self.peek(1).text == "=" and self.peek(1).kind == lx.PUNCT:
                # Decision 4: `inputPrecision = tf32` inside the operand clause.
                key = t.text
                self.i += 2
                value = self.single_value_text()
                if value:
                    attrs[key] = value
                first = False
                continue
            if t.kind == lx.PUNCT and t.text == "(":
                lo = self.i
                self.skip_bracket_group()
                text = self.text_of(lo, self.i)
                inner = text[1:-1].strip()
                items = [s.strip() for s in inner.split(",") if s.strip()]
                if items and all(it.startswith("%") for it in items):
                    for it in items:
                        operands.append(it)
                        has_ssa = True
                    first = False
                    if self.peek().kind == lx.PUNCT and self.peek().text == ",":
                        self.i += 1
                    continue
            elif t.kind == lx.PUNCT and t.text == "[":
                lo = self.i
                self.skip_bracket_group()
                text = self.text_of(lo, self.i)
            else:
                text = t.text
                self.i += 1
            if text:
                operands.append(text)
                if t.kind == lx.SSA:
                    has_ssa = True
            first = False
            if self.peek().kind == lx.PUNCT and self.peek().text == ",":
                self.i += 1
        return operands, attrs, iv_name, iter_names, has_ssa

    def after_target(self) -> int:
        """Token index just past the `^name` (and its optional `(…)` group)."""
        k = self.i + 1
        tk = self.toks[k] if k < len(self.toks) else self.toks[-1]
        if tk.kind in (lx.IDENT, lx.NUMBER, lx.SSA):
            k += 1
        tk = self.toks[k] if k < len(self.toks) else self.toks[-1]
        if tk.kind == lx.PUNCT and tk.text == "(":
            end = self.find_matching(k, "(", ")")
            k = k if end < 0 else end + 1
        return k

    def is_label_at(self, k: int) -> bool:
        """Is the token at `k` the `:` that ends a block label?"""
        tk = self.toks[k] if k < len(self.toks) else self.toks[-1]
        return tk.kind == lx.PUNCT and tk.text == ":"

    def stops_operands(self, t: lx.Token) -> bool:
        if t.kind in (lx.EOF, lx.ERROR, lx.ARROW, lx.OPNAME):
            return True
        if t.kind == lx.PUNCT and t.text in _OPERAND_STOP_PUNCT:
            return True
        if t.kind == lx.PUNCT and t.text == "<" and self.peek(1).text == "{":
            return True
        if t.kind == lx.PUNCT and t.text == "(" and self.peek(1).text == "{":
            return True
        return t.kind == lx.IDENT and t.text in _CLAUSE_KEYWORDS

    def single_value_text(self) -> str:
        """One token, or a balanced bracket group, as written."""
        t = self.peek()
        if t.kind == lx.PUNCT and t.text in "([{<":
            lo = self.i
            self.skip_bracket_group()
            return self.text_of(lo, self.i)
        self.i += 1
        return t.text

    def parse_iter_args(self) -> tuple[list[str], list[str]]:
        """`(%name = %init, …)` -> `([names], [initial values])`."""
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "("):
            self.fail("'(' after `iter_args`")
            return [], []
        end = self.find_matching(self.i, "(", ")")
        if end < 0:
            self.fail("')' to close `iter_args(`")
            return [], []
        names: list[str] = []
        values: list[str] = []
        for lo, hi in self.split_top_level(self.i + 1, end):
            eq = self.find_top_level_assign(lo, hi)
            if eq < 0:
                names.append("")
                values.append(self.text_of(lo, hi))
                continue
            names.append(self.text_of(lo, eq))
            values.append(self.text_of(eq + 1, hi))
        self.i = end + 1
        return names, values

    def find_top_level_assign(self, lo: int, hi: int) -> int:
        depth = 0
        for k in range(lo, hi):
            t = self.toks[k]
            if t.kind != lx.PUNCT:
                continue
            if t.text in "(<[":
                depth += 1
            elif t.text in ")>]":
                depth -= 1
            elif t.text == "=" and depth == 0:
                return k
        return -1

    # ---------------------------------------------------------------- types
    def parse_arrow_types(self) -> list[str]:
        """`-> (T1, T2)` or the bare `-> T` form."""
        if self.peek().kind == lx.PUNCT and self.peek().text == "(":
            end = self.find_matching(self.i, "(", ")")
            if end < 0:
                self.fail("')' to close the result type list")
                return []
            parts = self.split_top_level(self.i + 1, end)
            self.i = end + 1
            return [self.text_of(lo, hi) for lo, hi in parts]
        return self.parse_type_list()

    def prev_text(self) -> str:
        """The text of the token before the cursor, or `""` at the start."""
        return self.toks[self.i - 1].text if self.i > 0 else ""

    def parse_type_list(self) -> list[str]:
        """A `:` type list, up to the next trailer.

        A dotted name at depth 0 normally means the next operation has started —
        except after `!`, where it is a type constructor (`!tt.ptr<f32>`). Missing
        that distinction splits `!tt.ptr<f32>` into a phantom operation, which is
        exactly the bug this rule exists to prevent.
        """
        lo = self.i
        depth = 0
        while not self.at_end():
            t = self.peek()
            if t.kind in (lx.EOF, lx.ERROR, lx.ARROW):
                break
            if depth == 0:
                if t.kind == lx.IDENT and t.text in ("loc", "attributes", "else"):
                    break
                if t.kind == lx.OPNAME and self.prev_text() != "!":
                    break
                if t.kind == lx.SSA and self.peek(1).text == "=":
                    break  # the next operation's `%x =`
            if t.kind == lx.PUNCT:
                if t.text in "(<[":
                    depth += 1
                elif t.text in ")>]":
                    if depth == 0:
                        break
                    depth -= 1
                elif t.text == "}":
                    # A `}` closes a region, and a region is never inside a type
                    # list. Without this, an operation with no trailing `loc(…)`
                    # swallows the brace that closes its own block — invisible
                    # on the corpus, where a loc always follows the type, and
                    # fatal on anything hand-written.
                    break
                elif t.text == "^":
                    break  # a block label never belongs to a type list
                elif depth == 0 and t.text == "{":
                    break
            self.i += 1
        hi = self.i
        if hi <= lo:
            self.fail("at least one type after ':'")
            return []
        return [self.text_of(a, b) for a, b in self.split_top_level(lo, hi)]

    # ------------------------------------------------------- attribute dicts
    def looks_like_attr_dict(self) -> bool:
        """Decision 5: a `{` is a dictionary iff it opens with `name =`.

        `{}` is an empty region (the printer emits regions, never `{}` attrs).
        """
        inner = self.peek(1)
        if inner.kind in (lx.SSA, lx.OPNAME, lx.EOF):
            return False
        if inner.kind == lx.PUNCT and inner.text in ("}", "^"):
            return False
        if inner.kind in (lx.IDENT, lx.STRING):
            after = self.peek(2)
            return after.kind == lx.PUNCT and after.text == "="
        return False

    def parse_brace_dict(self) -> dict[str, str]:
        """`{ key = value, … }`, values kept as written (nested dicts included)."""
        out: dict[str, str] = {}
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "{"):
            self.fail("'{' to open the attribute dictionary")
            return out
        self.i += 1
        while not self.at_end():
            t = self.peek()
            if t.kind == lx.PUNCT and t.text == "}":
                self.i += 1
                return out
            if t.kind not in (lx.IDENT, lx.STRING, lx.NUMBER):
                self.fail("an attribute name", tok=t)
                return out
            key = lx.string_value(t) if t.kind == lx.STRING else t.text
            self.i += 1
            if not (self.peek().kind == lx.PUNCT and self.peek().text == "="):
                self.fail("'=' after the attribute name")
                return out
            self.i += 1
            lo = self.i
            depth = 0
            while not self.at_end():
                t = self.peek()
                if t.kind == lx.PUNCT:
                    if t.text in "(<[{":
                        depth += 1
                    elif t.text in ")>]":
                        depth -= 1
                    elif t.text == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif t.text == "," and depth == 0:
                        break
                self.i += 1
            if self.i == lo:
                self.fail("an attribute value")
                return out
            out[key] = self.text_of(lo, self.i)
            if self.peek().kind == lx.PUNCT and self.peek().text == ",":
                self.i += 1
        self.fail("'}' to close the attribute dictionary")
        return out

    # ------------------------------------------------------------- locations
    def maybe_loc(self) -> RawLoc | None:
        if self.peek().kind == lx.IDENT and self.peek().text == "loc" and self.peek(1).text == "(":
            return self.parse_loc()
        return None

    def parse_loc(self) -> RawLoc | None:
        """`loc(…)` in any of the printer's forms. Never fails the parse."""
        self.i += 1  # `loc`
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "("):
            return None
        end = self.find_matching(self.i, "(", ")")
        if end < 0:
            self.fail("')' to close `loc(`")
            return None
        lo, hi = self.i + 1, end
        self.i = end + 1
        return self.interpret_loc(lo, hi)

    def interpret_loc(self, lo: int, hi: int) -> RawLoc:
        """Decision 7: names are recovered; everything else stays verbatim."""
        toks = self.toks[lo:hi]
        if not toks:
            return RawLoc("unknown")
        if len(toks) == 1 and toks[0].kind == lx.LOCREF:
            return RawLoc(toks[0].text)  # a table reference, resolved by to_ir.py
        if len(toks) == 1 and toks[0].kind == lx.IDENT:
            return RawLoc(toks[0].text)  # loc(unknown)
        if toks[0].kind == lx.STRING:
            name = lx.string_value(toks[0])
            if len(toks) == 5 and toks[1].text == ":" and toks[3].text == ":":
                try:
                    return RawLoc(name, int(toks[2].text), int(toks[4].text))
                except ValueError:  # pragma: no cover - a non-numeric line:col
                    return RawLoc(name)
            return RawLoc(name)
        return RawLoc(self.text_of(lo, hi))

    def parse_loc_entry(self) -> None:
        """`#loc42 = loc(…)` — a table entry; both orders in the dump are legal."""
        key = self.peek().text
        self.i += 1
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "="):
            self.fail("'=' after the `#loc` key")
            return
        self.i += 1
        entry = self.parse_loc()
        if entry is None:
            self.fail("`loc(…)` after the `#loc` key")
            return
        self.loc_table[key] = entry

    # --------------------------------------------------------------- regions
    def parse_region(self, depth: int, first_args: list[tuple[str, str]] | None) -> RawRegion:
        """`{ … }`, blocks separated by `^bbN(…)` labels. Nesting is preserved."""
        if not (self.peek().kind == lx.PUNCT and self.peek().text == "{"):
            self.fail("'{' to open the region")
            return RawRegion(blocks=[])
        start = self.peek()
        self.i += 1
        blocks: list[RawBlock] = []
        args: list[tuple[str, str]] = list(first_args or [])
        ops: list[RawOp] = []

        while not self.at_end() and not self.diags:
            t = self.peek()
            if t.kind == lx.PUNCT and t.text == "}":
                break
            if t.kind == lx.ERROR:
                self.fail("a lexable token", found=f"lexical error: {t.text}", tok=t)
                break
            if t.kind == lx.PUNCT and t.text == "^":
                # A label opens a block. If operations have already been
                # collected they are the previous block; a second label in a row
                # is an empty block; but a label that opens the region *is* the
                # entry block, so no empty block is prepended for it.
                if ops or blocks:
                    blocks.append(self.make_block(args, ops))
                args, ops = self.parse_block_label(), []
                continue
            before = self.i
            op = self.parse_op(depth)
            if op is not None:
                ops.append(op)
            if self.i == before and not self.diags:
                self.fail("the region to make progress", found="no token consumed")
        blocks.append(self.make_block(args, ops))

        if not (self.peek().kind == lx.PUNCT and self.peek().text == "}"):
            self.fail(
                "'}' to close the region opened here",
                found=self.peek().text or self.peek().kind,
                tok=start,
            )
            return RawRegion(blocks=blocks)
        self.i += 1
        return RawRegion(blocks=blocks)

    @staticmethod
    def make_block(args: list[tuple[str, str]], ops: list[RawOp]) -> RawBlock:
        return RawBlock(args=args, ops=ops, terminator_index=len(ops) - 1 if ops else -1)

    def parse_block_label(self) -> list[tuple[str, str]]:
        """`^bb1(%a: i32, …):` -> the block's arguments."""
        self.i += 1  # `^`
        if self.peek().kind in (lx.IDENT, lx.NUMBER, lx.SSA):
            self.i += 1
        args: list[tuple[str, str]] = []
        if self.peek().kind == lx.PUNCT and self.peek().text == "(":
            end = self.find_matching(self.i, "(", ")")
            if end < 0:
                self.fail("')' to close the block argument list")
                return args
            for lo, hi in self.split_top_level(self.i + 1, end):
                eq = self.find_top_level_colon(lo, hi)
                if eq < 0:
                    raw_type = self.text_of(lo, hi)
                    args.append(("", raw_type.split(" loc(")[0].strip()))
                else:
                    raw_type = self.text_of(eq + 1, hi)
                    args.append((self.text_of(lo, eq).strip(), raw_type.split(" loc(")[0].strip()))
            self.i = end + 1
        if self.peek().kind == lx.PUNCT and self.peek().text == ":":
            self.i += 1
        return args

    def find_top_level_colon(self, lo: int, hi: int) -> int:
        depth = 0
        for k in range(lo, hi):
            t = self.toks[k]
            if t.kind != lx.PUNCT:
                continue
            if t.text in "(<[":
                depth += 1
            elif t.text in ")>]":
                depth -= 1
            elif t.text == ":" and depth == 0:
                return k
        return -1


def parse_raw(text: str, *, source_path: str = "<string>") -> RawModule:
    """Parse `ttir` text into a `RawModule`. Total: never raises, never hangs.

    A `RawModule` with a non-empty `diagnostics` list is unusable: the consumer
    (`ttir/to_ir.py`) must refuse it rather than build a partial IR.
    """
    if not isinstance(text, str):
        # Totality includes inputs that are not strings at all: a caller that
        # hands over bytes or None gets a diagnostic, not a TypeError from
        # somewhere three frames down.
        return RawModule(
            ops=[],
            loc_table={},
            source_path=source_path,
            triton_version=None,
            diagnostics=[
                ParseDiagnostic(
                    line=1,
                    col=1,
                    expected="`ttir` text (a str)",
                    found=type(text).__name__,
                    snippet="",
                )
            ],
        )
    module = _Parser(text, source_path).parse()
    if module.diagnostics:
        # A failed parse returns "no partial
        # module". A half-built tree would make the invariant a promise the
        # next stage has to keep (`if diagnostics: refuse`), and one forgotten
        # check would lower a tree that cannot be trusted. Dropping the ops
        # makes it structural instead: there is nothing to use by accident.
        # The `#loc` table is kept, because diagnostics refer into it.
        return replace(module, ops=[])
    return module


def with_triton_version(raw: RawModule, version: str | None) -> RawModule:
    """Attach the producing Triton version.

    Kept separate so `parse_raw`'s signature stays minimal; the
    texture extraction harness is the only caller that knows the version.
    """
    return replace(raw, triton_version=version)


__all__ = [
    "MAX_REGION_DEPTH",
    "MAX_RESULT_ARITY",
    "ParseDiagnostic",
    "RawBlock",
    "RawLoc",
    "RawModule",
    "RawOp",
    "RawRegion",
    "parse_raw",
    "with_triton_version",
]
