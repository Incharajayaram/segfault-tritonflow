"""`serialize` / `deserialize` / `disassemble` — the program's text forms.

Two text forms, deliberately different, because they answer different questions:

* :func:`serialize` is the **durable** form. It is parsed by
:func:`deserialize`, it is byte-stable across processes, and it is
  the artifact a consumer reads without the schema — hence the header's
  `total_cost`. Every free-text field is percent-quoted, so a constraint, a
  reason or a descriptor key can contain a space, a comma or a newline without
  ending the line's grammar.
* :func:`disassemble` is the **readable** form. Unquoted, aligned, carrying the
  `loc` names in trailing comments. It is deliberately *not* an input
  to `deserialize`: one format that is both pretty and exactly reversible is
  how a parser grows a special case for alignment whitespace.

**Grammar of the durable form.** One statement per line, keys in a fixed order,
no optional keys — a reader that has to decide whether a key is absent or empty
is a reader that will guess.

| Line | Form |
|---|---|
| header | `PROGRAM isa=<q> schema=<int> kernel=<q> total_cost=<repr(float)>` |
| inputs | `INPUTS=<name>,<name>,…` (may be empty) |
| loop open | `LOOP id=<int> iv=<name-or-none> lower=<operand> upper=<operand> step=<operand> iter_args=<names> inits=<names> results=<names> yields=<names> source=<source-or-none>` |
| loop close | `LOOPEND` |
| instruction | `INSTR name=<q> cost=<repr(float)> loop=<int-or-none> defs=<names> constraint=<q-or-none> constrained_on=<q-or-none> source=<source-or-none>` |
| instruction operand | two-space indent, `<role>=<operand>` |
| epilogue section | `EPILOGUE`, then top-level `INSTR` lines that belong to the epilogue |
| marker | `UNSUPPORTED op=<q> kind=<kind> reason=<q> loc=<q-or-none> source=<source-or-none>` |
| program end | `END` |

The `EPILOGUE` section line exists because `Program.epilogue` is a separate
container, not a flag on an instruction: without it, a round-trip would file
every post-loop instruction under `instrs` and postcondition 3 would fail on the
one kernel (T2) that has an epilogue at all.

`MemRef.access` — the recognition descriptor object — is deliberately not written:
it has no textual inverse, so `access_key` is what the file carries (see
`emit.ir.MemRef`).

`<q>` is `urllib.parse.quote(text, safe="")`. `<names>` is a comma-separated
list of SSA names written unquoted; `_check_name` refuses a name containing a
space, comma, `=` or newline, because a name the form cannot represent is a
refusal rather than a corrupted file. `<source>` is
`<q:op_name>:<line>:<col>:<q:loc_name-or-none>`. `<operand>` is `ssa:<name>`,
`imm:<repr(value)>` or `mem:<q:space>:<q:base>:<q:access_key-or-none>`.

**Two properties that are asserted rather than assumed.**
`deserialize(serialize(p)) == p` (contract postcondition 3) and
`serialize(deserialize(serialize(p))) == serialize(p)` — the second catches what
the first cannot: a serialiser that drops a field plus a deserialiser that
defaults it back would make `==` pass while the file had silently lost
information. Nothing here iterates a `set`, and operand roles are written in
sorted order, so the bytes do not depend on dict construction order or on
`PYTHONHASHSEED`.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

from .ir import (
    AssemblyError,
    Imm,
    Instr,
    Loop,
    MemRef,
    Operand,
    Program,
    SourceRef,
    SsaRef,
    UnsupportedMarker,
    validate_program,
)

#: Characters a bare SSA/storage name may not contain. `%` is absent on
#: purpose: it opens every SSA name, and inside a quoted field it introduces an
#: escape rather than a literal.
_FORBIDDEN_IN_NAME = (" ", "\t", "\n", "\r", ",", "=")

_HEADER_KEYS = ("isa", "schema", "kernel", "total_cost")
_LOOP_KEYS = (
    "id",
    "iv",
    "lower",
    "upper",
    "step",
    "iter_args",
    "inits",
    "results",
    "yields",
    "source",
)
_INSTR_KEYS = (
    "name",
    "cost",
    "loop",
    "defs",
    "constraint",
    "constrained_on",
    "source",
)
_MARKER_KEYS = ("op", "kind", "reason", "loc", "source")


# --------------------------------------------------------------------------- #
# Field codecs
# --------------------------------------------------------------------------- #


def _q(text: str) -> str:
    return quote(text, safe="")


def _check_name(name: str, *, what: str) -> str:
    for bad in _FORBIDDEN_IN_NAME:
        if bad in name:
            raise AssemblyError(
                f"{what} {name!r} contains {bad!r}, which the durable form cannot "
                "represent; refusing rather than writing a file that reads back wrong"
            )
    return name


def _names(names: tuple[str, ...], *, what: str) -> str:
    return ",".join(_check_name(name, what=what) for name in names)


def _parse_names(text: str, *, what: str) -> tuple[str, ...]:
    if text == "":
        return ()
    return tuple(_check_name(part, what=what) for part in text.split(","))


def _fmt_float(value: float) -> str:
    """`repr`, so the text round-trips exactly (and `-0.0` stays `-0.0`)."""
    return repr(float(value))


def _fmt_operand(operand: Operand) -> str:
    if isinstance(operand, SsaRef):
        return f"ssa:{_check_name(operand.name, what='ssa name')}"
    if isinstance(operand, Imm):
        return f"imm:{operand.value!r}"
    return (
        f"mem:{_q(operand.space)}:{_q(_check_name(operand.base, what='memory base'))}"
        f":{'none' if operand.access_key is None else _q(operand.access_key)}"
    )


def _parse_operand(text: str) -> Operand:
    head, _, rest = text.partition(":")
    if head == "ssa":
        return SsaRef(name=_check_name(rest, what="ssa name"))
    if head == "imm":
        return Imm(value=_parse_number(rest))
    if head == "mem":
        parts = rest.split(":")
        if len(parts) != 3:
            raise AssemblyError(
                f"malformed memory operand {text!r}: expected mem:space:base:access"
            )
        space, base, access = parts
        return MemRef(
            space=unquote(space),
            base=_check_name(unquote(base), what="memory base"),
            access_key=None if access == "none" else unquote(access),
        )
    raise AssemblyError(f"unknown operand form {text!r}: expected ssa:, imm: or mem:")


def _parse_number(text: str) -> int | float:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError as exc:
        raise AssemblyError(f"{text!r} is neither an int nor a float") from exc


def _fmt_source(source: SourceRef | None) -> str:
    if source is None:
        return "none"
    loc = "none" if source.loc_name is None else _q(source.loc_name)
    return f"{_q(source.op_name)}:{source.line}:{source.col}:{loc}"


def _parse_source(text: str) -> SourceRef | None:
    if text == "none":
        return None
    parts = text.split(":")
    if len(parts) != 4:
        raise AssemblyError(f"malformed source {text!r}: expected op:line:col:loc")
    op_name, line, col, loc = parts
    try:
        return SourceRef(
            op_name=unquote(op_name),
            line=int(line),
            col=int(col),
            loc_name=None if loc == "none" else unquote(loc),
        )
    except ValueError as exc:
        raise AssemblyError(f"malformed source {text!r}: {exc}") from exc


def _pairs(line: str, keys: tuple[str, ...], *, what: str) -> dict[str, str]:
    """Parse `key=value` pairs, refusing anything the grammar does not fix.

    Strict on purpose: an unknown key is a *format* we do not understand, and
    the contract's failure mode for text we cannot interpret is refusal rather
    than best effort. A value never contains ` key=`, because the values that
    could contain a space are percent-quoted, so the cut is unambiguous.
    """
    found: dict[str, str] = {}
    remaining = line
    while remaining.strip():
        remaining = remaining.lstrip()
        for key in keys:
            prefix = f"{key}="
            if not remaining.startswith(prefix):
                continue
            rest = remaining[len(prefix) :]
            cut = len(rest)
            for other in keys:
                index = rest.find(f" {other}=")
                if index != -1:
                    cut = min(cut, index)
            found[key] = rest[:cut]
            remaining = rest[cut:]
            break
        else:
            raise AssemblyError(f"unknown key in {what} line: {remaining!r}")
    missing = [key for key in keys if key not in found]
    if missing:
        raise AssemblyError(f"{what} line is missing {missing}: {line!r}")
    return found


# --------------------------------------------------------------------------- #
# serialize
# --------------------------------------------------------------------------- #


def serialize(program: Program) -> str:
    """The durable text form. Byte-stable for equal programs."""
    validate_program(program)

    lines = [
        f"PROGRAM isa={_q(program.isa_name)} schema={program.schema_version} kernel={_q(program.kernel_name)} total_cost={_fmt_float(program.total_cost)}",
        f"INPUTS={_names(program.inputs, what='input name')}",
    ]

    for item in program.body:
        if isinstance(item, Loop):
            lines.extend(_serialize_loop(item))
        else:
            lines.extend(_serialize_instr(item))
    if program.epilogue:
        lines.append("EPILOGUE")
        for instr in program.epilogue:
            lines.extend(_serialize_instr(instr))
    for marker in program.unsupported:
        lines.append(_serialize_marker(marker))
    lines.append("END")
    return "\n".join(lines) + "\n"


def _serialize_loop(loop: Loop) -> list[str]:
    iv = (
        "none"
        if loop.induction_var is None
        else _check_name(loop.induction_var, what="induction variable")
    )
    header = " ".join(
        (
            "LOOP",
            f"id={loop.id}",
            f"iv={iv}",
            f"lower={_fmt_operand(loop.lower) if loop.lower is not None else 'none'}",
            f"upper={_fmt_operand(loop.upper) if loop.upper is not None else 'none'}",
            f"step={_fmt_operand(loop.step) if loop.step is not None else 'none'}",
            f"iter_args={_names(loop.iter_args, what='iter_arg')}",
            f"inits={_names(loop.inits, what='initialiser')}",
            f"results={_names(loop.results, what='loop result')}",
            f"yields={_names(loop.yields, what='yielded value')}",
            f"source={_fmt_source(loop.source)}",
        )
    )
    body: list[str] = [header]
    for item in loop.body:
        if isinstance(item, Instr):
            body.extend(_serialize_instr(item, indent="  "))
        else:
            body.append(f"  {_serialize_marker(item)}")
    body.append("LOOPEND")
    return body


def _serialize_instr(instr: Instr, *, indent: str = "") -> list[str]:
    loop = "none" if instr.loop is None else str(instr.loop)
    header = " ".join(
        (
            f"{indent}INSTR",
            f"name={_q(instr.name)}",
            f"cost={_fmt_float(instr.cost)}",
            f"loop={loop}",
            f"defs={_names(instr.defs, what='defined name')}",
            f"constraint={'none' if instr.constraint is None else _q(instr.constraint)}",
            f"constrained_on={'none' if instr.constrained_on is None else _q(instr.constrained_on)}",
            f"source={_fmt_source(instr.source)}",
        )
    )
    operands = [f"{indent}  {role}={_fmt_operand(instr.operands[role])}" for role in instr.roles]
    return [header, *operands]


def _serialize_marker(marker: UnsupportedMarker) -> str:
    loc = "none" if marker.loc_name is None else _q(marker.loc_name)
    return " ".join(
        (
            "UNSUPPORTED",
            f"op={_q(marker.op_name)}",
            f"kind={marker.kind}",
            f"reason={_q(marker.reason)}",
            f"loc={loc}",
            f"source={_fmt_source(marker.source)}",
        )
    )


# --------------------------------------------------------------------------- #
# deserialize
# --------------------------------------------------------------------------- #


def deserialize(text: str, *, expect_schema_version: int | None = None) -> Program:
    """The inverse of :func:`serialize`, refusing anything it cannot read.

    `expect_schema_version` is the contract's "refuse with a clear error; never
    reinterpret": given the schema actually loaded, a file written by
    another revision is not approximated, it is rejected.
    """
    lines = [line for line in text.splitlines() if line.strip() != ""]
    if not lines:
        raise AssemblyError("empty program text; expected a PROGRAM header")
    if not lines[0].startswith("PROGRAM "):
        raise AssemblyError(f"expected a PROGRAM header, found {lines[0]!r}")

    head = _pairs(lines[0][len("PROGRAM ") :], _HEADER_KEYS, what="PROGRAM")
    try:
        schema_version = int(head["schema"])
    except ValueError as exc:
        raise AssemblyError(f"schema version {head['schema']!r} is not an integer") from exc
    if expect_schema_version is not None and schema_version != expect_schema_version:
        raise AssemblyError(
            f"program was written for schema version {schema_version}, but "
            f"{expect_schema_version} is loaded; refusing to reinterpret it"
        )
    try:
        total_cost = float(head["total_cost"])
    except ValueError as exc:
        raise AssemblyError(f"total_cost {head['total_cost']!r} is not a number") from exc

    if len(lines) < 2 or not lines[1].startswith("INPUTS="):
        raise AssemblyError("expected an INPUTS line directly after the PROGRAM header")
    inputs = _parse_names(lines[1][len("INPUTS=") :], what="input name")

    loops: list[Loop] = []
    instrs: list[Instr] = []
    epilogue: list[Instr] = []
    markers: list[UnsupportedMarker] = []
    index = 2
    in_epilogue = False

    while index < len(lines):
        line = lines[index]
        if line == "END":
            index += 1
            break
        if line == "EPILOGUE":
            in_epilogue = True
            index += 1
            continue
        if line.startswith("LOOP "):
            if in_epilogue:
                raise AssemblyError(
                    "a LOOP after EPILOGUE; the epilogue is post-loop by definition"
                )
            loop, index = _parse_loop(lines, index)
            loops.append(loop)
            continue
        if line.startswith("INSTR "):
            instr, index = _parse_instr(lines, index, indent="")
            (epilogue if in_epilogue else instrs).append(instr)
            continue
        if line.startswith("UNSUPPORTED "):
            markers.append(_parse_marker(line))
            index += 1
            continue
        if line.startswith((" ", "\t")):
            raise AssemblyError(f"operand line with no INSTR to belong to: {line!r}")
        raise AssemblyError(f"unexpected line: {line!r}")
    else:
        raise AssemblyError("program text has no END line; truncated file?")

    if index < len(lines):
        raise AssemblyError(f"content after END: {lines[index]!r}")

    return Program(
        isa_name=unquote(head["isa"]),
        schema_version=schema_version,
        kernel_name=unquote(head["kernel"]),
        loops=tuple(loops),
        instrs=tuple(instrs),
        epilogue=tuple(epilogue),
        unsupported=tuple(markers),
        total_cost=total_cost,
        inputs=inputs,
    )


def _parse_loop(lines: list[str], index: int) -> tuple[Loop, int]:
    fields = _pairs(lines[index][len("LOOP ") :], _LOOP_KEYS, what="LOOP")
    try:
        loop_id = int(fields["id"])
    except ValueError as exc:
        raise AssemblyError(f"loop id {fields['id']!r} is not an integer") from exc

    index += 1
    body: list[Instr | UnsupportedMarker] = []
    while True:
        if index >= len(lines):
            raise AssemblyError(f"loop {loop_id} is never closed with LOOPEND")
        line = lines[index]
        if line == "LOOPEND":
            index += 1
            break
        if line.startswith("  INSTR "):
            instr, index = _parse_instr(lines, index, indent="  ")
            if instr.loop != loop_id:
                raise AssemblyError(
                    f"{instr.name} sits inside loop {loop_id} but its line says "
                    f"loop={instr.loop}; refusing to guess which is right"
                )
            body.append(instr)
            continue
        if line.startswith("  UNSUPPORTED "):
            body.append(_parse_marker(line[len("  ") :]))
            index += 1
            continue
        if line.startswith("LOOP "):
            raise AssemblyError(
                "nested scf.for is not representable in this program format; the "
                "emitter marks it UNSUPPORTED rather than flattening it"
            )
        raise AssemblyError(f"unexpected line inside loop {loop_id}: {line!r}")

    return (
        Loop(
            id=loop_id,
            induction_var=None if fields["iv"] == "none" else fields["iv"],
            lower=None if fields["lower"] == "none" else _parse_operand(fields["lower"]),
            upper=None if fields["upper"] == "none" else _parse_operand(fields["upper"]),
            step=None if fields["step"] == "none" else _parse_operand(fields["step"]),
            iter_args=_parse_names(fields["iter_args"], what="iter_arg"),
            inits=_parse_names(fields["inits"], what="initialiser"),
            results=_parse_names(fields["results"], what="loop result"),
            yields=_parse_names(fields["yields"], what="yielded value"),
            body=tuple(body),
            source=_parse_source(fields["source"]),
        ),
        index,
    )


def _parse_instr(lines: list[str], index: int, *, indent: str) -> tuple[Instr, int]:
    fields = _pairs(lines[index][len(indent) + len("INSTR ") :], _INSTR_KEYS, what="INSTR")
    try:
        cost = float(fields["cost"])
    except ValueError as exc:
        raise AssemblyError(f"cost {fields['cost']!r} is not a number") from exc
    if fields["loop"] == "none":
        loop_id = None
    else:
        try:
            loop_id = int(fields["loop"])
        except ValueError as exc:
            raise AssemblyError(f"loop id {fields['loop']!r} is not an integer") from exc

    name = unquote(fields["name"])
    index += 1
    operands: dict[str, Operand] = {}
    while index < len(lines) and lines[index].startswith(f"{indent}  "):
        role, sep, value = lines[index].strip().partition("=")
        if not sep or not role:
            raise AssemblyError(f"malformed operand line: {lines[index]!r}")
        if role in operands:
            raise AssemblyError(f"role {role!r} is given twice for {name}")
        operands[role] = _parse_operand(value)
        index += 1

    source = _parse_source(fields["source"])
    instr = Instr(
        name=name,
        operands=operands,
        loop=loop_id,
        cost=cost,
        source_ops=() if source is None else (source,),
        defs=_parse_names(fields["defs"], what="defined name"),
        constraint=None if fields["constraint"] == "none" else unquote(fields["constraint"]),
        constrained_on=(
            None if fields["constrained_on"] == "none" else unquote(fields["constrained_on"])
        ),
    )
    return instr, index


def _parse_marker(line: str) -> UnsupportedMarker:
    fields = _pairs(line[len("UNSUPPORTED ") :], _MARKER_KEYS, what="UNSUPPORTED")
    return UnsupportedMarker(
        op_name=unquote(fields["op"]),
        reason=unquote(fields["reason"]),
        loc_name=None if fields["loc"] == "none" else unquote(fields["loc"]),
        kind=fields["kind"],
        source=_parse_source(fields["source"]),
    )


# --------------------------------------------------------------------------- #
# disassemble
# --------------------------------------------------------------------------- #


def disassemble(program: Program) -> str:
    """The readable form. Not an input to:func:`deserialize`."""
    inputs = ", ".join(program.inputs) if program.inputs else "(none)"
    lines = [
        f"; {program.isa_name} v{program.schema_version}  kernel={program.kernel_name}  "
        f"total_cost={program.total_cost}",
        f"; inputs: {inputs}",
    ]

    for item in program.body:
        if isinstance(item, Loop):
            lines.extend(_readable_loop(item))
        else:
            lines.append(_readable_instr(item))
    lines.extend(_readable_instr(instr) for instr in program.epilogue)
    lines.extend(_readable_marker(marker) for marker in program.unsupported)
    lines.append("end")
    return "\n".join(lines) + "\n"


def _readable_loop(loop: Loop) -> list[str]:
    lower = "?" if loop.lower is None else str(loop.lower)
    upper = "?" if loop.upper is None else str(loop.upper)
    step = "" if loop.step is None else f" step {loop.step}"
    pairs = ", ".join(f"{arg} = {init}" for init, arg in loop.iter_arg_pairs)
    yields = ", ".join(loop.results) if loop.results else "(nothing)"
    loc = f"    ; {loop.source}" if loop.source else ""
    lines = [
        f"loop {loop.id}: {loop.induction_var} = {lower} .. {upper}{step}{loc}",
        f"    iter_args({pairs}) -> ({yields})",
    ]
    for item in loop.body:
        rendered = _readable_instr(item) if isinstance(item, Instr) else _readable_marker(item)
        lines.append(f"    {rendered}")
    lines.append("loopend")
    return lines


def _readable_instr(instr: Instr) -> str:
    roles = "  ".join(f"{role}={instr.operands[role]}" for role in instr.roles)
    loc = f"    ; {instr.source}" if instr.source else ""
    return f"{instr.name} cost={instr.cost}  {roles}{loc}".rstrip()


def _readable_marker(marker: UnsupportedMarker) -> str:
    loc = f"    ; {marker.source}" if marker.source else ""
    return f"! {marker.kind} {marker.op_name}: {marker.reason}{loc}"


__all__ = ["deserialize", "disassemble", "serialize"]
