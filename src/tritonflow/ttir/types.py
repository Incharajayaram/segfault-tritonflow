"""Type text -> :class:`TypeExpr`. The semantic layer's half of the parser/IR seam.

The syntax layer keeps type text as *text* ("the syntax layer never
decides what a type means; the semantic layer never looks at a character of
source"). This module is the only place that decides what a type means, and
it is where the following edge cases are answered:

| Edge case | Answer |
|---|---|
| `-> tensor<64x64xf32>` | ``shape == (64, 64)``, ``dtype == "f32"`` |
| `!tt.ptr<f32, 1>` | ``ptr_space == 1`` |
| `tensor<64x64x!tt.ptr<f32>>` | ``shape == (64, 64)`` **and** ``element.kind == "ptr"`` |

Two deliberate choices, both because a type is shared between many values:

* ``shape`` is a tuple, not the list the data-model writes. Immutability is the
  point — a value's type is handed to several consumers and none of them may
  edit it — and a tuple is hashable, which a list is not.
* ``dtype`` is the *immediate* element type as written, so a pointer element is
  visible (``"!tt.ptr<f32>"``) instead of flattened away; :attr:`TypeExpr.scalar_dtype`
  gives the innermost scalar type for callers that want it.

**Totality.** Like the syntax layer, nothing here raises: unrecognised text comes
back as ``kind == "unknown"`` with ``raw`` preserved. Deciding that an unknown
type is *invalid* is a judgement, and judgements belong to :mod:`to_ir`, so that
a dynamic shape (``tensor<?x64xf32>``, legal) and an impossible element
(``tensor<64x64x???>``, not) cannot be confused with each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SCALAR_RE = re.compile(r"(?:[iuf]|si|ui)\d+(?:[a-z]+\d+)*|bf16|index|void")
r"""Scalar type names, as a pattern rather than a list so the ``f8e4m3`` family
does not silently become "unknown" the day a fixture uses one. The trailing
``(?:[a-z]+\d+)*`` is what covers ``f8e4m3`` / ``f8e5m2``; without it the
docstring would promise something the expression did not do."""

NUMBER_RE = re.compile(r"(\d+|\?)$")


@dataclass(frozen=True)
class TypeExpr:
    """A parsed type. ``raw`` is always exactly the text that was read."""

    raw: str
    kind: str = "unknown"  # tensor | ptr | scalar | unknown
    shape: tuple[int | None, ...] = ()
    dtype: str = ""
    ptr_space: int | None = None
    element: TypeExpr | None = field(default=None, repr=False)

    # -- convenience ---------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """Whether this is a type at all (see the module docstring)."""
        return self.kind != "unknown" and (self.element is None or self.element.is_valid)

    @property
    def scalar_dtype(self) -> str:
        """The innermost scalar dtype: ``f32`` for both ``f32`` and
        ``tensor<64x64x!tt.ptr<f32>>``."""
        node = self
        while node.element is not None:
            node = node.element
        return node.dtype

    @property
    def is_pointer_like(self) -> bool:
        """A pointer, or a tensor whose elements are pointers.

        Both shapes mean "memory operand" to the recogniser, and both occur in
        the corpus (`!tt.ptr<f32>` as a function argument, `tensor<64x32x!tt.ptr<f32>>`
        as a block of addresses).
        """
        return self.kind == "ptr" or (self.element is not None and self.element.kind == "ptr")

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.raw


def parse_type(raw: str) -> TypeExpr:
    """Parse one type as written in `ttir`. Total: never raises."""
    text = raw.strip()
    if not text:
        return TypeExpr(raw=raw, kind="unknown")

    if text.startswith("!"):
        return _parse_ptr(text, raw)

    if text.startswith("tensor<"):
        inner, closed = _unwrap(text, "tensor<")
        if not closed:
            return TypeExpr(raw=raw, kind="unknown")
        dims, element_text = _split_dims(inner)
        if not dims:
            return TypeExpr(raw=raw, kind="unknown")
        element = parse_type(element_text)
        return TypeExpr(
            raw=raw,
            kind="tensor",
            shape=tuple(dims),
            dtype=element.raw.strip(),
            element=element,
        )

    if SCALAR_RE.fullmatch(text):
        return TypeExpr(raw=raw, kind="scalar", dtype=text)

    return TypeExpr(raw=raw, kind="unknown")


def _parse_ptr(text: str, raw: str) -> TypeExpr:
    """``!tt.ptr<f32>`` / ``!tt.ptr<f32, 1>``.

    The address space is the second, comma-separated parameter, and its absence
 is *not* the same as address space 0 — distinguishes a flag that is
    set from one that was never printed, so ``None`` is kept.
    """
    if not text.startswith("!tt.ptr<"):
        return TypeExpr(raw=raw, kind="unknown")
    inner, closed = _unwrap(text, "!tt.ptr<")
    if not closed:
        return TypeExpr(raw=raw, kind="unknown")
    parts = _split_top_level(inner, ",")
    element_text = parts[0].strip()
    space: int | None = None
    if len(parts) > 1:
        candidate = parts[1].strip()
        if candidate.lstrip("-").isdigit():
            space = int(candidate)
        else:
            return TypeExpr(raw=raw, kind="unknown")
    element = parse_type(element_text)
    return TypeExpr(
        raw=raw,
        kind="ptr",
        dtype=element.raw.strip(),
        ptr_space=space,
        element=element,
    )


def _unwrap(text: str, prefix: str) -> tuple[str, bool]:
    """Content of ``<...>`` after `prefix`, handling nesting. `closed` is False
    when the closing ``>`` is missing, which is a malformed type, not a crash."""
    start = len(prefix)
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                if index != len(text) - 1:
                    return "", False  # trailing junk after the type
                return text[start:index], True
    return "", False


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on `separator` outside ``<>`` so nested types stay intact."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _split_dims(inner: str) -> tuple[list[int | None], str]:
    """Split ``64x32x!tt.ptr<f32>`` into dims and element text.

    Leading ``x``-separated dimensions are taken while they look like
    dimensions; everything after the last such ``x`` is the element type, joined
    back together so a type containing ``x`` cannot be truncated.
    """
    pieces = _split_top_level(inner, "x")
    dims: list[int | None] = []
    for index, piece in enumerate(pieces):
        if NUMBER_RE.fullmatch(piece.strip()):
            token = piece.strip()
            dims.append(None if token == "?" else int(token))
        else:
            return dims, "x".join(pieces[index:])
    return dims, ""


__all__ = ["SCALAR_RE", "TypeExpr", "parse_type"]
