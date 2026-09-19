"""Tokenizer for the textual Triton IR dump (`ttir`).

The lexer is the parser's lowest layer. Three properties matter more than speed:

* **Total.** `tokenize` returns for any string. Bad input becomes an `ERROR`
  token, never an exception (contract `raw-module.md`, postcondition 1).
* **Deterministic.** Same bytes -> same token list, every process. No dict or
 set iteration, no locale, no regex backtracking (``).
* **Text-preserving.** Every token records `offset`/`end_offset` into the
  *original* string, so the parser can slice type text and attribute values out
  verbatim instead of re-printing them. Re-printing is how a second dialect
  appears (contract postcondition 2).

`strip_comments` replaces comments with spaces rather than deleting them, so
offsets stay valid and reported columns keep pointing at the original text.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Token vocabulary. The parser switches on these; nothing else may invent kinds.
# --------------------------------------------------------------------------
OPNAME = "opname"  # dotted name: tt.load, arith.addi, foo.bar, scf.for
IDENT = "ident"  # bare word: module, public, to, step, iter_args, attributes
SSA = "ssa"  # %acc, %0, %_k, %acc_25#2, %acc_25:3
LOCREF = "locref"  # #loc, #loc42
SYMBOL = "symbol"  # @matmul
STRING = "string"  # "a_ptrs"
NUMBER = "number"  # 64, 0.0, 1e-3, -1, 0x1f
LITERAL = "literal"  # dense<0.000000e+00> — captured whole, deliberately opaque
PUNCT = "punct"  # = , : { } ( ) < > [ ] *
ARROW = "arrow"  # ->
ERROR = "error"  # the input is not lexable here; carries the reason in `text`
EOF = "eof"

# `?` is a real part of the dialect, not junk: `tensor<?x64xf32>` is a dynamic
# shape. Leaving it out of the punctuation set made it a lexical error, which
# would have blamed bad *type text* on the syntax layer — the one thing the
# contract's failure-route table forbids (that case belongs to the def-use graph layer).
_PUNCT = set("=,:{}()<>[]*!^?")
_WHITESPACE = " \t\f\v"
_DIGITS = set("0123456789")
_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CONT = _IDENT_START | _DIGITS
# SSA names as Triton prints them. '#' covers the multi-result use form
# (`%acc_25#2`); ':' is handled separately as the arity suffix.
_SSA_CONT = _IDENT_CONT | set("#$.")
# Words that introduce an opaque balanced-`<>` literal.
_LITERAL_PREFIXES = ("dense", "sparse", "array")

#: A dump larger than this is refused rather than tokenised, so a hostile input
#: cannot grow memory without bound (contract: ">10 MB -> parses or refuses").
MAX_TOKENS = 1_000_000


@dataclass(frozen=True)
class Token:
    """One lexical token, addressed in the original string."""

    kind: str
    text: str
    line: int
    col: int
    offset: int
    end_offset: int

    def __str__(self) -> str:  # only used in diagnostics
        return f"{self.kind}({self.text!r}) @{self.line}:{self.col}"


def strip_comments(text: str) -> str:
    """Blank out `//` comments, preserving length and line structure.

    A `//` inside a `"…"` string is not a comment. `\\` escapes the next
    character inside a string, so `loc("a\\"b")` stays one string.
    """
    out = list(text)
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_string = False
            elif ch in "\r\n":
                in_string = False  # unterminated string: do not eat the next line
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _scan_balanced(text: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index just past the `close_ch` matching the `open_ch` at `i`.

    Returns `len(text)` if the input ends first; the caller decides whether that
    is an error, so this function never raises and never loops.
    """
    depth = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def tokenize(text: str) -> list[Token]:
    """Tokenise `text`. Always ends with exactly one `EOF` token."""
    if not isinstance(text, str):  # pragma: no cover - defensive, contract is `str`
        return [Token(ERROR, "input is not a string", 1, 1, 0, 0), Token(EOF, "", 1, 1, 0, 0)]

    text = strip_comments(text)
    if text.startswith("\ufeff"):
 #: a UTF-8 BOM is tolerated, not treated as a token. Replaced by
        # a space so every offset in the file stays valid.
        text = " " + text[1:]

    tokens: list[Token] = []
    i = 0
    n = len(text)
    line = 1
    line_start = 0

    def emit(kind: str, start: int, end: int, ln: int, col: int) -> None:
        tokens.append(Token(kind, text[start:end], ln, col, start, end))

    while i < n:
        if len(tokens) >= MAX_TOKENS:
            tokens.append(
                Token(
                    ERROR,
                    f"input exceeds the token limit ({MAX_TOKENS}); refused, not truncated",
                    line,
                    i - line_start + 1,
                    i,
                    i,
                )
            )
            break

        ch = text[i]
        col = i - line_start + 1

        # --- line structure ------------------------------------------------
        if ch == "\r":
            i += 1
            if i < n and text[i] == "\n":
                i += 1
            line += 1
            line_start = i
            continue
        if ch == "\n":
            i += 1
            line += 1
            line_start = i
            continue
        if ch in _WHITESPACE:
            i += 1
            continue

        # --- arrow (before '-' can start a number) -------------------------
        if ch == "-" and i + 1 < n and text[i + 1] == ">":
            emit(ARROW, i, i + 2, line, col)
            i += 2
            continue

        # --- opaque balanced literals: dense<…> ---------------------------
        if ch in _IDENT_START:
            j = i
            while j < n and text[j] in _IDENT_CONT:
                j += 1
            word = text[i:j]
            if word in _LITERAL_PREFIXES and j < n and text[j] == "<":
                k = _scan_balanced(text, j, "<", ">")
                emit(LITERAL, i, k, line, col)
                i = k
                continue

        # --- strings -------------------------------------------------------
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == '"':
                    break
                if text[j] in "\r\n":
                    break
                j += 1
            if j < n and text[j] == '"':
                emit(STRING, i, j + 1, line, col)
                i = j + 1
            else:
                emit(ERROR, i, j, line, col)
                tokens[-1] = Token(ERROR, "unterminated string literal", line, col, i, j)
                i = j
            continue

        # --- SSA value: %name, %name#2, %name:3 ----------------------------
        if ch == "%":
            j = i + 1
            while j < n and text[j] in _SSA_CONT:
                j += 1
            if j == i + 1:
                emit(ERROR, i, i + 1, line, col)
                tokens[-1] = Token(ERROR, "'%' with no name", line, col, i, i + 1)
                i += 1
                continue
            if j < n and text[j] == ":" and j + 1 < n and text[j + 1] in _DIGITS:
                j += 1
                while j < n and text[j] in _DIGITS:
                    j += 1
            emit(SSA, i, j, line, col)
            i = j
            continue

        # --- loc / attribute reference: #loc42 -----------------------------
        if ch == "#":
            j = i + 1
            while j < n and text[j] in _SSA_CONT:
                j += 1
            if j == i + 1:
                emit(PUNCT, i, i + 1, line, col)
                i += 1
                continue
            emit(LOCREF, i, j, line, col)
            i = j
            continue

        # --- symbol: @matmul ----------------------------------------------
        if ch == "@":
            j = i + 1
            while j < n and text[j] in _SSA_CONT:
                j += 1
            if j == i + 1:
                emit(ERROR, i, i + 1, line, col)
                tokens[-1] = Token(ERROR, "'@' with no symbol name", line, col, i, i + 1)
                i += 1
                continue
            emit(SYMBOL, i, j, line, col)
            i = j
            continue

        # --- numbers -------------------------------------------------------
        signed = ch in "+-" and i + 1 < n and text[i + 1] in _DIGITS
        if ch in _DIGITS or signed:
            j = i + 1 if signed else i
            if text[j] == "0" and j + 1 < n and text[j + 1] in "xX":
                j += 2
                while j < n and (text[j] in _DIGITS or text[j] in "abcdefABCDEF"):
                    j += 1
            else:
                while j < n and text[j] in _DIGITS:
                    j += 1
                if j < n and text[j] == "." and j + 1 < n and text[j + 1] in _DIGITS:
                    j += 1
                    while j < n and text[j] in _DIGITS:
                        j += 1
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k] in _DIGITS:
                    j = k
                    while j < n and text[j] in _DIGITS:
                        j += 1
            emit(NUMBER, i, j, line, col)
            i = j
            continue

        # --- words: op names (dotted) and plain identifiers ----------------
        if ch in _IDENT_START:
            j = i
            while j < n and text[j] in _IDENT_CONT:
                j += 1
            dotted = False
            while j < n and text[j] == "." and j + 1 < n and text[j + 1] in _IDENT_START:
                dotted = True
                j += 1
                while j < n and text[j] in _IDENT_CONT:
                    j += 1
            emit(OPNAME if dotted else IDENT, i, j, line, col)
            i = j
            continue

        # --- punctuation ---------------------------------------------------
        if ch in _PUNCT:
            emit(PUNCT, i, i + 1, line, col)
            i += 1
            continue

        # --- anything else: one character of error, and we keep going -------
        emit(PUNCT, i, i + 1, line, col)
        tokens[-1] = Token(ERROR, f"unrecognised character {ch!r}", line, col, i, i + 1)
        i += 1

    tokens.append(Token(EOF, "", line, n - line_start + 1, n, n))
    return tokens


def string_value(token: Token) -> str:
    """The contents of a `STRING` token, with `\\"` and `\\\\` unescaped.

    A loc name containing an escaped quote must not terminate the
    string early, and the recovered name is the *value*, not the quoted text.
    """
    if token.kind != STRING or len(token.text) < 2:
        return token.text
    body = token.text[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


__all__ = [
    "ARROW",
    "EOF",
    "ERROR",
    "IDENT",
    "LITERAL",
    "LOCREF",
    "MAX_TOKENS",
    "NUMBER",
    "OPNAME",
    "PUNCT",
    "SSA",
    "STRING",
    "SYMBOL",
    "Token",
    "string_value",
    "strip_comments",
    "tokenize",
]
