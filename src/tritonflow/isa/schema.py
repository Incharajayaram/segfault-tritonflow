"""The ISA schema: declarative instructions, a decidable predicate language,
fail-closed admissibility, and total costs.

Three design decisions are worth reading before the code, because each is a
place where a plausible shortcut would have been a lie:

1. **The predicate language is not an expression evaluator over Python.** It is a
   small grammar: terms, integer comparisons, `aligned`,
   the two `in_bounds` forms, and `all_of`/`any_of`. Deliberately not
   Turing-complete — a full addressing solver would be integer constraint
   programming, and a CP solver does not fit this project's budget. A decidable
   subset is what makes selection *auditable*: the rejected candidate names the
   exact predicate that refused it, which `eval` could never do.

2. **Fail-closed is a construction, not a check.** Every term resolves
   to a `maybe` value: an integer, or `unknown`. Arithmetic on `unknown` gives
   `unknown`; a comparison involving `unknown` gives `"unknown"`; the selector
   treats `"unknown"` as inadmissible. There is no code path that turns a
   missing fact into a pass — the only way to be admissible is for every
   predicate to decide `True`.

3. **Symbolic descriptors decide only what a launch environment grounds.** A
   descriptor's strides are `int | SymExpr`, so `%sam % 4 == 0`
   is `unknown` unless the caller supplies the launch environment
   (`{"%sam": 128, ...}`). That is the honest behaviour: the *same* program can
   have an aligned or misaligned access depending on launch parameters, and a
   selector that guessed would make the choice unfalsifiable. Fail-closed means
   the conservative variant (DMA1D, MAC8) is chosen until the environment says
   otherwise — and the corpus's environment is a declared fact in the checks,
   never guessed at the call site.

**Why instructions carry both `kind` and `rule`.** The schema's coarse
axis is `memory | compute`; the recogniser's binding kinds are the three
*lowering sets* `memory | mac | elementwise` (recognize/op_shapes.py). A selector
asked for `compute` and handed both a MAC and an EPI would be answering two
questions at once — the same defect class as a rule table that conflated them.
`rule` is therefore the grouping the selector enumerates; `kind` remains the
axis the validator's minimum-variety rule counts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from tritonflow.isa.semantics import is_parseable

from ..recognize.walk import SymExpr

__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "AccumulateSpec",
    "AddressingForm",
    "CostError",
    "CostExpr",
    "DataModel",
    "Instruction",
    "IsaSchema",
    "MemorySpace",
    "Predicate",
    "PredicateError",
    "SchemaError",
    "SchemaViolation",
    "cost_of",
    "evaluate",
    "load_builtin",
    "load_schema",
    "validate_schema",
]

#: Verdicts. `unknown` is a *value*, not an exception, so a caller can collect
#: every verdict before deciding — and the selector can attribute an
#: inadmissibility to the predicate that would not decide.
Verdict = Literal[True, False]
UNKNOWN: str = "unknown"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SchemaError(Exception):
    """A schema could not be loaded or is structurally invalid."""


class PredicateError(SchemaError):
    """A predicate or cost expression failed to parse."""


class CostError(SchemaError):
    """A cost expression cannot be evaluated for this operand.

    Costs are total: a cost that cannot be evaluated (a division by a zero
    tile dimension, an `unknown` term) is a schema error, not a runtime
    surprise.
    """


@dataclass(frozen=True)
class SchemaViolation:
    """One invalidity of one schema, with the key path that owns it."""

    path: str
    problem: str

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.path}: {self.problem}"


# --------------------------------------------------------------------------- #
# Numeric core: `maybe` integers and the arithmetic/comparison grammar
# --------------------------------------------------------------------------- #

#: An operand of arithmetic: a concrete integer, `None` (= unknown), or `"div0"`
#: (a defined computation that divided by zero — propagated by `_div`, not by
#: `None`, so "we don't know" and "the answer is division by zero" stay distinct).
Maybe = int | None


def _add(a: Maybe, b: Maybe) -> Maybe:
    if a is None or b is None:
        return None
    return a + b


def _sub(a: Maybe, b: Maybe) -> Maybe:
    if a is None or b is None:
        return None
    return a - b


def _mul(a: Maybe, b: Maybe) -> Maybe:
    if a is None or b is None:
        return None
    return a * b


def _div(a: Maybe, b: Maybe) -> Maybe:
    """True division on `maybe` ints, with an *exactness* rule.

    A non-exact quotient would silently make `stride[1] == 1` decide `False` for
    `stride[1] = 3/2` — a value that does not exist. Non-exact division is
    therefore `None` (undecidable), and division by zero is the string marker.
    """
    if a is None or b is None:
        return None
    if b == 0:
        return "div0"
    quotient, remainder = divmod(a, b)
    return quotient if remainder == 0 else None


def _modulo(a: Maybe, b: Maybe) -> Maybe:
    if a is None or b is None:
        return None
    if b == 0:
        return "div0"
    return a % b


def _neg(a: Maybe) -> Maybe:
    return None if a is None else -a


def _compare(a: Maybe, b: Maybe, operator: str) -> bool | str:
    """One comparison. `unknown` propagates; `div0` is an *error*, not unknown.

    A `div0` reaching a comparison means the predicate computed division by
    zero — that is a malformed predicate against this descriptor, and returning
    `"unknown"` would misreport a bug as a missing fact.
    """
    if a == "div0" or b == "div0":
        raise PredicateError(f"division by zero inside the predicate {operator!r}")
    if a is None or b is None:
        return UNKNOWN
    if operator == "==":
        return a == b
    if operator == "!=":
        return a != b
    if operator == "<":
        return a < b
    if operator == "<=":
        return a <= b
    if operator == ">":
        return a > b
    if operator == ">=":
        return a >= b
    raise PredicateError(f"unknown comparison operator {operator!r}")


# --------------------------------------------------------------------------- #
# Tokenizer + precedence climber, shared by predicates and cost expressions
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(
    r"""
    \s*(?:
      (?P<num>\d+(?:\.\d+)?)
    | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<op>==|!=|<=|>=|<|>|%|\+|-|\*|/|\(|\)|\[|\]|,)
    )
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            rest = text[position:].strip()
            if not rest:
                break
            raise PredicateError(f"cannot tokenize {rest[:24]!r} in {text!r}")
        kind = match.lastgroup or ""
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


#: Binary operator precedence, loosest first. `%` binds like `*`/`/`.
_PRECEDENCE: dict[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}


@dataclass(frozen=True)
class Num:
    value: int | float


@dataclass(frozen=True)
class Name:
    text: str


@dataclass(frozen=True)
class BinOp:
    operator: str
    left: object
    right: object


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: object


class _Parser:
    """Recursive descent over the token list.

    Grammar (precedence climbing, `%` at multiplicative level):

        comparison := sum (('=='|'!='|'<'|'<='|'>'|'>=') sum)?
        sum        := product (('+'|'-') product)*
        product    := unary (('%'|'*'|'/') unary)*
        unary      := ('-')? atom
        atom       := NUM | NAME | NAME '[' NUM ']' | NAME '(' args ')' | '(' comparison ')'
        args       := comparison (',' comparison)*
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.position = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise PredicateError(f"unexpected end of expression in {self.text!r}")
        self.position += 1
        return token

    def expect_op(self, symbol: str) -> None:
        token = self.peek()
        if token is None or token[0] != "op" or token[1] != symbol:
            raise PredicateError(f"expected {symbol!r} in {self.text!r}, found {token}")
        self.take()

    def parse(self) -> object:
        node = self.comparison()
        if self.position != len(self.tokens):
            raise PredicateError(
                f"trailing tokens in {self.text!r}: {self.tokens[self.position :]}"
            )
        return node

    def comparison(self) -> object:
        left = self.sum()
        token = self.peek()
        if (
            token is not None
            and token[0] == "op"
            and token[1] in ("==", "!=", "<", "<=", ">", ">=")
        ):
            self.take()
            right = self.sum()
            return BinOp(token[1], left, right)
        return left

    def sum(self) -> object:
        node = self.product()
        while True:
            token = self.peek()
            if token is not None and token[0] == "op" and token[1] in ("+", "-"):
                self.take()
                node = BinOp(token[1], node, self.product())
            else:
                return node

    def product(self) -> object:
        node = self.unary()
        while True:
            token = self.peek()
            if token is not None and token[0] == "op" and token[1] in ("*", "/", "%"):
                self.take()
                node = BinOp(token[1], node, self.unary())
            else:
                return node

    def unary(self) -> object:
        token = self.peek()
        if token is not None and token == ("op", "-"):
            self.take()
            return Unary("-", self.unary())
        return self.atom()

    def atom(self) -> object:
        token = self.peek()
        if token == ("op", "("):
            # A parenthesised group: `m * n * k / (8*8)` from the schema doc.
            self.take()
            node = self.comparison()
            self.expect_op(")")
            return node
        token = self.take()
        if token[0] == "num":
            text = token[1]
            return Num(float(text) if "." in text else int(text))
        if token[0] != "name":
            raise PredicateError(f"expected a term or number in {self.text!r}, found {token}")
        name = token[1]
        nxt = self.peek()
        if nxt == ("op", "("):
            self.take()
            args: list[object] = []
            if self.peek() != ("op", ")"):
                args.append(self.comparison())
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.comparison())
            self.expect_op(")")
            return Call(name, tuple(args))
        if nxt == ("op", "["):
            self.take()
            index_token = self.take()
            if index_token[0] != "num" or not str(index_token[1]).isdigit():
                raise PredicateError(f"an index must be an integer literal in {self.text!r}")
            self.expect_op("]")
            return Index(name, int(index_token[1]))
        return Name(name)


@dataclass(frozen=True)
class Call:
    function: str
    args: tuple[object, ...]


@dataclass(frozen=True)
class Index:
    name: str
    index: int


def _walk_terms(node: object, into: set[str]) -> None:
    if isinstance(node, Name):
        into.add(node.text)
    elif isinstance(node, Index):
        into.add(node.name)
    elif isinstance(node, BinOp):
        _walk_terms(node.left, into)
        _walk_terms(node.right, into)
    elif isinstance(node, Unary):
        _walk_terms(node.operand, into)
    elif isinstance(node, Call):
        for argument in node.args:
            _walk_terms(argument, into)


class Predicate:
    """A parsed constraint expression, carrying its original text.

    The text is load-bearing: a rejected candidate names *the predicate that
    failed*, e.g. `"by": "cost(1.00*words) > 0.60*words"` — the author's
    text, not a rendering of an AST.
    """

    __slots__ = ("ast", "text")

    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise PredicateError("a predicate must be non-empty text")
        self.text = text.strip()
        self.ast = _Parser(self.text).parse()

    @property
    def terms(self) -> frozenset[str]:
        seen: set[str] = set()
        _walk_terms(self.ast, seen)
        return frozenset(seen)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"Predicate({self.text!r})"


def _parse_predicate(text: str) -> Predicate:
    return text if isinstance(text, Predicate) else Predicate(text)


# --------------------------------------------------------------------------- #
# Cost expressions
# --------------------------------------------------------------------------- #


class CostExpr:
    """A parsed cost expression (`"0.60 * words"`) with its text.

    Distinct from `Predicate` on purpose: a cost must evaluate to a *number*
    (postcondition 3), so it rejects comparison operators at parse time rather
    than at evaluation time.
    """

    __slots__ = ("ast", "text")

    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise PredicateError("a cost expression must be non-empty text")
        self.text = text.strip()
        parsed = _Parser(self.text).parse()
        if isinstance(parsed, BinOp) and parsed.operator in ("==", "!=", "<", "<=", ">", ">="):
            raise PredicateError(f"a cost expression cannot contain a comparison: {self.text!r}")
        self.ast = parsed

    @property
    def terms(self) -> frozenset[str]:
        seen: set[str] = set()
        _walk_terms(self.ast, seen)
        return frozenset(seen)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"CostExpr({self.text!r})"


# --------------------------------------------------------------------------- #
# Term resolution: descriptor + tile + env → `maybe`
# --------------------------------------------------------------------------- #


def _as_int(value: object, env: dict[str, int] | None) -> Maybe:
    """A descriptor value (`int | SymExpr`) → `maybe`, grounding symbols by `env`."""
    if isinstance(value, bool):  # pragma: no cover - defensive
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, SymExpr):
        if env is None:
            return None
        constant = value.as_int()
        if constant is not None:
            return constant
        total: Maybe = 0
        for scale, symbols in value.terms:
            resolved: Maybe = scale
            for name in symbols:
                grounding = env.get(name)
                if grounding is None or isinstance(grounding, str):
                    return None
                resolved = _mul(resolved, grounding)
            total = _add(total, resolved)
        return total
    return None


def _scalar_dtype_bits(descriptor: object) -> Maybe:
    dtype = getattr(descriptor, "dtype", "") or ""
    bits = {
        "f8": 8,
        "f16": 16,
        "bf16": 16,
        "f32": 32,
        "f64": 64,
        "i1": 1,
        "i8": 8,
        "i16": 16,
        "i32": 32,
        "i64": 64,
    }
    return bits.get(dtype.replace("!", "").strip())


def _sequence(name: str, descriptor: object) -> tuple[object, ...] | None:
    value: object
    if name == "shape":
        value = getattr(descriptor, "shape", ())
    elif name == "stride":
        value = getattr(descriptor, "strides", ())
    elif name == "offset":
        value = getattr(descriptor, "offsets", ())
    elif name == "size":
        value = getattr(descriptor, "sizes", ())
    else:
        return None
    return tuple(value)  # type: ignore[arg-type]


def _length(descriptor: object, env: dict[str, int] | None) -> Maybe:
    """`words`/`elements`: the total number of elements the access moves.

    The product of the descriptor's sizes. Every cost expression in the corpus
    is per-element, so this — not `sizes[0]` — is the quantity the unit names.
    """
    sizes = getattr(descriptor, "sizes", ()) or ()
    if not sizes:
        return 1
    total: Maybe = 1
    for size in sizes:
        value = _as_int(size, env)
        if value is None:
            return None
        total = _mul(total, max(value, 0))
    return total


def _terms_of(
    descriptor: object,
    tile: tuple[int, ...] | None,
    env: dict[str, int] | None,
) -> dict[str, Maybe]:
    """Every term the language names, resolved against this operand.

    An absent term is `None` (= unknown), never a default: `dtype_bits` for a
    descriptor whose dtype is unrecorded, `m` for an operand with no tile, a
    stride index past the rank — all unknown, and every predicate over them
    comes back `"unknown"` (fail-closed) rather than over some assumed value.
    """
    rank = len(getattr(descriptor, "sizes", ()) or ())
    tile_values: list[Maybe] = [None, None, None]
    if tile:
        for axis in range(min(len(tile), 3)):
            if tile[axis] is not None:
                tile_values[axis] = int(tile[axis])
    return {
        "shape": None,
        "stride": None,
        "offset": None,
        "size": None,
        "base": _as_int(getattr(descriptor, "base_num", None), env)
        if hasattr(descriptor, "base_num")
        else None,
        "length": _length(descriptor, env),
        "words": _length(descriptor, env),
        "elements": _length(descriptor, env),
        "m": tile_values[0],
        "n": tile_values[1],
        "k": tile_values[2],
        "dtype_bits": _scalar_dtype_bits(descriptor),
        "acc_dtype": None,
        "op_dtype": None,
        "rank": rank,
    }


def _resolve(
    node: object, descriptor: object, tile: tuple[int, ...] | None, env: dict[str, int] | None
) -> Maybe:
    """Evaluate an arithmetic node to a `maybe`. Comparisons are not values."""
    if isinstance(node, Num):
        value = node.value
        if isinstance(value, float):
            # Costs may be fractional; predicate arithmetic may not. A float
            # reaching a predicate term is a schema bug, and returning it would
            # make `stride[0] == 1.5` decidable.
            raise PredicateError(f"a non-integer literal {value!r} reached predicate arithmetic")
        return value
    if isinstance(node, Unary):
        return _neg(_resolve(node.operand, descriptor, tile, env))
    if isinstance(node, BinOp):
        left = _resolve(node.left, descriptor, tile, env)
        right = _resolve(node.right, descriptor, tile, env)
        if node.operator == "+":
            return _add(left, right)
        if node.operator == "-":
            return _sub(left, right)
        if node.operator == "*":
            return _mul(left, right)
        if node.operator == "/":
            return _div(left, right)
        if node.operator == "%":
            return _modulo(left, right)
        raise PredicateError(f"operator {node.operator!r} has no arithmetic meaning here")
    if isinstance(node, Index):
        sequence = _sequence(node.name, descriptor)
        if sequence is None:
            return None
        if not 0 <= node.index < len(sequence):
            return None
        return _as_int(sequence[node.index], env)
    if isinstance(node, Name):
        if env and node.text in env and isinstance(env[node.text], int):
            return env[node.text]
        if node.text in _POINTER_TERMS:
            # A pointer operand as an arithmetic value: the address is the
            # allocator's fact, not the descriptor's (see `aligned`). Unknown.
            return None
        if node.text in ("acc_dtype", "op_dtype"):
            # Dtype names compare through `==` as their bit widths. Both sides
            # of the comparison ground from the same descriptor here, so
            # `acc_dtype == op_dtype` decides "the accumulator and the operand
            # are the same width" — the EPI constraint's actual content.
            return _scalar_dtype_bits(descriptor)
        if node.text == "base":
            return _terms_of(descriptor, tile, env)["base"]
        return _terms_of(descriptor, tile, env).get(node.text)
    if isinstance(node, Call):
        raise PredicateError(f"{node.function}() is a predicate form, not an arithmetic term")
    raise PredicateError(f"cannot evaluate node {node!r}")


# --------------------------------------------------------------------------- #
# Predicate evaluation
# --------------------------------------------------------------------------- #


def _non_negative(value: object, env: dict[str, int] | None) -> bool | None:
    """Structural sign of a descriptor value, or `None` when undecidable.

    Per term of a `SymExpr` (`Σ scale · ∏ symbols`), grounded ints multiply into
    a concrete coefficient; a symbol declared with the class "pid" is non-
    negative by hardware definition and contributes no sign. A term is signed
    when its coefficient is: unknown coefficient, or a *negative* coefficient,
    makes the whole expression's sign undecidable/negative respectively. All
    terms non-negative ⇒ the expression is non-negative.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, SymExpr):
        if value.as_int() is not None:
            return value.as_int() >= 0  # type: ignore[operator]
        if env is None:
            return None
        all_terms_signed = True
        for _scale, symbols in value.terms:
            coefficient: int | None = 1
            symbolic_nonneg = False
            for name in symbols:
                grounding = env.get(name)
                if isinstance(grounding, str):
                    if grounding == "pid":
                        symbolic_nonneg = True
                        continue
                    all_terms_signed = False
                    break
                if grounding is None:
                    all_terms_signed = False
                    break
                if coefficient is not None:
                    coefficient *= grounding
            if not all_terms_signed:
                break
            if coefficient is None:
                all_terms_signed = False
                break
            if coefficient < 0 and not symbolic_nonneg:
                return False
            if coefficient < 0:
                # A negative scale on a program id is still not provably ≥ 0.
                return False
            if coefficient == 0:
                continue  # the term vanishes; sign contribution is zero
            if coefficient > 0:
                continue  # non-negative whether or not a pid symbol remains
            all_terms_signed = all_terms_signed and symbolic_nonneg
        return all_terms_signed if all_terms_signed else None
    return None


def _decided(value: object, env: dict[str, int] | None) -> bool | str:
    """Whether `value`'s sign question is fully answered by `_non_negative`."""
    if isinstance(value, SymExpr) and value.as_int() is None:
        if env is None:
            return UNKNOWN
        for _scale, symbols in value.terms:
            for name in symbols:
                grounding = env.get(name)
                if not isinstance(grounding, int) and grounding != "pid":
                    return UNKNOWN
        return True
    return True


def _eval_call(
    node: Call, descriptor: object, tile: tuple[int, ...] | None, env: dict[str, int] | None
) -> bool | str:
    """The predicate forms. Every argument decides, or the whole call is unknown."""
    name = node.function

    if name == "all_of":
        verdicts = [_eval(a, descriptor, tile, env) for a in node.args]
        if any(v == UNKNOWN for v in verdicts):
            return UNKNOWN
        return all(bool(v) for v in verdicts)

    if name == "any_of":
        verdicts = [_eval(a, descriptor, tile, env) for a in node.args]
        if any(v is True for v in verdicts):
            return True
        if any(v == UNKNOWN for v in verdicts):
            return UNKNOWN
        return False

    if name == "aligned":
        if len(node.args) != 2:
            raise PredicateError("aligned(X, k) takes exactly two arguments")
        value = _resolve(node.args[0], descriptor, tile, env)
        alignment = _resolve(node.args[1], descriptor, tile, env)
        if value is None:
            # Pointer terms (`a_base`, `b_base`, `base`) name specific operands
            # whose *addresses* no descriptor carries — a MAC's A and B tiles
            # live at two addresses the allocator assigns. The machine's
            # data model declares `alignment_words: 4` for every allocation,
            # so alignment up to that promise is a property of the *machine*,
            # decidable here, and asserted by the emulator's allocator at
            # execution. Deciding it False would reject every MAC on every
            # machine; deciding an ungrounded address True without a declared
            # promise would hide the allocator's job. The promise is the
            # decision.
            if alignment is None:
                return UNKNOWN
            return alignment <= _current_alignment_promise()
        result = _compare(_modulo(value, alignment), 0, "==")
        return result

    if name == "in_bounds":
        if len(node.args) != 2:
            raise PredicateError("in_bounds(base, length) takes exactly two arguments")
        base = _resolve(node.args[0], descriptor, tile, env)
        length = _resolve(node.args[1], descriptor, tile, env)
        # An unresolvable base is the allocator's address, non-negative by the
        # same flat-model promise as `aligned` above; the decidable content
        # here is a non-degenerate length. Real upper bounds are asserted at
        # execution, not guessed here.
        ge0 = True if base is None else _compare(base, 0, ">=")
        gt0 = _compare(length, 0, ">")
        if ge0 == UNKNOWN or gt0 == UNKNOWN:
            return UNKNOWN
        return bool(ge0) and bool(gt0)

    if name == "in_bounds_all":
        # The flat-memory model has no buffer-extent term, so the decidable
        # content of "in bounds" is non-degeneracy: every offset non-negative,
        # every extent at least 1. An actual upper bound is a *launch* property
        # and the emulator asserts it at execution; writing a fake extent here
        # would make the predicate decide things it cannot know.
        #
        # An offset may be *symbolic* (%pid_m*64*%sam). Sign is decidable from
        # structure when the env declares the term's class: a program id is
        # non-negative by hardware definition (`tt.get_program_id` returns an
        # unsigned tile coordinate), a kernel parameter is non-negative because
        # the launcher typed it so. `launch_env.json` declares the class as the
        # string "pid"; an unclassed symbol leaves the sign unknown — and
        # unknown stays inadmissible, never assumed positive.
        for name_seq, comparator in (("offset", ">="), ("size", ">")):
            sequence = _sequence(name_seq, descriptor) or ()
            for element in sequence:
                if _non_negative(element, env) is False:
                    return False
                verdict = _decided(element, env)
                if verdict == UNKNOWN:
                    return UNKNOWN
                value = _as_int(element, env)
                bound = _compare(value, 0, comparator)
                if bound is True:
                    continue
                if bound == UNKNOWN and _non_negative(element, env):
                    continue
                if bound is False:
                    return False
                return UNKNOWN
        return True

    raise PredicateError(f"unknown predicate form {name!r}")

    raise PredicateError(f"unknown predicate form {name!r}")


def _eval(
    node: object, descriptor: object, tile: tuple[int, ...] | None, env: dict[str, int] | None
) -> bool | str:
    if isinstance(node, BinOp) and node.operator in ("==", "!=", "<", "<=", ">", ">="):
        left = _resolve(node.left, descriptor, tile, env)
        right = _resolve(node.right, descriptor, tile, env)
        return _compare(left, right, node.operator)
    if isinstance(node, Call):
        return _eval_call(node, descriptor, tile, env)
    if isinstance(node, Name) and node.text in ("acc_dtype", "op_dtype"):
        # A bare dtype name compares through `==` only; reaching here means the
        # schema used one as a boolean, which is malformed.
        raise PredicateError(f"{node.text} is a value term, not a predicate")
    raise PredicateError(f"{node!r} is not a predicate form")


def evaluate(
    predicate: Predicate | str,
    descriptor: object,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
) -> Verdict | str:
    """`True`, `False`, or `"unknown"` — never an exception, never a guess.

    `False` carries no reason because it needs none: the *predicate's own text*
    is the reason, which is what the selector records. `"unknown"` is
    the fail-closed verdict: the selector treats it as inadmissible,
    and it is the *only* way a missing fact can influence a decision.
    """
    parsed = _parse_predicate(predicate)
    return _eval(parsed.ast, descriptor, tile, env)


#: The allocator's alignment promise, from the ISA-1 data model
#: (`data_model.memory_spaces[0].alignment_words: 4`). `aligned(<pointer>, k)`
#: decides against this promise because pointer *addresses* are the allocator's
#: facts, not the descriptor's. ISA-2's banked model parametrises this per space.
ALLOCATOR_ALIGNMENT_WORDS = 4

#: Per-schema allocator promises, keyed by schema name. A schema whose data model
#: declares a different `alignment_words` registers its promise here at load
#: time; `aligned(X, k)` decides against the promise of the schema being
#: selected against. ISA-1 stays the module default (4); tritonflow2 declares 8.
#: This is the mechanism the banked data model needs: the promise is a property
#: of the *machine*, and two machines may promise differently.
SCHEMA_ALIGNMENT_WORDS: dict[str, int] = {"tritonflow1": 4}

#: The schema currently being selected against. `aligned(X, k)` is decided
#: during selection, which is per-schema, but the predicate evaluator is
#: module-level — so `assemble` sets this around each selection pass (the
#: same pattern as a decimal context). Default: ISA-1's promise, so every
#: existing caller is unchanged.
_ACTIVE_SCHEMA: str = "tritonflow1"


def set_active_schema(name: str) -> None:
    """Point `aligned()`'s promise at this schema's data model."""
    global _ACTIVE_SCHEMA
    _ACTIVE_SCHEMA = name


def _current_alignment_promise() -> int:
    return SCHEMA_ALIGNMENT_WORDS.get(_ACTIVE_SCHEMA, ALLOCATOR_ALIGNMENT_WORDS)


def register_alignment_promise(schema_name: str, alignment_words: int) -> None:
    """Record a schema's allocator promise so `aligned()` can decide against it."""
    SCHEMA_ALIGNMENT_WORDS[schema_name] = alignment_words


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #


def _cost_terms(
    descriptor: object, tile: tuple[int, ...] | None, env: dict[str, int] | None
) -> dict[str, float]:
    length = _length(descriptor, env)
    tile_values = [0.0, 0.0, 0.0]
    if tile:
        for axis in range(min(len(tile), 3)):
            if tile[axis] is not None:
                tile_values[axis] = float(tile[axis])
    return {
        "words": float(length) if length is not None else 0.0,
        "elements": float(length) if length is not None else 0.0,
        "m": tile_values[0],
        "n": tile_values[1],
        "k": tile_values[2],
    }


def cost_of(
    instr: Instruction,
    descriptor: object,
    tile: tuple[int, ...] | None = None,
    env: dict[str, int] | None = None,
) -> float:
    """The instruction's cost for this operand. Total, or `CostError`.

    Totality is the contract (postcondition 3): every cost must evaluate for
    every operand it is ever considered against. A zero-length access costs 0
    (an empty move is free); a division by a zero tile dimension is a schema
    error raised *here*, with the instruction's name on it.
    """
    cost_expr = getattr(instr, "select_cost", None) or getattr(instr, "cost", None)
    if isinstance(cost_expr, (int, float)) and not isinstance(cost_expr, bool):
        return float(cost_expr)
    if instr.rule == "mac" and tile is None:
        raise CostError(f"{instr.name}: cost needs a tile; got None")

    from .cost import CostQuery, CostResultError, CostUnknown, evaluate_cost
    query = CostQuery(instruction=instr, access=descriptor, tile=tile, env=env or {})
    try:
        res = evaluate_cost(query)
        return float(res.select_cost)
    except (CostUnknown, CostResultError) as err:
        raise CostError(str(err)) from err


@dataclass(frozen=True)
class MemorySpace:
    name: str
    kind: str  # flat | scratchpad | accumulator | banked
    dtype: str
    alignment_words: int
    banks: int = 1
    interleave_bytes: int = 4


@dataclass(frozen=True)
class DataModel:
    memory_spaces: tuple[MemorySpace, ...]
    tile_dims: tuple[str, ...]
    max_dims: int


@dataclass(frozen=True)
class AccumulateSpec:
    precision: str
    reduce_dim: str
    order: str


@dataclass(frozen=True)
class AddressingForm:
    form: str
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Instruction:
    """One instruction of one ISA, exactly as the YAML declared it.

    `admissible` is the *only* admissibility gate and it is fail-closed by
    construction: it is `evaluate(...) is True` — `"unknown"` is not `True`, so
    an undecidable constraint is inadmissible without a special case anywhere.
    """

    name: str
    kind: str  # memory | compute (the schema document's axis)
    addressing: AddressingForm
    constraint: Predicate
    cost: CostExpr
    semantics: str
    rule: str  # memory | mac | elementwise (what the selector enumerates by)
    time: CostExpr | None = None
    computation_attrs: tuple[str, ...] = ()
    accumulate: AccumulateSpec | None = None
    tile: dict[str, int] | None = None
    ops: tuple[str, ...] = ()
    declaration_index: int = 0
    is_async: bool = False
    completion: str | None = None
    sparse: bool = False
    compression_ratio: float | None = None
    format: str | None = None
    block_scale_size: int | None = None
    metadata_req: str | None = None
    encoding: dict[str, Any] | None = None
    backend: str | None = None
    direction: str | None = None
    select_cost: CostExpr | None = None
    #: Legal `(source space, destination space)` moves of a memory/async_copy
    #: instruction, read from its `transfers:` key. `None` means "not declared"
    #: (compute instructions); `()` means "moves no data" (a barrier).
    transfers: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if self.select_cost is None and self.cost is not None:
            object.__setattr__(self, "select_cost", self.cost)
        elif self.cost is None and self.select_cost is not None:
            object.__setattr__(self, "cost", self.select_cost)

    def reads_from(self, space: str) -> bool:
        """Can this instruction take data out of `space`?"""
        return any(src == space for src, _ in (self.transfers or ()))

    def writes_to(self, space: str) -> bool:
        """Can this instruction put data into `space`?"""
        return any(dst == space for _, dst in (self.transfers or ()))

    def serves(self, direction: str | None) -> bool:
        """Can this instruction lower a kernel-buffer `load` or `store`?

        Kernel buffers live in the `global` space, so a load needs a transfer that
        reads `global` and a store needs one that writes `global`. An instruction
        whose only transfers are scratch-side (LDS, STS, LDS2D) cannot serve either.
        """
        if direction is None:
            return True
        wanted = direction.lower()
        if self.direction is not None and self.direction.lower() != wanted:
            return False
        if self.transfers is None:
            return True
        if wanted == "load":
            return self.reads_from("global")
        if wanted == "store":
            return self.writes_to("global")
        return True

    def admissible_for(
        self,
        descriptor: object,
        tile: tuple[int, ...] | None = None,
        env: dict[str, int] | None = None,
    ) -> Verdict | str:
        return evaluate(self.constraint, descriptor, tile, env)

    def cost_for(
        self,
        descriptor: object,
        tile: tuple[int, ...] | None = None,
        env: dict[str, int] | None = None,
    ) -> float:
        return cost_of(self, descriptor, tile, env)


@dataclass(frozen=True)
class IsaSchema:
    schema_version: int
    name: str
    data_model: DataModel
    instructions: dict[str, Instruction]
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    csr_registers: dict[str, int] = field(default_factory=dict)

    def of_kind(self, rule: str) -> tuple[Instruction, ...]:
        """Candidates for one lowering set, in declaration order.

        Declaration order is the deterministic tie-break (selector contract
        postcondition 3), which is why this is a tuple and not a set.
        """
        found = [i for i in self.instructions.values() if i.rule == rule]
        found.sort(key=lambda i: i.declaration_index)
        return tuple(found)

    def evaluate(
        self,
        predicate: Predicate | str,
        descriptor: object,
        tile: tuple[int, ...] | None = None,
        env: dict[str, int] | None = None,
    ) -> Verdict | str:
        """The emitter's independent re-validation seam.

        `check_constraint` re-checks the chosen instruction's constraint through
        the schema rather than trusting selection's own verdict — the same
        fail-closed module-level `evaluate`, so `"unknown"` is a violation here
        too. Accepts a `Predicate` (as constraints arrive) or its text (as the
        emitted `Instr` records it), so a round-tripped program can be
        re-validated without re-parsing the schema.
        """
        resolved = predicate
        if isinstance(resolved, str):
            resolved = _parse_predicate(resolved)
        return evaluate(resolved, descriptor, tile, env)

    def instruction(self, name: str) -> Instruction | None:
        return self.instructions.get(name)


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #

DEFAULT_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "tritonflow1.yaml")

_KINDS = ("memory", "compute")
_RULES = ("memory", "mac", "elementwise", "async_copy")
_SPACE_KINDS = ("flat", "scratchpad", "accumulator")
_ORDERS = ("k_major_sequential", "k_blocked")


def load_schema(path: str | os.PathLike[str]) -> IsaSchema:
    """Load and validate a schema document. Raises `SchemaError`, never defaults."""
    import yaml

    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as error:
        raise SchemaError(f"cannot read schema file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise SchemaError(f"schema file {path} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise SchemaError(f"schema file {path} must be a mapping")
    schema = _build(raw, source=str(path))
    violations = validate_schema(schema)
    if violations:
        details = "; ".join(f"{v.path}: {v.problem}" for v in violations)
        raise SchemaError(f"schema {path} is invalid: {details}")
    # Register the machine's allocator promise: `aligned(X, k)` decides against
    # the promise of the schema being selected against, and a second machine may
    # promise differently from ISA-1's 4 (tritonflow2 declares 8, banked).
    promise = schema.data_model.memory_spaces[0].alignment_words if schema.data_model.memory_spaces else 4
    register_alignment_promise(schema.name, promise)
    return schema


def load_builtin(name: str = "tritonflow1") -> IsaSchema:
    """Load a schema shipped with the package (`tritonflow1`, later `tritonflow2`)."""
    path = os.path.join(os.path.dirname(__file__), "schemas", f"{name}.yaml")
    return load_schema(path)


def _build(raw: dict[str, Any], source: str) -> IsaSchema:
    """Raw YAML → `IsaSchema`. Parse failures raise `SchemaError` with the key path."""
    try:
        version = int(raw["schema_version"])
        name = str(raw["name"])
    except KeyError as error:
        raise SchemaError(f"{source}: missing required key {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise SchemaError(f"{source}: schema_version must be an integer") from error

    model_raw = raw.get("data_model") or {}
    spaces = tuple(
        MemorySpace(
            name=str(space["name"]),
            kind=str(space.get("kind", "flat")),
            dtype=str(space.get("dtype", "")),
            alignment_words=int(space.get("alignment_words", 1)),
            banks=int(space.get("banks", 1)),
            interleave_bytes=int(space.get("interleave_bytes", 4)),
        )
        for space in model_raw.get("memory_spaces", ())
    )
    model = DataModel(
        memory_spaces=spaces,
        tile_dims=tuple(model_raw.get("tile_dims", ())),
        max_dims=int(model_raw.get("max_dims", 4)),
    )

    instructions: dict[str, Instruction] = {}
    for index, entry in enumerate(raw.get("instructions", ())):
        try:
            instruction_name = str(entry["name"])
            kind = str(entry["kind"])
            constraint_text = entry.get("constraint")
            cost_text = (
                entry.get("cost", {}).get("expr") if isinstance(entry.get("cost"), dict) else None
            )
        except (KeyError, TypeError) as error:
            raise SchemaError(f"{source}: instructions[{index}] is malformed: {error}") from error
        if instruction_name in instructions:
            raise SchemaError(f"{source}: duplicate instruction name {instruction_name!r}")
        try:
            constraint = Predicate(str(constraint_text)) if constraint_text is not None else None
            cost = CostExpr(str(cost_text)) if cost_text is not None else None
            select_cost_raw = entry.get("select_cost")
            select_cost = CostExpr(str(select_cost_raw)) if select_cost_raw is not None else cost
            time_expr = None
            time_raw = entry.get("time")
            if isinstance(time_raw, dict) and time_raw.get("expr") is not None:
                time_expr = CostExpr(str(time_raw["expr"]))
            elif isinstance(time_raw, str):
                time_expr = CostExpr(time_raw)
        except PredicateError as error:
            raise SchemaError(f"{source}: instruction {instruction_name!r}: {error}") from error
        accumulate_raw = entry.get("accumulate")
        accumulate = (
            AccumulateSpec(
                precision=str(accumulate_raw.get("precision", "")),
                reduce_dim=str(accumulate_raw.get("reduce_dim", "")),
                order=str(accumulate_raw.get("order", "")),
            )
            if isinstance(accumulate_raw, dict)
            else None
        )
        tile_raw = entry.get("tile")
        addressing_raw = entry.get("addressing") or {}
        encoding_raw = entry.get("encoding")
        instructions[instruction_name] = Instruction(
            name=instruction_name,
            kind=kind,
            addressing=AddressingForm(
                form=str(addressing_raw.get("form", "")),
                fields=tuple(addressing_raw.get("fields", ())),
            ),
            constraint=constraint,
            cost=cost,
            semantics=str(entry.get("semantics", "")),
            rule=str(entry.get("rule", "")),
            time=time_expr,
            computation_attrs=tuple(entry.get("computation_attrs", ())),
            accumulate=accumulate,
            tile=dict(tile_raw) if isinstance(tile_raw, dict) else None,
            ops=tuple(entry.get("op", ())),
            declaration_index=index,
            is_async=bool(entry.get("async", False)),
            completion=entry.get("completion"),
            sparse=bool(entry.get("sparse", False)),
            compression_ratio=float(entry["compression_ratio"]) if "compression_ratio" in entry else None,
            format=entry.get("format"),
            block_scale_size=int(entry["block_scale_size"]) if "block_scale_size" in entry else None,
            metadata_req=entry.get("metadata_req"),
            encoding=dict(encoding_raw) if isinstance(encoding_raw, dict) else None,
            backend=entry.get("backend"),
            direction=entry.get("direction"),
            select_cost=select_cost,
            transfers=_parse_transfers(instruction_name, entry.get("transfers"), source),
        )

    csr_regs: dict[str, int] = {}
    for k, v in (raw.get("csr_registers") or {}).items():
        if isinstance(v, int):
            csr_regs[str(k)] = v
        else:
            csr_regs[str(k)] = int(str(v), 0)

    return IsaSchema(
        schema_version=version,
        name=name,
        data_model=model,
        instructions=instructions,
        description=str(raw.get("description", "")),
        config=dict(raw.get("config") or {}),
        csr_registers=csr_regs,
    )


def _parse_transfers(
    name: str, raw: Any, source: str
) -> tuple[tuple[str, str], ...] | None:
    """`transfers: ["global>register", ...]` -> `(("global", "register"), ...)`."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise SchemaError(f"{source}: instruction {name!r}: transfers must be a list, got {raw!r}")
    parsed: list[tuple[str, str]] = []
    for item in raw:
        parts = str(item).split(">")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise SchemaError(
                f"{source}: instruction {name!r}: transfer {item!r} must look like 'src>dst'"
            )
        parsed.append((parts[0].strip(), parts[1].strip()))
    return tuple(parsed)


#: Known schema versions. Forward compatibility is *not* assumed (contract
#: schema-errors table, last row): an unlisted version is a violation, because
#: interpreting a document under a grammar it was not written for is how a
#: predicate silently changes meaning.
_KNOWN_VERSIONS = (1,)


def validate_schema(schema: IsaSchema) -> list[SchemaViolation]:
    """Every invalidity, with the key path that owns it. Empty list == valid."""
    problems: list[SchemaViolation] = []

    if schema.schema_version not in _KNOWN_VERSIONS:
        problems.append(
            SchemaViolation(
                "schema_version",
                f"{schema.schema_version} is not a known version {_KNOWN_VERSIONS}",
            )
        )

    memory_count = sum(1 for i in schema.instructions.values() if i.kind == "memory")
    compute_instructions = [i for i in schema.instructions.values() if i.kind == "compute"]
    mac_count = sum(1 for i in compute_instructions if i.rule == "mac")
    if memory_count < 2:
        problems.append(
            SchemaViolation(
                "instructions", f"memory instruction count must be >= 2, is {memory_count}"
            )
        )
    if mac_count < 2:
        problems.append(
            SchemaViolation(
                "instructions",
                f"compute MAC instruction count must be >= 2, is {mac_count}",
            )
        )

    descriptor = _ValidationDescriptor()
    for instruction in schema.instructions.values():
        path = f"instructions.{instruction.name}"
        if instruction.constraint is None:
            problems.append(SchemaViolation(path, "has no constraint"))
        if instruction.cost is None:
            problems.append(SchemaViolation(path, "has no cost"))
        if instruction.kind not in _KINDS:
            problems.append(
                SchemaViolation(f"{path}.kind", f"{instruction.kind!r} is not one of {_KINDS}")
            )
        if not instruction.rule:
            # The coarse `kind` axis cannot drive selection (module docstring);
            # a schema that omits `rule` falls back to it only for compute.
            instruction = _with_rule(instruction)
        if instruction.rule not in _RULES:
            problems.append(
                SchemaViolation(f"{path}.rule", f"{instruction.rule!r} is not one of {_RULES}")
            )
        if (
            instruction.accumulate is None
            and instruction.kind == "compute"
            and instruction.rule == "mac"
        ):
            problems.append(
                SchemaViolation(
                    f"{path}.accumulate", "a MAC instruction must declare accumulate"
                )
            )
        if instruction.accumulate is not None:
            if not instruction.accumulate.order:
                problems.append(
                    SchemaViolation(
                        f"{path}.accumulate.order",
                        "missing order makes the emulator's tolerance unfalsifiable",
                    )
                )
            elif instruction.accumulate.order not in _ORDERS:
                problems.append(
                    SchemaViolation(
                        f"{path}.accumulate.order",
                        f"{instruction.accumulate.order!r} is not one of {_ORDERS}",
                    )
                )
        if instruction.tile is not None:
            for axis in ("m", "n"):
                value = instruction.tile.get(axis)
                if value is not None and int(value) <= 0:
                    problems.append(
                        SchemaViolation(
                            f"{path}.tile.{axis}", f"tile {axis} must be >= 1, is {value}"
                        )
                    )
        if instruction.rule in ("memory", "async_copy"):
            known_spaces = {space.name for space in schema.data_model.memory_spaces} | {"register"}
            if instruction.transfers is None:
                problems.append(
                    SchemaViolation(
                        f"{path}.transfers",
                        "a memory instruction must declare which spaces it moves data between "
                        "(use an empty list for one that moves none)",
                    )
                )
            else:
                for src, dst in instruction.transfers:
                    for space_name in (src, dst):
                        if space_name not in known_spaces:
                            problems.append(
                                SchemaViolation(
                                    f"{path}.transfers",
                                    f"space {space_name!r} is not declared in data_model "
                                    f"(known: {sorted(known_spaces)})",
                                )
                            )
        if instruction.constraint is not None:
            undefined = sorted(
                term for term in instruction.constraint.terms if term not in _KNOWN_TERMS
            )
            if undefined:
                problems.append(
                    SchemaViolation(
                        f"{path}.constraint",
                        f"references undefined term(s): {', '.join(undefined)}",
                    )
                )
            else:
                # Evaluate once against a null descriptor: every term is unknown,
                # so the only failure this can surface is a *malformed* predicate
                # (division by zero, a bad call form) — exactly the class that
                # must be a schema error and not a runtime surprise.
                try:
                    instruction.constraint.admissible() if hasattr(
                        instruction.constraint, "admissible"
                    ) else evaluate(instruction.constraint, descriptor)
                except PredicateError as error:
                    problems.append(SchemaViolation(f"{path}.constraint", str(error)))
        for cexpr, field_name in (
            (instruction.cost, "cost"),
            (getattr(instruction, "select_cost", None), "select_cost"),
            (instruction.time, "time"),
        ):
            if cexpr is not None:
                undefined = sorted(
                    term for term in cexpr.terms if not is_known_cost_term(term)
                )
                if undefined:
                    problems.append(
                        SchemaViolation(
                            f"{path}.{field_name}", f"references undefined term(s): {', '.join(undefined)}"
                        )
                    )
        # Task A6: semantics must be present and parseable
        if not instruction.semantics or not is_parseable(instruction.semantics):
            problems.append(
                SchemaViolation(
                    f"{path}.semantics",
                    f"instruction has missing or unparseable semantics: {instruction.semantics!r}",
                )
            )

        # Task C1: check that declared op entries can actually be expressed by instruction semantics
        if getattr(instruction, "ops", None) and instruction.semantics:
            sem = instruction.semantics.lower()
            unexpressible_by_arithmetic = {
                "constant", "get_program_id", "make_range", "splat",
                "cmpi", "select", "expand_dims", "broadcast"
            }
            for op in instruction.ops:
                op_str = str(op).lower()
                if op_str in unexpressible_by_arithmetic and op_str not in sem and not (op_str == "constant" and "value" in sem):
                    problems.append(
                        SchemaViolation(
                            f"{path}.op",
                            f"op {op!r} cannot be expressed by semantics {instruction.semantics!r}",
                        )
                    )
                elif ("+" in sem or "add" in sem) and "*" not in sem and "/" not in sem and "-" not in sem and "%" not in sem:
                    if op_str in ("mul", "mulf", "muli", "div", "divsi", "divui", "sub", "subf", "subi", "mod", "remsi", "remui"):
                        problems.append(
                            SchemaViolation(
                                f"{path}.op",
                                f"op {op!r} cannot be expressed by addition semantics {instruction.semantics!r}",
                            )
                        )
    return problems


def _with_rule(instruction: Instruction) -> Instruction:
    """The documented fallback: `rule` defaults from `kind`, `compute` → `mac`.

    The fallback is *deliberately* wrong for an EPI-style instruction so that a
    schema relying on it is visible: an elementwise compute instruction without
    an explicit `rule` would enumerate against MAC tiles and fail its own
    constraints. Explicit `rule` on every instruction is the honest document.
    """
    fallback = "mac" if instruction.kind == "compute" else "memory"
    return Instruction(
        name=instruction.name,
        kind=instruction.kind,
        addressing=instruction.addressing,
        constraint=instruction.constraint,
        cost=instruction.cost,
        semantics=instruction.semantics,
        rule=fallback,
        time=instruction.time,
        computation_attrs=instruction.computation_attrs,
        accumulate=instruction.accumulate,
        tile=instruction.tile,
        ops=instruction.ops,
        declaration_index=instruction.declaration_index,
    )


class _ValidationDescriptor:
    """A null descriptor: every term resolves to unknown.

    Its only job is to let `validate_schema` run each constraint once so a
    *malformed* predicate (division by zero, a bad call) is a schema error at
    load time rather than a runtime surprise (postcondition 3's failure-modes
    row). All terms being unknown is correct here: validity is about form.
    """

    base = None
    dtype = ""
    loop_carried = False
    increment = None
    provenance = ()

    def __init__(self) -> None:
        self.sizes = ()
        self.strides = ()
        self.offsets = ()
        self.shape = ()


#: The term vocabulary the predicate language recognizes. A predicate naming
#: anything else is undefined, and that must be named as such rather than
#: silently resolving to `unknown`.
_KNOWN_TERMS = frozenset(
    {
        "shape",
        "stride",
        "offset",
        "size",
        "base",
        "length",
        "words",
        "elements",
        "m",
        "n",
        "k",
        "dtype_bits",
        "acc_dtype",
        "op_dtype",
        "rank",
        # The MAC's pointer operands: `aligned(a_base, 4)`.
        # They resolve through the allocator contract in `aligned`, and are
        # unknown as arithmetic values.
        "a_base",
        "b_base",
        "desc_slot",
        "bar_id",
        "cta_mask",
        "sparse",
        "block_scale_size",
    }
)
_KNOWN_COST_TERMS = frozenset({
    "words",
    "elements",
    "bytes",
    "transactions",
    "coalescing_efficiency",
    "bank_conflicts",
    "contiguous",
    "alignment",
    "m",
    "n",
    "k",
    "mac_ops",
    "trip_count",
    "stride",
    "size",
    "offset",
    "shape",
    "base",
    "length",
})


def is_known_cost_term(term: str) -> bool:
    if term in _KNOWN_COST_TERMS:
        return True
    if term.startswith("machine."):
        return True
    try:
        from .cost import term_registry
        if term in term_registry():
            return True
    except Exception:
        pass
    return False

#: Terms that name allocated addresses rather than descriptor fields.
_POINTER_TERMS = frozenset({"base", "a_base", "b_base"})
