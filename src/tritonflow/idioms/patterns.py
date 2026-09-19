"""The two patterns, as data: what each one requires, and what a match looks like.

The patterns are written down as *requirements* rather than as
detection code because the requirements are what a reader has to check — "does
Tier 0 match the MAC pattern?" is answerable from this file alone, which is what
makes the two true negatives (Tier 0 and Tier 3) meaningful rather than
incidental.

**`multiplicity` is defined here, once.** Several matches of one pattern can exist
in one module (two independent `tt.dot`s:), and the contract requires both
to be reported rather than the first. So `multiplicity` counts the match *sites*
found for this pattern in one detection pass, and every `MatchResult` from that
pass carries the same number. A reader who sees only one of them cannot tell
"there was one" from "we stopped at the first".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..ttir.ssa import Operation, SsaValue

#: The pattern identifiers. `ReductionPattern.id` / `EpiloguePattern.id`.
MAC_ID = "mac"
EPILOGUE_ID = "epilogue"

#: `ReductionPattern.required` — the conjunction that makes a loop a MAC.
MAC_REQUIRED: Mapping[str, str] = MappingProxyType(
    {
        "loop": "one scf.for",
        "iter_arg": "an iter_arg of that loop used as tt.dot's accumulator operand",
        "dot": "exactly one tt.dot consuming that iter_arg",
        "yield": "scf.yield re-threading the dot's result at the accumulator's position",
    }
)

#: `ReductionPattern.optional` — present in the corpus, not required for a match.
MAC_OPTIONAL: tuple[str, ...] = (
    "a pre-load splat/broadcast chain building the pointer tile",
    "a post-loop elementwise epilogue",
)

#: `EpiloguePattern.required`.
EPILOGUE_REQUIRED: Mapping[str, str] = MappingProxyType(
    {
        "after": "operations applied to the accumulator after the loop",
        "reachable": "each one reachable from the loop's accumulator result",
    }
)

EPILOGUE_OPTIONAL: tuple[str, ...] = ("a further elementwise chain after the first op",)

#: `EpiloguePattern.classify`'s four values.
EPILOGUE_CLASSES = ("add", "sub", "mul", "relu", "other")

#: Op name → class. Names, not semantics: this table is the *decision* about what
#: counts as `add` and what counts as `relu`, and writing it as data means an
#: addition to it is a reviewed change rather than a new `elif`.
#:
#: `arith.maxnumf` is `relu`: Triton lowers `tl.maximum(x, 0)` to it, and the
#: distinction that matters is *which* clamp is used — `maxnumf` propagates a NaN
#: operand where `maximumf` does not, so they are different instructions and
#: conflating them would be a numerics difference dressed as a classification.
EPILOGUE_CLASS_OF: Mapping[str, str] = MappingProxyType(
    {
        "arith.addf": "add",
        "arith.addi": "add",
        "arith.subf": "sub",
        "arith.subi": "sub",
        "arith.mulf": "mul",
        "arith.muli": "mul",
        "arith.divf": "other",
        "arith.divsi": "other",
        "arith.maxnumf": "relu",
        "arith.maximumf": "relu",
        "arith.minnumf": "other",
        "math.exp": "other",
        "math.exp2": "other",
        "math.log": "other",
        "math.tanh": "other",
        "tt.sigmoid": "other",
    }
)

#: The `tt.dot` operand roles, positionally: `tt.dot %a, %b, %acc`.
DOT_OPERAND_ROLES: tuple[str, ...] = ("a", "b", "acc")

#: Operations that do *not* extend an epilogue chain: the memory path is the
#: DMA's, the rank manipulation belongs to whichever instruction consumes it, and
#: a container or a terminator is not an operation applied to a value at all.
#:
#: `scf.for` is in this tuple because of a real defect this table fixes: a second
#: loop that reads the first loop's accumulator was pulled into the first loop's
#: epilogue chain, so a two-dot module reported an epilogue classified `other`
#: whose only operation was the second `scf.for`. A container is *structure*, and
#: a chain of elementwise operations over the accumulator is not.
NON_COMPUTE_EPILOGUE_OPS: tuple[str, ...] = (
    "tt.load",
    "tt.store",
    "tt.func",
    "scf.for",
    "scf.if",
    "tt.return",
    "scf.yield",
)


def classify(op: Operation) -> str:
    """The epilogue class of one operation. Total: `"other"` when unnamed.

    `"other"` rather than a refusal, deliberately: an unsupported epilogue *op* is
    a property of the ISA, not of the analysis, and the coverage report wants to
    say "we saw an epilogue we could not classify" rather than crash on it.
    """
    return EPILOGUE_CLASS_OF.get(op.name, "other")


def tile_of(op: Operation) -> tuple[int, int, int] | None:
    """`(m, n, k)` of a `tt.dot`, read from the IR. `None` when not derivable.

    Read, not assumed: `tt.dot %a, %b, %acc : tensor<64x32xf32> * tensor<32x64xf32>
    -> tensor<64x64xf32>` gives `(64, 64, 32)` from the result and the left-hand
    operand's own types. A partially-known shape gives `None`, which is visibly
    wrong in a failure instead of plausible.
    """
    if not op.results or len(op.operands) < 2:
        return None
    result_shape = op.results[0].type.shape if op.results[0].type is not None else ()
    lhs_shape = op.operands[0].type.shape if op.operands[0].type is not None else ()
    if len(result_shape) < 2 or len(lhs_shape) < 2:
        return None
    m, n, k = result_shape[0], result_shape[1], lhs_shape[1]
    if m is None or n is None or k is None:
        return None
    return int(m), int(n), int(k)


@dataclass(frozen=True)
class MatchResult:
    """One idiom occurrence, plus the two fields that implies.

    `bindings` holds `SsaValue`s by role — `{"a": …, "b": …, "acc": …}` for a MAC —
    and `tile` holds `(m, n, k)`. Both are populated: `bindings["tile"]`
    is what an annotation reads, `tile_shape` is what a report prints.

    `classify` is `EpiloguePattern.classify`, carried on the match rather than
    on the pattern so a report can say which chain was classified *how*.
    """

    pattern_id: str
    ops: tuple[Operation, ...] = ()
    bindings: dict[str, object] = field(default_factory=dict)
    tile_shape: tuple[int, int, int] | None = None
    dtype: str = ""
    input_precision: str = "ieee"
    multiplicity: int = 1
    classify: str | None = None

    def __post_init__(self) -> None:
        if self.pattern_id not in (MAC_ID, EPILOGUE_ID):
            raise ValueError(
                f"unknown pattern id {self.pattern_id!r}; only "
                f"{MAC_ID!r} and {EPILOGUE_ID!r} are defined"
            )

    @property
    def is_mac(self) -> bool:
        return self.pattern_id == MAC_ID

    def value(self, role: str) -> SsaValue | None:
        got = self.bindings.get(role)
        return got if isinstance(got, SsaValue) else None

    def source_lines(self) -> tuple[int, ...]:
        return tuple(op.line for op in self.ops)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        roles = ", ".join(f"{name}={value}" for name, value in sorted(self.bindings.items()))
        return f"{self.pattern_id}[{roles}] x{self.multiplicity}"


__all__ = [
    "DOT_OPERAND_ROLES",
    "EPILOGUE_CLASSES",
    "EPILOGUE_CLASS_OF",
    "EPILOGUE_ID",
    "EPILOGUE_OPTIONAL",
    "EPILOGUE_REQUIRED",
    "MAC_ID",
    "MAC_OPTIONAL",
    "MAC_REQUIRED",
    "NON_COMPUTE_EPILOGUE_OPS",
    "MatchResult",
    "classify",
    "tile_of",
]
