"""Precision policy and derived tolerance.

This module makes one demand that rules out the usual shortcut: **a declared
precision must be implemented, not tolerated.** If the kernel's `tt.dot`
declares `inputPrecision = tf32`, the emulator truncates the multiply inputs to
tf32's mantissa width *before* multiplying, and the difference against an fp32
reference is then a *structural* quantity with a formula behind it — not noise
absorbed by a hand-picked epsilon.

Three facts drive the whole module.

1. **tf32 is fp32 with a shorter mantissa.** Same 8-bit exponent, 10 explicit
   mantissa bits instead of 23. The conversion therefore drops the low 13 bits,
   and round-half-to-even on those bits is the exact operation the tensor core
   performs on its inputs. That is `tf32_truncate`.
2. **The error is bounded, so a tolerance can be *derived*.** A truncated input
   is off by at most one half-ulp of the 10-bit mantissa (`2**-11` relative), a
   product of two truncated inputs by at most `2 * 2**-11`, and a reduction of
   length `n` compounds that `n` times. The fp32 accumulator adds its own
   `n * 2**-24`. Both terms are written out in :func:`derive_tolerance`, so the
   number in the report has a derivation next to it (postcondition 5) — a
   tolerance with no derivation is a contract violation, which is why
   :class:`Tolerance` carries the string rather than just the float.
3. **The accumulation order is declared, not chosen here.** Reassociation is
   then *not* an unmodelled error source (postcondition 4): if the emulator and
   the reference disagree, the cause is precision, and the tolerance is
   meaningful. `accumulate` walks the reduction in the declared order.

The measured anchor for the tf32 formula is on the record: on the RTX 4060 the
corpus matmul sits at `2.708e-02` vs eager fp32 at K=32 (`gpu_audit/`, E5).
`2 * 2**-11 * 32 = 3.125e-02` bounds it, which is the point — the formula is
conservative, and it is derived rather than fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Explicit mantissa bits: tf32 mantissa truncation is 10 explicit bits vs 23.
TF32_EXPLICIT_MANTISSA_BITS = 10
FP32_EXPLICIT_MANTISSA_BITS = 23

#: Unit roundoff of the two formats. `2**-(bits + 1)` is the half-ulp of a
#: format with `bits` explicit mantissa bits (the implicit leading 1 counts).
_TF32_HALF_ULP = 2.0 ** -(TF32_EXPLICIT_MANTISSA_BITS + 1)
_FP32_HALF_ULP = 2.0 ** -(FP32_EXPLICIT_MANTISSA_BITS + 1)

#: The two values `input_precision` may take.
InputPrecision = Literal["tf32", "ieee"]

#: Orders the ISA schema may declare (`isa/schemas/tritonflow1.yaml` declares
#: `k_major_sequential`; ISA-2 declares `k_blocked(4)`). The emulator implements
#: the sequential order; a declared order it does not implement is refused by
#: :meth:`PrecisionPolicy.accumulate` rather than silently reinterpreted.
SUPPORTED_ORDERS = ("k_major_sequential",)


@dataclass(frozen=True)
class Tolerance:
    """A derived tolerance, with the derivation that produced it.

    `value` is a **relative** bound: the comparison admits an absolute error up
    to `value * max(1, max|reference|)`. `absolute` is set for the integer and
    low-precision paths, where the contract requires exactness (`tolerance == 0`)
    and a relative reading would be wrong.
    """

    value: float
    derivation: str
    absolute: bool = False

    def bound(self, reference: np.ndarray) -> float:
        """The absolute error this tolerance admits for `reference`."""
        if self.absolute:
            return self.value
        peak = float(np.max(np.abs(np.asarray(reference, dtype=np.float64)))) if reference.size else 0.0
        return self.value * max(1.0, peak)

    def __str__(self) -> str:
        kind = "absolute" if self.absolute else "relative"
        return f"{self.value:.6g} ({kind}): {self.derivation}"


def derive_tolerance(
    input_precision: InputPrecision,
    reduction_length: int,
    dtype: str = "f32",
) -> Tolerance:
    """The tolerance for one reduction, and the formula behind it.

    Integer and low-precision paths are exact (`tolerance == 0`, absolute): a
    10-bit multiply is exact on 10-bit mantissas, and an integer dot is exact in
    fp32 up to 2**24. The float paths get the two-term bound described in the
    module docstring.
    """
    if reduction_length < 0:
        raise ValueError(f"reduction_length must be non-negative, got {reduction_length}")

    if input_precision not in ("tf32", "ieee"):
        raise ValueError(
            f"input_precision must be 'tf32' or 'ieee', got {input_precision!r}; "
            "an unknown precision has no defensible tolerance"
        )

    if "int" in dtype or dtype in ("i8", "i32", "i64", "u8", "u32"):
        return Tolerance(
            value=0.0,
            derivation=f"integer path ({dtype}): exact, tolerance is 0 by contract postcondition 5",
            absolute=True,
        )

    if input_precision == "ieee":
        value = reduction_length * _FP32_HALF_ULP
        return Tolerance(
            value=value,
            derivation=(
                f"ieee fp32: {reduction_length} sequential accumulation steps, each bounded "
                f"by the fp32 half-ulp 2**-{FP32_EXPLICIT_MANTISSA_BITS + 1} "
                f"= {_FP32_HALF_ULP:.3e}; "
                f"bound = {reduction_length} * {_FP32_HALF_ULP:.3e} = {value:.6e}"
            ),
        )

    truncation = 2.0 * _TF32_HALF_ULP
    accumulation = _FP32_HALF_ULP
    value = reduction_length * (truncation + accumulation)
    return Tolerance(
        value=value,
        derivation=(
            f"tf32 inputs truncated to {TF32_EXPLICIT_MANTISSA_BITS} explicit mantissa bits "
            f"({FP32_EXPLICIT_MANTISSA_BITS} dropped): each input off by at most the tf32 "
            f"half-ulp 2**-{TF32_EXPLICIT_MANTISSA_BITS + 1} = {_TF32_HALF_ULP:.3e}, so each "
            f"product by at most 2*{_TF32_HALF_ULP:.3e} = {truncation:.3e}; accumulated over "
            f"{reduction_length} products with fp32 step error {accumulation:.3e}: "
            f"bound = {reduction_length} * {truncation + accumulation:.3e} = {value:.6e}"
        ),
    )


def tf32_truncate(values: np.ndarray) -> np.ndarray:
    """Round `values` to tf32 precision, round-half-to-even.

    Implemented on the raw bits because that is what the hardware does and it is
    exactly reversible to reason about: fp32 and tf32 share the exponent field,
    so dropping the low 13 mantissa bits is a single mask plus the round bit.
    Non-finite values pass through untouched (adding the rounding term to an
    infinity would carry into the exponent field and turn it into a quiet NaN).
    """
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).copy()
    dropped = FP32_EXPLICIT_MANTISSA_BITS - TF32_EXPLICIT_MANTISSA_BITS
    mask = np.uint32((1 << dropped) - 1)
    # Round-half-to-even: add (half - 1) + the kept least-significant bit, then
    # mask the dropped bits off. Ties therefore round to the even mantissa.
    lsb = (bits >> np.uint32(dropped)) & np.uint32(1)
    rounding = np.uint32(1 << (dropped - 1)) - np.uint32(1) + lsb
    rounded = (bits + rounding) & ~mask
    finite = np.isfinite(array)
    bits = np.where(finite, rounded, bits)
    return bits.view(np.float32).reshape(array.shape)


def accumulate(
    terms: np.ndarray,
    axis: int,
    order: str = "k_major_sequential",
    precision: InputPrecision = "ieee",
) -> np.ndarray:
    """Sum `terms` along `axis` in the declared order, in fp32.

    `k_major_sequential` is the reduction the ISA-1 schema declares: walk the
    reduction axis in increasing index order, adding one fp32 term at a time.
    Deliberately a loop and not `np.sum`: NumPy sums pairwise, which is a
    *different* reassociation than the declared one, and using it here would
    put an unmodelled error source back into the comparison the tolerance is
    supposed to bound (postcondition 4).
    """
    if order not in SUPPORTED_ORDERS:
        raise ValueError(
            f"accumulation order {order!r} is not implemented; the emulator refuses to "
            f"reinterpret a declared order it does not perform (known: {SUPPORTED_ORDERS})"
        )
    moved = np.moveaxis(np.asarray(terms, dtype=np.float32), axis, 0)
    if moved.shape[0] == 0:
        return np.zeros(moved.shape[1:], dtype=np.float32)
    total = moved[0].astype(np.float32)
    for index in range(1, moved.shape[0]):
        total = (total + moved[index]).astype(np.float32)
    return total


@dataclass(frozen=True)
class PrecisionPolicy:
    """The precision contract one program is executed under.

    `accumulation_order` arrives from the ISA schema and `input_precision` from
    idiom detection's reading of `tt.dot` (`idioms/patterns.py::MatchResult`).
    Neither is chosen here: the emulator's job is to *implement* the declared
    pair so that a disagreement with the reference is attributable.
    """

    input_precision: InputPrecision = "ieee"
    accumulation_order: str = "k_major_sequential"
    reduction_length: int = 0

    def __post_init__(self) -> None:
        if self.input_precision not in ("tf32", "ieee"):
            raise ValueError(
                f"input_precision must be 'tf32' or 'ieee', got {self.input_precision!r}"
            )

    @classmethod
    def for_tile(
        cls,
        *,
        input_precision: InputPrecision,
        reduction_length: int,
        accumulation_order: str = "k_major_sequential",
    ) -> PrecisionPolicy:
        return cls(
            input_precision=input_precision,
            accumulation_order=accumulation_order,
            reduction_length=reduction_length,
        )

    def truncate(self, values: np.ndarray) -> np.ndarray:
        """Apply the declared input precision to a multiply operand."""
        if self.input_precision == "tf32":
            return tf32_truncate(values)
        return np.asarray(values, dtype=np.float32)

    def tolerance_for(self, dtype: str = "f32") -> Tolerance:
        """The tolerance for this policy's declared reduction length."""
        return derive_tolerance(self.input_precision, self.reduction_length, dtype)

    @property
    def derivation(self) -> str:
        """Human-readable, recorded in the report next to any compared number."""
        return self.tolerance_for().derivation

    def multiply_accumulate(self, a: np.ndarray, b: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """`acc += truncate(a) @ truncate(b)` under the declared order.

        The single place the emulator performs a tensor-core multiply, so the
        declared precision and the declared accumulation order cannot drift
        apart across call sites. Inputs are truncated **before** multiplying
        (postcondition 3), and the reduction is walked in order (postcondition 4).
        """
        left = np.asarray(self.truncate(a), dtype=np.float32)
        right = np.asarray(self.truncate(b), dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError(
                f"multiply_accumulate expects 2-D tiles, got {left.shape} and {right.shape}"
            )
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"reduction dimension disagrees: a has k={left.shape[1]}, b has k={right.shape[0]}"
            )
        total = np.asarray(acc, dtype=np.float32).copy()
        for k in range(left.shape[1]):
            total = (total + np.outer(left[:, k], right[k, :])).astype(np.float32)
        return total

    def compare(
        self,
        actual: np.ndarray,
        reference: np.ndarray,
        dtype: str = "f32",
    ) -> Comparison:
        """The differential check the precision policy requires."""
        a = np.asarray(actual, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        if a.shape != r.shape:
            raise ShapeMismatch(
                f"emulated result has shape {a.shape}, reference has {r.shape}",
                expected=str(r.shape),
                got=str(a.shape),
            )
        tolerance = self.tolerance_for(dtype)
        error = float(np.max(np.abs(a - r))) if a.size else 0.0
        bound = tolerance.bound(r)
        return Comparison(
            ok=bool(error <= bound),
            max_error=error,
            bound=bound,
            tolerance=tolerance,
            derivation=tolerance.derivation,
        )


@dataclass(frozen=True)
class Comparison:
    """What a differential comparison observed, derivation included."""

    ok: bool
    max_error: float
    bound: float
    tolerance: Tolerance
    derivation: str

    def __str__(self) -> str:
        verdict = "within" if self.ok else "BEYOND"
        return (
            f"max |diff|={self.max_error:.6e} {verdict} bound {self.bound:.6e} "
            f"[{self.tolerance.value:.6e} {self.tolerance.derivation}]"
        )


class ShapeMismatch(ValueError):
    """An input's shape disagrees with the descriptor that addresses it."""

    def __init__(self, message: str, *, expected: str, got: str) -> None:
        super().__init__(message)
        self.expected = expected
        self.got = got


__all__ = [
    "FP32_EXPLICIT_MANTISSA_BITS",
    "SUPPORTED_ORDERS",
    "TF32_EXPLICIT_MANTISSA_BITS",
    "Comparison",
    "InputPrecision",
    "PrecisionPolicy",
    "ShapeMismatch",
    "Tolerance",
    "accumulate",
    "derive_tolerance",
    "tf32_truncate",
]
