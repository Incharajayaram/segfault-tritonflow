"""semantics.py — Executable semantics interpreter for ISA instruction specifications.

Every instruction in a schema declares what it computes via its `semantics:` string.
This module turns those declarative descriptions into an executable oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from tritonflow.emu.precision import PrecisionPolicy


class SemanticsParseError(ValueError):
    """Raised when a semantics string cannot be parsed into a known form."""


class SemanticsEvalError(RuntimeError):
    """Raised when evaluation of a parsed semantics expression fails."""


@dataclass(frozen=True)
class ParsedSemantics:
    raw: str
    kind: str  # "binary", "unary", "compare", "clamp", "mac", "memory_1d", "memory_2d", "async", "barrier"
    dst: str
    srcs: tuple[str, ...]
    details: dict[str, Any]


def _clean_semantics(s: str) -> str:
    """Strip comments and trailing commentary."""
    s = s.strip()
    s = re.sub(r"\s*—.*$", "", s)
    s = re.sub(r"\s*--.*$", "", s)
    s = re.sub(r"\s+\(k in.*?\)$", "", s)
    return s.strip()


def parse_semantics(raw: str) -> ParsedSemantics:
    """Parse an ISA semantics string into a structured ParsedSemantics object."""
    clean = _clean_semantics(raw)

    # 1. Barriers
    if clean.startswith("vx_bar"):
        m = re.match(r"^([a-zA-Z0-9_]+)\((.*?)\)$", clean)
        if m:
            func = m.group(1)
            args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
            return ParsedSemantics(raw=raw, kind="barrier", dst="", srcs=args, details={"func": func})

    # 2. Replicate / meta operations
    if clean.startswith("replicate_lmem_writes"):
        m = re.match(r"^replicate_lmem_writes\((.*?)\)$", clean)
        if m:
            args = tuple(a.strip() for a in m.group(1).split(",") if a.strip())
            return ParsedSemantics(raw=raw, kind="async", dst="lmem", srcs=args, details={"func": "replicate_lmem_writes"})

    if clean.startswith("VX_tcu_sp_meta"):
        return ParsedSemantics(
            raw=raw, kind="memory_1d", dst="VX_tcu_sp_meta", srcs=("lmem",), details={"op": "meta_ld"}
        )

    # 3. MAC / Matmul forms: acc[m,n] += sum_k ...
    if "+=" in clean or "sum_k" in clean:
        m = re.match(r"^([a-zA-Z0-9_]+)\[(.*?)\]\s*\+=\s*sum_k\s+(.*)$", clean)
        if m:
            dst = m.group(1)
            rhs = m.group(3).strip()
            is_tf32 = "tf32" in rhs
            is_sparse = "decompress" in rhs
            is_scaled = "scale_" in rhs
            return ParsedSemantics(
                raw=raw,
                kind="mac",
                dst=dst,
                srcs=("a", "b", dst),
                details={
                    "is_tf32": is_tf32,
                    "is_sparse": is_sparse,
                    "is_scaled": is_scaled,
                    "rhs": rhs,
                },
            )

    # Assignments
    if "=" not in clean:
        raise SemanticsParseError(f"No '=' found in semantics string: {raw!r}")

    lhs, rhs = [part.strip() for part in clean.split("=", 1)]

    # Constant materialisation: dst[0:length] = value or dst = value
    if rhs in ("value", "imm"):
        return ParsedSemantics(
            raw=raw, kind="constant", dst=lhs, srcs=(), details={"op": "constant"}
        )

    # 4. Async memory forms: scratch[...] = async_gmem_...
    if "async_gmem_" in rhs:
        m_call = re.match(r"^([a-zA-Z0-9_]+)\((.*?)\)$", rhs)
        if m_call:
            func = m_call.group(1)
            args = tuple(a.strip() for a in m_call.group(2).split(",") if a.strip())
            return ParsedSemantics(
                raw=raw,
                kind="async",
                dst=lhs,
                srcs=args,
                details={"func": func, "lhs": lhs},
            )

    # 5. Memory 2D: dst[i, 0:sizes[1]] = src[offsets[0]+i*strides[0], ...]
    if "offsets" in rhs or "strides" in rhs or ("i," in lhs or "i ," in lhs):
        return ParsedSemantics(
            raw=raw,
            kind="memory_2d",
            dst=lhs,
            srcs=(rhs,),
            details={"lhs": lhs, "rhs": rhs},
        )

    # 6. Memory 1D: dst[0:length] = src[0:length] or global[base:base+length] = src[0:length]
    if (":" in lhs and ":" in rhs) or ("0:length" in lhs or "0:length" in rhs) or ("base:base+length" in lhs or "base:base+length" in rhs) or ("smem_base:smem_base+length" in lhs):
        return ParsedSemantics(
            raw=raw,
            kind="memory_1d",
            dst=lhs,
            srcs=(rhs,),
            details={"lhs": lhs, "rhs": rhs},
        )

    # 7. Elementwise clamp: dst[i] = min(max(lo, src[i]), hi) or min(max(src[i], 0), 6)
    if "min(" in rhs and "max(" in rhs:
        m_clamp = re.search(r"min\s*\(\s*max\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*,\s*([^)]+)\)", rhs)
        if m_clamp:
            c1, c2, c3 = m_clamp.group(1).strip(), m_clamp.group(2).strip(), m_clamp.group(3).strip()
            # Could be (lo, src[i]), hi or (src[i], 0), 6
            if "src" in c2:
                src = c2
                lo = c1
            else:
                src = c1
                lo = c2
            hi = c3
            return ParsedSemantics(
                raw=raw,
                kind="clamp",
                dst=lhs,
                srcs=("src",),
                details={"lo": lo, "hi": hi, "src": src},
            )

    # 8. Elementwise max / min
    if "max(" in rhs and "min(" not in rhs:
        m_max = re.search(r"max\s*\(\s*([^,]+)\s*,\s*([^)]+)\)", rhs)
        if m_max:
            a1, a2 = m_max.group(1).strip(), m_max.group(2).strip()
            if "0" in (a1, a2):
                return ParsedSemantics(
                    raw=raw, kind="unary", dst=lhs, srcs=("src",), details={"op": "relu", "rhs": rhs}
                )
            return ParsedSemantics(
                raw=raw, kind="binary", dst=lhs, srcs=("src0", "src1"), details={"op": "max", "sym": "max"}
            )
    if "min(" in rhs and "max(" not in rhs:
        m_min = re.search(r"min\s*\(\s*([^,]+)\s*,\s*([^)]+)\)", rhs)
        if m_min:
            return ParsedSemantics(
                raw=raw, kind="binary", dst=lhs, srcs=("src0", "src1"), details={"op": "min", "sym": "min"}
            )



    # 8b. Elementwise unary neg / abs: dst[i] = -src[i] or -src0[i] or abs(src[i])
    if rhs.startswith("-src") or rhs.startswith("- src") or rhs.startswith("-"):
        return ParsedSemantics(
            raw=raw,
            kind="unary",
            dst=lhs,
            srcs=("src",),
            details={"op": "neg", "rhs": rhs},
        )
    if rhs.startswith("abs(") and rhs.endswith(")"):
        return ParsedSemantics(
            raw=raw,
            kind="unary",
            dst=lhs,
            srcs=("src",),
            details={"op": "abs", "rhs": rhs},
        )


    # Cast / Convert: dst[i] = cast(src0[i]) or dst[i] = cast(src[i])
    if rhs.startswith("cast(") and rhs.endswith(")"):
        return ParsedSemantics(
            raw=raw,
            kind="unary",
            dst=lhs,
            srcs=("src0",),
            details={"op": "cast", "rhs": rhs},
        )

    # Reduction: dst = reduce(src0)
    if rhs.startswith("reduce(") and rhs.endswith(")"):
        return ParsedSemantics(
            raw=raw,
            kind="unary",
            dst=lhs,
            srcs=("src0",),
            details={"op": "reduce", "rhs": rhs},
        )

    # 8c. Compare: dst[i] = cmp(pred, src0[i], src1[i]) -- the predicate is an operand
    if rhs.startswith("cmp(") and rhs.endswith(")"):
        return ParsedSemantics(
            raw=raw,
            kind="compare",
            dst=lhs,
            srcs=("src0", "src1"),
            details={"op": "compare", "rhs": rhs},
        )

    # 9. Elementwise generic op: dst[i] = op(src[i])
    if rhs.startswith("op(") and rhs.endswith(")"):
        return ParsedSemantics(
            raw=raw,
            kind="unary",
            dst=lhs,
            srcs=("src",),
            details={"op": "generic", "rhs": rhs},
        )

    # 10. Elementwise binary: dst[i] = src0[i] <op> src1[i]
    for op_sym, op_name in [
        ("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div"), ("%", "mod"),
        ("==", "eq"), ("!=", "ne"), ("<=", "le"), (">=", "ge"), ("<", "lt"), (">", "gt")
    ]:
        pattern = rf"^src0\[i\]\s*{re.escape(op_sym)}\s*src1\[i\]$"
        if re.match(pattern, rhs):
            return ParsedSemantics(
                raw=raw,
                kind="binary",
                dst=lhs,
                srcs=("src0", "src1"),
                details={"op": op_name, "sym": op_sym},
            )

    raise SemanticsParseError(f"Cannot parse semantics string: {raw!r}")


def is_parseable(raw: str | None) -> bool:
    """Return True if the semantics string can be parsed by parse_semantics."""
    if not raw or not isinstance(raw, str):
        return False
    try:
        parse_semantics(raw)
        return True
    except SemanticsParseError:
        return False


def eval_semantics(
    parsed_or_raw: ParsedSemantics | str,
    operands: dict[str, Any],
    env: dict[str, Any] | None = None,
    policy: PrecisionPolicy | None = None,
) -> Any:
    """Execute the parsed semantics expression on the provided operands."""
    parsed = parse_semantics(parsed_or_raw) if isinstance(parsed_or_raw, str) else parsed_or_raw
    env = env or {}
    policy = policy or PrecisionPolicy()

    if parsed.kind == "barrier":
        return None

    if parsed.kind == "binary":
        op = parsed.details["op"]
        s0 = operands.get("src0") if "src0" in operands else operands.get("in0")
        s1 = operands.get("src1") if "src1" in operands else operands.get("in1")
        if s0 is None or s1 is None:
            vals = [v for k, v in operands.items() if k not in ("dst", "out")]
            if len(vals) >= 2:
                s0, s1 = vals[0], vals[1]
            else:
                raise SemanticsEvalError(f"Binary op {op} requires two operands, got {list(operands.keys())}")
        a0 = np.asarray(s0)
        a1 = np.asarray(s1)
        if op == "add":
            return a0 + a1
        elif op == "sub":
            return a0 - a1
        elif op == "mul":
            return a0 * a1
        elif op == "div":
            return a0 / a1
        elif op == "mod":
            return np.mod(a0, a1)
        elif op == "max":
            return np.maximum(a0, a1)
        elif op == "eq":
            return np.equal(a0, a1)
        elif op == "ne":
            return np.not_equal(a0, a1)
        elif op == "lt":
            return np.less(a0, a1)
        elif op == "le":
            return np.less_equal(a0, a1)
        elif op == "gt":
            return np.greater(a0, a1)
        elif op == "ge":
            return np.greater_equal(a0, a1)
        elif op == "min":
            return np.minimum(a0, a1)
        raise SemanticsEvalError(f"Unknown binary op: {op}")

    if parsed.kind == "unary":
        op = parsed.details["op"]
        src = operands.get("src") if "src" in operands else operands.get("in0")
        if src is None:
            vals = [v for k, v in operands.items() if k not in ("dst", "out")]
            if vals:
                src = vals[0]
            else:
                raise SemanticsEvalError(f"Unary op requires operand, got {list(operands.keys())}")
        arr = np.asarray(src)
        if op == "relu":
            return np.maximum(0, arr)
        elif op == "neg":
            return -arr
        elif op == "abs":
            return np.abs(arr)
        elif op == "cast":
            return arr
        elif op == "reduce":
            return np.sum(arr)
        elif op == "generic":
            op_fn = env.get("op", lambda x: x)
            return op_fn(arr)
        raise SemanticsEvalError(f"Unknown unary op: {op}")

    if parsed.kind == "constant":
        return operands.get("value")

    if parsed.kind == "compare":
        from .predicates import PredicateError, evaluate

        family = env.get("compare_family")
        if family not in ("arith.cmpi", "arith.cmpf"):
            raise SemanticsEvalError(
                "a compare needs env['compare_family'] = 'arith.cmpi' | 'arith.cmpf'; "
                "the predicate number alone does not say which table applies"
            )
        if "predicate" not in operands:
            raise SemanticsEvalError("a compare needs a 'predicate' operand; there is no default")
        s0 = operands.get("src0") if "src0" in operands else operands.get("in0")
        s1 = operands.get("src1") if "src1" in operands else operands.get("in1")
        if s0 is None or s1 is None:
            raise SemanticsEvalError(f"compare needs two operands, got {list(operands.keys())}")
        try:
            return evaluate(family, int(operands["predicate"]), s0, s1)
        except PredicateError as error:
            raise SemanticsEvalError(str(error)) from error

    if parsed.kind == "clamp":
        src = operands.get("src") if "src" in operands else operands.get("in0")
        if src is None:
            vals = [v for k, v in operands.items() if k not in ("dst", "out", "lo", "hi")]
            if vals:
                src = vals[0]
            else:
                raise SemanticsEvalError(f"Clamp requires src operand, got {list(operands.keys())}")
        arr = np.asarray(src)
        if "lo" not in operands or "hi" not in operands:
            raise SemanticsEvalError(
                f"Clamp requires both 'lo' and 'hi' operands with no defaults, got {list(operands.keys())}"
            )
        lo_val = operands["lo"]
        hi_val = operands["hi"]
        try:
            lo = float(lo_val)
            hi = float(hi_val)
        except (TypeError, ValueError) as exc:
            raise SemanticsEvalError(f"Invalid clamp bounds: lo={lo_val}, hi={hi_val}") from exc
        return np.clip(arr, lo, hi)

    if parsed.kind == "mac":
        a = operands.get("a") if "a" in operands else operands.get("in0")
        b = operands.get("b") if "b" in operands else operands.get("in1")
        acc = operands.get("acc", operands.get("c"))
        if a is None or b is None:
            raise SemanticsEvalError(f"MAC requires 'a' and 'b', got {list(operands.keys())}")
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if parsed.details.get("is_tf32"):
            a_arr = policy.truncate(a_arr)
            b_arr = policy.truncate(b_arr)
        prod = a_arr @ b_arr
        if acc is not None:
            return np.asarray(acc) + prod
        return prod

    if parsed.kind in ("memory_1d", "memory_2d", "async"):
        src = operands.get("src") if "src" in operands else operands.get("global")
        if src is None:
            vals = [v for k, v in operands.items() if k not in ("dst", "scratch", "out")]
            src = vals[0] if vals else np.zeros(16, dtype=np.float32)
        return np.copy(np.asarray(src))

    raise SemanticsEvalError(f"Unhandled semantics kind: {parsed.kind}")
