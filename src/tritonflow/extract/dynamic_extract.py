"""Dynamic TTIR extraction, Path 1: real Triton compilation, no GPU required.

Two mechanisms, and they answer different questions.

**Direct extraction.** Triton's pip wheel carries the whole compiler, so a
kernel *source* can be compiled here and now against an explicit
``GPUTarget("cuda", 80, 32)`` and read back as ``compiled.asm["ttir"]``. No
driver, no device, no MLIR bindings to hold onto: the text comes out of the same
compiler that would have produced it on real hardware. This is what the seam
uses at plan time, and it is why the lowering for a shape nobody recorded is a
compilation and not a lookup.

**Compile interception.** :class:`CompileSpy` wraps ``triton.compiler.compile``
for the duration of a ``with`` block and records the TTIR of every compilation
that passes through it — including a compilation performed by somebody else's
codegen (Inductor's, for instance). It is an *observation* hook: it does not
rewrite the kernel, it records what was compiled. What it cannot see is a
compilation that happened before installation or through a direct
``from triton.compiler import compile`` alias captured earlier, and that limit
is stated rather than papered over.

**The launch environment is part of the contract.** A TTIR text says which
parameters exist; it does not say what they were bound to. ``prepare()`` in the
seam derives buffer extents from ``M``/``N``/``K`` and needs the *padded* extents
a launch actually addresses, while the caller holds the *logical* tensor. So an
:class:`Extracted` carries both — ``problem`` is what the caller has, ``padded``
is what the program addresses, ``tile`` is what is in between — and the seam is
what reconciles them. Getting this wrong is not a crash, it is a wrong-shaped
answer: an elementwise kernel handed ``n=1024`` with a buffer extent of one
block computes 64 lanes and the seam reports success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_TILE",
    "ELEMENTWISE_OPS",
    "SUPPORTED_OPS",
    "Capability",
    "CompileRecord",
    "CompileSpy",
    "Extracted",
    "ExtractionError",
    "ExtractionUnavailable",
    "capability",
    "compile_triton_kernel",
    "extract_elementwise",
    "extract_for_op",
    "extract_linear",
    "extract_matmul",
    "is_supported",
    "is_triton_available",
    "record_compilations",
    "require_triton",
]

#: The tile the extraction compiles against. Equal to the recorded Tier-1
#: kernel's tile on purpose: a dynamic lowering of a shape that *is* recorded
#: assembles the same program, which is what makes the two routes comparable
#: rather than merely both present.
DEFAULT_TILE = (64, 64, 32)

#: Torch-level op names the extractor can produce TTIR for.
ELEMENTWISE_OPS = frozenset({"add", "sub", "mul", "div", "relu", "neg", "abs", "clamp"})
_MATMUL_OPS = frozenset({"mm", "matmul", "linear", "addmm"})
SUPPORTED_OPS = frozenset(_MATMUL_OPS | ELEMENTWISE_OPS)


class ExtractionUnavailable(RuntimeError):
    """Triton is not installed, so nothing can be extracted.

    Distinct from :class:`ExtractionError` because the two call for different
    answers: an unavailable compiler means *degrade to the recorded lowerings
    and say so*, while a failed compilation of a kernel we asked for is a
    fallback with the compiler's own diagnostic attached.
    """


class ExtractionError(RuntimeError):
    """Triton is present and refused to compile what was asked of it."""


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Capability:
    """What the extraction can do here, and why when it cannot."""

    available: bool
    reason: str
    triton_version: str | None = None
    target: str | None = None

    def __bool__(self) -> bool:
        return self.available


_CAPABILITY: Capability | None = None


def capability(*, refresh: bool = False) -> Capability:
    """Probe for a usable Triton compiler, once per process unless refreshed.

    The probe imports and *calls* nothing on the public path: `import triton` is
    the expensive part and it is exactly what this decides.
    """
    global _CAPABILITY
    if _CAPABILITY is not None and not refresh:
        return _CAPABILITY
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource  # noqa: F401
        from triton.compiler import compile as _compile  # noqa: F401
    except Exception as exc:
        _CAPABILITY = Capability(
            available=False,
            reason=f"triton is not importable: {type(exc).__name__}: {exc}",
        )
        return _CAPABILITY
    version = getattr(triton, "__version__", "unknown")
    try:
        target = GPUTarget("cuda", 80, 32)
    except Exception as exc:
        _CAPABILITY = Capability(
            available=False,
            reason=f"triton {version} has no usable GPUTarget: {type(exc).__name__}: {exc}",
            triton_version=version,
        )
        return _CAPABILITY
    _CAPABILITY = Capability(
        available=True,
        reason=f"triton {version} can compile ahead of time for an explicit target",
        triton_version=version,
        target=str(target),
    )
    return _CAPABILITY


def is_triton_available() -> bool:
    """Whether dynamic extraction can run at all. Never raises."""
    return capability().available


def require_triton() -> Capability:
    """The capability, or :class:`ExtractionUnavailable` carrying the reason."""
    cap = capability()
    if not cap.available:
        raise ExtractionUnavailable(cap.reason)
    return cap


# --------------------------------------------------------------------------- #
# The kernel sources
# --------------------------------------------------------------------------- #
#
# Compiled lazily so that importing this module never needs Triton. The bodies
# are only defined when the compiler is present, because `@triton.jit` is the
# compiler's own decorator, not a marker.

_KERNELS: dict[str, Any] = {}


def _build_kernels() -> dict[str, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + (rm[:, None] * sam + rk[None, :] * sak)
        b_ptrs = b_ptr + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, input_precision="tf32")
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
        tl.store(c_ptrs, acc)

    @triton.jit
    def _linear_kernel(
        x_ptr, w_ptr, bias_ptr, out_ptr,
        M, N, K,
        sxm, sxk, swk, swn, som, son,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        x_ptrs = x_ptr + (rm[:, None] * sxm + rk[None, :] * sxk)
        w_ptrs = w_ptr + (rk[:, None] * swk + rn[None, :] * swn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, tl.cdiv(K, BK)):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc = tl.dot(x, w, acc, input_precision="tf32")
            x_ptrs += BK * sxk
            w_ptrs += BK * swk
        # `HAS_BIAS` is constexpr, so this branch is resolved by the compiler and
        # the load simply does not exist when there is no bias to add.
        if HAS_BIAS:
            bias = tl.load(bias_ptr + rn)
            acc = acc + bias[None, :]
        out_ptrs = out_ptr + (rm[:, None] * som + rn[None, :] * son)
        tl.store(out_ptrs, acc)

    # Four kernels, written out, rather than one factory taking a Python
    # callable: `@triton.jit` compiles the function *body*, so a lambda passed as
    # a default argument is not something the compiler can lower — it fails at
    # compile time, not at call time.

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def _sub_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x - y, mask=mask)

    @triton.jit
    def _mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * y, mask=mask)

    @triton.jit
    def _div_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x / y, mask=mask)

    @triton.jit
    def _relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        zero = 0.0
        tl.store(out_ptr + offs, tl.maximum(x, zero), mask=mask)

    @triton.jit
    def _neg_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, -x, mask=mask)

    @triton.jit
    def _abs_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, tl.abs(x), mask=mask)

    @triton.jit
    def _clamp_kernel(x_ptr, min_ptr, max_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        lo = tl.load(min_ptr + offs, mask=mask)
        hi = tl.load(max_ptr + offs, mask=mask)
        res = tl.minimum(tl.maximum(x, lo), hi)
        tl.store(out_ptr + offs, res, mask=mask)

    kernels: dict[str, Any] = {
        "matmul": _matmul_kernel,
        "linear": _linear_kernel,
        "relu": _relu_kernel,
        "neg": _neg_kernel,
        "abs": _abs_kernel,
        "clamp": _clamp_kernel,
        "bin_add": _add_kernel,
        "bin_sub": _sub_kernel,
        "bin_mul": _mul_kernel,
        "bin_div": _div_kernel,
    }
    return kernels


def _kernel(kind: str) -> Any:
    if kind not in _KERNELS:
        try:
            _KERNELS.update(_build_kernels())
        except (ImportError, ModuleNotFoundError) as err:
            raise ExtractionUnavailable(f"Triton is not available: {err}") from err
    return _KERNELS[kind]


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


def compile_triton_kernel(
    fn: Any,
    signature: Mapping[str, str],
    constexprs: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
) -> str:
    """Compile a Triton JIT kernel ahead of time and return its TTIR text.

    ``target`` is explicit, which is the whole trick: Triton resolves a target
    from the *driver* when none is given, and there is no driver here. Passing
    one means the compiler is exercised for real with no GPU present.

    Raises :class:`ExtractionUnavailable` when Triton is absent and
    :class:`ExtractionError` when it is present and refused — two different
    answers for two different situations, neither of them a silent ``None``.
    """
    require_triton()
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tt_compile

    source = ASTSource(fn=fn, signature=dict(signature), constexprs=dict(constexprs or {}))
    kwargs: dict[str, Any] = {"target": GPUTarget("cuda", 80, 32)}
    if options:
        kwargs["options"] = dict(options)
    try:
        compiled = tt_compile(source, **kwargs)
    except Exception as exc:
        name = getattr(fn, "__name__", str(fn))
        raise ExtractionError(f"{name}: triton refused to compile it: {exc}") from exc
    asm = getattr(compiled, "asm", None) or {}
    ttir = asm.get("ttir")
    if not ttir:
        raise ExtractionError(
            f"{getattr(fn, '__name__', fn)}: triton compiled it but exposed no 'ttir' "
            f"assembly (keys: {sorted(asm)})"
        )
    return str(ttir)


# --------------------------------------------------------------------------- #
# Compile interception
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompileRecord:
    """One compilation observed by :class:`CompileSpy`."""

    name: str
    ttir: str
    ttgir: str | None
    signature: Mapping[str, str] = field(default_factory=dict)
    constexprs: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"CompileRecord({self.name}, {len(self.ttir)} chars of ttir)"


class CompileSpy:
    """Observe ``triton.compiler.compile`` for the duration of a ``with`` block.

    Wraps the *module attribute* and restores it on exit. An alias bound by
    ``from triton.compiler import compile`` **before** the block started keeps
    pointing at the original function and therefore stays invisible; the raw
    module attribute is the only place this can honestly sit without reaching
    into bytecode.
    """

    def __init__(self) -> None:
        self.records: list[CompileRecord] = []
        self.errors: list[str] = []
        self._module: Any = None
        self._original: Any = None

    def __enter__(self) -> CompileSpy:
        try:
            import triton.compiler as tc
        except Exception as exc:
            self.errors.append(f"triton.compiler is not importable: {exc}")
            return self
        self._module = tc
        self._original = getattr(tc, "compile", None)
        if self._original is None:  # pragma: no cover - a Triton that moved its entry point
            self.errors.append("triton.compiler has no 'compile' attribute to wrap")
            return self
        spy = self
        original = self._original

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            compiled = original(*args, **kwargs)
            try:
                source = args[0] if args else kwargs.get("src")
                asm = getattr(compiled, "asm", None) or {}
                ttir = str(asm.get("ttir") or "")
                if ttir:
                    spy.records.append(
                        CompileRecord(
                            name=str(getattr(getattr(source, "fn", None), "__name__", "?")),
                            ttir=ttir,
                            ttgir=(str(asm["ttgir"]) if asm.get("ttgir") else None),
                            signature=dict(getattr(source, "signature", {}) or {}),
                            constexprs=dict(getattr(source, "constants", {}) or {}),
                        )
                    )
            except Exception as exc:
                spy.errors.append(f"could not record a compilation: {type(exc).__name__}: {exc}")
            return compiled

        tc.compile = wrapper
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        if self._module is not None and self._original is not None:
            self._module.compile = self._original
        return False


def record_compilations() -> CompileSpy:
    """A fresh :class:`CompileSpy`, for ``with record_compilations() as spy:``."""
    return CompileSpy()


# --------------------------------------------------------------------------- #
# Extraction per op
# --------------------------------------------------------------------------- #


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return max(1, value)
    return max(multiple, math.ceil(value / multiple) * multiple)


@dataclass(frozen=True)
class Extracted:
    """TTIR plus everything a launch needs to bind it to a caller's tensors.

    ``problem`` is the caller's logical shape, ``padded`` is what the program
    addresses, and ``tile`` is the granularity between them. All three are
    carried because the seam must pad by exactly the amount the program was
    compiled for: pad more and the extents no longer match, pad less and the
    index arithmetic walks off the buffer.
    """

    name: str
    kind: str
    op: str
    ttir: str
    env: Mapping[str, int]
    tile: tuple[int, int, int]
    problem: tuple[int, int, int]
    padded: tuple[int, int, int]
    has_bias: bool = False

    @property
    def is_elementwise(self) -> bool:
        return self.kind == "elementwise"

    @property
    def pads(self) -> tuple[int, int, int]:
        """`(rows, inner, cols)` padding, the amount the caller must add."""
        return tuple(p - q for p, q in zip(self.padded, self.problem, strict=False))  # type: ignore[return-value]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"Extracted({self.op}, problem={self.problem}, padded={self.padded}, "
            f"tile={self.tile}, {len(self.ttir.splitlines())} ttir lines)"
        )


def _tile_env(tile: tuple[int, int, int]) -> dict[str, int]:
    bm, bn, bk = tile
    return {"%__tile_m__": bm, "%__tile_n__": bn, "%__tile_k__": bk}


def extract_matmul(
    M: int, N: int, K: int, *, tile: tuple[int, int, int] = DEFAULT_TILE
) -> Extracted:
    """TTIR for `(M, K) @ (K, N)`, compiled for these extents.

    Strides are the contiguous ones for the *padded* buffers, because that is
    what the seam allocates: A is `(rows, inner)` so its k-stride is `inner`;
    B is `(inner, cols)` so its k-stride is `cols`; C is `(rows, cols)` so its
    row stride is `cols`. The previous version of this module carried the
    logical strides next to padded extents, which addresses the wrong rows.
    """
    if min(M, N, K) <= 0:
        raise ExtractionError(f"matmul extents must be positive, got M={M} N={N} K={K}")
    bm, bn, bk = tile
    rows, inner, cols = _round_up(M, bm), _round_up(K, bk), _round_up(N, bn)
    signature = {
        "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
        "M": "i32", "N": "i32", "K": "i32",
        "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32", "scm": "i32", "scn": "i32",
    }
    ttir = compile_triton_kernel(
        _kernel("matmul"), signature, {"BM": bm, "BN": bn, "BK": bk}
    )
    env = {
        "M": rows, "N": cols, "K": inner,
        "%M": rows, "%N": cols, "%K": inner,
        "sam": inner, "sak": 1, "sbk": cols, "sbn": 1, "scm": cols, "scn": 1,
        "%sam": inner, "%sak": 1, "%sbk": cols, "%sbn": 1, "%scm": cols, "%scn": 1,
        **_tile_env(tile),
    }
    return Extracted(
        name=f"dyn_matmul_{M}x{N}x{K}",
        kind="matmul",
        op="matmul",
        ttir=ttir,
        env=env,
        tile=tile,
        problem=(M, K, N),
        padded=(rows, inner, cols),
    )


def extract_linear(
    M: int,
    N: int,
    K: int,
    *,
    has_bias: bool = True,
    tile: tuple[int, int, int] = DEFAULT_TILE,
) -> Extracted:
    """TTIR for `x (M, K) @ w.T (K, N) + bias (N)`.

    `w` is the torch weight `(N, K)`; the caller transposes it, so the strides
    here describe `(K, N)` contiguous. The bias load sits outside the reduction
    loop and is padded to the tile width by the seam, which is why a bias of
    zeros is a no-op even when `cols > N`.
    """
    if min(M, N, K) <= 0:
        raise ExtractionError(f"linear extents must be positive, got M={M} N={N} K={K}")
    bm, bn, bk = tile
    rows, inner, cols = _round_up(M, bm), _round_up(K, bk), _round_up(N, bn)
    signature = {
        "x_ptr": "*fp32", "w_ptr": "*fp32", "bias_ptr": "*fp32", "out_ptr": "*fp32",
        "M": "i32", "N": "i32", "K": "i32",
        "sxm": "i32", "sxk": "i32", "swk": "i32", "swn": "i32", "som": "i32", "son": "i32",
    }
    ttir = compile_triton_kernel(
        _kernel("linear"),
        signature,
        {"BM": bm, "BN": bn, "BK": bk, "HAS_BIAS": bool(has_bias)},
    )
    env = {
        "M": rows, "N": cols, "K": inner,
        "%M": rows, "%N": cols, "%K": inner,
        "sxm": inner, "sxk": 1, "swk": cols, "swn": 1, "som": cols, "son": 1,
        "%sxm": inner, "%sxk": 1, "%swk": cols, "%swn": 1, "%som": cols, "%son": 1,
        **_tile_env(tile),
    }
    return Extracted(
        name=f"dyn_linear_{M}x{N}x{K}" + ("_bias" if has_bias else ""),
        kind="linear",
        op="linear",
        ttir=ttir,
        env=env,
        tile=tile,
        problem=(M, K, N),
        padded=(rows, inner, cols),
        has_bias=has_bias,
    )


def extract_elementwise(op: str, n: int, *, block: int = 64) -> Extracted:
    """TTIR for a 1-D elementwise op over `n` lanes.

    The env carries two different numbers on purpose. `n` is the *logical* lane
    count, because the kernel's masks are written against it and using the
    padded count would make every masked load read past the caller's tensor.
    `%__flat_width__` is the padded buffer extent, because that is what the
    seam allocates and what the program's addresses must stay inside.
    """
    if op not in ELEMENTWISE_OPS:
        raise ExtractionError(
            f"{op!r} is not a supported elementwise extraction; have {sorted(ELEMENTWISE_OPS)}"
        )
    if n <= 0:
        raise ExtractionError(f"elementwise lane count must be positive, got {n}")
    padded = _round_up(n, block)
    if op in ("relu", "neg", "abs"):
        signature = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
        kind = op
    elif op == "clamp":
        signature = {"x_ptr": "*fp32", "min_ptr": "*fp32", "max_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
        kind = op
    else:
        signature = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
        kind = f"bin_{op}"
    ttir = compile_triton_kernel(_kernel(kind), signature, {"BLOCK": block})
    env = {
        "n": n, "%n": n,
        "M": padded, "N": 1,
        "%__flat_width__": padded,
        "%__block__": block,
        "%__tile_m__": block, "%__tile_n__": 1, "%__tile_k__": 1,
    }
    return Extracted(
        name=f"dyn_{op}_{n}",
        kind="elementwise",
        op=op,
        ttir=ttir,
        env=env,
        tile=(block, 1, 1),
        problem=(n, 0, 0),
        padded=(padded, 0, 0),
    )


def is_supported(op_name: str) -> bool:
    """Whether the extractor has a kernel for this torch-level op name."""
    return op_name in SUPPORTED_OPS


def extract_for_op(
    op_name: str,
    shapes: Sequence[Sequence[int]],
    *,
    has_bias: bool | None = None,
    tile: tuple[int, int, int] = DEFAULT_TILE,
) -> Extracted | None:
    """Extract TTIR for a torch op and its operand shapes, or `None`.

    `None` means "this op/shape is outside what the extractor knows", which the
    seam turns into a `FallbackRecord` naming the op. It never means "the
    compiler failed": that raises, because a compilation that failed is a fact
    the report should carry and a wrong answer should not hide.
    """
    op_name = str(op_name)
    shapes = [tuple(int(d) for d in shape) for shape in shapes]
    if op_name in ("mm", "matmul", "linear", "addmm"):
        # **Canonical operand order, stated as a precondition**: `shapes[0]` is the
        # activation `(M, K)` and `shapes[1]` is `(K, N)`. The caller resolves which
        # tensor plays which role, because only the caller knows the op — Dynamo
        # hands a `linear` over as `(weight, x)` and an `addmm` as
        # `(bias, mat1, mat2)`, so deciding it here by shape alone would swap M and
        # N on one of them and emit a plausibly-shaped, wrong program.
        if len(shapes) < 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2:
            return None
        M, K = shapes[0]
        K2, N = shapes[1]
        if K != K2:
            return None
        if op_name in ("mm", "matmul"):
            return extract_matmul(M, N, K, tile=tile)
        # `linear` and `addmm` both carry a bias, and the linear kernel is the one
        # that adds it inside the program (`addmm(bias, mat1, mat2)` is exactly
        # `mat1 @ mat2 + bias`).
        bias = has_bias if has_bias is not None else (op_name == "addmm")
        return extract_linear(M, N, K, has_bias=bool(bias), tile=tile)
    if op_name in ELEMENTWISE_OPS:
        if not shapes:
            return None
        lanes = 1
        for dim in shapes[0]:
            lanes *= dim
        return extract_elementwise(op_name, lanes)
    return None
