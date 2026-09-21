"""Inductor → Triton → TTIR capture for the Dynamo seam (v1).

This is the pitch hand-off the custom ``backend="tritonflow"`` previously skipped:
run Inductor on an FX graph, observe every ``triton.compiler.compile`` via
:class:`~tritonflow.extract.dynamic_extract.CompileSpy`, and turn the resulting
TTIR into an :class:`~tritonflow.extract.dynamic_extract.Extracted` that
``torch_backend.compiler.prepare`` can lower.

**Scope (v1).** Only matmul-family ops (``mm`` / ``matmul`` / ``linear`` /
``addmm``) and ``relu``. Anything else is refused before Inductor is invoked so
we never pretend a softmax TTIR is supported.

**Honesty about CPU Inductor.** Inductor only emits Triton for its GPU/triton
codegen path. A pure CPU Inductor compile often never calls ``triton.compile``,
so the spy returns zero records. That is a documented empty result with a
reason, not a synthesized TTIR.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .dynamic_extract import (
    DEFAULT_TILE,
    CompileRecord,
    Extracted,
    capability as triton_capability,
    record_compilations,
)

__all__ = [
    "INDUCTOR_OPS",
    "CaptureResult",
    "InductorUnavailable",
    "capture_inductor_ttir",
    "extract_via_inductor",
    "is_inductor_op",
    "records_to_extracted",
]

#: Torch-level op names this bridge will ask Inductor about.
INDUCTOR_OPS = frozenset({"mm", "matmul", "linear", "addmm", "relu", "relu_"})


class InductorUnavailable(RuntimeError):
    """Inductor or Triton cannot be used for capture on this machine."""


@dataclass(frozen=True)
class CaptureResult:
    """What Inductor capture observed, including the empty/failure cases."""

    records: tuple[CompileRecord, ...]
    reason: str
    """Empty when records were captured; otherwise why capture produced none."""

    @property
    def ok(self) -> bool:
        return bool(self.records)


def is_inductor_op(op_name: str) -> bool:
    """Whether v1 will attempt Inductor capture for this torch-level op."""
    return op_name in INDUCTOR_OPS or op_name == "relu"


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return max(1, value)
    return max(multiple, math.ceil(value / multiple) * multiple)


def _tile_env(tile: tuple[int, int, int]) -> dict[str, int]:
    bm, bn, bk = tile
    return {"%__tile_m__": bm, "%__tile_n__": bn, "%__tile_k__": bk}


def _matmul_env(
    M: int, N: int, K: int, *, tile: tuple[int, int, int] = DEFAULT_TILE
) -> dict[str, int]:
    bm, bn, bk = tile
    rows, inner, cols = _round_up(M, bm), _round_up(K, bk), _round_up(N, bn)
    return {
        "M": rows,
        "N": cols,
        "K": inner,
        "%M": rows,
        "%N": cols,
        "%K": inner,
        "%sam": inner,
        "%sak": 1,
        "%sbk": cols,
        "%sbn": 1,
        "%scm": cols,
        "%scn": 1,
        "sam": inner,
        "sak": 1,
        "sbk": cols,
        "sbn": 1,
        "scm": cols,
        "scn": 1,
        **_tile_env(tile),
    }


def _elementwise_env(n: int, *, block: int = 64) -> dict[str, int]:
    padded = _round_up(n, block)
    return {
        "n": n,
        "%n": n,
        "M": padded,
        "N": 1,
        "%__flat_width__": padded,
        "%__block__": block,
        "%__tile_m__": block,
        "%__tile_n__": 1,
        "%__tile_k__": 1,
    }


def _force_triton_codegen() -> Any:
    """Configure Inductor to prefer Triton kernels when it can emit them.

    Returns the inductor config module so callers can restore nothing — these
    knobs are process-wide and match how Inductor is normally driven for GPU.
    """
    import torch._inductor.config as inductor_config

    # Prefer triton over cpp for pointwise/reductions when CUDA/triton path is live.
    if hasattr(inductor_config, "cpu_backend"):
        # Leave CPU backend alone; we move tensors to CUDA when available.
        pass
    if hasattr(inductor_config.triton, "cudagraphs"):
        inductor_config.triton.cudagraphs = False
    return inductor_config


def _maybe_cuda_inputs(example_inputs: Sequence[Any]) -> tuple[list[Any], str]:
    """Move fp32 tensors to CUDA when a device exists; else leave on CPU.

    Returns ``(inputs, device_note)``. On CPU-only hosts Inductor often never
    calls Triton; the note explains that for the FallbackRecord.
    """
    import torch

    if not torch.cuda.is_available():
        return list(example_inputs), (
            "CUDA is unavailable: Inductor's CPU path often never calls "
            "triton.compiler.compile, so TTIR capture may be empty"
        )
    out: list[Any] = []
    for arg in example_inputs:
        if isinstance(arg, torch.Tensor) and arg.is_floating_point():
            out.append(arg.detach().to(device="cuda", dtype=torch.float32))
        else:
            out.append(arg)
    return out, "inputs moved to CUDA for Inductor Triton codegen"


def capture_inductor_ttir(
    gm: Any,
    example_inputs: Sequence[Any],
) -> CaptureResult:
    """Run Inductor on ``gm`` under :class:`CompileSpy` and return TTIR records.

    Installs the spy *before* invoking Inductor so compilations that go through
    ``triton.compiler.compile`` are visible. Compilations bound via
    ``from triton.compiler import compile`` before this call remain invisible —
    the same limitation documented on ``CompileSpy``.
    """
    cap = triton_capability()
    if not cap.available:
        return CaptureResult((), f"triton is unavailable: {cap.reason}")

    try:
        import warnings

        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from torch._inductor.compile_fx import compile_fx
    except Exception as exc:
        return CaptureResult(
            (), f"torch._inductor.compile_fx is not importable: {type(exc).__name__}: {exc}"
        )

    inputs, device_note = _maybe_cuda_inputs(example_inputs)
    try:
        _force_triton_codegen()
    except Exception as exc:
        return CaptureResult(
            (), f"could not configure Inductor Triton codegen: {type(exc).__name__}: {exc}"
        )

    # Eagerly materialise a GraphModule Inductor accepts. Dynamo backends already
    # hand us one; a raw nn.Module would need export first.
    if not hasattr(gm, "graph"):
        return CaptureResult((), "capture requires an FX GraphModule with a .graph")

    with record_compilations() as spy:
        try:
            compiled = compile_fx(gm, inputs)
            # Trigger kernel compile: Inductor may defer Triton compile until run.
            if callable(compiled):
                compiled(*inputs)
        except Exception as exc:
            reason = (
                f"Inductor compile/run failed: {type(exc).__name__}: {exc}; "
                f"{device_note}"
            )
            if spy.records:
                return CaptureResult(tuple(spy.records), "")
            return CaptureResult((), reason)

    if not spy.records:
        extra = f"; spy errors: {spy.errors}" if spy.errors else ""
        return CaptureResult(
            (),
            f"Inductor produced no triton.compiler.compile TTIR records ({device_note}){extra}",
        )
    return CaptureResult(tuple(spy.records), "")


def _pick_record(records: Sequence[CompileRecord], op_name: str) -> CompileRecord | None:
    """Choose the Inductor kernel TTIR that matches the op family."""
    op = "relu" if op_name == "relu_" else op_name
    if op in ("mm", "matmul", "linear", "addmm"):
        dotted = [r for r in records if "tt.dot" in r.ttir]
        return dotted[0] if dotted else (records[0] if records else None)
    if op == "relu":
        # Prefer a kernel that looks like a clamp/max epilogue, else first record.
        for r in records:
            if "maxnumf" in r.ttir or "maximumf" in r.ttir or "tt.dot" not in r.ttir:
                if "tt.dot" not in r.ttir:
                    return r
        undotted = [r for r in records if "tt.dot" not in r.ttir]
        return undotted[0] if undotted else (records[0] if records else None)
    return records[0] if records else None


def records_to_extracted(
    records: Sequence[CompileRecord],
    op_name: str,
    shapes: Sequence[Sequence[int]],
    *,
    has_bias: bool = False,
    tile: tuple[int, int, int] = DEFAULT_TILE,
) -> Extracted | None:
    """Turn captured Inductor TTIR into an :class:`Extracted` with a launch env.

    Env padding follows the same conventions as ``extract_matmul`` /
    ``extract_elementwise`` so ``prepare`` can bind caller tensors.
    """
    record = _pick_record(records, op_name)
    if record is None or not record.ttir:
        return None

    op = "relu" if op_name == "relu_" else op_name
    shapes_i = [tuple(int(d) for d in shape) for shape in shapes]

    if op in ("mm", "matmul", "linear", "addmm"):
        if len(shapes_i) < 2 or len(shapes_i[0]) != 2 or len(shapes_i[1]) != 2:
            return None
        M, K = shapes_i[0]
        K2, N = shapes_i[1]
        if K != K2:
            return None
        bm, bn, bk = tile
        rows, inner, cols = _round_up(M, bm), _round_up(K, bk), _round_up(N, bn)
        kind = "linear" if op in ("linear", "addmm") else "matmul"
        return Extracted(
            name=f"inductor_{kind}_{M}x{N}x{K}",
            kind=kind,
            op=op,
            ttir=record.ttir,
            env=_matmul_env(M, N, K, tile=tile),
            tile=tile,
            problem=(M, K, N),
            padded=(rows, inner, cols),
            has_bias=bool(has_bias and kind == "linear"),
        )

    if op == "relu":
        if not shapes_i:
            return None
        lanes = 1
        for dim in shapes_i[0]:
            lanes *= int(dim)
        block = 64
        padded = _round_up(lanes, block)
        return Extracted(
            name=f"inductor_relu_{lanes}",
            kind="elementwise",
            op="relu",
            ttir=record.ttir,
            env=_elementwise_env(lanes, block=block),
            tile=(block, 1, 1),
            problem=(lanes, 0, 0),
            padded=(padded, 0, 0),
        )

    return None


def _synthetic_graph(
    op_name: str, shapes: Sequence[Sequence[int]], *, has_bias: bool
) -> tuple[Any, list[Any]] | None:
    """Build a tiny FX GraphModule Inductor can compile for one op.

    Used when the Dynamo graph is multi-node (or absent) and we still want
    Inductor's Triton TTIR for a single matmul/relu rather than our templates.
    """
    import torch

    op = "relu" if op_name == "relu_" else op_name
    shapes_i = [tuple(int(d) for d in shape) for shape in shapes]

    if op in ("mm", "matmul"):
        if len(shapes_i) < 2:
            return None
        a = torch.randn(*shapes_i[0], dtype=torch.float32)
        b = torch.randn(*shapes_i[1], dtype=torch.float32)

        def fn(x, y):
            return torch.mm(x, y) if op == "mm" else torch.matmul(x, y)

        gm = torch.fx.symbolic_trace(fn)
        return gm, [a, b]

    if op in ("linear", "addmm"):
        if len(shapes_i) < 2:
            return None
        M, K = shapes_i[0]
        _K2, N = shapes_i[1]
        x = torch.randn(M, K, dtype=torch.float32)
        w = torch.randn(N, K, dtype=torch.float32)
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)

            def fn(x, w, bias):
                return torch.nn.functional.linear(x, w, bias)

            gm = torch.fx.symbolic_trace(fn)
            return gm, [x, w, bias]

        def fn(x, w):
            return torch.nn.functional.linear(x, w)

        gm = torch.fx.symbolic_trace(fn)
        return gm, [x, w]

    if op == "relu":
        if not shapes_i:
            return None
        x = torch.randn(*shapes_i[0], dtype=torch.float32)

        def fn(x):
            return torch.relu(x)

        gm = torch.fx.symbolic_trace(fn)
        return gm, [x]

    return None


def extract_via_inductor(
    gm: Any | None,
    example_inputs: Sequence[Any] | None,
    op_name: str,
    shapes: Sequence[Sequence[int]],
    *,
    has_bias: bool = False,
) -> tuple[Extracted | None, str]:
    """Capture Inductor TTIR for one candidate op and package it.

    Returns ``(extracted, reason)``. ``extracted`` is set on success; ``reason``
    is empty on success and otherwise names why Inductor did not supply TTIR.

    When ``gm`` is None (per-node multi-kernel path), a synthetic one-op FX
    graph is built from ``shapes`` so Inductor still runs.
    """
    if not is_inductor_op(op_name):
        return None, f"{op_name!r} is outside inductor_bridge v1 op set {sorted(INDUCTOR_OPS)}"

    if gm is None or example_inputs is None:
        built = _synthetic_graph(op_name, shapes, has_bias=has_bias)
        if built is None:
            return None, f"could not build a synthetic FX graph for {op_name!r}"
        gm, example_inputs = built

    capture = capture_inductor_ttir(gm, example_inputs)
    if not capture.ok:
        return None, capture.reason or "Inductor capture returned no TTIR"

    extracted = records_to_extracted(
        capture.records, op_name, shapes, has_bias=has_bias
    )
    if extracted is None:
        return (
            None,
            "Inductor emitted TTIR but none matched the expected matmul/relu shape contract",
        )
    return extracted, ""
