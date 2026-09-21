"""Integration: Inductor → Triton TTIR capture into the Dynamo seam.

Needs the `extract` extra (`torch`, `triton`). Skips cleanly without them.

Two behaviours are asserted:

1. When Inductor emits Triton TTIR (GPU/triton codegen path), provenance is
   ``inductor`` and the lowered result matches eager within the tf32 band.
2. When Inductor emits no Triton (typical CPU Inductor), the seam records an
   inductor note/fallback and still lowers via dynamic/recorded — it must not
   crash or invent TTIR.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tritonflow.extract import inductor_bridge as ib
from tritonflow.torch_backend import compiler as seam


def tf32_band(reduction: int) -> float:
    return reduction * (2.0 * 2.0**-11 + 2.0**-24)


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    peak = max(1e-9, float(want.detach().abs().max()))
    return float((got.detach() - want.detach()).abs().max()) / peak


def exported(fn, *args):
    graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(*args)
    return graph


def test_capture_inductor_ttir_is_honest_when_empty_or_populated() -> None:
    """Spy either returns TTIR with tt.dot, or an empty result with a reason."""
    a = torch.randn(64, 32)
    b = torch.randn(32, 64)

    def mm(x, y):
        return torch.mm(x, y)

    gm = torch.fx.symbolic_trace(mm)
    capture = ib.capture_inductor_ttir(gm, [a, b])
    if capture.ok:
        assert any("tt.dot" in r.ttir for r in capture.records), [
            r.name for r in capture.records
        ]
        extracted = ib.records_to_extracted(
            capture.records, "mm", [(64, 32), (32, 64)]
        )
        assert extracted is not None
        assert "tt.dot" in extracted.ttir
        kernel = seam.prepare(
            extracted.name, extracted.ttir, extracted.env, provenance="inductor"
        )
        assert kernel.provenance == "inductor"
        assert not kernel.program.markers()
    else:
        assert capture.reason
        assert any(
            key in capture.reason
            for key in ("TTIR", "Inductor", "CUDA", "importable", "compile")
        ), capture.reason


def test_plan_graph_prefers_inductor_or_notes_why() -> None:
    """plan_graph tries inductor before templates; notes explain misses."""
    a = torch.randn(128, 64)
    b = torch.randn(64, 128)

    def mm(x, y):
        return torch.mm(x, y)

    gm = exported(mm, a, b)
    plan = seam.plan_graph(gm, [a, b])
    assert plan.lowered or plan.fallbacks

    if plan.lowered:
        prov = plan.lowered[0].provenance
        assert prov in seam.EXTRACTED_PATHS
        if prov != "inductor":
            assert any(n.stage == "inductor" for n in plan.notes), plan.notes
    else:
        assert any(f.stage == "inductor" for f in plan.fallbacks) or any(
            f.stage in ("extract", "match", "flaggems") for f in plan.fallbacks
        )


def test_backend_matmul_matches_eager_with_inductor_or_fallback() -> None:
    """End-to-end: backend runs; numbers match eager; provenance is recorded."""
    a = torch.randn(64, 32)
    b = torch.randn(32, 64)

    def mm(x, y):
        return torch.mm(x, y)

    gm = exported(mm, a, b)
    run = seam.tritonflow_backend(gm, [a, b])
    out = run(a, b)
    if isinstance(out, (list, tuple)):
        out = out[0]
    want = mm(a, b)
    assert relative_error(torch.as_tensor(out), want) <= tf32_band(32)

    plan = getattr(run, "tritonflow_plan", None)
    assert plan is not None
    if plan.lowered:
        assert plan.lowered[0].provenance in seam.EXTRACTED_PATHS


def test_extract_via_inductor_synthetic_relu() -> None:
    """Per-op synthetic path used by multi-kernel interpret."""
    extracted, reason = ib.extract_via_inductor(
        None, None, "relu", [(8, 64)], has_bias=False
    )
    if extracted is None:
        assert reason
        return
    assert extracted.op == "relu"
    kernel = seam.prepare(
        extracted.name, extracted.ttir, extracted.env, provenance="inductor"
    )
    assert kernel.provenance == "inductor"
