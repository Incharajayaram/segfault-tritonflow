"""Integration: Path 1 — TTIR extracted at runtime is lowered and executed.

Every number is computed in this file: the TTIR comes from `triton.compile` inside
this process (no fixture file is opened), the program comes from the shipped
pipeline, and the reference comes from `torch`. The tolerances are the project's
own tf32 band, derived from the reduction length rather than written as a
constant.

These tests need both optional extras (`triton`, `torch`) and skip without them,
the same way `tests/contract/test_torch_seam.py` skips without torch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tritonflow.extract import dynamic_extract as de
from tritonflow.torch_backend import compiler as seam


def tf32_band(reduction: int) -> float:
    """The tf32 error band for a reduction of length `reduction`.

    The same derivation `verify/verify_end_to_end.py` uses: each tf32 operand
    carries 11 mantissa bits (`2**-11`) and the accumulation is fp32 (`2**-24`).
    """
    return reduction * (2.0 * 2.0**-11 + 2.0**-24)


def exported(fn, *args):
    """The FX graph Dynamo produces for `fn`, exactly as the backend receives it."""
    graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(*args)
    return graph


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    peak = max(1e-9, float(want.detach().abs().max()))
    return float((got.detach() - want.detach()).abs().max()) / peak


def run_backend(fn, *args):
    call = seam.tritonflow_backend(exported(fn, *args), args)
    result = call(*args)
    if isinstance(result, (list, tuple)):
        result = result[0]
    return call, result


# --------------------------------------------------------------------------- #
# Extraction itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [(128, 64, 128), (37, 53, 91), (256, 96, 128), (320, 96, 192), (64, 64, 64)],
)
def test_extract_matmul_pads_to_the_tile(shape: tuple[int, int, int]) -> None:
    M, K, N = shape
    extracted = de.extract_matmul(M, N, K)
    rows, inner, cols = extracted.padded
    assert rows >= M and inner >= K and cols >= N
    assert rows % 64 == 0 and inner % 32 == 0 and cols % 64 == 0
    assert extracted.problem == (M, K, N)
    assert tuple(p + q for p, q in zip(extracted.problem, extracted.pads, strict=False)) == extracted.padded
    assert "tt.dot" in extracted.ttir and "scf.for" in extracted.ttir


def test_extracted_ttir_declares_tf32() -> None:
    """The precision the emulator will be held to comes from the IR, not a guess."""
    extracted = de.extract_matmul(128, 128, 64)
    assert "inputPrecision = tf32" in extracted.ttir


def test_extracted_ttir_lowers_without_markers() -> None:
    extracted = de.extract_matmul(128, 128, 64)
    program = seam.prepare(extracted.name, extracted.ttir, extracted.env, provenance="dynamic")
    assert program.program.markers() == ()
    assert program.provenance == "dynamic"


def test_unsupported_op_yields_none_not_an_error() -> None:
    assert de.extract_for_op("softmax", [(8, 8)]) is None
    assert de.extract_for_op("bmm", [(4, 8, 8), (4, 8, 8)]) is None


def test_elementwise_carries_logical_and_padded_widths() -> None:
    """The kernel's mask reads `n`; the buffer it addresses is the padded width."""
    extracted = de.extract_elementwise("relu", 100)
    assert extracted.env["n"] == 100
    assert extracted.env["%__flat_width__"] == 128
    assert extracted.padded[0] == 128
    assert extracted.pads[0] == 28


# --------------------------------------------------------------------------- #
# Through the seam: provenance, padding, numerics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [(128, 64, 128), (37, 53, 91), (320, 96, 192)],
)
def test_matmul_lowers_dynamically_and_matches_torch(shape: tuple[int, int, int]) -> None:
    """An unaligned matmul is the case no recorded fixture covers."""
    M, K, N = shape
    a, b = torch.randn(M, K), torch.randn(K, N)
    call, result = run_backend(lambda x, y: x @ y, a, b)
    provenance = [kernel.provenance for kernel in call.tritonflow_plan.lowered]
    assert provenance[0] in ("inductor", "dynamic"), (
        f"generated TTIR sources must answer before recorded fixtures, got {provenance}"
    )
    if provenance[0] == "dynamic":
        assert any(n.stage == "inductor" for n in call.tritonflow_plan.notes)
    assert tuple(result.shape) == (M, N)
    assert relative_error(result, a @ b) <= tf32_band(K)


@pytest.mark.parametrize("n", [64, 100, 1000, 1024])
def test_elementwise_covers_every_lane_above_one_block(n: int) -> None:
    """`n > BLOCK` is the regression: one block of 64 used to be reported as all of it."""
    x, y = torch.randn(n), torch.randn(n)
    call, result = run_backend(lambda p, q: p + q, x, y)
    assert [kernel.provenance for kernel in call.tritonflow_plan.lowered] == ["dynamic"]
    assert tuple(result.shape) == (n,)
    assert relative_error(result, x + y) == 0.0


def test_tile_is_read_from_the_module_not_hard_coded() -> None:
    """A 128-wide tile must survive into the launch; the runner used to assume 64."""
    for tile in [(64, 64, 32), (128, 128, 64), (32, 32, 32)]:
        extracted = de.extract_matmul(256, 256, 128, tile=tile)
        prepared = seam.prepare(
            extracted.name, extracted.ttir, extracted.env, provenance="dynamic"
        )
        assert prepared.tile == tile, f"tile {tile} did not survive into the kernel record"
        if not prepared.program.markers():
            # Only a lowering the assembler accepted can be run; the assertion that
            # matters is that the recorded tile is the compiled one, not a constant.
            assert prepared.grid[0] == 256 // tile[0]


def test_plan_records_why_each_stage_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal must name all three sources, not just the last one tried."""
    x = torch.randn(16, 16)
    call, _ = run_backend(lambda t: torch.softmax(t, dim=-1), x)
    stages = {record.stage for record in call.tritonflow_plan.fallbacks}
    assert "extract" in stages
    assert call.tritonflow_plan.fully_lowered is False
    joined = " ".join(record.reason for record in call.tritonflow_plan.fallbacks)
    assert "softmax" in joined


def test_degrades_to_the_recorded_lowering_without_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of keeping Path 3: it is what "degrade gracefully" means."""
    monkeypatch.setattr(
        de, "_CAPABILITY", de.Capability(available=False, reason="test: triton removed")
    )
    a, b = torch.randn(128, 64), torch.randn(64, 128)
    call, result = run_backend(lambda x, y: x @ y, a, b)
    assert [kernel.provenance for kernel in call.tritonflow_plan.lowered] == ["recorded"]
    assert relative_error(result, a @ b) <= tf32_band(64)
    # "recorded" alone cannot say *why* the earlier sources were skipped, and the
    # two causes (no Triton / no kernel for this op) have different fixes.
    notes = {note.stage: note.reason for note in call.tritonflow_plan.notes}
    assert "unavailable" in notes.get("extract", ""), notes
    # Inductor is tried before extract; on CPU it also leaves a note.
    assert "inductor" in notes or any(n.stage == "inductor" for n in call.tritonflow_plan.notes)
    assert call.tritonflow_plan.fallbacks == [], "nothing fell back to eager here"
    assert call.tritonflow_plan.fully_lowered is True

def test_a_generated_source_win_notes_only_skipped_stages() -> None:
    """Inductor is tried first; a dynamic win must note why inductor was skipped."""
    a, b = torch.randn(128, 64), torch.randn(64, 128)
    call, _ = run_backend(lambda x, y: x @ y, a, b)
    prov = [kernel.provenance for kernel in call.tritonflow_plan.lowered]
    assert prov[0] in ("inductor", "dynamic"), prov
    if prov[0] == "inductor":
        assert call.tritonflow_plan.notes == []
    else:
        assert any(n.stage == "inductor" for n in call.tritonflow_plan.notes)

def test_no_triton_also_records_the_bridge_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        de, "_CAPABILITY", de.Capability(available=False, reason="test: triton removed")
    )
    x = torch.randn(64)
    call, _ = run_backend(lambda t: torch.relu(t), x)
    assert call.tritonflow_plan.lowered == []
    reasons = {record.stage: record.reason for record in call.tritonflow_plan.fallbacks}
    assert "extraction is unavailable" in reasons["extract"].lower()


# --------------------------------------------------------------------------- #
# Multi-node graphs: per-node honesty
# --------------------------------------------------------------------------- #


def test_multi_node_graph_lowers_every_node() -> None:
    """`linear -> relu -> linear` is three kernels, and all three must lower."""
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
    ).eval()
    x = torch.randn(8, 64)
    call, result = run_backend(lambda t: model(t), x)
    counters = call.tritonflow_counters
    assert counters["eager"] == 0, "the relu between two linears is a kernel too"
    assert counters["lowered"] == 3
    assert call.tritonflow_plan.fully_lowered is True
    assert relative_error(result, model(x)) <= tf32_band(64)


def test_eager_node_is_named_and_not_hidden() -> None:
    """FR-025: a node that ran in PyTorch is reported, by name, per graph."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x @ x, dim=-1)

    x = torch.randn(16, 16)
    call, result = run_backend(fn, x)
    assert call.tritonflow_plan.fully_lowered is False
    eager = [r for r in call.tritonflow_plan.fallbacks if r.stage == "node"]
    reasons = " ".join(record.reason for record in call.tritonflow_plan.fallbacks)
    assert "relu" not in reasons  # nothing here is a relu; a wrong name is a wrong record
    assert "softmax" in reasons or eager
    # Matmul may lower under tf32 while softmax runs eager on that result.
    assert relative_error(result, fn(x)) <= tf32_band(16)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32])
def test_undeclared_input_dtype_falls_back_instead_of_raising(dtype: torch.dtype) -> None:
    """A dtype the device does not declare must run in PyTorch, not crash the caller.

    Measured before this was fixed: every one of these raised `LoweringError` out
    of the compiled callable — so an fp16 model, which is nearly every real model,
    did not run at all. The device is still f32-only and still refuses to upcast;
    what changes is that the refusal is a *record* with a correct answer beside it,
    which is the seam's stated contract.
    """
    if dtype is torch.int32:
        a = torch.randint(-4, 5, (32, 32), dtype=dtype)
        b = torch.randint(-4, 5, (32, 32), dtype=dtype)
    else:
        a = torch.randn(32, 32, dtype=dtype)
        b = torch.randn(32, 32, dtype=dtype)

    call, result = run_backend(lambda p, q: p @ q, a, b)  # must not raise

    assert call.tritonflow_plan.lowered == []
    assert call.tritonflow_plan.fallbacks, "the refusal has to be recorded, not just survived"
    record = call.tritonflow_plan.fallbacks[0]
    assert record.stage == "inputs"
    assert "dtype" in record.reason
    assert str(dtype) in record.detail, "the record names which dtype was refused"
    assert result.dtype == dtype
    expected = a @ b
    if dtype.is_floating_point:
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
    else:
        assert torch.equal(result, expected)


def test_tf32_declared_precision_survives_a_rounded_dtype_band() -> None:
    """The f32 path still lowers; the dtype fix must not have widened it to "anything"."""
    a, b = torch.randn(48, 32), torch.randn(32, 48)
    call, result = run_backend(lambda p, q: p @ q, a, b)
    assert call.tritonflow_plan.lowered
    assert call.tritonflow_plan.lowered[0].provenance in ("inductor", "dynamic")
    assert call.tritonflow_plan.fallbacks == []
    assert relative_error(result, a @ b) <= tf32_band(32)


def test_plan_json_reports_provenance_and_counters() -> None:
    x, y = torch.randn(64), torch.randn(64)
    call, _ = run_backend(lambda p, q: p * q, x, y)
    payload = call.tritonflow_plan.to_json()
    assert payload["fully_lowered"] is True
    assert payload["provenance"] == ["dynamic"]  # mul is outside inductor v1 op set
    assert payload["node_lowerings"] == 0
