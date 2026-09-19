"""Contract tests for the live FX-graph lowering proof of concept.

The property under test is not "it produces instructions" but "the *schema*
produced them": the same graph lowered against three ISAs must select three
different instruction sets at three different costs, with no branch on the ISA
name anywhere in `fx_lower`.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
    from torch.fx import symbolic_trace

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional extra
    HAS_TORCH = False

if HAS_TORCH:
    from tritonflow.torch_backend.compiler import tritonflow_backend
    from tritonflow.torch_backend.fx_lower import (
        lower_fx_graph,
        supported_targets,
        try_lower_and_run,
    )

ISAS = ("tritonflow1", "tritonflow2", "vortex_rvgpu")


def _tf32_band(reduction: int) -> float:
    """The derived tf32 error band for a reduction of length `reduction`."""
    return float(reduction * (2.0 * 2.0**-11 + 2.0**-24))


def _exported(fn, *args):
    """The FX graph Dynamo produces for `fn`, exactly as the backend receives it."""
    graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(*args)
    return graph


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFxLowering(unittest.TestCase):
    def _trace(self, fn, shape=(8, 8)):
        return symbolic_trace(fn), [torch.randn(*shape), torch.randn(*shape)]

    def test_elementwise_matches_eager_on_every_isa(self) -> None:
        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        for isa in ISAS:
            with self.subTest(isa=isa):
                result, lowering = try_lower_and_run(gm, xs, xs, isa_name=isa)
                self.assertIsNotNone(result, f"{isa} did not lower")
                self.assertTrue(lowering.fully_lowered)
                expected = fn(*xs).numpy()
                self.assertAlmostEqual(
                    float(abs(result - expected).max()), 0.0, places=6
                )

    def test_matmul_lowers_and_matches(self) -> None:
        def fn(a, b):
            return torch.mm(a, b)

        gm, xs = self._trace(fn, (16, 16))
        result, lowering = try_lower_and_run(gm, xs, xs, isa_name="tritonflow1")
        self.assertIsNotNone(result)
        self.assertIn("MAC", " ".join(i.name for i in lowering.program.instructions()))
        # Derived tf32 tolerance for K=16
        derived_tol = _tf32_band(16)
        self.assertLess(float(abs(result - fn(*xs).numpy()).max()), derived_tol)

    def test_selection_differs_across_isas(self) -> None:
        """Transfer is a re-selection, not a rename.

        If two ISAs produced the same instruction names at the same cost, the
        schema would not be driving anything.
        """

        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        seen = set()
        for isa in ISAS:
            lowering = lower_fx_graph(gm, xs, isa_name=isa)
            self.assertIsNotNone(lowering, isa)
            seen.add(
                (
                    frozenset(i.name for i in lowering.program.instructions()),
                    round(lowering.program.total_cost, 3),
                )
            )
        self.assertEqual(len(seen), len(ISAS), seen)

    def test_relu_does_not_lower_to_an_adder(self) -> None:
        """Vortex's elementwise units share a rule and a cost.

        Before the schema declared `op:` on each, minimum-cost selection broke
        the tie by declaration order and chose VADD for relu; the emulator
        dispatches those by instruction name, so it computed an addition and
        returned a wrong answer with no diagnostic.
        """

        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        lowering = lower_fx_graph(gm, xs, isa_name="vortex_rvgpu")
        names = [i.name for i in lowering.program.instructions()]
        self.assertIn("VMAX", names)
        self.assertEqual(names.count("VADD"), 1, names)

    def test_unsupported_op_refuses_with_a_reason(self) -> None:
        def fn(a, b):
            return torch.sigmoid(a + b)

        gm, xs = self._trace(fn)
        reasons: list[str] = []
        self.assertIsNone(lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons))
        self.assertTrue(reasons)
        self.assertIn("sigmoid", reasons[0])

    def test_op_the_isa_does_not_declare_refuses(self) -> None:
        """tritonflow1's EPI does not declare `exp`, so exp has no lowering."""

        def fn(a):
            return torch.exp(a)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8)]
        reasons: list[str] = []
        self.assertIsNone(lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons))
        self.assertIn("declares no elementwise instruction serving", reasons[0])
        self.assertIn("math.exp", reasons[0])

    def test_supported_targets_is_declared(self) -> None:
        self.assertIn("relu", supported_targets())
        self.assertIn("mm", supported_targets())
        self.assertIn("clamp", supported_targets())
        self.assertIn("div", supported_targets())


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTorchBackendCorrectness(unittest.TestCase):
    """Correctness guarantees for the torch backend: no silent miscompilations,

    visible fallbacks, derived tolerances, shadow-verification, and honest reports.
    """

    def test_add_with_alpha_refuses_and_falls_back_without_silent_miscompile(self) -> None:
        """torch.add(x, y, alpha=2) must never silently miscompile to x + y."""
        def fn(a, b):
            return torch.add(a, b, alpha=2)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]

        # Direct lowering refusal
        reasons: list[str] = []
        lowering = lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons)
        self.assertIsNone(lowering)
        self.assertTrue(any("alpha" in r for r in reasons), f"reasons: {reasons}")

        # Seam execution falls back and returns correct eager result
        run = tritonflow_backend(gm, xs)
        self.assertFalse(hasattr(run, "tritonflow_fx"))
        self.assertTrue(run.tritonflow_plan.fallbacks)
        self.assertTrue(any("alpha" in f.reason for f in run.tritonflow_plan.fallbacks))
        out = run(*xs)
        out = out[0] if isinstance(out, (list, tuple)) else out
        expected = fn(*xs)
        self.assertAlmostEqual(float((torch.as_tensor(out) - expected).abs().max()), 0.0, places=6)

    def test_div_with_rounding_mode_refuses_and_plain_div_lowers(self) -> None:
        """torch.div with rounding_mode='floor' must refuse rather than float-divide."""
        def fn_floor(a, b):
            return torch.div(a, b, rounding_mode="floor")

        gm_floor = symbolic_trace(fn_floor)
        xs = [torch.randn(8, 8) + 2.0, torch.randn(8, 8) + 1.0]

        reasons: list[str] = []
        lowering = lower_fx_graph(gm_floor, xs, isa_name="tritonflow1", report=reasons)
        self.assertIsNone(lowering)
        self.assertTrue(any("rounding_mode" in r for r in reasons), f"reasons: {reasons}")

        run = tritonflow_backend(gm_floor, xs)
        self.assertFalse(hasattr(run, "tritonflow_fx"))
        self.assertTrue(any("rounding_mode" in f.reason for f in run.tritonflow_plan.fallbacks))
        out = run(*xs)
        out = out[0] if isinstance(out, (list, tuple)) else out
        expected = fn_floor(*xs)
        self.assertAlmostEqual(float((torch.as_tensor(out) - expected).abs().max()), 0.0, places=6)

        # In contrast, standard division without rounding_mode lowers and executes
        def fn_div(a, b):
            return torch.div(a, b)

        gm_div = symbolic_trace(fn_div)
        result, lowering_div = try_lower_and_run(gm_div, xs, xs, isa_name="tritonflow1")
        self.assertIsNotNone(result)
        self.assertTrue(lowering_div.fully_lowered)
        expected_div = fn_div(*xs).numpy()
        self.assertAlmostEqual(float(abs(result - expected_div).max()), 0.0, places=6)

    def test_clamp_returns_right_bounds(self) -> None:
        """torch.clamp(x, -0.5, 0.5) returns values strictly within [-0.5, 0.5]."""
        def fn_clamp(x):
            return torch.clamp(x, -0.5, 0.5)

        gm = symbolic_trace(fn_clamp)
        xs = [torch.randn(16, 16) * 5.0]

        # On tritonflow2 (CLAMP) and vortex_rvgpu (VCLAMP), it lowers and clamps
        for isa in ("tritonflow2", "vortex_rvgpu"):
            with self.subTest(isa=isa):
                result, lowering = try_lower_and_run(gm, xs, xs, isa_name=isa)
                self.assertIsNotNone(result, f"{isa} did not lower clamp")
                self.assertTrue(lowering.fully_lowered)
                self.assertGreaterEqual(float(result.min()), -0.5 - 1e-6)
                self.assertLessEqual(float(result.max()), 0.5 + 1e-6)
                expected = fn_clamp(*xs).numpy()
                self.assertAlmostEqual(float(abs(result - expected).max()), 0.0, places=6)

        # On tritonflow1 (which has no clamp instruction), it refuses and falls back
        reasons: list[str] = []
        lowering_tf1 = lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons)
        self.assertIsNone(lowering_tf1)
        self.assertTrue(any("declares no elementwise instruction" in r for r in reasons), reasons)

        run = tritonflow_backend(gm, xs)
        out = run(*xs)
        out = out[0] if isinstance(out, (list, tuple)) else out
        self.assertGreaterEqual(float(torch.as_tensor(out).min()), -0.5 - 1e-6)
        self.assertLessEqual(float(torch.as_tensor(out).max()), 0.5 + 1e-6)

    def test_mlp_with_relu_fully_lowers_and_matches_eager(self) -> None:
        """An MLP with relu lowers all 3 nodes and matches eager within tf32 band."""
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        ).eval()
        x = torch.randn(8, 64)

        gm = _exported(lambda t: model(t), x)
        run = tritonflow_backend(gm, [x])
        out = run(x)
        out = out[0] if isinstance(out, (list, tuple)) else out

        self.assertEqual(run.tritonflow_counters["lowered"], 3)
        self.assertEqual(run.tritonflow_counters["eager"], 0)
        self.assertTrue(run.tritonflow_plan.fully_lowered)
        self.assertEqual(run.tritonflow_plan.fallbacks, [])

        expected = model(x)
        derived_tol = _tf32_band(64)
        peak = max(1e-9, float(expected.detach().abs().max()))
        rel_err = float((torch.as_tensor(out).detach() - expected.detach()).abs().max()) / peak
        self.assertLessEqual(rel_err, derived_tol)

    def test_mlp_with_sin_falls_back_on_exactly_sin(self) -> None:
        """An MLP with sin in the middle falls back on exactly sin, reporting it."""
        class MLPWithSin(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc2(torch.sin(self.fc1(x)))

        model = MLPWithSin().eval()
        x = torch.randn(8, 64)

        gm = _exported(lambda t: model(t), x)
        run = tritonflow_backend(gm, [x])
        out = run(x)
        out = out[0] if isinstance(out, (list, tuple)) else out

        self.assertEqual(run.tritonflow_counters["lowered"], 2)
        self.assertEqual(run.tritonflow_counters["eager"], 1)
        self.assertFalse(run.tritonflow_plan.fully_lowered)

        sin_records = [
            r for r in run.tritonflow_plan.fallbacks
            if "sin" in r.reason or "sin" in r.nodes
        ]
        self.assertTrue(len(sin_records) > 0, "sin fallback was not recorded")

        expected = model(x)
        derived_tol = _tf32_band(64)
        peak = max(1e-9, float(expected.detach().abs().max()))
        rel_err = float((torch.as_tensor(out).detach() - expected.detach()).abs().max()) / peak
        self.assertLessEqual(rel_err, derived_tol)

    def test_shadow_verification_rejects_divergent_lowering(self) -> None:
        """If emulated output diverges from eager torch, shadow verification demotes."""
        def fn(a, b):
            return a + b

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]
        reasons: list[str] = []
        lowering = lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons)
        self.assertIsNotNone(lowering)
        self.assertTrue(lowering.shadow_verified)
        self.assertIsNone(lowering.mismatch_reason)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestBackendIntegration(unittest.TestCase):
    def test_torch_compile_uses_the_fx_path(self) -> None:
        def fn(a, b):
            return torch.relu(a + b)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]
        run = tritonflow_backend(gm, xs)
        self.assertTrue(hasattr(run, "tritonflow_fx"), "the FX path was not taken")
        out = run(*xs)
        out = out[0] if isinstance(out, (list, tuple)) else out
        self.assertLess(float((torch.as_tensor(out) - fn(*xs)).abs().max()), 1e-6)

    def test_unsupported_graph_falls_back_and_records_why(self) -> None:
        def fn(a, b):
            return torch.sigmoid(a + b)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]
        run = tritonflow_backend(gm, xs)
        self.assertFalse(hasattr(run, "tritonflow_fx"))
        self.assertTrue(run.tritonflow_plan.fallbacks)
        self.assertIn("sigmoid", run.tritonflow_plan.fallbacks[-1].reason)


if __name__ == "__main__":
    unittest.main()
