"""E-graph selection against real ISA schemas, and the synthetic target.

Two things are locked in here.

First, the measured negative result: on all three shipped schemas the e-graph
ties the greedy selector on every arithmetic island in every fixture. It never
wins, because those schemas name a 1:1 instruction for every operation the
fixtures use and price every elementwise instruction at the same flat rate, so no
rewrite can reduce the cost. If that ever changes -- a schema gains an
instruction the IR does not name, or a non-flat cost -- `test_egraph_ties_greedy`
starts failing and the claim in `reports/egraph_selection.md` has to be rewritten
rather than quietly drifting out of date.

Second, that the mechanism nonetheless works: against a clearly-labelled
synthetic target with a cost structure the real schemas lack, the e-graph both
finds a cheaper lowering and lowers a term greedy has to refuse.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tritonflow.isa.egraph import (
    INFEASIBLE,
    Expr,
    build_exprs_from_module,
    egraph_select,
    greedy_cost,
    leaf,
)
from tritonflow.isa.schema import load_builtin, load_schema
from tritonflow.ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parent.parent
ISAS = ("tritonflow1", "tritonflow2", "vortex_rvgpu")
TIERS = ("t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo")
SYNTHETIC = ROOT / "tests" / "data" / "synthetic_fused.yaml"


def _islands(tier: str):
    path = ROOT / "fixtures" / f"{tier}.ttir"
    module = parse_module(path.read_text(encoding="utf-8"), source_path=str(path)).module
    return build_exprs_from_module(module)


def _env(tier: str) -> dict[str, int]:
    launch = json.loads((ROOT / "fixtures" / "launch_env.json").read_text())
    return {k: v for k, v in launch.get(tier, {}).items() if isinstance(v, int)}


class TestAgainstRealSchemas(unittest.TestCase):
    def test_fixtures_yield_arithmetic_islands(self) -> None:
        """Guard the premise: a comparison over an empty set would prove nothing."""
        for tier in TIERS:
            with self.subTest(tier=tier):
                self.assertTrue(_islands(tier), f"{tier} produced no arithmetic island")

    def test_egraph_never_costs_more_than_greedy(self) -> None:
        """Property: saturation retains the original term, so it cannot do worse.

        The original form is always in the e-graph, so min-cost extraction is over
        a superset of what greedy considers. A violation means extraction is
        wrong, not that the e-graph made a worse choice.
        """
        for tier in TIERS:
            env = _env(tier)
            for isa in ISAS:
                schema = load_builtin(isa)
                for isl in _islands(tier):
                    kw = dict(
                        env=env, dtype_class=isl.dtype_class,
                        sizes=isl.sizes, dtype=isl.dtype,
                    )
                    g = greedy_cost(schema, isl.expr, **kw)
                    e = egraph_select(schema, isl.expr, **kw).cost
                    with self.subTest(tier=tier, isa=isa, expr=str(isl.expr)):
                        self.assertLessEqual(
                            e, g, f"e-graph {e} worse than greedy {g} for {isl.expr}"
                        )

    def test_egraph_ties_greedy_on_every_shipped_schema(self) -> None:
        """The measured result: no rewrite pays on any real fixture.

        This is an honest negative. It is asserted so that a schema change which
        makes rewriting worthwhile shows up as a failure here.
        """
        for tier in TIERS:
            env = _env(tier)
            for isa in ISAS:
                schema = load_builtin(isa)
                for isl in _islands(tier):
                    kw = dict(
                        env=env, dtype_class=isl.dtype_class,
                        sizes=isl.sizes, dtype=isl.dtype,
                    )
                    g = greedy_cost(schema, isl.expr, **kw)
                    e = egraph_select(schema, isl.expr, **kw).cost
                    with self.subTest(tier=tier, isa=isa, expr=str(isl.expr)):
                        self.assertEqual(
                            g, e,
                            "shipped schemas are expected to give greedy the optimum; "
                            "if this now differs, update reports/egraph_selection.md",
                        )

    def test_saturation_reaches_a_fixed_point_on_every_fixture(self) -> None:
        """No fixture island is large enough to hit a bound, so no result is truncated."""
        for tier in TIERS:
            env = _env(tier)
            schema = load_builtin("vortex_rvgpu")
            for isl in _islands(tier):
                sel = egraph_select(
                    schema, isl.expr, env=env, dtype_class=isl.dtype_class,
                    sizes=isl.sizes, dtype=isl.dtype,
                )
                with self.subTest(tier=tier, expr=str(isl.expr)):
                    self.assertEqual(sel.saturation.status, "saturated")

    def test_every_island_is_priced_not_refused(self) -> None:
        """Vortex names every arithmetic op the fixtures use, so none may refuse."""
        schema = load_builtin("vortex_rvgpu")
        for tier in TIERS:
            env = _env(tier)
            for isl in _islands(tier):
                cost = greedy_cost(
                    schema, isl.expr, env=env, dtype_class=isl.dtype_class,
                    sizes=isl.sizes, dtype=isl.dtype,
                )
                with self.subTest(tier=tier, expr=str(isl.expr)):
                    self.assertLess(cost, INFEASIBLE, f"unexpected refusal for {isl.expr}")


class TestSyntheticTarget(unittest.TestCase):
    """The mechanism works where the cost structure rewards it."""

    def setUp(self) -> None:
        self.schema = load_schema(SYNTHETIC)
        self.kw = dict(env={"%n": 1024}, dtype_class="int", sizes=(1024,), dtype="i32")

    def test_rewrites_max_zero_into_the_dedicated_relu(self) -> None:
        expr = Expr("max", (leaf("%x"), Expr("const:0", ())))
        greedy = greedy_cost(self.schema, expr, **self.kw)
        sel = egraph_select(self.schema, expr, **self.kw)

        self.assertLess(sel.cost, greedy, "SYN_RELU is cheaper than SYN_MAX")
        self.assertEqual(sel.extracted.op, "relu")
        self.assertEqual(sel.instructions.get("relu"), "SYN_RELU")

    def test_lowers_a_subtraction_the_greedy_selector_must_refuse(self) -> None:
        """The target has no SUB; only a rewrite to add(a, neg(b)) can lower it."""
        expr = Expr("sub", (leaf("%a"), leaf("%b")))

        self.assertEqual(
            greedy_cost(self.schema, expr, **self.kw), INFEASIBLE,
            "greedy must refuse: the schema names no subtract",
        )
        sel = egraph_select(self.schema, expr, **self.kw)
        self.assertTrue(sel.feasible)
        self.assertEqual(set(sel.extracted.ops()), {"add", "neg"})

    def test_leaves_an_already_optimal_term_alone(self) -> None:
        """Control: with nothing to gain, the cost must not move."""
        expr = Expr("add", (leaf("%a"), leaf("%b")))
        self.assertEqual(
            egraph_select(self.schema, expr, **self.kw).cost,
            greedy_cost(self.schema, expr, **self.kw),
        )

    def test_rewrite_gain_disappears_without_the_rule(self) -> None:
        """Mutation check: the win comes from max0-to-relu, not from somewhere else.

        Re-running with that rule removed must restore the greedy cost. Without
        this, a bug that made every extraction cheap would still pass the test
        above.
        """
        from tritonflow.isa import egraph as eg_mod

        expr = Expr("max", (leaf("%x"), Expr("const:0", ())))
        original = eg_mod.RULES
        try:
            eg_mod.RULES = tuple(r for r in original if r.name != "max0-to-relu")
            without = egraph_select(self.schema, expr, **self.kw)
        finally:
            eg_mod.RULES = original

        self.assertEqual(
            without.cost, greedy_cost(self.schema, expr, **self.kw),
            "removing max0-to-relu must remove the gain",
        )
        self.assertLess(
            egraph_select(self.schema, expr, **self.kw).cost, without.cost,
            "and restoring it must bring the gain back",
        )

    def test_reports_the_reason_an_operator_is_unimplementable(self) -> None:
        from tritonflow.isa.egraph import SchemaCostModel

        model = SchemaCostModel(self.schema, env={"%n": 1024}, dtype_class="int")
        from tritonflow.isa.egraph import ENode

        self.assertEqual(model.cost_of(ENode("div", (0, 1))), INFEASIBLE)
        self.assertTrue(model.refusals, "a refusal must carry a stated reason")


if __name__ == "__main__":
    unittest.main()
