"""Core e-graph mechanics: union-find, hashcons, congruence, saturation, extraction.

These tests are about the data structure, not about any ISA. The schema-facing
behaviour is in `test_egraph_selection.py`.
"""

from __future__ import annotations

import unittest

from tritonflow.isa.egraph import (
    INFEASIBLE,
    EGraph,
    ENode,
    Expr,
    PatNode,
    PatVar,
    Rule,
    ematch,
    extract,
    leaf,
    rules_for,
    saturate,
)


def unit_cost(node: ENode) -> float:
    """One unit per operator, nothing for a leaf."""
    return 0.0 if node.is_leaf() else 1.0


class TestUnionFindAndHashcons(unittest.TestCase):
    def test_identical_nodes_share_a_class(self) -> None:
        eg = EGraph()
        a = eg.add(ENode("x"))
        b = eg.add(ENode("x"))
        self.assertEqual(a, b, "hashcons must return the existing class for an equal node")

    def test_structurally_equal_terms_share_a_class(self) -> None:
        eg = EGraph()
        first = eg.add_expr(Expr("add", (leaf("a"), leaf("b"))))
        second = eg.add_expr(Expr("add", (leaf("a"), leaf("b"))))
        self.assertEqual(eg.find(first), eg.find(second))

    def test_find_is_idempotent_and_canonical(self) -> None:
        eg = EGraph()
        a = eg.add(ENode("a"))
        b = eg.add(ENode("b"))
        eg.merge(a, b)
        eg.rebuild()
        self.assertEqual(eg.find(a), eg.find(b))
        self.assertEqual(eg.find(eg.find(a)), eg.find(a))

    def test_merge_is_symmetric(self) -> None:
        eg = EGraph()
        a, b = eg.add(ENode("a")), eg.add(ENode("b"))
        eg.merge(b, a)
        eg.rebuild()
        self.assertEqual(eg.find(a), eg.find(b))


class TestCongruenceClosure(unittest.TestCase):
    def test_congruent_parents_merge_after_children_merge(self) -> None:
        """If a == b then f(a) == f(b). This is the property hashcons alone misses."""
        eg = EGraph()
        a = eg.add(ENode("a"))
        b = eg.add(ENode("b"))
        fa = eg.add(ENode("f", (a,)))
        fb = eg.add(ENode("f", (b,)))
        eg.rebuild()
        self.assertNotEqual(eg.find(fa), eg.find(fb), "f(a) and f(b) start distinct")

        eg.merge(a, b)
        eg.rebuild()
        self.assertEqual(eg.find(fa), eg.find(fb), "congruence must merge f(a) with f(b)")

    def test_congruence_propagates_upward_transitively(self) -> None:
        eg = EGraph()
        a, b = eg.add(ENode("a")), eg.add(ENode("b"))
        ga = eg.add(ENode("g", (eg.add(ENode("f", (a,))),)))
        gb = eg.add(ENode("g", (eg.add(ENode("f", (b,))),)))
        eg.rebuild()
        eg.merge(a, b)
        eg.rebuild()
        self.assertEqual(eg.find(ga), eg.find(gb), "congruence must reach g(f(x))")

    def test_invariants_hold_after_merging(self) -> None:
        eg = EGraph()
        a, b = eg.add(ENode("a")), eg.add(ENode("b"))
        eg.add(ENode("f", (a,)))
        eg.add(ENode("f", (b,)))
        eg.merge(a, b)
        eg.rebuild()
        eg.check_invariants()

    def test_detects_a_broken_invariant(self) -> None:
        """check_invariants must actually fail on a corrupted graph, not pass blindly.

        The merge has to absorb `a`'s class rather than keep it as the root, or
        `f(a)`'s child stays canonical and there is nothing to detect. Union is by
        class size, so `b`'s class is padded first to make it the survivor.
        """
        eg = EGraph()
        a = eg.add(ENode("a"))
        b = eg.add(ENode("b"))
        eg.merge(b, eg.add(ENode("b_extra")))
        eg.rebuild()

        eg.add(ENode("f", (a,)))
        eg.merge(a, b)
        # Deliberately skip rebuild, leaving f's child pointing at an absorbed class.
        self.assertNotEqual(eg.find(a), a, "a's class must have been absorbed")
        with self.assertRaises(AssertionError):
            eg.check_invariants()


class TestEMatching(unittest.TestCase):
    def test_variable_pattern_matches_every_class(self) -> None:
        eg = EGraph()
        eg.add_expr(Expr("add", (leaf("a"), leaf("b"))))
        eg.rebuild()
        self.assertEqual(len(ematch(eg, PatVar("x"))), eg.total_classes())

    def test_node_pattern_binds_arguments(self) -> None:
        eg = EGraph()
        root = eg.add_expr(Expr("add", (leaf("a"), leaf("b"))))
        eg.rebuild()
        matches = ematch(eg, PatNode("add", (PatVar("p"), PatVar("q"))))
        self.assertEqual(len(matches), 1)
        eid, subst = matches[0]
        self.assertEqual(eg.find(eid), eg.find(root))
        self.assertEqual(set(subst), {"p", "q"})

    def test_repeated_variable_requires_the_same_class(self) -> None:
        """`add(x, x)` must not match `add(a, b)` when a and b differ."""
        eg = EGraph()
        eg.add_expr(Expr("add", (leaf("a"), leaf("b"))))
        eg.rebuild()
        self.assertEqual(ematch(eg, PatNode("add", (PatVar("x"), PatVar("x")))), [])

        eg2 = EGraph()
        eg2.add_expr(Expr("add", (leaf("a"), leaf("a"))))
        eg2.rebuild()
        self.assertEqual(len(ematch(eg2, PatNode("add", (PatVar("x"), PatVar("x"))))), 1)


class TestSaturation(unittest.TestCase):
    def test_reaches_a_fixed_point_on_a_small_term(self) -> None:
        eg = EGraph()
        eg.add_expr(Expr("sub", (leaf("x"), leaf("y"))))
        eg.rebuild()
        result = saturate(eg, rules_for("float"))
        self.assertEqual(result.status, "saturated")
        self.assertTrue(result.reached_fixpoint)
        eg.check_invariants()

    def test_sub_becomes_add_of_negation(self) -> None:
        eg = EGraph()
        root = eg.add_expr(Expr("sub", (leaf("x"), leaf("y"))))
        eg.rebuild()
        saturate(eg, rules_for("float"))
        ops = {n.op for n in eg.eclasses()[eg.find(root)]}
        self.assertIn("add", ops, "sub-to-add-neg must put an add in the root class")
        self.assertIn("sub", ops, "the original form must be retained, not replaced")

    def test_node_limit_stops_growth_and_is_reported(self) -> None:
        chain = leaf("v0")
        for i in range(1, 8):
            chain = Expr("add", (chain, leaf(f"v{i}")))
        eg = EGraph()
        eg.add_expr(chain)
        eg.rebuild()
        result = saturate(eg, rules_for("int"), node_limit=200, iter_limit=50)
        self.assertEqual(result.status, "node_limit")
        self.assertFalse(result.reached_fixpoint)
        eg.check_invariants()

    def test_iter_limit_stops_and_is_reported(self) -> None:
        chain = leaf("v0")
        for i in range(1, 8):
            chain = Expr("add", (chain, leaf(f"v{i}")))
        eg = EGraph()
        eg.add_expr(chain)
        eg.rebuild()
        result = saturate(eg, rules_for("int"), node_limit=10**9, iter_limit=2)
        self.assertEqual(result.status, "iter_limit")
        self.assertEqual(result.iterations, 2)

    def test_always_terminates_on_associative_commutative_chains(self) -> None:
        """Property: bounded saturation returns for every chain length, never hangs.

        Associativity plus commutativity is the classic non-terminating rewrite
        system; the bound is what makes termination unconditional.
        """
        for length in range(2, 7):
            chain = leaf("v0")
            for i in range(1, length):
                chain = Expr("add", (chain, leaf(f"v{i}")))
            eg = EGraph()
            eg.add_expr(chain)
            eg.rebuild()
            result = saturate(eg, rules_for("int"), node_limit=2000, iter_limit=15)
            with self.subTest(length=length):
                self.assertIn(result.status, ("saturated", "node_limit", "iter_limit"))
                self.assertLessEqual(result.iterations, 15)
                eg.check_invariants()

    def test_records_which_rules_fired(self) -> None:
        eg = EGraph()
        eg.add_expr(Expr("sub", (leaf("x"), leaf("y"))))
        eg.rebuild()
        result = saturate(eg, rules_for("float"))
        self.assertIn("sub-to-add-neg", result.rule_firings)
        self.assertEqual(result.applications, sum(result.rule_firings.values()))


class TestRuleSoundness(unittest.TestCase):
    def test_float_excludes_inexact_rules_by_default(self) -> None:
        names = {r.name for r in rules_for("float")}
        for inexact in ("assoc-add", "assoc-mul", "fma-contract", "div-to-recip-mul"):
            self.assertNotIn(inexact, names, f"{inexact} changes float results")

    def test_float_includes_exact_rules(self) -> None:
        names = {r.name for r in rules_for("float")}
        for exact in ("commute-add", "sub-to-add-neg", "mul2-to-add", "max0-to-relu"):
            self.assertIn(exact, names)

    def test_opt_in_admits_inexact_rules_for_float(self) -> None:
        names = {r.name for r in rules_for("float", allow_inexact=True)}
        self.assertIn("assoc-add", names)
        self.assertIn("fma-contract", names)

    def test_integers_admit_reassociation(self) -> None:
        """Two's-complement addition is associative, so the rule is exact there."""
        names = {r.name for r in rules_for("int")}
        self.assertIn("assoc-add", names)
        self.assertIn("fma-contract", names)

    def test_no_float_reassociation_reaches_the_egraph(self) -> None:
        """The gate must hold end to end, not just in the rule list."""
        expr = Expr("add", (Expr("add", (leaf("a"), leaf("b"))), leaf("c")))
        eg = EGraph()
        eg.add_expr(expr)
        eg.rebuild()
        result = saturate(eg, rules_for("float"))
        self.assertNotIn("assoc-add", result.rule_firings)

    def test_every_rule_declares_its_soundness(self) -> None:
        from tritonflow.isa.egraph import RULES

        for rule in RULES:
            with self.subTest(rule=rule.name):
                self.assertTrue(rule.note, "each rule must record why it is sound")
                self.assertIn(rule.dtype_class, ("any", "int", "float"))


class TestExtraction(unittest.TestCase):
    def test_picks_the_cheaper_of_two_equal_forms(self) -> None:
        eg = EGraph()
        root = eg.add_expr(Expr("sub", (leaf("x"), leaf("y"))))
        eg.rebuild()
        saturate(eg, rules_for("float"))

        expensive_sub = extract(
            eg, root, lambda n: 0.0 if n.is_leaf() else {"sub": 100.0}.get(n.op, 1.0)
        )
        self.assertEqual(expensive_sub.term.op, "add")

        cheap_sub = extract(
            eg, root, lambda n: 0.0 if n.is_leaf() else {"sub": 1.0}.get(n.op, 50.0)
        )
        self.assertEqual(cheap_sub.term.op, "sub")

    def test_refuses_when_every_form_is_infeasible(self) -> None:
        eg = EGraph()
        root = eg.add_expr(Expr("mystery", (leaf("x"),)))
        eg.rebuild()
        result = extract(eg, root, lambda n: 0.0 if n.is_leaf() else INFEASIBLE)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.term)
        self.assertEqual(result.cost, INFEASIBLE)

    def test_never_returns_a_term_containing_an_infeasible_node(self) -> None:
        """Property: a feasible extraction contains only priced operators."""
        eg = EGraph()
        root = eg.add_expr(Expr("sub", (leaf("x"), leaf("y"))))
        eg.rebuild()
        saturate(eg, rules_for("float"))

        def cost_of(node: ENode) -> float:
            if node.is_leaf():
                return 0.0
            return INFEASIBLE if node.op == "sub" else 1.0

        result = extract(eg, root, cost_of)
        self.assertTrue(result.feasible)
        self.assertNotIn("sub", set(result.term.ops()))

    def test_terminates_on_a_cyclic_egraph(self) -> None:
        """`x` and `mul(x, 1)` share a class, so the class graph has a cycle."""
        eg = EGraph()
        root = eg.add_expr(Expr("mul", (leaf("x"), Expr("const:1", ()))))
        eg.rebuild()
        saturate(eg, rules_for("float"))
        leaf_class = eg.find(eg.add(ENode("x")))
        self.assertEqual(eg.find(root), leaf_class, "mul1-identity should merge them")

        result = extract(eg, root, unit_cost)
        self.assertTrue(result.feasible)
        self.assertEqual(str(result.term), "x", "the cheapest form is the bare leaf")

    def test_cost_is_additive_over_the_chosen_tree(self) -> None:
        eg = EGraph()
        root = eg.add_expr(Expr("add", (Expr("mul", (leaf("a"), leaf("b"))), leaf("c"))))
        eg.rebuild()
        result = extract(eg, root, unit_cost)
        self.assertEqual(result.cost, 2.0, "one add plus one mul, leaves free")


class TestMalformedRules(unittest.TestCase):
    def test_unbound_rhs_variable_is_skipped_not_crashed(self) -> None:
        bad = Rule("bad", PatNode("add", (PatVar("a"), PatVar("b"))), PatVar("zzz"))
        eg = EGraph()
        eg.add_expr(Expr("add", (leaf("x"), leaf("y"))))
        eg.rebuild()
        result = saturate(eg, [bad], iter_limit=3)
        self.assertEqual(result.applications, 0)
        eg.check_invariants()


if __name__ == "__main__":
    unittest.main()
