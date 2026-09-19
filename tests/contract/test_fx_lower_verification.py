"""Schema-derived op support and shadow verification for the torch FX backend.

Support is asked of the loaded schema, not read from a typed table, and a lowered
program is only reported as lowered when the emulated result agrees with eager
torch on the example input *and* on a second seeded random input, within a bound
derived from fp32 rounding.
"""

from __future__ import annotations

import dataclasses
import operator

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.fx import Graph, GraphModule, symbolic_trace  # noqa: E402

import tritonflow.torch_backend.compiler as compiler  # noqa: E402
import tritonflow.torch_backend.fx_lower as fx_lower  # noqa: E402
from tritonflow.isa.schema import load_builtin  # noqa: E402
from tritonflow.torch_backend.fx_lower import lower_fx_graph, supported_targets  # noqa: E402

ISAS = ("tritonflow1", "tritonflow2", "vortex_rvgpu")


def _inputs(*shapes, shift=0.0):
    torch.manual_seed(7)
    return [torch.randn(*s) + shift for s in shapes]


def _true_div(a, b):
    return a / b


def _relu(a):
    return torch.relu(a)


def _neg(a):
    return -a


def _abs(a):
    return abs(a)


def _relu_matmul(a, b):
    return torch.relu(a @ b)


# --------------------------------------------------------------------------- #
# Support comes from the schema
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize(
    "fn,shapes,shift",
    [(_true_div, [(8, 16), (8, 16)], 3.0), (_neg, [(8, 16)], 0.0), (_abs, [(8, 16)], 0.0)],
    ids=["truediv", "neg", "abs"],
)
def test_ops_the_schema_claims_lower_and_match_eager(isa, fn, shapes, shift):
    xs = _inputs(*shapes, shift=shift)
    reasons: list[str] = []
    lowering = lower_fx_graph(symbolic_trace(fn), xs, isa_name=isa, report=reasons)
    assert lowering is not None, reasons
    assert lowering.fully_lowered
    assert lowering.verified_inputs == ("example", f"seed={fx_lower.VERIFY_SEED}")


def _schema_without(name: str, op: str):
    schema = load_builtin("tritonflow1")
    instructions = {
        n: dataclasses.replace(i, ops=tuple(o for o in i.ops if o != op)) if op in i.ops else i
        for n, i in schema.instructions.items()
    }
    return dataclasses.replace(schema, instructions=instructions)


@pytest.mark.parametrize(
    "fn,shapes,removed,served",
    [(_neg, [(8, 16)], "negf", "arith.negf"), (_abs, [(8, 16)], "absf", "math.absf")],
    ids=["neg", "abs"],
)
def test_removing_an_op_from_a_schema_copy_makes_the_backend_refuse_it(monkeypatch, fn, shapes, removed, served):
    xs = _inputs(*shapes)
    assert lower_fx_graph(symbolic_trace(fn), xs, isa_name="tritonflow1") is not None

    modified = _schema_without("tritonflow1", removed)
    monkeypatch.setattr(fx_lower, "load_builtin", lambda name: modified)
    reasons: list[str] = []
    assert lower_fx_graph(symbolic_trace(fn), xs, isa_name="tritonflow1", report=reasons) is None
    assert any("declares no elementwise instruction serving" in r and served in r for r in reasons), reasons


def test_supported_targets_follow_the_schema():
    tf1, tf2 = set(supported_targets("tritonflow1")), set(supported_targets("tritonflow2"))
    assert {"truediv", "neg", "abs", "relu", "mm"} <= tf1
    assert "clamp" not in tf1 and "clamp" in tf2
    assert "exp" not in tf1 | tf2 | set(supported_targets("vortex_rvgpu"))


def test_operator_module_targets_are_named():
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.output(graph.call_function(operator.sub, (graph.call_function(operator.mul, (x, y)), y)))
    gm = GraphModule(torch.nn.Module(), graph)
    xs = _inputs((8, 8), (8, 8))
    assert lower_fx_graph(gm, xs, isa_name="tritonflow1") is not None


# --------------------------------------------------------------------------- #
# Shadow verification
# --------------------------------------------------------------------------- #


def _corrupting_emulate(factor, only_after=0):
    real = fx_lower.emulate
    calls = {"n": 0}

    def wrapper(program, storage, **kwargs):
        out = real(program, storage, **kwargs)
        calls["n"] += 1
        if calls["n"] > only_after:
            key = next(k for k in out if k.endswith("fx_out"))
            out = dict(out)
            out[key] = out[key] * np.float32(factor)
        return out

    return wrapper


def test_a_wrong_by_1e3_emulated_result_is_caught(monkeypatch):
    monkeypatch.setattr(fx_lower, "emulate", _corrupting_emulate(1.001))
    xs = _inputs((8, 16), (8, 16))
    reasons: list[str] = []
    assert lower_fx_graph(symbolic_trace(lambda a, b: a + b), xs, isa_name="tritonflow1", report=reasons) is None
    assert any("shadow verification mismatch" in r for r in reasons), reasons


def test_a_wrong_by_1e3_matmul_is_caught_because_the_bound_is_fp32_not_tf32(monkeypatch):
    """The tf32 band (k*2*2^-11) would admit this; the ieee run is judged by its own arithmetic."""
    monkeypatch.setattr(fx_lower, "emulate", _corrupting_emulate(1.001))
    xs = _inputs((8, 16), (16, 8))
    reasons: list[str] = []
    assert lower_fx_graph(symbolic_trace(lambda a, b: a @ b), xs, isa_name="tritonflow1", report=reasons) is None
    assert any("shadow verification mismatch" in r for r in reasons), reasons


def test_a_program_that_matches_only_the_example_input_is_caught(monkeypatch):
    monkeypatch.setattr(fx_lower, "emulate", _corrupting_emulate(1.01, only_after=1))
    xs = _inputs((8, 16))
    reasons: list[str] = []
    assert lower_fx_graph(symbolic_trace(_relu), xs, isa_name="tritonflow1", report=reasons) is None
    assert any(f"seed={fx_lower.VERIFY_SEED}" in r for r in reasons), reasons


def test_the_bound_is_derived_from_the_precision_policy():
    a = np.random.default_rng(1).standard_normal((4, 16)).astype(np.float32)
    b = np.random.default_rng(2).standard_normal((16, 4)).astype(np.float32)
    from tritonflow.emit.ir import SsaRef

    spec = [fx_lower._NodeSpec("%c", "tt.dot", {"a": SsaRef("%a"), "b": SsaRef("%b")})]
    ieee = fx_lower._error_analysis(spec, {"%a": a, "%b": b}, "ieee")["%c"][1]
    tf32 = fx_lower._error_analysis(spec, {"%a": a, "%b": b}, "tf32")["%c"][1]
    assert np.all(tf32 > 100 * ieee)
    ideal = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    assert np.all(ieee <= 16 * 2.0**-24 * ideal * 1.001 + 1e-30)


def test_shadow_verification_does_not_mutate_the_callers_tensors():
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.output(graph.call_function(torch.Tensor.add_, (x, y)))
    gm = GraphModule(torch.nn.Module(), graph)
    xs = _inputs((8, 16), (8, 16))
    before = [t.clone() for t in xs]
    lower_fx_graph(gm, xs, isa_name="tritonflow1")
    for kept, original in zip(before, xs, strict=True):
        assert torch.equal(kept, original)


def test_matmul_reduction_length_is_the_operand_inner_dimension():
    xs = _inputs((8, 32), (32, 16))
    lowering = lower_fx_graph(symbolic_trace(lambda a, b: a @ b), xs, isa_name="tritonflow1")
    assert lowering is not None and lowering.fully_lowered
    assert lowering.output_shape == (8, 16)


def test_fully_lowered_needs_a_match():
    xs = _inputs((8, 16))
    lowering = lower_fx_graph(symbolic_trace(_relu), xs, isa_name="tritonflow1")
    assert lowering.fully_lowered
    assert not dataclasses.replace(lowering, shadow_verified=False).fully_lowered
    assert not dataclasses.replace(lowering, mismatch_reason="x").fully_lowered


def test_non_float32_placeholder_is_refused():
    reasons: list[str] = []
    xs = [torch.randn(8, 8, dtype=torch.float64)]
    assert lower_fx_graph(symbolic_trace(_relu), xs, isa_name="tritonflow1", report=reasons) is None
    assert any("not float32" in r for r in reasons), reasons


# --------------------------------------------------------------------------- #
# The seam: partial lowering and per-node verification
# --------------------------------------------------------------------------- #


class _MLPWithSin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(torch.sin(self.fc1(x)))


def _plan_of(model, x):
    holder = {}

    def backend(graph, example_inputs):
        fn = compiler.tritonflow_backend(graph, example_inputs)
        holder["fn"] = fn
        return fn

    out = torch.compile(model, backend=backend)(x)
    return holder["fn"].tritonflow_plan, out


def test_mlp_with_sin_falls_back_on_exactly_that_node():
    torch.manual_seed(0)
    model = _MLPWithSin().eval()
    x = torch.randn(8, 16)
    plan, out = _plan_of(model, x)
    assert torch.allclose(out, model(x), atol=1e-2, rtol=1e-1)
    fallen = {n for fb in plan.fallbacks for n in fb.nodes}
    assert fallen == {"sin"}, fallen
    assert not plan.fully_lowered
    assert plan.node_lowerings >= 2 and plan.node_fallbacks == 1


def test_a_lowered_node_that_disagrees_with_eager_is_demoted_and_eager_is_returned(monkeypatch):
    real = compiler._run_with_padding

    def wrong(*args, **kwargs):
        return real(*args, **kwargs) * 1.001

    monkeypatch.setattr(compiler, "_run_with_padding", wrong)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)).eval()
    x = torch.randn(8, 16)
    plan, out = _plan_of(model, x)
    assert torch.allclose(out, model(x), atol=1e-6, rtol=1e-6), "the eager value must be what the graph returns"
    assert not plan.fully_lowered
    assert any(fb.stage == "shadow" for fb in plan.fallbacks), plan.fallbacks


def test_the_fully_supported_mlp_is_verified_and_fully_lowered():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)).eval()
    x = torch.randn(8, 16)
    plan, out = _plan_of(model, x)
    assert plan.fully_lowered, plan.fallbacks
    assert torch.allclose(out, model(x), atol=1e-2, rtol=1e-1)
