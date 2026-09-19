"""Test per-node lowering, fallback reporting, and honest fully_lowered flag (Task E3).

Verifies:
  1. A model with an unsupported op in the middle (MLP with sin) executes correctly,
     names the unsupported op in fallback records, and reports fully_lowered == False.
  2. A fully supported model (MLP with relu) lowers all nodes, reports fully_lowered == True.
  3. Small attention block and convolution stack lower supported operations, fall back
     honestly on unsupported operations, and report exact lowered fractions.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from typing import Any

import torch
import torch.nn as nn

import tritonflow.torch_backend.compiler as compiler

compiler.register_backend()


class MLPWithUnsupported(nn.Module):
    """Linear -> sin (unsupported on device) -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = torch.sin(h)  # Deliberately unsupported operation in the middle
        return self.fc2(h)


class FullySupportedMLP(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = torch.relu(h)
        return self.fc2(h)


class SmallAttentionBlock(nn.Module):
    """QKV Linear -> Matmul -> Softmax (unsupported) -> Matmul."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)  # Softmax runs eager
        return torch.matmul(probs, v)


class ConvStack(nn.Module):
    """Conv2d (unsupported) -> ReLU -> Conv2d (unsupported)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.relu(h)
        return self.conv2(h)




class EmbeddingLayerNormModel(nn.Module):
    """Embedding -> LayerNorm -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.fc = nn.Linear(16, 8)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer_norm(self.embedding(idxs)))


def test_mlp_with_deliberate_fallback() -> None:
    """Acceptance test: unsupported op in the middle executes correctly and is named."""
    torch.manual_seed(42)
    model = MLPWithUnsupported().eval()
    x = torch.randn(8, 16)

    last_plan = None

    def hooked(graph: Any, example_inputs: Any) -> Any:
        nonlocal last_plan
        fn = compiler.tritonflow_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    compiled = torch.compile(model, backend=hooked)
    eager_out = model(x)
    comp_out = compiled(x)

    # 1. Numerically correct result within tolerance
    assert torch.allclose(eager_out, comp_out, atol=1e-2, rtol=1e-1)

    # 2. fully_lowered CANNOT be true while any node fell back
    assert last_plan is not None
    assert last_plan.fully_lowered is False, "fully_lowered must be False when a node fell back"

    # 3. Exactly named in fallback records
    fallback_ops = [n for fb in last_plan.fallbacks for n in fb.nodes]
    assert any("sin" in op for op in fallback_ops), f"sin must be named in fallbacks, got {fallback_ops}"

    # 4. Correct counts and fraction
    summary = last_plan.compilation_summary()
    assert summary["nodes_lowered"] >= 2  # fc1 and fc2
    assert summary["nodes_fallen_back"] >= 1  # sin
    assert 0.0 < summary["lowered_fraction"] < 1.0


def test_fully_supported_mlp() -> None:
    """A fully supported model lowers every node and reports fully_lowered == True."""
    torch.manual_seed(42)
    model = FullySupportedMLP().eval()
    x = torch.randn(8, 16)

    last_plan = None

    def hooked(graph: Any, example_inputs: Any) -> Any:
        nonlocal last_plan
        fn = compiler.tritonflow_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    compiled = torch.compile(model, backend=hooked)
    eager_out = model(x)
    comp_out = compiled(x)

    assert torch.allclose(eager_out, comp_out, atol=1e-2, rtol=1e-1)
    assert last_plan is not None
    assert last_plan.fully_lowered is True, "fully_lowered must be True when all nodes lowered"
    assert last_plan.node_fallbacks == 0
    assert len(last_plan.fallbacks) == 0
    summary = last_plan.compilation_summary()
    assert summary["lowered_fraction"] == 1.0


def test_attention_block_partial_lowering() -> None:
    """Attention block lowers linear projections and matmuls, falling back on softmax."""
    torch.manual_seed(42)
    model = SmallAttentionBlock().eval()
    x = torch.randn(8, 16)

    last_plan = None

    def hooked(graph: Any, example_inputs: Any) -> Any:
        nonlocal last_plan
        fn = compiler.tritonflow_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    compiled = torch.compile(model, backend=hooked)
    eager_out = model(x)
    comp_out = compiled(x)

    assert torch.allclose(eager_out, comp_out, atol=1e-2, rtol=1e-1)
    assert last_plan is not None
    assert last_plan.fully_lowered is False
    assert last_plan.node_lowerings > 0
    assert last_plan.node_fallbacks > 0
    summary = last_plan.compilation_summary()
    assert 0.0 < summary["lowered_fraction"] < 1.0


def test_conv_stack_partial_lowering() -> None:
    """Conv stack lowers elementwise relu while conv2d runs in eager."""
    torch.manual_seed(42)
    model = ConvStack().eval()
    x = torch.randn(2, 1, 8, 8)

    last_plan = None

    def hooked(graph: Any, example_inputs: Any) -> Any:
        nonlocal last_plan
        fn = compiler.tritonflow_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    compiled = torch.compile(model, backend=hooked)
    eager_out = model(x)
    comp_out = compiled(x)

    assert torch.allclose(eager_out, comp_out, atol=1e-2, rtol=1e-1)
    assert last_plan is not None
    assert last_plan.fully_lowered is False
    summary = last_plan.compilation_summary()
    assert summary["nodes_fallen_back"] >= 2  # conv1 and conv2
    assert summary["lowered_fraction"] < 1.0


def test_embedding_layernorm_partial_lowering() -> None:
    """Embedding+LayerNorm falls back on unsupported ops, executes correctly."""
    torch.manual_seed(42)
    model = EmbeddingLayerNormModel().eval()
    idxs = torch.randint(0, 100, (4, 8))

    last_plan = None

    def hooked(graph: Any, example_inputs: Any) -> Any:
        nonlocal last_plan
        fn = compiler.tritonflow_backend(graph, example_inputs)
        last_plan = getattr(fn, "tritonflow_plan", None)
        return fn

    compiled = torch.compile(model, backend=hooked)
    eager_out = model(idxs)
    comp_out = compiled(idxs)

    assert torch.allclose(eager_out, comp_out, atol=1e-2, rtol=1e-1)
    assert last_plan is not None
    assert last_plan.fully_lowered is False
    summary = last_plan.compilation_summary()
    assert summary["nodes_fallen_back"] >= 1
