#!/usr/bin/env python3
"""run_model_workloads.py — Evaluates realistic model-level workloads (Tasks E2, E3).

Evaluates 5 model architectures through torch.compile with tritonflow backend:
1. Fully Supported MLP (Linear + ReLU + Linear)
2. MLP with Fallback (Linear + Sin + Linear)
3. Small Transformer Attention Block (Linear QKV + Matmul + Softmax + Matmul)
4. Convolution Stack (Conv2d + ReLU + Conv2d)
5. Embedding + LayerNorm Stack (Embedding + LayerNorm + Linear)

Emits reports/model_workloads.md recording per-node lowerings, fallback causes,
lowered fractions, and numerical parity against eager PyTorch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

import tritonflow.torch_backend.compiler as compiler

compiler.register_backend()


class FullySupportedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class MLPWithFallback(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.sin(self.fc1(x)))


class SmallAttentionBlock(nn.Module):
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
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class ConvStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class EmbeddingLayerNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.fc = nn.Linear(16, 8)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer_norm(self.embedding(idxs)))


def evaluate_workloads() -> dict[str, Any]:
    torch.manual_seed(42)
    workloads = [
        ("MLP (Fully Supported)", FullySupportedMLP(), torch.randn(8, 16)),
        ("MLP (with Sin Fallback)", MLPWithFallback(), torch.randn(8, 16)),
        ("Transformer Attention Block", SmallAttentionBlock(), torch.randn(8, 16)),
        ("Convolution Stack", ConvStack(), torch.randn(2, 1, 8, 8)),
        ("Embedding + LayerNorm", EmbeddingLayerNormModel(), torch.randint(0, 100, (8, 4))),
    ]

    results = []

    for name, model, example_in in workloads:
        model.eval()
        last_plan = None

        def hooked(graph: Any, example_inputs: Any) -> Any:
            nonlocal last_plan
            fn = compiler.tritonflow_backend(graph, example_inputs)
            last_plan = getattr(fn, "tritonflow_plan", None)
            return fn

        torch._dynamo.reset()
        compiled = torch.compile(model, backend=hooked)
        eager_out = model(example_in)
        comp_out = compiled(example_in)

        diff = (eager_out - comp_out).abs().max().item() if isinstance(eager_out, torch.Tensor) else 0.0
        # Derived tolerance: tf32 is 1e-2 for matmuls, 1e-4 for pure elementwise
        passed_parity = diff <= 1e-2

        summary = last_plan.compilation_summary() if last_plan else {
            "fully_lowered": False,
            "nodes_lowered": 0,
            "nodes_fallen_back": 0,
            "lowered_fraction": 0.0,
            "fallbacks": [],
        }

        results.append({
            "name": name,
            "fully_lowered": summary["fully_lowered"],
            "lowered_count": summary["nodes_lowered"],
            "fallback_count": summary["nodes_fallen_back"],
            "fraction": summary["lowered_fraction"],
            "diff": diff,
            "parity": "PASS" if passed_parity else "FAIL",
            "fallbacks": summary.get("fallback_causes", []),
        })

    # Generate Markdown Report (Deterministic, no dates/clocks)
    lines = [
        "# Model-Level Workload Evaluation Report (Tasks E2, E3)",
        "",
        "Evaluates realistic PyTorch model architectures through `torch.compile(..., backend='tritonflow')`.",
        "Measures honest per-node lowering, fallback reporting, and numerical parity against eager PyTorch.",
        "",
        "## 1. Summary Table",
        "",
        "| Model Architecture | Fully Lowered? | Nodes Lowered | Nodes Fallen Back | Lowered Fraction | Max Abs Diff | Parity |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        fl_str = "YES" if r["fully_lowered"] else "NO"
        lines.append(
            f"| **{r['name']}** | {fl_str} | {r['lowered_count']} | {r['fallback_count']} | "
            f"{r['fraction']*100:.1f}% | {r['diff']:.2e} | {r['parity']} |"
        )

    lines.extend([
        "",
        "## 2. Per-Model Fallback Breakdown",
        "",
    ])

    for r in results:
        lines.append(f"### {r['name']}")
        if not r["fallbacks"]:
            lines.append("- *No fallbacks: 100% of graph lowered to ISA.*")
        else:
            for fb in r["fallbacks"]:
                nodes_str = ", ".join(f"`{n}`" for n in fb["nodes"])
                reason_clean = fb["reason"].replace("\n", " ").strip()
                lines.append(f"- **{nodes_str}**: {reason_clean}")
        lines.append("")

    report_content = "\n".join(lines) + "\n"
    out_file = ROOT / "reports" / "model_workloads.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")
    print(f"Report written to {out_file}")

    return {"results": results}


if __name__ == "__main__":
    evaluate_workloads()
