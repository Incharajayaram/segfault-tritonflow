#!/usr/bin/env python3
"""Demo: Inductor → Triton TTIR → TritonFlow lower (judge-facing).

    PYTHONPATH=src python tools/demo_inductor_ttir.py

Prints whether Inductor emitted TTIR, provenance used by the seam, and a TTIR
snippet. On CPU-only hosts Inductor often emits no Triton — that is reported
honestly and the seam falls through to dynamic/recorded templates.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as exc:
        print(f"need torch+triton (pip install -e '.[extract]'): {exc}")
        return 1

    from tritonflow.extract import inductor_bridge as ib
    from tritonflow.torch_backend import compiler as seam

    a = torch.randn(64, 32)
    b = torch.randn(32, 64)

    def mm(x, y):
        return torch.mm(x, y)

    gm = torch.fx.symbolic_trace(mm)
    print("=== Inductor capture ===")
    capture = ib.capture_inductor_ttir(gm, [a, b])
    print(f"records: {len(capture.records)}")
    if capture.reason:
        print(f"reason:  {capture.reason}")
    for i, rec in enumerate(capture.records[:3]):
        print(f"--- record[{i}] name={rec.name} ({len(rec.ttir)} chars) ---")
        for line in rec.ttir.splitlines()[:12]:
            print(line)
        if "tt.dot" in rec.ttir:
            print("... (contains tt.dot)")

    print("\n=== seam plan_graph ===")
    graph, _ = torch._dynamo.export(mm, tracing_mode="real", aten_graph=False)(a, b)
    plan = seam.plan_graph(graph, [a, b])
    if plan.lowered:
        k = plan.lowered[0]
        print(f"provenance: {k.provenance}")
        print(f"kernel:     {k.name}")
        print(f"instrs:     {len(k.program.instructions())}")
    else:
        print("not lowered; fallbacks:")
        for f in plan.fallbacks[:5]:
            print(f"  [{f.stage}] {f.reason}")
    if plan.notes:
        print("notes:")
        for n in plan.notes[:5]:
            print(f"  [{n.stage}] {n.reason}: {n.detail[:120]}")

    run = seam.tritonflow_backend(graph, [a, b])
    out = run(a, b)
    if isinstance(out, (list, tuple)):
        out = out[0]
    ref = mm(a, b)
    diff = float((torch.as_tensor(out) - ref).abs().max())
    print(f"\nmax abs diff vs eager: {diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
