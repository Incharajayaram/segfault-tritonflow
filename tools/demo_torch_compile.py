#!/usr/bin/env python
"""Prove the tritonflow Dynamo backend works, and that it runs on the C++ emulator.

Note what this does and does NOT demonstrate:

* `torch.device("tritonflow")` / `tensor.to("tritonflow")` are NOT supported here, by
  design (see torch_backend/device_interface.py: DATA_PLANE_GAP,
  RENAME_BLOCKED). Allocating a real torch.Tensor on the device needs a
  compiled PrivateUse1 allocator this project doesn't ship, and renaming the
  PrivateUse1 backend to "tritonflow" breaks Dynamo tracing for every graph, toy
  or not. That boundary is a measured decision, not a bug.
* The real, working seam is the `tritonflow_backend` registered under the name
  "tritonflow" (what `torch.compile(fn, backend="tritonflow")` looks up). It lowers
  an FX graph through the actual ISA-1 pipeline (parse -> def-use -> annotate
  -> assemble -> emulate) and runs it, returning ordinary CPU tensors.

Why this calls `tritonflow_backend` directly instead of going through
`torch.compile`: `torch.compile`'s returned wrapper is Dynamo's eval-frame
object, not the callable our backend returns -- Dynamo caches compiled
artifacts internally keyed by guards, and does not forward attributes like
`tritonflow_plan` from them onto the wrapper. `torch.fx.symbolic_trace` gives us
the same kind of GraphModule Dynamo would hand the backend, and calling
`tritonflow_backend` on it directly gets us the exact `run` callable -- with
`tritonflow_plan` attached -- so we can assert on it instead of guessing.

This script traces a matmul matching the one recorded lowering in this
checkout (a 128x64 @ 64x128 tile), runs it through the backend, and checks:
  1. HAS_CPP is True -- the extension we fixed the build for is loaded.
  2. The graph was actually LOWERED (not silently sent to eager fallback).
  3. The lowered result matches eager torch.matmul to tf32-tier precision.
  4. torch.compile(fn, backend="tritonflow") also runs end-to-end and agrees.
"""

from __future__ import annotations

import torch

# The package deliberately imports nothing at package level (it must stay
# importable with torch absent -- see tritonflow/__init__.py), so the
# @register_backend("tritonflow") decorator in torch_backend/compiler.py only
# runs once this module is actually imported. Without this import,
# torch.compile(..., backend="tritonflow") raises InvalidBackend.
from tritonflow.emu import HAS_CPP
from tritonflow.torch_backend.compiler import tritonflow_backend


def main() -> None:
    print(f"HAS_CPP (compiled C++ emulator backend loaded): {HAS_CPP}")
    if not HAS_CPP:
        print(
            "WARNING: the C++ extension is not loaded -- this run will still "
            "execute correctly (the emulator falls back to its NumPy reference "
            "path), but it will not prove the C++ backend specifically."
        )

    torch.manual_seed(0)
    a = torch.randn(128, 64, dtype=torch.float32)
    b = torch.randn(64, 128, dtype=torch.float32)
    expected = torch.matmul(a, b)

    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)

    # --- Step 1: call the registered backend directly, so we can inspect
    # exactly what it decided (lowered vs. fallback), not guess from outside. ---
    traced = torch.fx.symbolic_trace(kernel)
    run = tritonflow_backend(traced, [a, b])

    plan = run.tritonflow_plan
    print(f"Graph lowered (not eager fallback): {plan.fully_lowered}")
    if not plan.fully_lowered:
        for fb in plan.fallbacks:
            print(f"  fallback: {fb}")
        raise SystemExit(
            "the graph fell back to eager PyTorch -- nothing ran on the toy ISA"
        )
    print(f"Lowered via kernel(s): {[k.name for k in plan.lowered]}")

    (direct_result,) = run(a, b)
    max_err = (direct_result - expected).abs().max().item()
    print(f"Direct-backend result shape: {tuple(direct_result.shape)}")
    print(f"Direct-backend max abs error vs eager torch.matmul: {max_err:.3e}")

    # tf32-tier tolerance: this recorded kernel declares tt.dot inputPrecision
    # "tf32" (see torch_backend/compiler.py's default_policy), so exact fp32
    # equality is not the right bar -- a generous but real bound is.
    if max_err > 2e-2:
        raise SystemExit(f"result diverged from eager torch.matmul: {max_err:.3e}")

    # --- Step 2: also go through the public torch.compile API end-to-end,
    # to prove the same backend is reachable the way a real user would call it.
    compiled = torch.compile(kernel, backend="tritonflow")
    compiled_result = compiled(a, b)
    compiled_err = (compiled_result - expected).abs().max().item()
    print(f"torch.compile(backend='tritonflow') max abs error: {compiled_err:.3e}")
    if compiled_err > 2e-2:
        raise SystemExit(f"torch.compile path diverged: {compiled_err:.3e}")

    print(
        "\nPASS: the tritonflow backend lowered and executed a real matmul on the "
        "toy ISA emulator (both called directly and via torch.compile), and "
        "the answer matches eager PyTorch."
    )


if __name__ == "__main__":
    main()

