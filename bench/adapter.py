"""The one place bench/ touches the pipeline.

`bench/run.py` never imports project modules directly: it asks this adapter for
a `RunContext`. That keeps the protocol in one file and lets every track land
its part without editing the runner.

`lower_fixture` runs the real pipeline (`tritonflow.pipeline`). A metric that cannot be
grounded is recorded as `unavailable` with a reason, so `bench/results.json` always has
every row.

DETERMINISM CONTRACT:
  * inputs come from `make_inputs`, never from a bare `np.random` call
  * the seed is the one in bench/cases.yaml
  * no GPU, no network, no environment-dependent behaviour
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SEED = 24173


@dataclass
class RunContext:
    """Everything a bench metric may read. Filled by the pipeline, not by bench."""

    tier: str
    module: Any = None          # ttir.ssa.Module
    program: Any = None         # emit.ir.Program
    annotations: dict = field(default_factory=dict)
    unsupported: list = field(default_factory=list)
    total_cost: float | None = None
    reference_cost: float | None = None
    oracle_cost: float | None = None
    value_ops: int = 0
    annotated_value_ops: int = 0
    largest_subgraph_ops: int = 0
    emitted_instructions: int = 0
    raw_op_count: int = 0
    emu_outputs: dict | None = None
    reference_outputs: dict | None = None
    tolerance: float | None = None
    schema: str | None = None


def lower_fixture(tier: str, isa_name: str = "tritonflow1") -> RunContext | None:
    """Run the real pipeline on one frozen fixture. Exceptions propagate to the runner.

    `reference_cost` and `oracle_cost` are left `None`: there is no hand-written
    reference program and no exhaustive enumerator behind this function, so the two
    cost-ratio metrics report `unavailable` instead of a made-up denominator.
    """
    from tritonflow.pipeline import check_parity, compile_fixture

    result = compile_fixture(tier, isa_name)
    ctx = RunContext(
        tier=tier,
        module=result.module,
        program=result.program,
        unsupported=list(result.unsupported),
        total_cost=result.total_cost,
        value_ops=result.value_ops,
        annotated_value_ops=result.annotated_value_ops,
        largest_subgraph_ops=result.annotated_value_ops,
        emitted_instructions=result.emitted_instructions,
        raw_op_count=result.raw_op_count,
        schema=isa_name,
    )
    parity = check_parity(result, seed=SEED)
    ctx.__dict__["parity_max_rel_err"] = parity.max_rel_err
    ctx.__dict__["parity_reason"] = parity.reason
    ctx.__dict__["parity_bound"] = parity.bound
    return ctx
