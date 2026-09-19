"""The compiler driver: TTIR text -> parsed module -> annotations -> emitted program.

This is the only place the stages are composed for a fixture. It reports what the
stages produced and nothing else: costs come from the selector, refusals from the
emitted markers. Reference outputs are the caller's business (tests and the bench
compare against an independent NumPy/fp64 computation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .emit.assemble import assemble
from .emit.ir import Program
from .idioms.detect import AnnotationSet, annotate
from .isa.schema import load_builtin
from .ttir.graph import build_def_use
from .ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT / "fixtures"

_STRUCTURAL = {"tt.return", "tt.func", "scf.yield", "builtin.module"}


@dataclass(frozen=True)
class CompileResult:
    tier: str
    isa_name: str
    module: Any = None
    program: Program | None = None
    annotations: AnnotationSet | None = None
    unsupported: tuple[str, ...] = ()
    total_cost: float | None = None
    value_ops: int = 0
    annotated_value_ops: int = 0
    emitted_instructions: int = 0
    raw_op_count: int = 0

    @property
    def fully_lowered(self) -> bool:
        return self.program is not None and not self.unsupported


def _walk(region: Any) -> list[Any]:
    ops: list[Any] = []
    for block in region.blocks:
        for op in block.operations:
            ops.append(op)
            for sub in op.regions:
                ops.extend(_walk(sub))
    return ops


def _launch_env(tier: str) -> dict[str, int]:
    path = FIXTURES_DIR / "launch_env.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).get(tier, {}).items() if isinstance(v, int)}


def compile_text(
    text: str, isa_name: str, *, tier: str = "", env: dict[str, int] | None = None, source_path: str = ""
) -> CompileResult:
    parsed = parse_module(text, source_path=source_path or tier)
    if not parsed.ok or parsed.module is None:
        return CompileResult(tier=tier, isa_name=isa_name, unsupported=(f"parse: {parsed.diagnostic}",))
    module = parsed.module
    graph = build_def_use(module)
    annotations = annotate(module, graph)
    program = assemble(module, graph, annotations, load_builtin(isa_name), env=env or {})

    ops = [op for op in _walk(module.body) if op.name not in _STRUCTURAL]
    markers = program.markers()
    body = list(program.instrs) + list(program.epilogue) + [i for lp in program.loops for i in lp.body]
    return CompileResult(
        tier=tier,
        isa_name=isa_name,
        module=module,
        program=program,
        annotations=annotations,
        unsupported=tuple(f"{m.op_name}: {m.reason}" for m in markers),
        total_cost=program.total_cost,
        value_ops=len(ops),
        annotated_value_ops=max(0, len(ops) - len(markers)),
        emitted_instructions=sum(1 for i in body if hasattr(i, "operands")),
        raw_op_count=len(ops),
    )


def compile_fixture(tier_or_path: str, isa_name: str = "vortex_rvgpu") -> CompileResult:
    is_path = "/" in tier_or_path or tier_or_path.endswith(".ttir")
    path = Path(tier_or_path) if is_path else FIXTURES_DIR / f"{tier_or_path}.ttir"
    tier = path.stem
    return compile_text(
        path.read_text(encoding="utf-8"), isa_name, tier=tier, env=_launch_env(tier), source_path=str(path)
    )


@dataclass(frozen=True)
class ParityResult:
    executed: bool
    max_rel_err: float | None = None
    bound: float | None = None
    reason: str | None = None

    @property
    def within_bound(self) -> bool:
        return self.executed and self.max_rel_err is not None and self.max_rel_err <= (self.bound or 0.0)


SEED = 24173


def check_parity(result: CompileResult, *, seed: int = SEED, use_cpp: bool = False) -> ParityResult:
    """Run the emitted program and compare with an fp64 NumPy computation of the same kernel.

    The reference never touches the emulator or the emitted program. Tolerance is
    derived: exact for elementwise fp32, and k*(2*2^-11 + 2^-24) for a tf32 matmul
    (operand truncation error 2^-11 per input, accumulated across the k terms).
    """
    import numpy as np

    from .emu.exec import ProgramNotExecutable, emulate
    from .emu.precision import PrecisionPolicy

    if result.program is None:
        return ParityResult(False, reason="no program was emitted")
    if result.unsupported:
        return ParityResult(False, reason=f"program is not lowered: {result.unsupported[0]}")

    rng = np.random.default_rng(seed)
    tier = result.tier
    if tier == "t0_vecadd":
        x, y = (rng.standard_normal(1024, dtype=np.float32) for _ in range(2))
        inputs = {"%x_ptr": x, "%y_ptr": y, "%out_ptr": np.zeros(1024, np.float32), "%n": 1024}
        out_key, sel = "%out_ptr", slice(None)
        ref = (x.astype(np.float64) + y.astype(np.float64))
        policy, bound = PrecisionPolicy(input_precision="ieee"), 2.0**-24
    elif tier in ("t1_matmul", "t2_matmul_relu"):
        m, n, k = 128, 128, 64
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        inputs = {
            "%a_ptr": a, "%b_ptr": b, "%c_ptr": np.zeros((m, n), np.float32),
            "%M": m, "%N": n, "%K": k,
            "%sam": k, "%sak": 1, "%sbk": n, "%sbn": 1, "%scm": n, "%scn": 1,
        }
        ref = a[:64].astype(np.float64) @ b[:, :64].astype(np.float64)
        if tier == "t2_matmul_relu":
            bias = rng.standard_normal(n, dtype=np.float32)
            inputs["%bias_ptr"] = bias
            ref = np.maximum(ref + bias[:64].astype(np.float64)[None, :], 0.0)
        out_key, sel = "%c_ptr", (slice(0, 64), slice(0, 64))
        policy, bound = PrecisionPolicy(input_precision="tf32"), k * (2.0 * 2.0**-11 + 2.0**-24)
    else:
        return ParityResult(False, reason=f"no independent reference is defined for {tier}")

    try:
        got = emulate(result.program, inputs, policy=policy, use_cpp=use_cpp)[out_key]
    except ProgramNotExecutable as exc:
        return ParityResult(False, reason=f"emulator refused: {exc}")
    got = np.asarray(got, dtype=np.float64)[sel]
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    return ParityResult(True, float(np.max(np.abs(got - ref))) / scale, bound)
