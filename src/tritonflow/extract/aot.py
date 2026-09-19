"""aot.py — Ahead-of-Time compilation and evaluation entry point for arbitrary Triton kernels (Task E1).

Generalises the proof in verify/verify_mlir_bindings.py into a fully supported entry point:
compiles any @triton.jit function ahead of time for an explicit GPUTarget with no GPU
allocated, producing TTIR, and evaluates it through the full compiler pipeline:
1. Triton AOT compilation (extract TTIR)
2. TTIR parsing (parse_module)
3. Def-Use analysis (build_def_use)
4. Access recognition (annotate)
5. Instruction selection & assembly (assemble)
6. Execution (emulate)
7. Numerical parity comparison against NumPy reference
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tritonflow.emit.assemble import assemble
from tritonflow.emit.ir import Program
from tritonflow.emu.exec import emulate
from tritonflow.emu.precision import PrecisionPolicy
from tritonflow.extract.dynamic_extract import compile_triton_kernel, require_triton
from tritonflow.idioms.detect import annotate
from tritonflow.isa.schema import load_builtin
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module


@dataclass
class StageOutcome:
    stage: str
    status: str  # "OK", "FAIL", "REFUSED", "UNSUPPORTED", "SKIPPED"
    detail: str = ""


@dataclass
class AotResult:
    name: str
    ttir: str | None
    program: Program | None
    outcomes: dict[str, StageOutcome]
    outputs: dict[str, np.ndarray] | None = None
    parity_error: float | None = None

    @property
    def fully_lowered(self) -> bool:
        return (
            self.outcomes.get("extract", StageOutcome("extract", "FAIL")).status == "OK"
            and self.outcomes.get("parse", StageOutcome("parse", "FAIL")).status == "OK"
            and self.outcomes.get("recognise", StageOutcome("recognise", "FAIL")).status == "OK"
            and self.outcomes.get("select", StageOutcome("select", "FAIL")).status == "OK"
            and self.outcomes.get("assemble", StageOutcome("assemble", "FAIL")).status == "OK"
            and self.outcomes.get("execute", StageOutcome("execute", "FAIL")).status == "OK"
        )

    def summary_row(self) -> dict[str, str]:
        return {
            "name": self.name,
            "extract": self.outcomes.get("extract", StageOutcome("extract", "FAIL")).status,
            "parse": self.outcomes.get("parse", StageOutcome("parse", "FAIL")).status,
            "recognise": self.outcomes.get("recognise", StageOutcome("recognise", "FAIL")).status,
            "select": self.outcomes.get("select", StageOutcome("select", "FAIL")).status,
            "assemble": self.outcomes.get("assemble", StageOutcome("assemble", "FAIL")).status,
            "execute": self.outcomes.get("execute", StageOutcome("execute", "FAIL")).status,
            "parity": self.outcomes.get("parity", StageOutcome("parity", "SKIPPED")).status,
        }


def compile_aot(
    fn: Any,
    signature: Mapping[str, str],
    constexprs: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    *,
    name: str = "aot_kernel",
    isa_name: str = "vortex_rvgpu",
    env: Mapping[str, int] | None = None,
    inputs: dict[str, np.ndarray] | None = None,
    reference_fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
    grid: tuple[int, int, int] = (0, 0, 0),
    tolerance: float = 1e-4,
) -> AotResult:
    """Compile an arbitrary @triton.jit function ahead of time and run through pipeline."""
    outcomes: dict[str, StageOutcome] = {}
    env_dict = dict(env or {})
    ttir: str | None = None
    program: Program | None = None
    outputs: dict[str, np.ndarray] | None = None
    parity_err: float | None = None

    # Stage 1: Triton compilation -> TTIR
    try:
        require_triton()
        ttir = compile_triton_kernel(fn, signature, constexprs, options)
        outcomes["extract"] = StageOutcome("extract", "OK", f"{len(ttir.splitlines())} lines TTIR")
    except Exception as exc:
        outcomes["extract"] = StageOutcome("extract", "FAIL", f"{type(exc).__name__}: {exc}")
        outcomes["parse"] = StageOutcome("parse", "SKIPPED", "No TTIR extracted")
        outcomes["recognise"] = StageOutcome("recognise", "SKIPPED", "No TTIR extracted")
        outcomes["select"] = StageOutcome("select", "SKIPPED", "No TTIR extracted")
        outcomes["assemble"] = StageOutcome("assemble", "SKIPPED", "No TTIR extracted")
        outcomes["execute"] = StageOutcome("execute", "SKIPPED", "No TTIR extracted")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "No TTIR extracted")
        return AotResult(name=name, ttir=None, program=None, outcomes=outcomes)

    # Stage 2: TTIR parsing
    try:
        parsed = parse_module(ttir)
        if not parsed.ok:
            diag = parsed.diagnostic
            detail = f"{diag.kind} at line {diag.line}: {diag.expected} vs {diag.found}" if diag else "Parse error"
            outcomes["parse"] = StageOutcome("parse", "FAIL", detail)
            outcomes["recognise"] = StageOutcome("recognise", "SKIPPED", "Parse failed")
            outcomes["select"] = StageOutcome("select", "SKIPPED", "Parse failed")
            outcomes["assemble"] = StageOutcome("assemble", "SKIPPED", "Parse failed")
            outcomes["execute"] = StageOutcome("execute", "SKIPPED", "Parse failed")
            outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Parse failed")
            return AotResult(name=name, ttir=ttir, program=None, outcomes=outcomes)
        module = parsed.module
        outcomes["parse"] = StageOutcome("parse", "OK", f"{len(module.functions())} func(s)")
    except Exception as exc:
        outcomes["parse"] = StageOutcome("parse", "FAIL", f"{type(exc).__name__}: {exc}")
        outcomes["recognise"] = StageOutcome("recognise", "SKIPPED", "Parse exception")
        outcomes["select"] = StageOutcome("select", "SKIPPED", "Parse exception")
        outcomes["assemble"] = StageOutcome("assemble", "SKIPPED", "Parse exception")
        outcomes["execute"] = StageOutcome("execute", "SKIPPED", "Parse exception")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Parse exception")
        return AotResult(name=name, ttir=ttir, program=None, outcomes=outcomes)

    # Stage 3: Def-Use & Recognise
    try:
        graph = build_def_use(module)
        annotations = annotate(module, graph)
        outcomes["recognise"] = StageOutcome("recognise", "OK", f"{annotations.annotated_count} annotation(s)")
    except Exception as exc:
        outcomes["recognise"] = StageOutcome("recognise", "REFUSED", f"{type(exc).__name__}: {exc}")
        outcomes["select"] = StageOutcome("select", "SKIPPED", "Recognition failed")
        outcomes["assemble"] = StageOutcome("assemble", "SKIPPED", "Recognition failed")
        outcomes["execute"] = StageOutcome("execute", "SKIPPED", "Recognition failed")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Recognition failed")
        return AotResult(name=name, ttir=ttir, program=None, outcomes=outcomes)

    # Stage 4: Select & Assemble
    try:
        schema = load_builtin(isa_name)
        program = assemble(module, graph, annotations, schema, env=env_dict)
        markers = program.markers()
        if markers:
            m = markers[0]
            outcomes["select"] = StageOutcome("select", "UNSUPPORTED", f"{m.kind}: {m.reason}")
            outcomes["assemble"] = StageOutcome("assemble", "REFUSED", f"{m.kind}: {m.reason}")
            outcomes["execute"] = StageOutcome("execute", "SKIPPED", "Has markers")
            outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Has markers")
            return AotResult(name=name, ttir=ttir, program=program, outcomes=outcomes)
        outcomes["select"] = StageOutcome("select", "OK", f"{len(program.instructions())} instrs")
        outcomes["assemble"] = StageOutcome("assemble", "OK", f"{len(program.instructions())} instrs assembled")
    except Exception as exc:
        outcomes["select"] = StageOutcome("select", "FAIL", f"{type(exc).__name__}: {exc}")
        outcomes["assemble"] = StageOutcome("assemble", "FAIL", f"{type(exc).__name__}: {exc}")
        outcomes["execute"] = StageOutcome("execute", "SKIPPED", "Assembly exception")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Assembly exception")
        return AotResult(name=name, ttir=ttir, program=None, outcomes=outcomes)

    # Stage 5: Execution
    if inputs is None:
        outcomes["execute"] = StageOutcome("execute", "SKIPPED", "No test inputs provided")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "No test inputs provided")
        return AotResult(name=name, ttir=ttir, program=program, outcomes=outcomes)

    try:
        policy = PrecisionPolicy()
        outputs = emulate(program, inputs, policy=policy, grid=grid)
        outcomes["execute"] = StageOutcome("execute", "OK", f"{len(outputs)} output(s) written")
    except Exception as exc:
        outcomes["execute"] = StageOutcome("execute", "FAIL", f"{type(exc).__name__}: {exc}")
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "Execution failed")
        return AotResult(name=name, ttir=ttir, program=program, outcomes=outcomes, outputs=None)

    # Stage 6: Reference parity
    if reference_fn is not None and outputs is not None:
        try:
            ref_outputs = reference_fn(inputs)
            max_err = 0.0
            for key, emu_arr in outputs.items():
                ref_key = key.lstrip("%")
                if ref_key in ref_outputs:
                    ref_arr = ref_outputs[ref_key]
                    denom = max(1.0, float(np.max(np.abs(ref_arr))))
                    err = float(np.max(np.abs(emu_arr - ref_arr))) / denom
                    max_err = max(max_err, err)
            parity_err = max_err
            status = "PASS" if max_err <= tolerance else "FAIL"
            outcomes["parity"] = StageOutcome("parity", status, f"rel_err={max_err:.4e}")
        except Exception as exc:
            outcomes["parity"] = StageOutcome("parity", "FAIL", f"Ref error: {exc}")
    else:
        outcomes["parity"] = StageOutcome("parity", "SKIPPED", "No reference fn")

    return AotResult(
        name=name,
        ttir=ttir,
        program=program,
        outcomes=outcomes,
        outputs=outputs,
        parity_error=parity_err,
    )
