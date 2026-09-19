"""Proof of concept: lower a live `torch.compile` FX graph to the target ISA.

**Scope.** This is a rudimentary path, deliberately. It exists to answer one
question that the recorded-lowering route cannot: *can an arbitrary graph that
Dynamo hands us be lowered to a declaratively described ISA at compile time,
with no frozen TTIR anywhere?* It handles a handful of aten operations on 1-D
and 2-D float32 tensors. Anything else returns `None` and the caller falls back
to eager, which is the same contract the rest of the seam keeps.

**What it is not.** It is not a replacement for the TTIR route. The TTIR path
gets tiling, masking, loop structure and `#loc` provenance from Triton; this one
gets none of those, because an FX graph does not carry them. A graph lowered
here runs one program over whole tensors rather than a grid over tiles. Where
the two disagree, the TTIR path is the one to trust.

**What it does not do — and this is the point.** It does not choose instruction
names. Every instruction is selected by `isa.select.select` against the loaded
schema, from a synthesised access descriptor, exactly as the TTIR path does. So:

    lower_fx_graph(gm, inputs, isa_name="tritonflow1")   -> DMA1D / MAC16 / EPI
    lower_fx_graph(gm, inputs, isa_name="tritonflow2")   -> LDG / OPU32 / VPU

with no branch on `isa_name` anywhere in this file. That is the property worth
demonstrating; a version of this module that mapped `aten.add -> "EPI"` would
prove nothing, and is the exact mistake `lower.py` used to make.

The emulator still decides *what arithmetic to perform* from each instruction's
`source_ops` provenance, so the aten op is recorded there under the `arith.*`
name the machine knows. That is the known abstraction leak (KNOWN_GAPS G-EPI),
not something this module introduces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..emit.ir import Imm, Instr, MemRef, Program, SourceRef, SsaRef
from ..emu.exec import ProgramNotExecutable, emulate
from ..emu.precision import TF32_EXPLICIT_MANTISSA_BITS, PrecisionPolicy
from ..isa.schema import load_builtin
from ..isa.select import _matches_op, select
from ..recognize.descriptor import AccessDescriptor

__all__ = ["FxLowering", "lower_fx_graph", "supported_targets"]

#: torch/aten call target (name only, no overload) -> (schema rule, TTIR op it
#: means). This table is *naming only*: it says what an operation is, never
#: whether a chip can do it. Support is decided per node by asking the loaded
#: schema whether any instruction claims the op (`schema_claims`), so adding an
#: op to a schema makes the backend accept it, and removing one makes it refuse.
_TORCH_TARGETS: dict[str, tuple[str, str]] = {
    "add": ("elementwise", "arith.addf"),
    "add_": ("elementwise", "arith.addf"),
    "sub": ("elementwise", "arith.subf"),
    "subtract": ("elementwise", "arith.subf"),
    "mul": ("elementwise", "arith.mulf"),
    "multiply": ("elementwise", "arith.mulf"),
    "div": ("elementwise", "arith.divf"),
    "div_": ("elementwise", "arith.divf"),
    "truediv": ("elementwise", "arith.divf"),
    "true_divide": ("elementwise", "arith.divf"),
    "neg": ("elementwise", "arith.negf"),
    "negative": ("elementwise", "arith.negf"),
    "abs": ("elementwise", "math.absf"),
    "absolute": ("elementwise", "math.absf"),
    "relu": ("elementwise", "arith.maxnumf"),
    "maximum": ("elementwise", "arith.maxnumf"),
    "minimum": ("elementwise", "arith.minnumf"),
    "clamp": ("elementwise", "tt.clamp"),
    "clamp_": ("elementwise", "tt.clamp"),
    "clip": ("elementwise", "tt.clamp"),
    "exp": ("elementwise", "math.exp"),
    "mm": ("mac", "tt.dot"),
    "matmul": ("mac", "tt.dot"),
}

#: Precision the emulator runs a lowered matmul at, and the one the tolerance is
#: derived for. One constant, used by both, so they cannot drift apart.
MATMUL_PRECISION = "ieee"

#: Fixed seed for the second, randomised verification input.
VERIFY_SEED = 24173


def _target_name(target: Any) -> str:
    raw = getattr(target, "__name__", str(target))
    if raw.startswith("aten."):
        raw = raw[len("aten."):]
    return raw.split(".")[0]


def schema_claims(schema: Any, rule: str, source_op: str) -> bool:
    """Does any instruction of `schema` claim `source_op`? (exact op match, as selection uses.)"""
    instructions = schema.of_kind(rule)
    if rule == "mac":
        return bool(instructions)
    return any(
        _matches_op(entry, source_op)
        for instruction in instructions
        for entry in getattr(instruction, "ops", ())
    )


def supported_targets(isa_name: str | None = None) -> tuple[str, ...]:
    """Torch operation names this backend can name; with `isa_name`, only those the schema claims."""
    if isa_name is None:
        return tuple(sorted(_TORCH_TARGETS))
    schema = load_builtin(isa_name)
    return tuple(
        sorted(name for name, (rule, op) in _TORCH_TARGETS.items() if schema_claims(schema, rule, op))
    )


@dataclass
class FxLowering:
    """One lowered FX graph: the program, how to feed it, and why anything failed."""

    program: Program
    inputs: tuple[str, ...]
    output: str
    isa_name: str
    node_count: int
    #: Shape of the output buffer. The emulator refuses to default a buffer the
    #: program writes, so `run` has to allocate it at the right extent.
    output_shape: tuple[int, ...] = ()
    lowered_nodes: tuple[str, ...] = ()
    refusals: list[str] = field(default_factory=list)
    #: Every selection made, as `(node, rule, chosen, rejected...)`. Kept because
    #: "the generator chose" is only a claim if the alternatives are visible.
    decisions: list[str] = field(default_factory=list)
    shadow_verified: bool = False
    #: Which inputs the emulated result was compared on: "example", "seed=<n>".
    verified_inputs: tuple[str, ...] = ()
    mismatch_reason: str | None = None

    @property
    def fully_lowered(self) -> bool:
        return (
            not self.refusals
            and not self.program.markers()
            and self.shadow_verified
            and self.mismatch_reason is None
        )

    def run(self, tensors: Sequence[Any]) -> np.ndarray:
        """Execute the program over `tensors`, in placeholder order."""
        storage: dict[str, Any] = {}
        for name, tensor in zip(self.inputs, tensors, strict=False):
            storage[name] = _as_numpy(tensor)
        storage[self.output] = np.zeros(self.output_shape or (1,), dtype=np.float32)
        produced = emulate(self.program, storage, policy=PrecisionPolicy(input_precision=MATMUL_PRECISION))
        return produced[self.output]


def _descriptor(shape: tuple[int, ...], base: str = "%fx") -> AccessDescriptor:
    """A contiguous row-major descriptor for a whole tensor.

    Synthesised from the tensor shape rather than recovered from index
    arithmetic, because an FX graph has no index arithmetic to recover: the
    operands are whole tensors. That is the honest difference between this path
    and the TTIR one, and it is why the descriptor here is always contiguous.
    """
    sizes = shape or (1,)
    strides: list[int] = []
    running = 1
    for extent in reversed(sizes):
        strides.append(running)
        running *= extent
    strides.reverse()
    # `base` is load-bearing, not cosmetic: `emu.exec._pointer_inputs` reads the
    # buffer name from the *descriptor's* base field, not from `MemRef.base`, so
    # a descriptor left at a placeholder name sends the emulator looking for a
    # buffer nobody supplied.
    return AccessDescriptor(
        base=base,
        sizes=tuple(sizes),
        strides=tuple(strides),
        offsets=tuple(0 for _ in sizes),
        shape=tuple(sizes),
        order=tuple(range(len(sizes) - 1, -1, -1)),
        dtype="f32",
        loop_carried=False,
        increment=None,
    )


def _choose(
    schema: Any,
    rule: str,
    shape: tuple[int, ...],
    direction: str | None = None,
    base: str = "%fx",
    names: frozenset[str] = frozenset(),
    k: int | None = None,
):
    """Select an instruction of `rule` for a whole-tensor access. Returns a report.

    The selector's second parameter is named `kind` but enumerates by *rule*:
    `schema.of_kind("mac")` returns MAC8/MAC16 while `schema.of_kind("compute")`
    returns nothing. The rule is therefore what gets passed, and no rule->kind
    translation happens here — doing one is how this function first returned
    "no admissible compute instruction" for every graph.
    """
    descriptor = _descriptor(shape, base)
    tile = tuple(shape) if rule == "mac" else None
    env = {"words": int(np.prod(shape or (1,)))}
    if rule == "mac" and len(shape) == 2:
        if k is None:
            raise ValueError("a matmul selection needs the reduction length k")
        env.update({"m": shape[0], "n": shape[1], "k": int(k)})
    report = select(schema, rule, descriptor, tile, env, direction)
    admissible = [c for c in report.candidates if c.admissible and c.cost is not None]
    if names:
        declared = [c for c in admissible if getattr(c.instruction, "ops", ())]
        if declared:
            admissible = [c for c in declared if any(_matches_op(op_entry, n) for op_entry in c.instruction.ops for n in names)]
    if not admissible:
        return None, report
    best = min(admissible, key=lambda c: (c.cost, c.instruction.declaration_index))
    return best, report


def _decision_line(node_name: str, rule: str, best: Any, report: Any) -> str:
    rejected = [
        f"{c.instruction.name}({c.rejected_by or c.cost})"
        for c in report.candidates
        if c.instruction.name != best.instruction.name
    ]
    return (
        f"{node_name}: {rule} -> {best.instruction.name} cost={best.cost:g}"
        + (f"; rejected {', '.join(rejected)}" if rejected else "")
    )


@dataclass(frozen=True)
class _NodeSpec:
    """What one lowered node computes, kept so its rounding error can be bounded."""

    name: str
    source_op: str
    operands: dict[str, Any]


_U = 2.0**-24  # fp32 unit roundoff
_TF32_EPS = 2.0 ** -(TF32_EXPLICIT_MANTISSA_BITS + 1)  # half-ulp of a 10-bit mantissa


def _fetch(operand: Any, table: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(operand, SsaRef):
        return table[operand.name]
    return np.asarray(float(operand.value), dtype=np.float64), np.asarray(0.0)


def _error_analysis(
    specs: Sequence[_NodeSpec],
    inputs: dict[str, np.ndarray],
    precision: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Value in fp64 and a rigorous absolute bound on the fp32 result's error, per node.

    Each op contributes one rounding (`u = 2**-24` of the computed value), exact ops
    (neg, abs, max, min, clamp) contribute none and are 1-Lipschitz so they pass the
    incoming bound through, and a dot product of length `k` accumulated in fp32 adds
    `k*u/(1-k*u)` of the sum of absolute products. Under a tf32 policy each matmul
    operand additionally carries the tf32 half-ulp. Nothing here is tuned: change the
    graph or the precision and the bound changes with it.
    """
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (np.asarray(value, dtype=np.float64), np.zeros(np.shape(value))) for name, value in inputs.items()
    }
    eps = _TF32_EPS if precision == "tf32" else 0.0
    for spec in specs:
        op = spec.source_op
        ops = spec.operands
        if op == "tt.dot":
            a, ea = _fetch(ops["a"], table)
            b, eb = _fetch(ops["b"], table)
            k = a.shape[-1]
            gamma = k * _U / (1.0 - k * _U)
            reach = (np.abs(a) + ea) @ (np.abs(b) + eb)
            value = a @ b
            bound = np.abs(a) @ eb + ea @ np.abs(b) + ea @ eb + reach * (gamma + (2 * eps + eps * eps) * (1 + gamma))
        else:
            a, ea = _fetch(ops["in0"], table)
            if op in ("arith.negf", "math.absf", "tt.clamp"):
                if op == "arith.negf":
                    value = -a
                elif op == "math.absf":
                    value = np.abs(a)
                else:
                    value = np.clip(a, float(ops["lo"].value), float(ops["hi"].value))
                bound = np.broadcast_to(ea, np.shape(value)).astype(np.float64)
            else:
                b, eb = _fetch(ops["in1"], table)
                if op in ("arith.maxnumf", "arith.minnumf"):
                    value = np.maximum(a, b) if op == "arith.maxnumf" else np.minimum(a, b)
                    bound = np.maximum(ea, eb)
                elif op in ("arith.addf", "arith.subf"):
                    value = a + b if op == "arith.addf" else a - b
                    bound = ea + eb + _U * (np.abs(value) + ea + eb)
                elif op == "arith.mulf":
                    value = a * b
                    bound = np.abs(a) * eb + np.abs(b) * ea + ea * eb + _U * (np.abs(value) + np.abs(a) * eb + np.abs(b) * ea + ea * eb)
                elif op == "arith.divf":
                    value = a / b
                    denom = np.abs(b) - eb
                    with np.errstate(divide="ignore", invalid="ignore"):
                        core = (ea + np.abs(value) * eb) / denom
                    core = np.where(denom > 0, core, np.inf)
                    bound = core + _U * (np.abs(value) + core)
                else:
                    raise ValueError(f"no error model for {op!r}")
        table[spec.name] = (value, np.asarray(bound, dtype=np.float64))
    return table


def _as_numpy(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    return np.array(array, dtype=np.float32, copy=True)


def _shadow_verify(
    program: Program,
    graph_module: Any,
    example_inputs: Sequence[Any],
    input_names: Sequence[str],
    out_name: str,
    store_shape: tuple[int, ...],
    specs: Sequence[_NodeSpec],
    output_value: str,
    refusals: list[str],
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Run the emulated program and eager torch on the example and on a seeded random input.

    Eager runs on clones, so a graph containing in-place ops cannot change the
    caller's tensors. The result must agree element for element within twice the
    derived fp32 error bound (each side is within one bound of the exact value).
    """
    import torch

    tensor_positions = [i for i, x in enumerate(example_inputs) if hasattr(x, "shape")]
    rng = np.random.default_rng(VERIFY_SEED)
    input_sets: list[tuple[str, list[np.ndarray]]] = [
        ("example", [_as_numpy(example_inputs[i]) for i in tensor_positions]),
        (
            f"seed={VERIFY_SEED}",
            [rng.standard_normal(np.shape(example_inputs[i])).astype(np.float32) for i in tensor_positions],
        ),
    ]
    tiny = float(np.finfo(np.float32).tiny)
    verified: list[str] = []
    for label, arrays in input_sets:
        try:
            eager_args = list(example_inputs)
            for pos, array in zip(tensor_positions, arrays, strict=True):
                eager_args[pos] = torch.from_numpy(array.copy())
            eager = graph_module(*eager_args)
            eager = eager[0] if isinstance(eager, (tuple, list)) else eager
            reference = np.array(eager.detach().cpu().numpy(), dtype=np.float64)

            storage = {name: array.copy() for name, array in zip(input_names, arrays, strict=True)}
            storage[out_name] = np.zeros(store_shape or (1,), dtype=np.float32)
            emulated = emulate(program, storage, policy=PrecisionPolicy(input_precision=MATMUL_PRECISION))[out_name]
            emulated = np.asarray(emulated, dtype=np.float64).reshape(reference.shape)

            table = _error_analysis(specs, dict(zip(input_names, arrays, strict=True)), MATMUL_PRECISION)
            if output_value not in table:
                raise ValueError(f"output {output_value} has no error model")
            bound = 2.0 * np.broadcast_to(table[output_value][1], reference.shape)
        except Exception as exc:
            reason = f"shadow verification failed to execute on {label}: {exc}"
            refusals.append(reason)
            return False, reason, tuple(verified)

        if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(emulated))):
            reason = f"shadow verification on {label}: non-finite values, tolerance cannot be certified"
            refusals.append(reason)
            return False, reason, tuple(verified)
        scale = np.maximum(np.abs(reference), tiny)
        relative = np.abs(emulated - reference) / scale
        allowed = bound / scale
        if not np.all(relative <= allowed):
            worst = int(np.argmax(relative - allowed))
            reason = (
                f"shadow verification mismatch on {label}: relative error "
                f"{float(relative.reshape(-1)[worst]):.3g} exceeds the derived bound "
                f"{float(allowed.reshape(-1)[worst]):.3g} at flat index {worst}"
            )
            refusals.append(reason)
            return False, reason, tuple(verified)
        verified.append(label)
    return True, None, tuple(verified)


def lower_fx_graph(
    graph_module: Any,
    example_inputs: Sequence[Any],
    *,
    isa_name: str = "tritonflow1",
    report: list[str] | None = None,
) -> FxLowering | None:
    """Lower one FX graph to `isa_name`, or return `None` if it cannot be lowered.

    `report`, when given, collects the reason for every refusal -- the same
    out-parameter convention `emit.assemble.assemble` uses. Without it a `None`
    return says "fell back" and not "fell back *because* this ISA declares no
    multiply", which is the difference between a coverage report a reader can act
    on and one that only counts.

    `None` rather than an exception: an unlowerable graph is the eager-fallback
    case the seam already handles, not a defect. A graph that lowers *partially*
    is also `None` — all-or-nothing, because a half-lowered program would have to
    hand intermediate values back to eager and this path has no mechanism for
    that.
    """
    try:
        schema = load_builtin(isa_name)
    except Exception:  # pragma: no cover - a missing schema is a packaging fault
        return None

    nodes = list(graph_module.graph.nodes)
    shapes: dict[str, tuple[int, ...]] = {}
    input_names: list[str] = []
    instrs: list[Instr] = []
    decisions: list[str] = []
    refusals: list[str] = report if report is not None else []
    lowered: list[str] = []
    specs: list[_NodeSpec] = []
    total_cost = 0.0
    tensor_inputs = [t for t in example_inputs if hasattr(t, "shape")]
    position = 0
    output_value: str | None = None

    for node in nodes:
        if node.op == "placeholder":
            if position >= len(tensor_inputs):
                refusals.append(f"{node.name}: no example tensor supplied")
                return None
            name = f"%{node.name}"
            if str(getattr(tensor_inputs[position], "dtype", "")) != "torch.float32":
                refusals.append(
                    f"{node.name}: placeholder dtype {getattr(tensor_inputs[position], 'dtype', None)} "
                    "is not float32; the emulated machine computes in float32"
                )
                return None
            shapes[name] = tuple(int(d) for d in tensor_inputs[position].shape)
            input_names.append(name)
            position += 1
            continue

        if node.op == "output":
            arg = node.args[0]
            arg = arg[0] if isinstance(arg, (tuple, list)) else arg
            output_value = f"%{getattr(arg, 'name', arg)}"
            continue

        if node.op != "call_function":
            refusals.append(f"{node.name}: node kind {node.op!r} is not lowered")
            return None

        # Check for unsupported kwargs:
        if "alpha" in node.kwargs and node.kwargs["alpha"] != 1:
            refusals.append(
                f"{node.name}: unsupported kwarg 'alpha'={node.kwargs['alpha']!r}"
            )
            return None

        if "rounding_mode" in node.kwargs and node.kwargs["rounding_mode"] is not None:
            refusals.append(
                f"{node.name}: unsupported kwarg 'rounding_mode'={node.kwargs['rounding_mode']!r}"
            )
            return None

        if "out" in node.kwargs and node.kwargs["out"] is not None:
            refusals.append(f"{node.name}: unsupported kwarg 'out'")
            return None

        if "dtype" in node.kwargs:
            dt = node.kwargs["dtype"]
            if dt is not None and str(dt) not in ("torch.float32", "float32", "<class 'float'>"):
                refusals.append(f"{node.name}: unsupported non-default dtype {dt!r}")
                return None

        allowed_kwargs = {"min", "max", "alpha", "rounding_mode", "out", "dtype", "inplace"}
        unexpected = set(node.kwargs.keys()) - allowed_kwargs
        if unexpected:
            refusals.append(f"{node.name}: unsupported kwargs {sorted(unexpected)}")
            return None

        target = _target_name(node.target)

        entry = _TORCH_TARGETS.get(target)
        if entry is None:
            refusals.append(f"{node.name}: target {target!r} has no known TTIR meaning")
            return None
        rule, source_op = entry
        if not schema_claims(schema, rule, source_op):
            refusals.append(
                f"{node.name}: {isa_name} declares no {rule} instruction serving {source_op!r}"
            )
            return None

        # Reject unsupported positional kwargs
        if target in ("add", "add_", "sub", "sub_") and len(node.args) > 2:
            alpha_arg = node.args[2]
            if alpha_arg != 1:
                refusals.append(
                    f"{node.name}: unsupported positional alpha={alpha_arg!r}"
                )
                return None

        if target in ("div", "div_") and len(node.args) > 2:
            rm_arg = node.args[2]
            if rm_arg is not None:
                refusals.append(
                    f"{node.name}: unsupported positional rounding_mode={rm_arg!r}"
                )
                return None

        operands: dict[str, Any] = {}
        operand_shapes: list[tuple[int, ...]] = []
        reduction_k: int | None = None

        if target in ("clamp", "clamp_"):
            lo_val = None
            hi_val = None
            if len(node.args) >= 3:
                lo_val = node.args[1]
                hi_val = node.args[2]
            elif "min" in node.kwargs and "max" in node.kwargs:
                lo_val = node.kwargs["min"]
                hi_val = node.kwargs["max"]
            elif len(node.args) == 2 and "max" in node.kwargs:
                lo_val = node.args[1]
                hi_val = node.kwargs["max"]
            elif len(node.args) == 2 and "min" in node.kwargs:
                lo_val = node.kwargs["min"]
                hi_val = node.args[1]

            if lo_val is None or hi_val is None:
                refusals.append(f"{node.name}: clamp requires both min and max bounds")
                return None

            try:
                lo_f = float(lo_val)
                hi_f = float(hi_val)
            except (TypeError, ValueError):
                refusals.append(
                    f"{node.name}: clamp bounds must be numeric constants, got {lo_val!r}, {hi_val!r}"
                )
                return None

            arg0 = node.args[0]
            arg_name = f"%{getattr(arg0, 'name', arg0)}"
            if arg_name not in shapes:
                refusals.append(f"{node.name}: operand {arg_name} has no known shape")
                return None
            operand_shapes.append(shapes[arg_name])
            operands["in0"] = SsaRef(arg_name)
            operands["in1"] = Imm(value=lo_f)
            operands["in2"] = Imm(value=hi_f)
            operands["lo"] = Imm(value=lo_f)
            operands["hi"] = Imm(value=hi_f)
            result_shape = operand_shapes[0]
        else:
            for index, arg in enumerate(node.args):
                if index >= 2 and target in ("add", "add_", "sub", "sub_", "div", "div_"):
                    continue
                arg_name = f"%{getattr(arg, 'name', arg)}"
                if arg_name in shapes:
                    operand_shapes.append(shapes[arg_name])
                    role = ("a", "b")[index] if rule == "mac" and index < 2 else f"in{index}"
                    operands[role] = SsaRef(arg_name)
                elif isinstance(arg, (int, float)):
                    operands[f"in{index}"] = Imm(value=float(arg))
                else:
                    refusals.append(f"{node.name}: operand {arg_name} has no known shape")
                    return None

            if rule == "mac":
                if len(operand_shapes) != 2 or len(operand_shapes[0]) != 2:
                    refusals.append(f"{node.name}: matmul needs two 2-D operands")
                    return None
                if operand_shapes[0][1] != operand_shapes[1][0]:
                    refusals.append(
                        f"{node.name}: matmul inner dimensions differ {operand_shapes[0]} x {operand_shapes[1]}"
                    )
                    return None
                result_shape = (operand_shapes[0][0], operand_shapes[1][1])
                reduction_k = operand_shapes[0][1]
                operands["acc"] = Imm(value=0.0)
            else:
                reduction_k = None
                try:
                    result_shape = tuple(np.broadcast_shapes(*operand_shapes)) if operand_shapes else (1,)
                except ValueError:
                    refusals.append(f"{node.name}: operand shapes {operand_shapes} do not broadcast")
                    return None
                if source_op == "arith.maxnumf" and len(operands) == 1:
                    # relu is max(x, 0)
                    operands["in1"] = Imm(value=0.0)

        best, report_sel = _choose(
            schema, rule, result_shape, base=source_op, names=frozenset({source_op}), k=reduction_k
        )
        if best is None:
            refusals.append(
                f"{node.name}: no admissible {rule} instruction in {isa_name} for shape {result_shape}"
            )
            return None

        name = f"%{node.name}"
        shapes[name] = result_shape
        decisions.append(_decision_line(node.name, rule, best, report_sel))
        lowered.append(node.name)
        specs.append(_NodeSpec(name, source_op, dict(operands)))
        total_cost += float(best.cost or 0.0)
        instrs.append(
            Instr(
                name=best.instruction.name,
                operands=operands,
                cost=float(best.cost or 0.0),
                defs=(name,),
                source_ops=(SourceRef(source_op, len(instrs) + 1, 0, node.name),),
                constraint=str(getattr(best.instruction.constraint, "text", "")),
            )
        )

    if output_value is None or output_value not in shapes:
        return None

    store_shape = shapes[output_value]
    out_name = "%fx_out"
    store, store_report = _choose(
        schema, "memory", store_shape, direction="store", base=out_name
    )
    if store is None:
        refusals.append(f"output: no admissible memory instruction in {isa_name}")
        return None
    decisions.append(_decision_line("output", "memory", store, store_report))
    total_cost += float(store.cost or 0.0)
    instrs.append(
        Instr(
            name=store.instruction.name,
            operands={
                "dst": MemRef.of("global", out_name, _descriptor(store_shape, out_name)),
                "value": SsaRef(output_value),
            },
            cost=float(store.cost or 0.0),
            source_ops=(SourceRef("tt.store", len(instrs) + 1, 0, "output"),),
        )
    )

    program = Program(
        isa_name=isa_name,
        schema_version=schema.schema_version,
        kernel_name=getattr(graph_module, "_get_name", lambda: "fx_graph")(),
        instrs=tuple(instrs),
        inputs=(*input_names, out_name),
        total_cost=total_cost,
    )

    shadow_verified, mismatch_reason, verified_inputs = _shadow_verify(
        program, graph_module, example_inputs, input_names, out_name, store_shape, specs, output_value, refusals
    )
    if not shadow_verified:
        return None

    return FxLowering(
        program=program,
        inputs=tuple(input_names),
        output=out_name,
        isa_name=isa_name,
        node_count=len(nodes),
        output_shape=store_shape,
        lowered_nodes=tuple(lowered),
        refusals=refusals,
        decisions=decisions,
        shadow_verified=shadow_verified,
        mismatch_reason=mismatch_reason,
        verified_inputs=verified_inputs,
    )


def try_lower_and_run(
    graph_module: Any,
    example_inputs: Sequence[Any],
    tensors: Sequence[Any],
    *,
    isa_name: str = "tritonflow1",
) -> tuple[np.ndarray | None, FxLowering | None]:
    """Lower and execute, returning `(result, lowering)`; `(None, _)` means fall back."""
    lowering = lower_fx_graph(graph_module, example_inputs, isa_name=isa_name)
    if lowering is None or not lowering.fully_lowered:
        return None, lowering
    try:
        return lowering.run(tensors), lowering
    except ProgramNotExecutable:
        return None, lowering
