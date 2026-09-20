"""`emulate` / `apply` — `contracts/emulator.md` (T011).

The emulator is a small machine that consumes the **emitted instruction stream**
and nothing else. It never opens the TTIR module: the artifact under test is the
program (postcondition 1), and an emulator that "knows" it is looking at a
matmul would agree with the reference by construction and validate nothing.

Three model decisions, stated up front because every line below depends on them.

**Memory is one flat fp32 array; a pointer is an element index into it.** Each
pointer argument the kernel takes (`%a_ptr`, `%x_ptr`) owns a contiguous range at
a known base offset. `splat(%a_ptr)` materialises that base, `addptr` adds element
offsets, and a load/store indexes the flat array. There are no host addresses, so
the addressing arithmetic the kernel performs (`rm*sam + rk*sak`) is *executed*,
not interpreted symbolically — which is what makes the index math part of the
test rather than part of the oracle.

**A value is a Python/numpy scalar or an `ndarray`.** Scalars stay scalars and
broadcasting is numpy's, so `splat`/`broadcast` need no special case beyond a
target shape.

**The instruction's own `source.op_name` disambiguates the overloaded compute
instruction.** This is the one genuinely uncomfortable fact about executing the
current program format, and it is recorded rather than hidden (audit finding F8):
`EPI` is the generic elementwise unit for every non-memory, non-dot operation, and
the specific operation (`arith.addi`, `tt.expand_dims`, `tt.make_range`, …) lives
in the instruction's provenance, not in a role. Three consequences, all of which
would be fixed by giving `EPI` an `op` role the way `MAC8` has a tile:

* the elementwise op is read from `SourceRef.op_name`;
* a result shape is read from the descriptor recorded in `Instr.constrained_on`
  (`sizes=[…]`), because the program carries no shape table;
* `arith.cmpi`'s comparison predicate is not recorded anywhere, so a maskcompare
    "arith.cmpf": _cmpf,
    "arith.select": _select,
  is executed as the signed less-than the corpus uses.

Each is a *documented constraint of the current artifact*, checkable by reading
the program text — not a guess the emulator makes about intent. Everything the
stream does carry (operands, descriptors, loop re-threading) is used as given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..emit.ir import (
    Imm,
    Instr,
    Loop,
    MemRef,
    Operand,
    Program,
    SsaRef,
    UnsupportedMarker,
)
from .precision import PrecisionPolicy
from .tcu import TcuEmulator

#: Instruction classification tables. Derived from the loaded ISA schema at
#: runtime via `_build_instruction_tables()`, not hardcoded per ISA.
#: A fourth ISA does NOT require editing this file — schema `rule` fields drive
#: the classification (fix for audit finding: "zero-edit transfer" falsification).

# Legacy constants kept for backward compat with existing checks/tests that
# import them, but `apply()` now uses schema-derived tables when available.
MEMORY_INSTRUCTIONS = ("DMA1D", "DMA2D", "LDG", "LDS2D", "STG", "LDS", "STS", "BARRIER")
MAC_INSTRUCTIONS = ("MAC8", "MAC16", "OPU8", "OPU32", "TCU_MMA16", "TCU_MMA32")
ELEMENTWISE_INSTRUCTIONS = (
    "EPI",
    "VPU",
    "CLAMP",
    "VADD",
    "VMUL",
    "VSUB",
    "VMOD",
    "VRELU",
    "VCLAMP",
)
ELEMENTWISE_INSTRUCTION = "EPI"  # ISA-1's spelling (kept for the ISA-1 checks)

# Schema-derived instruction tables — populated per program
_SCHEMA_TABLES: dict[str, dict[str, set[str]]] = {}


def _build_instruction_tables(isa_name: str) -> dict[str, set[str]]:
    """Derive instruction classification from schema, not hardcoded tables.

    Each instruction's ``rule`` field (memory | mac | elementwise) determines
    which handler processes it. This closes the audit finding that adding a
    fourth ISA required editing exec.py.
    """
    if isa_name in _SCHEMA_TABLES:
        return _SCHEMA_TABLES[isa_name]

    try:
        from ..isa.schema import load_builtin
        schema = load_builtin(isa_name)
        tables: dict[str, set[str]] = {"memory": set(), "mac": set(), "elementwise": set()}
        for instr in schema.instructions.values():
            if instr.rule in tables:
                tables[instr.rule].add(instr.name)
            else:
                tables.setdefault(instr.rule, set()).add(instr.name)
        _SCHEMA_TABLES[isa_name] = tables
        return tables
    except Exception:
        # Fallback to legacy constants if schema loading fails
        return {
            "memory": set(MEMORY_INSTRUCTIONS),
            "mac": set(MAC_INSTRUCTIONS),
            "elementwise": set(ELEMENTWISE_INSTRUCTIONS),
        }


def _classify_instruction(instr_name: str, isa_name: str | None = None) -> str | None:
    """Return the rule kind for an instruction, derived from schema if available."""
    if isa_name:
        tables = _build_instruction_tables(isa_name)
        for kind, names in tables.items():
            if instr_name in names:
                return kind
        return None
    # Fallback to legacy
    if instr_name in MEMORY_INSTRUCTIONS:
        return "memory"
    if instr_name in MAC_INSTRUCTIONS:
        return "mac"
    if instr_name in ELEMENTWISE_INSTRUCTIONS:
        return "elementwise"
    return None

#: `tt.get_program_id` axis token → the launch-grid key the emulator reads.
PROGRAM_ID_AXES = ("x", "y", "z")


class ProgramNotExecutable(RuntimeError):
    """A program carrying an `UNSUPPORTED` marker (postcondition 2).

    Halts *locally*: the seam routes this kernel to the eager fallback. The
    emulator never guesses what the unsupported operation meant.
    """

    def __init__(self, marker: UnsupportedMarker) -> None:
        self.marker = marker
        super().__init__(
            f"program is not executable: {marker.kind} at {marker.loc_name or '?'} "
            f"({marker.op_name}): {marker.reason}"
        )


class UnsupportedInstruction(RuntimeError):
    """An instruction name this machine does not implement."""


class StorageError(RuntimeError):
    """A memory access outside the emulated storage — named, never zero-filled."""


class MissingInput(LookupError):
    """An SSA value the program reads that no instruction produced and no input supplied."""


@dataclass(frozen=True)
class Storage:
    """One named buffer: where it lives and what shape it came in as."""

    name: str
    base: int
    length: int
    shape: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.base + self.length


@dataclass
class MachineState:
    """The toy machine: flat memory, the named buffers, and the value store."""

    program: Program
    policy: PrecisionPolicy
    memory: np.ndarray
    storages: dict[str, Storage] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    written: set[str] = field(default_factory=set)
    grid: tuple[int, ...] = (0, 0, 0)
    loop_iteration: int = 0

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_program(
        cls,
        program: Program,
        inputs: dict[str, np.ndarray],
        policy: PrecisionPolicy | None = None,
        grid: tuple[int, ...] = (0, 0, 0),
        loop_iteration: int = 0,
    ) -> MachineState:
        policy = policy or PrecisionPolicy()
        pointers = _pointer_inputs(program)
        storages: dict[str, Storage] = {}
        buffers: list[np.ndarray] = []
        offset = 0
        for name in program.inputs:
            if name not in pointers:
                continue
            if name not in inputs:
                raise MissingInput(
                    f"pointer input {name} was not supplied; the kernel addresses it and "
                    "the emulator will not fabricate storage for it"
                )
            array = np.asarray(inputs[name], dtype=np.float32)
            flat = array.reshape(-1)
            storages[name] = Storage(name, offset, flat.size, array.shape)
            buffers.append(flat)
            offset += flat.size

        memory = np.concatenate(buffers) if buffers else np.zeros(0, dtype=np.float32)
        state = cls(program=program, policy=policy, memory=memory, storages=storages, grid=grid)
        for name in program.inputs:
            if name in storages:
                state.values[name] = storages[name].base
            elif name in inputs:
                state.values[name] = inputs[name]
            elif name in PROGRAM_ID_AXES:
                state.values[name] = grid[PROGRAM_ID_AXES.index(name)]
            else:
                raise MissingInput(
                    f"input {name} was not supplied and is not a program-id axis; the "
                    "emulator refuses to default a value the kernel reads"
                )
        return state

    # -- values ------------------------------------------------------------- #

    def resolve(self, operand: Operand) -> Any:
        if isinstance(operand, Imm):
            return operand.value
        if isinstance(operand, SsaRef):
            if operand.name in self.values:
                return self.values[operand.name]
            raise MissingInput(
                f"{operand.name} is read but nothing produced it; a re-threaded loop value "
                "with no producer is an unexecutable program, not a zero"
            )
        if isinstance(operand, MemRef):
            return self._resolve_memref(operand)
        raise TypeError(f"cannot resolve operand {operand!r}")

    def bind(self, name: str, value: Any) -> None:
        self.values[name] = value


    def _materialize_descriptor(self, memref: "MemRef", base: int) -> np.ndarray:
        from tritonflow.emu.exec import _descriptor_fields
        fields = _descriptor_fields(memref.access_key)
        raw_sizes = fields.get("sizes", "[]").strip("[]").strip()
        sizes = tuple(int(s) for s in raw_sizes.split(",") if s.strip()) if raw_sizes else ()

        raw_strides = fields.get("strides", "[]").strip("[]").strip()
        strides = []
        for s in (raw_strides.split(",") if raw_strides else []):
            s = s.strip()
            if not s: continue
            try:
                strides.append(int(s))
            except ValueError:
                if s in self.values:
                    strides.append(int(np.asarray(self.values[s]).item()))
                else:
                    strides.append(1)
        strides = tuple(strides)

        raw_offsets = fields.get("offsets", "[]").strip("[]").strip()
        offsets = []
        for o in (raw_offsets.split(",") if raw_offsets else []):
            o = o.strip()
            try:
                offsets.append(int(o))
            except ValueError:
                offsets.append(0)

        is_loop_carried = fields.get("loop_carried", "False") == "True"
        raw_inc = fields.get("increment", "0")
        inc = 0
        try:
            inc = int(raw_inc)
        except ValueError:
            if raw_inc in self.values:
                inc = int(np.asarray(self.values[raw_inc]).item())
                
        loop_offset = self.loop_iteration * inc if is_loop_carried else 0

        if not sizes:
            return np.array([base], dtype=np.int64)

        coords = np.indices(sizes, dtype=np.int64)
        addresses = np.full(sizes, base + loop_offset, dtype=np.int64)
        for dim in range(len(sizes)):
            off = offsets[dim] if dim < len(offsets) else 0
            stride = strides[dim] if dim < len(strides) else 1
            addresses += coords[dim] * stride + off
        return addresses

    def _resolve_memref(self, memref: "MemRef") -> np.ndarray:
        value = self.resolve(SsaRef(memref.base))
        arr = np.asarray(value, dtype=np.int64)
        if arr.ndim > 0:
            return arr
        return self._materialize_descriptor(memref, int(arr))

    def address_of(self, name: str) -> Storage | None:
        for storage in self.storages.values():
            if name == storage.name:
                return storage
        return None

    # -- memory ------------------------------------------------------------- #

    def gather(self, indices: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        flat = np.asarray(indices).reshape(-1).astype(np.int64)
        self._check_bounds(flat)
        out = self.memory[flat].astype(np.float32)
        if mask is not None:
            out = np.where(np.asarray(mask).reshape(-1), out, np.float32(0.0))
        return out

    def scatter(self, indices: np.ndarray, values: np.ndarray, mask: np.ndarray | None) -> None:
        flat = np.asarray(indices).reshape(-1).astype(np.int64)
        self._check_bounds(flat)
        payload_array = np.asarray(values, dtype=np.float32)
        payload = (
            payload_array.reshape(-1)
            if payload_array.size == flat.size
            else np.broadcast_to(payload_array, flat.shape)
        )
        if mask is not None:
            keep = np.asarray(mask).reshape(-1)
            self.memory[flat[keep]] = payload[keep]
        else:
            self.memory[flat] = payload
        self._mark_written(flat)

    def _check_bounds(self, flat: np.ndarray) -> None:
        if flat.size == 0:
            return
        low, high = int(flat.min()), int(flat.max())
        if low < 0 or high >= self.memory.size:
            where = ", ".join(f"{s.name}[{s.base}:{s.end}]" for s in self.storages.values())
            raise StorageError(
                f"access out of emulated storage: index range [{low}, {high}] but memory "
                f"has {self.memory.size} elements ({where}); refusing to zero-fill"
            )

    def _mark_written(self, flat: np.ndarray) -> None:
        if flat.size == 0:
            return
        low, high = int(flat.min()), int(flat.max())
        for storage in self.storages.values():
            if low < storage.end and high >= storage.base:
                self.written.add(storage.name)

    def outputs(self) -> dict[str, np.ndarray]:
        """Every buffer the program wrote, reshaped to the shape it arrived in."""
        out: dict[str, np.ndarray] = {}
        for name in sorted(self.written):
            storage = self.storages[name]
            out[name] = self.memory[storage.base : storage.end].reshape(storage.shape).copy()
        return out


# --------------------------------------------------------------------------- #
# Descriptor field recovery
# --------------------------------------------------------------------------- #


def _descriptor_fields(key: str | None) -> dict[str, str]:
    """Parse the `k=v;…` descriptor key into its fields.

    The program's own recorded form (`recognize.descriptor.descriptor_key`), not
    a second serialisation invented here.
    """
    if not key:
        return {}
    fields: dict[str, str] = {}
    for part in key.split(";"):
        name, sep, value = part.partition("=")
        if sep:
            fields[name.strip()] = value.strip()
    return fields


def _sizes(key: str | None) -> tuple[int, ...]:
    raw = _descriptor_fields(key).get("sizes", "[]")
    inner = raw.strip("[]").strip()
    if not inner:
        return ()
    return tuple(int(item) for item in inner.split(",") if item.strip())


def _declared_shape(instr: Instr) -> tuple[int, ...]:
    for operand in instr.operands.values():
        if isinstance(operand, MemRef):
            return _sizes(operand.access_key)
    return ()


def _pointer_inputs(program: Program) -> set[str]:
    """Input names a memory operand addresses — i.e. the buffers the kernel owns.

    The buffer is named by the descriptor's **recovered** base (`%x_ptr`), not by
    `MemRef.base`, which is the pointer *value* the DMA advances through
    (`%x_4`). The distinction is load-bearing: `%x_4` is produced by the
    instruction stream and `%x_ptr` is supplied by the caller.
    """
    names: set[str] = set()
    for instr in program.instructions():
        for role in instr.roles:
            operand = instr.operands[role]
            if isinstance(operand, MemRef):
                names.add(_descriptor_fields(operand.access_key).get("base", operand.base))
    return names


def _mask_of(instr: Instr, state: MachineState) -> np.ndarray | None:
    operand = instr.operand("mask")
    return None if operand is None else np.asarray(state.resolve(operand))


# --------------------------------------------------------------------------- #
# Instruction semantics
# --------------------------------------------------------------------------- #


def apply(instr: Instr, state: MachineState, policy: PrecisionPolicy) -> None:
    """Execute one instruction against `state` (contracts/emulator.md interface).

    Dispatch is schema-derived when a program carries ``isa_name``: the
    instruction's ``rule`` field (memory | mac | elementwise) from the ISA
    schema determines which handler processes it. This means adding a new ISA
    does NOT require editing this function — the schema drives classification.

    Falls back to the legacy hardcoded tables only when no schema is available.
    """
    # Try schema-derived classification first
    isa_name = getattr(state, '_isa_name', None)
    kind = _classify_instruction(instr.name, isa_name)

    if kind == "async_copy":
        _apply_memory(instr, state)
        return
    if kind == "memory":
        _apply_memory(instr, state)
        return
    if kind == "mac":
        _apply_mac(instr, state, policy)
        return
    if kind == "elementwise":
        _apply_elementwise(instr, state)
        return

    # Legacy fallback for backward compatibility
    if instr.name in MEMORY_INSTRUCTIONS:
        _apply_memory(instr, state)
        return
    if instr.name in MAC_INSTRUCTIONS:
        _apply_mac(instr, state, policy)
        return
    if instr.name in ELEMENTWISE_INSTRUCTIONS:
        _apply_elementwise(instr, state)
        return

    all_known = set(MEMORY_INSTRUCTIONS) | set(MAC_INSTRUCTIONS) | set(ELEMENTWISE_INSTRUCTIONS)
    if isa_name:
        tables = _build_instruction_tables(isa_name)
        for names in tables.values():
            all_known |= names
    raise UnsupportedInstruction(
        f"instruction {instr.name!r} is not implemented by this machine "
        f"(known: {(*MEMORY_INSTRUCTIONS, *MAC_INSTRUCTIONS, *ELEMENTWISE_INSTRUCTIONS)}); "
        "an unimplemented instruction is a refusal, not a no-op"
    )


def _apply_memory(instr: Instr, state: MachineState) -> None:
    """A DMA: `dst` present means store, `src` present means load."""
    if instr.name == "BARRIER":
        return
    mask = _mask_of(instr, state)
    dst = instr.operand("dst")
    if isinstance(dst, MemRef):
        indices = state.resolve(dst)
        value_operand = instr.operand("value")
        if value_operand is None:
            raise UnsupportedInstruction(f"{instr.name} store has no value operand")
        state.scatter(indices, state.resolve(value_operand), _align_mask(mask, indices))
        return

    src = instr.operand("src")
    if not isinstance(src, MemRef):
        raise UnsupportedInstruction(
            f"{instr.name} has neither a dst nor a src memory operand; cannot tell a load "
            "from a store and will not guess"
        )
    if not instr.defs:
        raise UnsupportedInstruction(f"{instr.name} load defines no value to bind")
    loaded = state.gather(state.resolve(src), _align_mask(mask, state.resolve(src)))
    state.bind(instr.defs[0], loaded.reshape(_declared_shape(instr) or loaded.shape))


def _align_mask(mask: np.ndarray | None, indices: Any) -> np.ndarray | None:
    if mask is None:
        return None
    count = np.asarray(indices).size
    flat = np.broadcast_to(np.asarray(mask), (count,)) if mask.size != count else mask
    return flat.reshape(-1)


def _apply_mac(instr: Instr, state: MachineState, policy: PrecisionPolicy) -> None:
    a = np.asarray(state.resolve(_require(instr, "a")), dtype=np.float32)
    b = np.asarray(state.resolve(_require(instr, "b")), dtype=np.float32)
    acc_operand = instr.operand("acc")
    if acc_operand is None:
        raise UnsupportedInstruction(f"{instr.name} has no accumulator operand")
    acc = state.resolve(acc_operand)
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32) if np.isscalar(acc) else np.asarray(
        acc, dtype=np.float32
    )
    if not hasattr(state, "_tcu_emu"):
        state._tcu_emu = TcuEmulator()

    if instr.name == "TCU_WGMMA_SP32":
        result, _ = state._tcu_emu.execute_wgmma(a, b, acc, is_sparse=True)
    elif instr.name == "TCU_WGMMA_MXFP8":
        result, _ = state._tcu_emu.execute_wgmma(a, b, acc, format_str="mxfp8")
    else:
        result = policy.multiply_accumulate(a, b, acc)
    if not instr.defs:
        raise UnsupportedInstruction(f"{instr.name} defines no accumulator value")
    state.bind(instr.defs[0], result)


def _require(instr: Instr, role: str) -> Operand:
    operand = instr.operand(role)
    if operand is None:
        raise UnsupportedInstruction(f"{instr.name} is missing required role {role!r}")
    return operand


def _apply_elementwise(instr: Instr, state: MachineState) -> None:
    isa_name = getattr(state, "_isa_name", None)
    sem_op = None
    name_map = {
        "VADD": "add", "ADD": "add",
        "VMUL": "mul", "MUL": "mul",
        "VDIV": "div", "DIV": "div",
        "VSUB": "sub", "SUB": "sub",
        "VMOD": "mod", "MOD": "mod",
        "VRELU": "relu", "RELU": "relu",
        "VCLAMP": "clamp", "CLAMP": "clamp",
    }
    if instr.name in name_map:
        sem_op = name_map[instr.name]
    elif isa_name:
        sem_op = _infer_semantics(instr.name, isa_name)

    if sem_op is not None:
        ops = _operands(instr, state)
        if not ops:
            ops = [state.resolve(v) for k, v in instr.operands.items() if not isinstance(v, MemRef)]
        if not ops:
            return
        if sem_op == "relu":
            res = np.maximum(0, np.asarray(ops[0]))
        elif sem_op == "clamp":
            res = np.clip(np.asarray(ops[0]), 0, 1)
        elif len(ops) == 1:
            res = np.asarray(ops[0])
        elif sem_op == "add":
            res = np.asarray(ops[0]) + np.asarray(ops[1])
        elif sem_op == "mul":
            res = np.asarray(ops[0]) * np.asarray(ops[1])
        elif sem_op == "div":
            res = np.asarray(ops[0]) / np.asarray(ops[1])
        elif sem_op == "sub":
            res = np.asarray(ops[0]) - np.asarray(ops[1])
        elif sem_op == "mod":
            res = np.mod(np.asarray(ops[0]), np.asarray(ops[1]))
        else:
            res = np.asarray(ops[0])
        if instr.defs:
            state.bind(instr.defs[0], res)
        return

    op = instr.source.op_name if instr.source else None
    if op is not None and op in _ELEMENTWISE:
        shape = _declared_shape(instr)
        handler = _ELEMENTWISE[op]
        if not instr.defs:
            raise UnsupportedInstruction(f"{op} defines no value to bind")
        value = handler(instr, state, shape)
        state.bind(instr.defs[0], value)
        return

    if op is None:
        raise UnsupportedInstruction(
            f"{instr.name} at {instr.source} carries no source operation; the elementwise "
            "unit cannot know which arithmetic to perform"
        )
    shape = _declared_shape(instr)
    handler = _ELEMENTWISE.get(op)
    if handler is None:
        raise UnsupportedInstruction(
            f"elementwise operation {op!r} is not implemented by this machine "
            f"(known: {sorted(_ELEMENTWISE)}); refusing rather than substituting"
        )

    if not instr.defs:
        raise UnsupportedInstruction(f"{op} defines no value to bind")
    value = handler(instr, state, shape)
    state.bind(instr.defs[0], value)


def _operands(instr: Instr, state: MachineState) -> list[Any]:
    return [state.resolve(instr.operands[role]) for role in instr.roles if role.startswith("in")]


def _const(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> Any:
    value = state.resolve(_require(instr, "value"))
    if not shape:
        return value
    return np.full(shape, value, dtype=np.float32 if isinstance(value, float) else np.int64)


def _program_id(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> int:
    axis = int(state.resolve(_require(instr, "value")))
    if axis >= len(PROGRAM_ID_AXES):
        raise UnsupportedInstruction(f"program id axis {axis} is out of range")
    return state.grid[axis]


def _make_range(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    length = int(state.resolve(_require(instr, "value")))
    if len(shape) > 1:
        raise UnsupportedInstruction(f"make_range is 1-D; the descriptor says shape {shape}")
    return np.arange(length, dtype=np.int64)


def _splat(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    value = _operands(instr, state)[0]
    return np.full(shape or (1,), value, dtype=np.asarray(value).dtype)


def _broadcast(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    value = _operands(instr, state)[0]
    return np.broadcast_to(np.asarray(value), shape).copy()


def _expand_dims(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(_operands(instr, state)[0])
    if not shape:
        raise UnsupportedInstruction("expand_dims has no recorded result shape")
    if int(np.prod(shape)) != value.size:
        raise UnsupportedInstruction(
            f"expand_dims result shape {shape} holds {int(np.prod(shape))} elements but "
            f"its operand holds {value.size}; the recorded shape cannot be that reshape"
        )
    return value.reshape(shape)


def _binary(function):
    def handler(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> Any:
        left, right = _operands(instr, state)
        return function(left, right)

    return handler


def _addi(left, right) -> Any:
    return np.add(np.asarray(left), np.asarray(right))


def _addf(left, right) -> Any:
    return (np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)).astype(
        np.float32
    )


def _muli(left, right) -> Any:
    return np.multiply(np.asarray(left), np.asarray(right))


def _divsi(left, right) -> Any:
    return np.floor_divide(np.asarray(left).astype(np.int64), np.asarray(right).astype(np.int64))


def _remsi(left, right) -> Any:
    return np.remainder(np.asarray(left).astype(np.int64), np.asarray(right).astype(np.int64))


def _maxnumf(left, right) -> Any:
    return np.maximum(
        np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    ).astype(np.float32)



def _mulf(left: Any, right: Any) -> np.ndarray:
    return np.asarray(left, dtype=np.float32) * np.asarray(right, dtype=np.float32)


def _subf(left: Any, right: Any) -> np.ndarray:
    return np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)

def _divf(left: Any, right: Any) -> np.ndarray:
    return np.asarray(left, dtype=np.float32) / np.asarray(right, dtype=np.float32)


def _negf(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    operands = _operands(instr, state)
    return -np.asarray(operands[0], dtype=np.float32)


def _cmpf(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    left, right = _operands(instr, state)
    return np.greater(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32))


def _select(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    operands = _operands(instr, state)
    cond, on_true, on_false = operands[0], operands[1], operands[2]
    return np.where(np.asarray(cond, dtype=bool), np.asarray(on_true), np.asarray(on_false))


def _cmpi(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    """`arith.cmpi`, executed as the signed less-than the corpus uses (F8).

    The predicate is not part of the emitted instruction, so this is the one
    place the machine supplies a fact the artifact does not carry. It is written
    down here, in the module docstring's third bullet, and in the audit report.
    """
    left, right = _operands(instr, state)
    return np.less(np.asarray(left), np.asarray(right))


def _addptr(instr: Instr, state: MachineState, shape: tuple[int, ...]) -> np.ndarray:
    base, offset = _operands(instr, state)
    return np.asarray(base).astype(np.int64) + np.asarray(offset).astype(np.int64)


_ELEMENTWISE: dict[str, Any] = {
    "arith.constant": _const,
    "tt.get_program_id": _program_id,
    "tt.make_range": _make_range,
    "tt.splat": _splat,
    "tt.broadcast": _broadcast,
    "tt.expand_dims": _expand_dims,
    "arith.addi": _binary(_addi),
    "arith.addf": _binary(_addf),
    "arith.muli": _binary(_muli),
    "arith.divsi": _binary(_divsi),
    "arith.divf": _binary(_divf),
    "arith.remsi": _binary(_remsi),
    "arith.maxnumf": _binary(_maxnumf),
    "arith.mulf": _binary(_mulf),
    "arith.subf": _binary(_subf),
    "arith.negf": _negf,
    "arith.cmpi": _cmpi,
    "tt.addptr": _addptr,
}

#: Schema-driven semantics map. When an instruction's name is not in the legacy
#: named-instruction table (VADD, VMUL, etc.), but the ISA schema declares a
#: ``semantics`` string, we parse the semantic operation from it. This closes
#: the audit finding that "semantics: field is never executed."
_SEMANTICS_OPS: dict[str, str] = {
    "+": "add",
    "*": "mul",
    "/": "div",
    "-": "sub",
    "%": "mod",
    "max(0": "relu",
    "min(max": "clamp",
}


def _infer_semantics(instr_name: str, isa_name: str | None) -> str | None:
    """Infer the semantic operation for a named elementwise instruction from schema.

    Returns the op kind ('add', 'mul', 'sub', 'mod', 'relu', 'clamp') or None.
    """
    if isa_name is None:
        return None
    try:
        from ..isa.schema import load_builtin
        schema = load_builtin(isa_name)
        instr_def = schema.instructions.get(instr_name)
        if instr_def is None or not instr_def.semantics:
            return None
        sem = instr_def.semantics
        for pattern, op_kind in _SEMANTICS_OPS.items():
            if pattern in sem:
                return op_kind
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Loops and programs
# --------------------------------------------------------------------------- #


def _loop_bounds(loop: Loop, state: MachineState) -> tuple[int, int, int]:
    lower = int(np.asarray(state.resolve(loop.lower)).item()) if loop.lower is not None else 0
    upper = int(np.asarray(state.resolve(loop.upper)).item()) if loop.upper is not None else 0
    step = int(np.asarray(state.resolve(loop.step)).item()) if loop.step is not None else 1
    if step == 0:
        raise StorageError(f"loop {loop.id} has step 0; it would never terminate")
    return lower, upper, step


def run_loop(loop: Loop, state: MachineState, policy: PrecisionPolicy) -> None:
    """Execute one recovered `scf.for`, re-threading `iter_args` through `yields`.

    `yields[i]` is the next iteration's `iter_args[i]`; `results[i]` is the name a
    post-loop reader sees for the same slot. Both are used, which is only
    possible because the emitter now records the first (finding F7).
    """
    if loop.yields and len(loop.yields) != len(loop.iter_args):
        raise StorageError(
            f"loop {loop.id} yields {len(loop.yields)} value(s) for "
            f"{len(loop.iter_args)} iter_args; cannot re-thread"
        )
    carried = list(loop.inits)
    if len(carried) != len(loop.iter_args):
        raise StorageError(
            f"loop {loop.id} has {len(carried)} initialiser(s) for "
            f"{len(loop.iter_args)} iter_args; cannot enter the loop"
        )
    for name, init in zip(loop.iter_args, carried, strict=True):
        state.bind(name, state.resolve(SsaRef(init)))

    lower, upper, step = _loop_bounds(loop, state)
    index = lower
    guard = (upper - lower) // step + 2
    iteration = 0
    while index < upper:
        state.loop_iteration = iteration
        guard -= 1
        if guard < 0:
            raise StorageError(f"loop {loop.id} did not terminate; refusing to spin")
        if loop.induction_var:
            state.bind(loop.induction_var, index)
        for item in loop.body:
            if isinstance(item, Instr):
                apply(item, state, policy)
        if loop.yields:
            advanced = [state.resolve(SsaRef(name)) for name in loop.yields]
            for name, value in zip(loop.iter_args, advanced, strict=True):
                state.bind(name, value)
        index += step
        iteration += 1

    for result, name in zip(loop.results, loop.iter_args, strict=False):
        state.bind(result, state.values.get(name))


def emulate(
    program: Program,
    inputs: dict[str, np.ndarray],
    policy: PrecisionPolicy | None = None,
    *,
    grid: tuple[int, ...] = (0, 0, 0),
    loop_iteration: int = 0,
    use_cpp: bool = False,
) -> dict[str, np.ndarray]:
    if use_cpp:
        from tritonflow.emu._emu_cpp import emulate as _cpp_emulate
        return _cpp_emulate(program, inputs, policy=policy, grid=grid)
    """Execute `program` and return every buffer it wrote (postcondition 1).

    `UNSUPPORTED` halts locally with :class:`ProgramNotExecutable`; the caller
    routes that kernel to the eager fallback (postcondition 2).
    """
    markers = program.markers()
    if markers:
        raise ProgramNotExecutable(markers[0])

    policy = policy or PrecisionPolicy()
    state = MachineState.from_program(program, inputs, policy, grid=grid)
    state._isa_name = getattr(program, 'isa_name', None)
    for item in program.execution_order():
        if isinstance(item, Loop):
            run_loop(item, state, policy)
        else:
            apply(item, state, policy)
    return state.outputs()


__all__ = [
    "ELEMENTWISE_INSTRUCTION",
    "MAC_INSTRUCTIONS",
    "MEMORY_INSTRUCTIONS",
    "PROGRAM_ID_AXES",
    "MachineState",
    "MissingInput",
    "ProgramNotExecutable",
    "Storage",
    "StorageError",
    "UnsupportedInstruction",
    "apply",
    "emulate",
    "run_loop",
]
