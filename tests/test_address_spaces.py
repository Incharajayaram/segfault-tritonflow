"""Address spaces: a kernel output must be written to `global`, a kernel input read from it.

Before this, the selector picked the cheapest instruction of the "memory" rule for an
output store. On `tritonflow2` and `vortex_rvgpu` that was `LDS2D` (a global-to-scratch
*load*) and `STS` (a scratchpad store), so a lowered kernel never wrote its result to
global memory, and shadow verification still passed because the emulator ignored
address spaces. These tests pin the schema declaration, the selection, and the
enforcement in both emulators, and each enforcement test has a mutation twin that
proves the check is the thing doing the refusing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritonflow.emit.ir import Instr, MemRef, Program, SourceRef, SsaRef
from tritonflow.emu import exec as emu_exec
from tritonflow.emu.exec import AddressSpaceViolation, emulate, transfer_table
from tritonflow.isa.schema import load_builtin
from tritonflow.pipeline import compile_fixture

ISAS = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
MEMORY_RULES = ("memory", "async_copy")


def _walk(program: Program) -> list[Instr]:
    return list(program.instrs) + list(program.epilogue) + [i for lp in program.loops for i in lp.body]


@pytest.mark.parametrize("isa", ISAS)
def test_every_memory_instruction_declares_its_transfers(isa: str) -> None:
    schema = load_builtin(isa)
    for instr in schema.instructions.values():
        if instr.rule in MEMORY_RULES:
            assert instr.transfers is not None, f"{isa}.{instr.name} declares no transfers"
        else:
            assert instr.transfers is None, f"{isa}.{instr.name} is not a memory instruction"


def test_transfers_match_each_instructions_own_semantics_string() -> None:
    """The declaration is derived from the semantics text, so the two cannot drift apart."""
    for isa in ISAS:
        for instr in load_builtin(isa).instructions.values():
            if instr.transfers is None or not instr.transfers:
                continue
            text = instr.semantics
            for src, dst in instr.transfers:
                # a "global>register" load appears as `dst = global[...]`; a "register>global" store
                # as `global[...] = src`; tritonflow1's flat DMA is spelled generically and is exempt.
                if isa == "tritonflow1":
                    continue
                if "async_gmem" in text or "replicate_lmem" in text or "tile2d" in text or "transposed" in text:
                    continue
                if src == "global" and dst != "global":
                    assert "global[" in text and text.index("global[") > text.index("="), (isa, instr.name, text)
                if dst == "global" and src != "global":
                    assert text.startswith("global["), (isa, instr.name, text)
                if src == "scratch" and dst == "register":
                    assert "scratch[" in text and text.index("scratch[") > text.index("="), (isa, instr.name, text)
                if dst == "scratch" and src == "register":
                    assert text.startswith("scratch["), (isa, instr.name, text)


@pytest.mark.parametrize("isa", ISAS)
def test_only_global_side_instructions_serve_kernel_loads_and_stores(isa: str) -> None:
    schema = load_builtin(isa)
    for instr in schema.instructions.values():
        if instr.rule != "memory":
            continue
        assert instr.serves("store") == instr.writes_to("global"), (isa, instr.name)
        assert instr.serves("load") == instr.reads_from("global"), (isa, instr.name)
    assert not schema.instructions.get("STS", None) or not schema.instructions["STS"].serves("store")
    assert not schema.instructions.get("LDS2D", None) or not schema.instructions["LDS2D"].serves("store")


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("tier", ["t0_vecadd", "t1_matmul", "t2_matmul_relu"])
def test_lowered_kernels_write_their_output_through_a_global_store(isa: str, tier: str) -> None:
    result = compile_fixture(tier, isa)
    if not result.fully_lowered:
        pytest.skip(f"{isa}/{tier} is refused: {result.unsupported[:1]}")
    schema = load_builtin(isa)
    stores = [i for i in _walk(result.program) if isinstance(i.operands.get("dst"), MemRef)]
    loads = [i for i in _walk(result.program) if isinstance(i.operands.get("src"), MemRef)]
    assert stores and loads
    for instr in stores:
        assert instr.operands["dst"].space == "global"
        assert schema.instructions[instr.name].writes_to("global"), (isa, instr.name)
    for instr in loads:
        assert instr.operands["src"].space == "global"
        assert schema.instructions[instr.name].reads_from("global"), (isa, instr.name)


def test_the_schemas_that_used_to_store_through_loads_now_pick_global_stores() -> None:
    for isa, forbidden in (("tritonflow2", {"LDG", "LDS2D"}), ("vortex_rvgpu", {"STS", "LDS"})):
        result = compile_fixture("t0_vecadd", isa)
        stores = [i.name for i in _walk(result.program) if isinstance(i.operands.get("dst"), MemRef)]
        assert stores and not (set(stores) & forbidden), (isa, stores)


def test_a_schema_with_no_global_store_refuses_instead_of_falling_back() -> None:
    """Drop every global-writing instruction from tritonflow2: the store must be refused, by name."""
    from dataclasses import replace

    from tritonflow.isa.select import select
    from tritonflow.recognize.descriptor import AccessDescriptor

    schema = load_builtin("tritonflow2")
    stripped = replace(
        schema,
        instructions={n: i for n, i in schema.instructions.items() if not i.writes_to("global")},
    )
    desc = AccessDescriptor(
        base="%out", sizes=(64,), strides=(1,), offsets=(0,), shape=(64,), order=(0,),
        dtype="f32", loop_carried=False, increment=None,
    )
    report = select(stripped, "memory", desc, None, {"words": 64}, "store")
    assert report.no_admissible_lowering
    assert all(not c.admissible for c in report.candidates)
    assert any("space mismatch" in (c.rejected_by or "") for c in report.candidates)


# --------------------------------------------------------------------------- #
# Enforcement in both emulators
# --------------------------------------------------------------------------- #


def _copy_program(isa: str, load_name: str, store_name: str, load_space: str = "global",
                  store_space: str = "global") -> tuple[Program, dict]:
    program = Program(
        isa_name=isa,
        schema_version=1,
        inputs=("A", "Out"),
        instrs=(
            Instr(name=load_name, operands={"src": MemRef(load_space, "A")}, defs=("V",),
                  source_ops=(SourceRef("tt.load"),)),
            Instr(name=store_name, operands={"dst": MemRef(store_space, "Out"), "value": SsaRef("V")},
                  source_ops=(SourceRef("tt.store"),)),
        ),
    )
    data = {"A": np.arange(4, dtype=np.float32), "Out": np.zeros(4, dtype=np.float32)}
    return program, data


def _run(program: Program, data: dict, use_cpp: bool) -> np.ndarray:
    return emulate(program, {k: v.copy() for k, v in data.items()}, use_cpp=use_cpp)["Out"]


@pytest.mark.parametrize("use_cpp", [False, True])
@pytest.mark.parametrize("isa,load,store", [
    ("tritonflow1", "DMA1D", "DMA1D"),
    ("tritonflow2", "LDG", "STG"),
    ("vortex_rvgpu", "LDG", "STG"),
])
def test_legal_global_moves_execute(isa: str, load: str, store: str, use_cpp: bool) -> None:
    program, data = _copy_program(isa, load, store)
    np.testing.assert_array_equal(_run(program, data, use_cpp), data["A"])


VIOLATIONS = [
    # (isa, load, store, why)
    ("tritonflow2", "LDG", "LDS2D", "a global-to-scratch load used to write the output"),
    ("vortex_rvgpu", "LDG", "STS", "a scratchpad store used to write the output"),
    ("vortex_rvgpu", "LDS", "STG", "a scratchpad read used to load a kernel input"),
    ("tritonflow2", "STG", "STG", "a scratch-to-global store used to load a kernel input"),
]


@pytest.mark.parametrize("use_cpp", [False, True])
@pytest.mark.parametrize("isa,load,store,why", VIOLATIONS)
def test_wrong_space_access_is_refused(isa: str, load: str, store: str, why: str, use_cpp: bool) -> None:
    program, data = _copy_program(isa, load, store)
    with pytest.raises(Exception, match="address space violation") as info:
        _run(program, data, use_cpp)
    assert "global" in str(info.value), why


def test_unknown_schema_refuses_every_memory_access() -> None:
    """No schema means no table, and no table means the access cannot be verified."""
    program, data = _copy_program("no_such_isa", "DMA1D", "DMA1D")
    for use_cpp in (False, True):
        with pytest.raises(Exception, match="address space violation"):
            _run(program, data, use_cpp)


def test_python_violation_is_the_typed_error() -> None:
    program, data = _copy_program("tritonflow2", "LDG", "LDS2D")
    with pytest.raises(AddressSpaceViolation):
        _run(program, data, False)


# --------------------------------------------------------------------------- #
# Mutation twins: break the enforcement, watch the violation stop being reported
# --------------------------------------------------------------------------- #


def test_mutation_python_enforcement_is_what_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    program, data = _copy_program("tritonflow2", "LDG", "LDS2D")
    with pytest.raises(AddressSpaceViolation):
        _run(program, data, False)
    monkeypatch.setattr(emu_exec, "_check_space", lambda *a, **k: None)
    np.testing.assert_array_equal(_run(program, data, False), data["A"])


def test_mutation_cpp_enforcement_is_what_refuses() -> None:
    """Hand the C++ emulator a table that wrongly lets LDS2D write global: the violation vanishes."""
    from tritonflow.emu._emu_cpp import emulate as cpp_emulate

    program, data = _copy_program("tritonflow2", "LDG", "LDS2D")
    honest = {n: [f"{s}>{d}" for s, d in tr] for n, tr in transfer_table("tritonflow2").items()}
    with pytest.raises(Exception, match="address space violation"):
        cpp_emulate(program, {k: v.copy() for k, v in data.items()}, transfers=honest)
    mutated = dict(honest)
    mutated["LDS2D"] = ["global>scratch", "scratch>global"]
    out = cpp_emulate(program, {k: v.copy() for k, v in data.items()}, transfers=mutated)
    np.testing.assert_array_equal(out["Out"], data["A"])

