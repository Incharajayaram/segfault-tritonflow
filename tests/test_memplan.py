"""Scratchpad allocation: sourcing, liveness, both exact solvers, and the checker.

The checker is the spine of this file. Every allocation any solver produces is
re-validated from the problem alone, so a solver that starts returning overlapping
offsets fails here rather than in a downstream emulator run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tritonflow.emit.ir import Imm, Instr, Loop, MemRef, Program, SourceRef, SsaRef
from tritonflow.isa import memplan
from tritonflow.isa.memplan import (
    AllocationProblem,
    CapacityUndeclared,
    MemPlanError,
    ScratchSpace,
    Tile,
    bank_of,
    check_allocation,
    plan,
    scratch_space,
    scratch_tiles,
    solve_cpsat,
    solve_greedy,
    solve_z3,
)
from tritonflow.isa.schema import load_builtin

ROOT = Path(__file__).resolve().parent.parent

EXACT_BACKENDS = [
    pytest.param("cpsat", marks=pytest.mark.skipif(not memplan.have_cpsat(), reason="ortools absent")),
    pytest.param("z3", marks=pytest.mark.skipif(not memplan.have_z3(), reason="z3 absent")),
]


def space(capacity=4096, banks=4, interleave=4, align=8) -> ScratchSpace:
    return ScratchSpace(
        isa="test", name="scratch", kind="banked", capacity_bytes=capacity,
        banks=banks, interleave_bytes=interleave, alignment_bytes=align,
        capacity_source="test fixture",
    )


def tile(name, size, first, last, align=8) -> Tile:
    return Tile(name=name, size_bytes=size, first=first, last=last, alignment_bytes=align)


# --------------------------------------------------------------------------- #
# Sourced parameters
# --------------------------------------------------------------------------- #


def test_vortex_capacity_is_read_from_the_vendored_upstream_config() -> None:
    """16 KB is not pasted here: it is 1 << VX_CFG_LMEM_LOG_SIZE from the snapshot."""
    config = ROOT / "third_party" / "vortex" / "VX_config.toml"
    declared = int(re.search(r"^\s*VX_CFG_LMEM_LOG_SIZE\s*=\s*(\d+)", config.read_text(), re.M).group(1))
    assert memplan._vortex_lmem_capacity_bytes() == 1 << declared

    resolved = scratch_space(load_builtin("vortex_rvgpu"))
    assert resolved.capacity_bytes == 1 << declared
    assert "VX_config.toml" in resolved.capacity_source


def test_vortex_bank_count_matches_upstream_lsu_lanes() -> None:
    """Upstream sets LMEM banks to the LSU lane count, which resolves to 4."""
    assert scratch_space(load_builtin("vortex_rvgpu")).banks == 4


def test_a_missing_config_yields_no_capacity_rather_than_a_default(tmp_path) -> None:
    assert memplan._vortex_lmem_capacity_bytes(tmp_path / "absent.toml") is None


# --------------------------------------------------------------------------- #
# Fail-closed
# --------------------------------------------------------------------------- #


def test_an_isa_with_no_scratchpad_is_refused_by_name() -> None:
    with pytest.raises(CapacityUndeclared, match="no scratchpad memory space is declared"):
        scratch_space(load_builtin("tritonflow1"))


def test_an_unsourceable_capacity_is_refused_and_says_how_to_fix_it() -> None:
    """tritonflow2 declares a scratchpad but no size, and has no upstream to consult."""
    with pytest.raises(CapacityUndeclared) as excinfo:
        scratch_space(load_builtin("tritonflow2"))
    message = str(excinfo.value)
    assert "tritonflow2.scratch" in message
    assert "refuses rather than assume" in message
    assert "MemorySpace" in message


def test_refusing_is_not_a_silent_zero_capacity() -> None:
    """The failure is an exception, never a space that would accept no tiles."""
    with pytest.raises(CapacityUndeclared):
        scratch_space(load_builtin("tritonflow2"))


# --------------------------------------------------------------------------- #
# Semantics parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "semantics,expected",
    [
        ("scratch[0:length] = global[0:length]", "scratch"),
        ("dst[0:length] = global[base:base+length]", "dst"),
        ("global[base:base+length] = src[0:length]", "global"),
        ("scratch[smem_base:smem_base+length] = async_gmem_read(slot, bar)", "scratch"),
        ("vx_bar(warp_group)", None),
        (None, None),
    ],
)
def test_destination_space_is_the_left_hand_side(semantics, expected) -> None:
    assert memplan._destination_space(semantics) == expected


def test_reads_space_sees_the_right_hand_side_only() -> None:
    assert memplan._reads_space("dst[0:n] = scratch[b:b+n]", "scratch") is True
    assert memplan._reads_space("scratch[b:b+n] = src[0:n]", "scratch") is False


# --------------------------------------------------------------------------- #
# Liveness
# --------------------------------------------------------------------------- #


def _descriptor(base, sizes, dtype="f32"):
    from tritonflow.recognize.descriptor import AccessDescriptor

    return AccessDescriptor(
        base=base, sizes=tuple(sizes), strides=tuple([1] * len(sizes)),
        offsets=tuple([0] * len(sizes)), shape=tuple([0] * len(sizes)),
        order=tuple(range(len(sizes))), dtype=dtype, loop_carried=False, increment=None,
    )


class _FakeInstruction:
    def __init__(self, semantics):
        self.semantics = semantics


class _FakeSchema:
    """Only the two attributes `scratch_tiles` reads."""

    name = "fake"

    def __init__(self, mapping):
        self.instructions = {k: _FakeInstruction(v) for k, v in mapping.items()}


def _load(name, base, sizes, loop=None, at_line=1):
    return Instr(
        name="LOADS", defs=(name,), loop=loop,
        operands={"src": MemRef.of("global", base, _descriptor(base, sizes))},
        source_ops=(SourceRef(op_name="tt.load", loc_name=f"#loc{at_line}", line=at_line),),
    )


SCRATCH_SCHEMA = _FakeSchema({
    "LOADS": "scratch[0:length] = global[0:length]",
    "USE": "acc = f(src)",
    "READS": "dst[0:n] = scratch[b:b+n]",
    "STORES": "scratch[b:b+n] = src[0:n]",
})


def test_liveness_finds_tiles_loaded_into_scratch_and_sizes_them_from_the_descriptor() -> None:
    program = Program(
        isa_name="fake", schema_version=1, inputs=("%a", "%b"),
        instrs=(
            _load("%t0", "%a", (64, 32), at_line=1),   # 2048 elems * 4 B
            _load("%t1", "%b", (32, 64), at_line=2),
            Instr(name="USE", defs=("%r",), operands={"x": SsaRef("%t0"), "y": SsaRef("%t1")},
                  source_ops=(SourceRef(op_name="tt.dot", loc_name="#loc3", line=3),)),
        ),
    )
    tiles = scratch_tiles(program, SCRATCH_SCHEMA, space())
    assert [t.name for t in tiles] == ["%t0", "%t1"]
    assert all(t.size_bytes == 64 * 32 * 4 for t in tiles)
    assert (tiles[0].first, tiles[0].last) == (0, 2)
    assert (tiles[1].first, tiles[1].last) == (1, 2)


def test_a_tile_whose_producer_is_not_a_scratch_writer_is_not_resident() -> None:
    schema = _FakeSchema({"LOADS": "dst[0:length] = global[0:length]"})  # writes a register
    program = Program(
        isa_name="fake", schema_version=1, inputs=("%a",),
        instrs=(_load("%t0", "%a", (16,)),),
    )
    assert scratch_tiles(program, schema, space()) == ()


def test_a_store_into_scratch_is_a_tile_even_though_it_defines_nothing() -> None:
    store = Instr(
        name="STORES", defs=(),
        operands={"dst": MemRef.of("scratch", "%s", _descriptor("%s", (128,))), "value": SsaRef("%v")},
        source_ops=(SourceRef(op_name="tt.store", loc_name="#loc1", line=1),),
    )
    read = Instr(
        name="READS", defs=("%r",),
        operands={"src": MemRef.of("scratch", "%s", _descriptor("%s", (128,)))},
        source_ops=(SourceRef(op_name="tt.load", loc_name="#loc2", line=2),),
    )
    program = Program(isa_name="fake", schema_version=1, inputs=("%v",), instrs=(store, read))
    tiles = scratch_tiles(program, SCRATCH_SCHEMA, space())
    assert [t.name for t in tiles] == ["%s"]
    assert tiles[0].size_bytes == 128 * 4
    # The later READS keeps it live: a store followed by a read spans both.
    assert (tiles[0].first, tiles[0].last) == (0, 1)


def test_a_tile_used_in_a_loop_is_live_for_the_whole_loop() -> None:
    """The point of the widening: a back-edge makes a value live on every iteration."""
    body = (
        _load("%t_in", "%a", (32,), loop=7, at_line=2),
        Instr(name="USE", defs=("%acc_next",), loop=7, operands={"x": SsaRef("%t_in")},
              source_ops=(SourceRef(op_name="tt.dot", loc_name="#loc3", line=3),)),
        # Two further instructions that never touch %t_in, so its *textual* range
        # ends at index 1 while the loop span runs to index 3.
        Instr(name="USE", defs=("%z1",), loop=7, operands={"x": SsaRef("%acc_next")},
              source_ops=(SourceRef(op_name="arith.addf", loc_name="#loc4", line=4),)),
        Instr(name="USE", defs=("%z2",), loop=7, operands={"x": SsaRef("%z1")},
              source_ops=(SourceRef(op_name="arith.mulf", loc_name="#loc5", line=5),)),
    )
    loop = Loop(id=7, induction_var="%k", lower=Imm(0), upper=Imm(4), step=Imm(1),
                iter_args=("%acc",), inits=("%c0",), results=("%out",), yields=("%acc_next",),
                body=body, source=SourceRef(op_name="scf.for", loc_name="#loc1", line=1))
    program = Program(isa_name="fake", schema_version=1, inputs=("%a", "%c0"), loops=(loop,))

    tiles = scratch_tiles(program, SCRATCH_SCHEMA, space())
    assert [t.name for t in tiles] == ["%t_in"]
    only = tiles[0]
    assert only.in_loop is True
    # Textually live [0, 1]; widened to the loop's full span because the tile is
    # reloaded on every iteration and so occupies its offset for the whole loop.
    assert (only.first, only.last) == (0, 3)


# --------------------------------------------------------------------------- #
# Checker
# --------------------------------------------------------------------------- #


def test_the_checker_catches_an_overlap_between_simultaneously_live_tiles() -> None:
    problem = AllocationProblem(space(), (tile("a", 64, 0, 5), tile("b", 64, 3, 9)))
    bad = memplan.Allocation(offsets={"a": 0, "b": 32}, solver="hand", feasible=True, peak_bytes=96)
    kinds = {v.kind for v in check_allocation(problem, bad)}
    assert "overlap" in kinds


def test_the_checker_allows_reuse_by_tiles_that_are_never_live_together() -> None:
    problem = AllocationProblem(space(), (tile("a", 64, 0, 2), tile("b", 64, 5, 9)))
    shared = memplan.Allocation(offsets={"a": 0, "b": 0}, solver="hand", feasible=True, peak_bytes=64)
    assert check_allocation(problem, shared) == ()


def test_the_checker_catches_capacity_alignment_and_unplaced_tiles() -> None:
    small = space(capacity=128)
    over = AllocationProblem(small, (tile("a", 100, 0, 1),))
    assert "capacity" in {v.kind for v in check_allocation(
        over, memplan.Allocation(offsets={"a": 64}, solver="hand", feasible=True))}

    misaligned = AllocationProblem(space(align=8), (tile("a", 16, 0, 1),))
    assert "alignment" in {v.kind for v in check_allocation(
        misaligned, memplan.Allocation(offsets={"a": 4}, solver="hand", feasible=True))}

    assert "unplaced" in {v.kind for v in check_allocation(
        misaligned, memplan.Allocation(offsets={}, solver="hand", feasible=True))}


def test_the_checker_catches_a_bank_collision_only_when_bank_spread_is_asked_for() -> None:
    tiles = (tile("a", 16, 0, 9, align=4), tile("b", 16, 0, 9, align=4))
    offsets = memplan.Allocation(offsets={"a": 0, "b": 16}, solver="hand", feasible=True)
    # 0 and 16 are both bank 0 with 4 banks of 4-byte words.
    assert bank_of(0, space()) == bank_of(16, space())
    assert "bank" not in {v.kind for v in check_allocation(AllocationProblem(space(), tiles), offsets)}
    spread = AllocationProblem(space(), tiles, bank_spread=True)
    assert "bank" in {v.kind for v in check_allocation(spread, offsets)}


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #


def _chain(n, size=64, span=2):
    """`n` tiles in a staircase, each overlapping the next `span`."""
    return tuple(tile(f"t{i}", size, i, i + span) for i in range(n))


def test_greedy_always_terminates_and_its_output_checks_out() -> None:
    problem = AllocationProblem(space(capacity=8192), _chain(40))
    result = solve_greedy(problem)
    assert result.feasible
    assert check_allocation(problem, result) == ()


def test_greedy_is_deterministic_across_runs() -> None:
    problem = AllocationProblem(space(capacity=8192), _chain(25))
    assert solve_greedy(problem).offsets == solve_greedy(problem).offsets


def test_greedy_reports_infeasible_rather_than_overflowing_capacity() -> None:
    problem = AllocationProblem(space(capacity=128), _chain(10, size=64, span=99))
    result = solve_greedy(problem)
    assert result.feasible is False
    assert "does not fit" in (result.reason or "")
    assert check_allocation(problem, result) == ()  # nothing to check when infeasible


@pytest.mark.parametrize("backend", EXACT_BACKENDS)
def test_each_exact_solver_produces_an_allocation_the_checker_accepts(backend) -> None:
    problem = AllocationProblem(space(capacity=4096), _chain(12))
    result = memplan.solve_exact(problem, timeout_ms=10_000, backend=backend)
    assert result.feasible, result.reason
    assert check_allocation(problem, result) == ()


@pytest.mark.parametrize("backend", EXACT_BACKENDS)
def test_an_over_capacity_problem_is_reported_infeasible_not_truncated(backend) -> None:
    """Three tiles all live at once, 64 B each, in a 96 B space: provably impossible."""
    tiles = (tile("a", 64, 0, 9, align=4), tile("b", 64, 0, 9, align=4), tile("c", 64, 0, 9, align=4))
    problem = AllocationProblem(space(capacity=96, align=4), tiles)
    result = memplan.solve_exact(problem, timeout_ms=10_000, backend=backend)
    assert result.feasible is False
    assert result.offsets == {}
    assert "no allocation satisfies" in (result.reason or "")


@pytest.mark.parametrize("backend", EXACT_BACKENDS)
def test_exact_never_uses_more_space_than_greedy(backend) -> None:
    """Optimality check: the exact peak is a lower bound on the greedy peak."""
    problem = AllocationProblem(space(capacity=16384), _chain(14, size=48, span=3))
    exact = memplan.solve_exact(problem, timeout_ms=20_000, backend=backend)
    greedy = solve_greedy(problem)
    assert exact.feasible and greedy.feasible
    assert exact.peak_bytes <= greedy.peak_bytes
    assert check_allocation(problem, exact) == ()


@pytest.mark.skipif(not memplan.have_cpsat(), reason="ortools absent")
def test_a_tiny_timeout_returns_rather_than_hanging() -> None:
    """The contract is that the exact solver always returns, even when starved."""
    problem = AllocationProblem(space(capacity=1 << 20), _chain(400, size=1024, span=60))
    result = solve_cpsat(problem, timeout_ms=1)
    assert isinstance(result, memplan.Allocation)
    assert result.solve_seconds < 5.0
    if not result.feasible:
        assert result.timed_out


@pytest.mark.skipif(not memplan.have_cpsat(), reason="ortools absent")
def test_cpsat_and_z3_agree_on_the_optimal_peak_when_both_are_available() -> None:
    if not memplan.have_z3():
        pytest.skip("z3 absent")
    problem = AllocationProblem(space(capacity=8192), _chain(10, size=64, span=2))
    a = solve_cpsat(problem, timeout_ms=30_000)
    b = solve_z3(problem, timeout_ms=30_000)
    assert a.feasible and b.feasible
    assert a.peak_bytes == b.peak_bytes  # two solver families, one optimum
    assert check_allocation(problem, a) == ()
    assert check_allocation(problem, b) == ()


# --------------------------------------------------------------------------- #
# plan()
# --------------------------------------------------------------------------- #


def test_plan_falls_back_to_greedy_and_still_returns_a_checked_allocation() -> None:
    problem = AllocationProblem(space(capacity=8192), _chain(20))
    result = plan(problem, prefer_exact=False)
    assert result.solver == "greedy"
    assert result.feasible
    assert check_allocation(problem, result) == ()


def test_plan_raises_when_a_solver_returns_something_invalid(monkeypatch) -> None:
    """A wrong answer must surface as a bug, never be swapped out quietly."""
    problem = AllocationProblem(space(capacity=4096), (tile("a", 64, 0, 5), tile("b", 64, 0, 5)))
    monkeypatch.setattr(
        memplan, "solve_greedy",
        lambda _p: memplan.Allocation(offsets={"a": 0, "b": 0}, solver="greedy", feasible=True, peak_bytes=64),
    )
    with pytest.raises(MemPlanError, match="invalid allocation"):
        plan(problem, prefer_exact=False)


def test_plan_on_an_empty_problem_is_a_valid_empty_allocation() -> None:
    problem = AllocationProblem(space(), ())
    result = plan(problem, prefer_exact=False)
    assert result.feasible and result.peak_bytes == 0
    assert check_allocation(problem, result) == ()


# --------------------------------------------------------------------------- #
# Real fixtures
# --------------------------------------------------------------------------- #


def test_the_shipped_fixtures_put_nothing_in_vortex_scratch_today() -> None:
    """Records the measured state: LDG/STG are selected, and neither writes scratch.

    This is not an aspiration -- it is what the selector currently does. If a
    later change starts routing tiles through LDS/STS/DXA_COPY, this test fails
    and the report that cites it has to be re-measured.
    """
    from tritonflow.pipeline import compile_fixture

    schema = load_builtin("vortex_rvgpu")
    resolved = scratch_space(schema)
    for tier in ("t0_vecadd", "t1_matmul", "t2_matmul_relu"):
        program = compile_fixture(tier, "vortex_rvgpu").program
        assert scratch_tiles(program, schema, resolved) == (), f"{tier} now uses scratch"
