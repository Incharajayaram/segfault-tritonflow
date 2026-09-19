"""Cost must depend on the shape of the access, through the real compiler.

Cost drives instruction selection (`isa/select.py` takes the minimum), so a cost that
ignores stride is a wrong instruction choice, not a cosmetic statistic. These tests
assemble real TTIR rather than calling the evaluator on a stub, so they fail if the
schemas stop using the transaction terms even when the evaluator itself is fine.
"""

from __future__ import annotations

import pytest

from tritonflow.emit.assemble import assemble
from tritonflow.idioms.detect import annotate
from tritonflow.isa.cost import (
    CostQuery,
    CostUnknown,
    MachineParams,
    evaluate_cost,
    load_machine_by_name,
)
from tritonflow.isa.schema import load_builtin
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module

LOAD_INSTRUCTIONS = {"vortex_rvgpu": "LDG", "tritonflow2": "LDG"}


def _strided_copy(stride: int) -> str:
    """One 64-element load through `stride`, stored contiguously."""
    index = "%o = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>\n"
    load_offsets = "%o"
    if stride != 1:
        index += (
            f"    %c = arith.constant dense<{stride}> : tensor<64xi32>\n"
            "    %o2 = arith.muli %o, %c : tensor<64xi32>\n"
        )
        load_offsets = "%o2"
    return f"""module {{
  tt.func public @k(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {{noinline = false}} {{
    {index}    %p = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a = tt.addptr %p, {load_offsets} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %a : tensor<64x!tt.ptr<f32>>
    %q = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b = tt.addptr %q, %o : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %b, %v : tensor<64x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def _load_cost(isa: str, stride: int) -> float:
    parsed = parse_module(_strided_copy(stride))
    assert parsed.ok, parsed.diagnostic
    module = parsed.module
    graph = build_def_use(module)
    program = assemble(module, graph, annotate(module, graph), load_builtin(isa), env={})
    assert not program.markers(), [str(m) for m in program.markers()]
    loads = [i for i in program.instrs if i.name == LOAD_INSTRUCTIONS[isa]]
    assert loads, f"no {LOAD_INSTRUCTIONS[isa]} emitted"
    return loads[0].cost


@pytest.mark.parametrize("isa", sorted(LOAD_INSTRUCTIONS))
def test_stride_17_costs_strictly_more_than_stride_1(isa: str) -> None:
    """The acceptance criterion: equal length, only the stride differs."""
    assert _load_cost(isa, 17) > _load_cost(isa, 1)


@pytest.mark.parametrize("isa", sorted(LOAD_INSTRUCTIONS))
def test_cost_never_decreases_as_stride_grows(isa: str) -> None:
    costs = [_load_cost(isa, s) for s in (1, 2, 4, 8, 16, 17, 32)]
    assert costs == sorted(costs), costs
    assert costs[0] < costs[-1]


def test_contiguous_cost_is_the_per_word_rate() -> None:
    """Contiguous access keeps the original 0.50/word price, so nothing else moves."""
    assert _load_cost("vortex_rvgpu", 1) == pytest.approx(0.50 * 64)


class _Access:
    def __init__(self, sizes, strides, dtype="f32", gather=False):
        self.sizes, self.strides, self.dtype = tuple(sizes), tuple(strides), dtype
        self.offsets, self.shape, self.base_num, self.base = (0,) * len(sizes), (0,) * len(sizes), 0, "%p"
        self.is_gather_scatter = gather


def _transactions(access, machine="vortex_rvgpu") -> float:
    m = load_machine_by_name(machine)
    ldg = load_builtin(machine).instruction("LDG")
    return evaluate_cost(CostQuery(instruction=ldg, access=access, machine=m), m).resources.transactions


def test_2d_tile_is_priced_by_its_contiguous_inner_row() -> None:
    """A (64, 32) tile with strides (64, 1) streams contiguously; it is not 'strided'.

    Regression: the first version took the first stride > 1 (64) as the tile's
    stride and priced this matmul-shaped access as if it were a scatter.
    """
    tile = _transactions(_Access((64, 32), (64, 1)))
    flat = _transactions(_Access((64 * 32,), (1,)))
    assert tile >= flat  # a tile touches at least as many lines as the flat copy
    assert tile == 64 * 2  # 64 rows x (32 floats = 128 B = 2 lines)
    assert tile < 64 * 32  # and is nowhere near one transaction per element


def test_gather_is_one_transaction_per_element() -> None:
    assert _transactions(_Access((32,), (0,), gather=True)) == 32


def test_element_width_changes_the_line_count() -> None:
    assert _transactions(_Access((64,), (1,), dtype="f16")) < _transactions(_Access((64,), (1,), dtype="f32"))


def test_missing_machine_parameter_refuses_instead_of_defaulting() -> None:
    """A target with no declared line size is unpriceable; it must not fall back to a guess."""
    empty = MachineParams(machine="x", fit_id="none", source_kind="declared", schema_version=1, params={})
    ldg = load_builtin("vortex_rvgpu").instruction("LDG")
    with pytest.raises(CostUnknown):
        evaluate_cost(CostQuery(instruction=ldg, access=_Access((64,), (1,)), machine=empty), empty)
