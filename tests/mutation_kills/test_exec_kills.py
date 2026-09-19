"""Behavioural checks on emulator address evaluation that mutation testing found unguarded."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tritonflow.emit.ir import MemRef, Program
from tritonflow.emu.exec import MachineState, _eval_descriptor_expr
from tritonflow.emu.precision import PrecisionPolicy


def _expr_state(grid: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(grid=grid, values={})


@pytest.mark.parametrize("name", ["pid_z", "pid_k", "_var_pid_z", "_var_pid_k", "%pid_z", "%pid_k"])
def test_third_grid_axis_is_readable_by_every_alias(name: str) -> None:
    assert _eval_descriptor_expr(name, _expr_state((4, 5, 6))) == 6


@pytest.mark.parametrize("name", ["pid_z", "pid_k", "_var_pid_z", "_var_pid_k"])
def test_missing_third_axis_reads_zero(name: str) -> None:
    assert _eval_descriptor_expr(name, _expr_state((4, 5))) == 0


def test_first_two_axes_and_arithmetic() -> None:
    state = _expr_state((4, 5, 6))
    assert _eval_descriptor_expr("%pid_m*3-2", state) == 10
    assert _eval_descriptor_expr("pid_n+7", state) == 12
    assert _eval_descriptor_expr("7-2*2", state) == 3
    assert _eval_descriptor_expr("6*7", state) == 42


def _machine(loop_iteration: int = 0) -> MachineState:
    program = Program(isa_name="tritonflow1", schema_version=1)
    return MachineState(
        program=program,
        policy=PrecisionPolicy(),
        memory=np.zeros(0, dtype=np.float32),
        loop_iteration=loop_iteration,
    )


def _addresses(base_name: str, key: str, iteration: int = 0) -> np.ndarray:
    return _machine(iteration)._materialize_descriptor(MemRef("global", base_name, key), 0)


def test_loop_increment_follows_the_stride_of_the_dimension_that_equals_the_increment() -> None:
    """sizes[0] is the increment: the loop advances along dim 0, one row of 8 elements per step."""
    key = "sizes=[4, 8];strides=[8, 1];offsets=[0, 0];loop_carried=True;increment=4"
    assert int(_addresses("%x", key, iteration=2)[0, 0]) == 2 * 4 * 8


def test_loop_increment_follows_dim_one_when_that_extent_equals_the_increment() -> None:
    """sizes[1] is the increment: the loop advances along dim 1, unit stride, whatever the buffer is called."""
    key = "sizes=[8, 4];strides=[8, 1];offsets=[0, 0];loop_carried=True;increment=4"
    assert int(_addresses("%b", key, iteration=2)[0, 0]) == 2 * 4 * 1


def test_offsets_shorter_than_the_rank_still_apply_to_the_leading_dimensions() -> None:
    key = "sizes=[2, 2];strides=[2, 1];offsets=[5]"
    expected = np.array([[5, 6], [7, 8]], dtype=np.int64)
    np.testing.assert_array_equal(_addresses("%x", key), expected)
