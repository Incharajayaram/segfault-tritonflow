"""Cost-model checks against independently computed values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tritonflow.isa import cost


class _Machine:
    def __init__(self, **values: int) -> None:
        self._values = values

    def get(self, name: str) -> int:
        return self._values[name]


MACHINE = _Machine(cache_line_bytes=64, lmem_banks=16)


def test_a_descriptor_that_says_nothing_about_gather_is_a_dense_access() -> None:
    """64 contiguous f32 words are four 64-byte lines, not 64 one-word transactions."""
    dense = SimpleNamespace(sizes=(64,), strides=(1,), dtype="f32")
    transactions, efficiency = cost._coalesce(dense, {}, MACHINE)
    assert transactions == 4.0
    assert efficiency == 1.0


def test_gather_costs_one_transaction_per_element_and_reports_its_efficiency() -> None:
    gather = SimpleNamespace(sizes=(64,), strides=(1,), dtype="f32", is_gather_scatter=True)
    transactions, efficiency = cost._coalesce(gather, {"words": 64}, MACHINE)
    assert transactions == 64.0
    assert efficiency == pytest.approx(64 * 4 / (64 * 64))


def _expected_conflicts(lanes: int, stride_words: int, banks: int) -> int:
    """Independent model: a bank serving n distinct words costs n - 1 extra cycles."""
    words_by_bank: dict[int, set[int]] = {}
    for lane in range(lanes):
        word = lane * stride_words
        words_by_bank.setdefault(word % banks, set()).add(word)
    return sum(len(words) - 1 for words in words_by_bank.values())


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 8, 16])
def test_bank_conflicts_match_an_independent_model(stride: int) -> None:
    descriptor = SimpleNamespace(sizes=(16,), strides=(stride,), dtype="f32")
    assert cost._bank_conflicts(descriptor, {}, MACHINE) == _expected_conflicts(16, stride, 16)


@pytest.mark.parametrize("record", ["MachineParam", "CostQuery", "CostResult"])
def test_cost_records_are_immutable(record: str) -> None:
    assert getattr(cost, record).__dataclass_params__.frozen is True
