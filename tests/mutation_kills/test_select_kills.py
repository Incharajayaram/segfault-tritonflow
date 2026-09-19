"""Selection-layer checks that mutation testing found unguarded."""

from __future__ import annotations

import dataclasses

import pytest

from tritonflow.isa.schema import load_builtin
from tritonflow.isa.select import (
    Candidate,
    SelectionReport,
    _matches_op,
    enumerate_candidates,
    select,
)
from tritonflow.recognize.descriptor import AccessDescriptor


def _descriptor(shape: tuple[int, ...]) -> AccessDescriptor:
    strides, running = [], 1
    for extent in reversed(shape):
        strides.append(running)
        running *= extent
    strides.reverse()
    return AccessDescriptor(
        base="%buf", sizes=shape, strides=tuple(strides), offsets=tuple(0 for _ in shape), shape=shape,
        order=tuple(range(len(shape) - 1, -1, -1)), dtype="f32", loop_carried=False, increment=None,
    )


@pytest.mark.parametrize(
    ("entry", "base", "expected"),
    [
        ("addf", "addf", True),
        ("addf", "arith.addf", True),
        ("arith.addf", "addf", True),
        ("arith.addf", "arith.addf", True),
        ("addf", "arith.subf", False),
        ("arith.addf", "subf", False),
        ("add", "arith.addf", False),
        ("addf", "arith.addi", False),
    ],
)
def test_op_matching_is_exact_on_the_op_suffix(entry: str, base: str, expected: bool) -> None:
    assert _matches_op(entry, base) is expected


def test_selection_records_are_immutable() -> None:
    report = select(load_builtin("tritonflow1"), "memory", _descriptor((8, 8)), None, {"words": 64}, "load")
    candidate = report.candidates[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.admissible = not candidate.admissible  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.gap = 1.0  # type: ignore[misc]


def test_a_report_is_not_flagged_unlowerable_unless_the_selector_says_so() -> None:
    assert SelectionReport(chosen=None, chosen_cost=None, candidates=()).no_admissible_lowering is False


def test_direction_mismatch_is_rejected_with_the_declared_direction() -> None:
    schema = load_builtin("vortex_rvgpu")
    candidates = enumerate_candidates(schema, "memory", _descriptor((8, 8)), None, {"words": 64}, direction="store")
    lds = next(c for c in candidates if c.instruction.name == "LDS")
    assert lds.admissible is False
    assert "declared direction" in (lds.rejected_by or "")


def test_an_instruction_without_parseable_semantics_is_never_admissible() -> None:
    real = load_builtin("tritonflow1")
    victim = next(iter(real.of_kind("memory")))

    class _Schema:
        name = real.name

        def of_kind(self, kind: str):  # noqa: ANN202
            return [dataclasses.replace(victim, semantics="this is not a semantics expression")]

    (candidate,) = enumerate_candidates(_Schema(), "memory", _descriptor((8, 8)), None, {"words": 64})
    assert isinstance(candidate, Candidate)
    assert candidate.admissible is False
    assert "unparseable semantics" in (candidate.rejected_by or "")
