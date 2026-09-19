"""Address arithmetic that a memory descriptor already encodes must not become instructions.

A structured load carries its base, strides, offsets and loop increment. The pointer
arithmetic that produced its address, including a pointer advanced across loop
iterations, is therefore redundant. Emitting it as ALU instructions is what made a
matmul 50 instructions long with one tensor-core operation, and (worse) let an
adder "execute" a broadcast. These tests pin the elision and, importantly, check
the *numbers* of the shrunken program against an independent reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tritonflow.emit.assemble import assemble
from tritonflow.emu.exec import emulate
from tritonflow.emu.precision import PrecisionPolicy
from tritonflow.idioms.detect import annotate, find_subsumed_address_ops, find_subsumed_loop_slots
from tritonflow.isa.schema import load_builtin
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parent.parent
LAUNCH = json.loads((ROOT / "fixtures" / "launch_env.json").read_text())


def _module(tier: str):
    parsed = parse_module((ROOT / "fixtures" / f"{tier}.ttir").read_text(), source_path=tier)
    assert parsed.ok, parsed.diagnostic
    return parsed.module


def _program(tier: str, isa: str):
    module = _module(tier)
    graph = build_def_use(module)
    env = {k: v for k, v in LAUNCH[tier].items() if isinstance(v, int)}
    return assemble(module, graph, annotate(module, graph), load_builtin(isa), env=env)


@pytest.mark.parametrize("tier", ["t1_matmul", "t2_matmul_relu"])
def test_both_loop_carried_pointers_are_eliminated(tier: str) -> None:
    module = _module(tier)
    graph = build_def_use(module)
    slots = find_subsumed_loop_slots(module, graph)
    assert sorted(j for _, j in slots) == [0, 1], "the a and b pointer slots, not the accumulator"


@pytest.mark.parametrize("tier", ["t1_matmul", "t2_matmul_relu"])
@pytest.mark.parametrize("isa", ["tritonflow1", "tritonflow2"])
def test_emitted_loop_carries_only_the_accumulator(tier: str, isa: str) -> None:
    (loop,) = _program(tier, isa).loops
    assert len(loop.iter_args) == 1 and loop.iter_args[0].startswith("%acc")
    assert len(loop.inits) == len(loop.results) == len(loop.yields) == 1


@pytest.mark.parametrize("isa", ["tritonflow1", "tritonflow2"])
def test_matmul_is_a_handful_of_instructions_not_fifty(isa: str) -> None:
    program = _program("t1_matmul", isa)
    total = len(program.instrs) + len(program.epilogue) + sum(len(loop.body) for loop in program.loops)
    assert total <= 12, total


def test_unstructured_access_subsumes_nothing() -> None:
    """The negative control: a modulo-wrapped address is not a descriptor, so it stays live."""
    module = _module("t3_modulo")
    assert find_subsumed_address_ops(module, build_def_use(module)) == set()


@pytest.mark.parametrize("isa", ["tritonflow1", "tritonflow2"])
def test_shrunken_matmul_still_computes_the_right_numbers(isa: str) -> None:
    """Compose assemble and emulate and compare with an fp64 reference.

    This is the check that never existed: elision changed the program 5x and no other
    test noticed, because nothing compared its output with an independent answer.
    """
    m, n, k = 128, 128, 64
    rng = np.random.default_rng(3)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    reference = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    inputs = {
        "%a_ptr": a.copy(), "%b_ptr": b.copy(), "%c_ptr": np.zeros((m, n), np.float32),
        "%M": np.int32(m), "%N": np.int32(n), "%K": np.int32(k),
        "%sam": np.int32(64), "%sak": np.int32(1), "%sbk": np.int32(128), "%sbn": np.int32(1),
        "%scm": np.int32(128), "%scn": np.int32(1),
    }
    out = emulate(_program("t1_matmul", isa), inputs, policy=PrecisionPolicy(input_precision="tf32"))["%c_ptr"]
    tile = (slice(0, 64), slice(0, 64))  # the fixture computes one 64x64 block
    scale = float(np.max(np.abs(reference[tile])))
    # tf32 keeps 10 mantissa bits: relative operand error 2**-11, accumulated over k.
    bound = k * (2.0 * 2.0**-11 + 2.0**-24)
    assert float(np.max(np.abs(out[tile] - reference[tile]))) / scale <= bound


_DUAL_USE = """module {
  tt.func public @k(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<i32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant dense<64> : tensor<64xi32>
    %zero = arith.constant dense<0> : tensor<64xi32>
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %res:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%off = %r, %acc = %zero) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
      %p = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %a = tt.addptr %p, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %v = tt.load %a : tensor<64x!tt.ptr<f32>>
      %acc2 = arith.addi %acc, %off : tensor<64xi32>
      %off2 = arith.addi %off, %c64 : tensor<64xi32>
      scf.yield %off2, %acc2 : tensor<64xi32>, tensor<64xi32>
    }
    %q = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %b = tt.addptr %q, %r : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    tt.store %b, %res#1 : tensor<64x!tt.ptr<i32>>
    tt.return
  }
}
"""


def test_a_carried_offset_that_is_also_data_is_not_subsumed() -> None:
    """`%off` addresses a load *and* feeds an integer accumulator.

    Eliding its chain would delete a value a live operation reads, so the slot has
    to stay in the loop. This is the case a descriptor cannot replace.
    """
    parsed = parse_module(_DUAL_USE)
    assert parsed.ok, parsed.diagnostic
    graph = build_def_use(parsed.module)
    assert find_subsumed_loop_slots(parsed.module, graph) == set()
