"""Pattern-detection checks on the frozen fixtures and on small synthetic modules."""

from __future__ import annotations

from pathlib import Path

from tritonflow.idioms.detect import (
    AnnotationSet,
    detect_epilogue,
    detect_mac,
    find_subsumed_address_ops,
)
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parents[2]


def _fixture(tier: str):
    parsed = parse_module((ROOT / "fixtures" / f"{tier}.ttir").read_text(), source_path=tier)
    assert parsed.ok, parsed.diagnostic
    return parsed.module


def _text(text: str):
    parsed = parse_module(text)
    assert parsed.ok, parsed.diagnostic
    return parsed.module


_LOOP = """    %r{i} = scf.for %i{i} = %c0 to %c1 step %c1 iter_args(%acc{i} = %z) -> (tensor<16x16xf32>)  : i32 {{
      %d{i} = tt.dot %a, %b, %acc{i} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      scf.yield %d{i} : tensor<16x16xf32>
    }}
"""

_HEAD = """module {
  tt.func public @k(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %z = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
"""

_TAIL = """    tt.return
  }
}
"""


def test_matmul_loop_is_one_mac_match_with_the_accumulator_bound() -> None:
    (match,) = detect_mac(_fixture("t1_matmul"))
    assert match.multiplicity == 1
    assert match.bindings["acc"].name.startswith("%acc")


def test_two_matmul_loops_are_counted_as_two() -> None:
    matches = detect_mac(_text(_HEAD + _LOOP.format(i=0) + _LOOP.format(i=1) + _TAIL))
    assert len(matches) == 2
    assert [m.multiplicity for m in matches] == [2, 2]


def test_a_dot_without_an_accumulator_operand_is_not_a_mac() -> None:
    loop = _LOOP.format(i=0).replace("%a, %b, %acc0", "%acc0, %acc0")
    assert detect_mac(_text(_HEAD + loop + _TAIL)) == ()


def test_a_dot_whose_result_is_not_yielded_is_not_a_mac() -> None:
    loop = _LOOP.format(i=0).replace("scf.yield %d0", "scf.yield %acc0")
    assert detect_mac(_text(_HEAD + loop + _TAIL)) == ()


def test_epilogue_chain_after_the_loop_is_found_in_order() -> None:
    (match,) = detect_epilogue(_fixture("t2_matmul_relu"))
    assert [op.name for op in match.ops] == ["arith.addf", "arith.maxnumf"]


def test_matmul_without_an_epilogue_has_no_epilogue_match() -> None:
    assert detect_epilogue(_fixture("t1_matmul")) == ()


def test_refusal_lookup_matches_on_the_operation_name() -> None:
    annotations = AnnotationSet(refusals=(("tt.load", "unstructured access"),))
    assert annotations.refusal_for(type("Op", (), {"name": "tt.load"})()) == "unstructured access"
    assert annotations.refusal_for(type("Op", (), {"name": "tt.store"})()) is None


def _names(module, subsumed) -> list[str]:
    graph = build_def_use(module)
    return sorted(op.name for op in graph.operations if id(op) in subsumed)


_MASKED_LOAD_ONLY = """module {
  tt.func public @k(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %nn = tt.splat %n : i32 -> tensor<64xi32>
    %m = arith.cmpi slt, %r, %nn : tensor<64xi32>
    %p = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %a, %m : tensor<64x!tt.ptr<f32>>
    %q = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b = tt.addptr %q, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %b, %v : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""

_MASKED_STORE_ONLY = _MASKED_LOAD_ONLY.replace("tt.load %a, %m", "tt.load %a").replace(
    "tt.store %b, %v :", "tt.store %b, %v, %m :"
)


def test_the_mask_of_a_load_is_part_of_the_address_cone() -> None:
    module = _text(_MASKED_LOAD_ONLY)
    assert "arith.cmpi" in _names(module, find_subsumed_address_ops(module, build_def_use(module)))


def test_the_mask_of_a_store_is_part_of_the_address_cone() -> None:
    module = _text(_MASKED_STORE_ONLY)
    assert "arith.cmpi" in _names(module, find_subsumed_address_ops(module, build_def_use(module)))


def test_stored_data_is_never_part_of_the_address_cone() -> None:
    module = _fixture("t0_vecadd")
    assert "arith.addf" not in _names(module, find_subsumed_address_ops(module, build_def_use(module)))
