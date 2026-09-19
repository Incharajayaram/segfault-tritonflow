"""kernels.py — 22 diverse Triton kernels covering patterns outside the 4 fixtures (Task E1).

Categories covered:
1. Reductions along axes (1D, 2D axis 0, 2D axis 1)
2. Broadcasts (1D to 2D, scalar to 2D)
3. Transposes (2D strided transpose)
4. Indirect memory access (gather and scatter)
5. Deep elementwise chains (depth 4, depth 6)
6. Data-dependent control flow (tl.where, conditional store mask)
7. Tiled 2D loops with non-unit stride and column-major layout
8. Mixed dtypes (fp16 to fp32, i32 to fp32)
9. Normalization & statistics (fused max, layer norm, vector dot)
10. Fused activation & residual connections
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import triton
import triton.language as tl


@dataclass
class CorpusKernel:
    name: str
    category: str
    description: str
    fn: Any
    signature: dict[str, str]
    constexprs: dict[str, Any]
    env: dict[str, Any]
    make_inputs: Callable[[], dict[str, np.ndarray]]
    reference: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]



# --------------------------------------------------------------------------- #
# 0. Fresh Dynamic Kernels (Lower to ISA with 0 refusals)
# --------------------------------------------------------------------------- #

@triton.jit
def _k00_dynamic_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# --------------------------------------------------------------------------- #
# 1. Reductions
# --------------------------------------------------------------------------- #

@triton.jit
def _k01_reduction_sum_1d(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr, s)

@triton.jit
def _k02_reduction_axis1_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + (rm[:, None] * N + rn[None, :]))
    row_sum = tl.sum(x, axis=1)
    tl.store(out_ptr + rm, row_sum)

@triton.jit
def _k03_reduction_axis0_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + (rm[:, None] * N + rn[None, :]))
    col_sum = tl.sum(x, axis=0)
    tl.store(out_ptr + rn, col_sum)


# --------------------------------------------------------------------------- #
# 2. Broadcasts
# --------------------------------------------------------------------------- #

@triton.jit
def _k04_broadcast_1d_to_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm)
    b = tl.broadcast_to(x[:, None], (M, N))
    tl.store(out_ptr + (rm[:, None] * N + rn[None, :]), b)

@triton.jit
def _k05_broadcast_scalar_to_2d(s_val, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    full = tl.full((M, N), s_val, dtype=tl.float32)
    tl.store(out_ptr + (rm[:, None] * N + rn[None, :]), full)


# --------------------------------------------------------------------------- #
# 3. Transpose
# --------------------------------------------------------------------------- #

@triton.jit
def _k06_transpose_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + (rm[:, None] * N + rn[None, :]))
    # Transposed store into (N, M) buffer
    tl.store(out_ptr + (rn[:, None] * M + rm[None, :]), tl.trans(x))


# --------------------------------------------------------------------------- #
# 4. Indirect indexing (gather / scatter)
# --------------------------------------------------------------------------- #

@triton.jit
def _k07_indirect_gather_1d(idx_ptr, tbl_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    idx = tl.load(idx_ptr + r)
    val = tl.load(tbl_ptr + idx)
    tl.store(out_ptr + r, val)

@triton.jit
def _k08_indirect_scatter_1d(idx_ptr, val_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    idx = tl.load(idx_ptr + r)
    val = tl.load(val_ptr + r)
    tl.store(out_ptr + idx, val)


# --------------------------------------------------------------------------- #
# 5. Deep elementwise chains
# --------------------------------------------------------------------------- #

@triton.jit
def _k09_elementwise_chain_depth4(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    y = ((x * 2.0 + 3.0) / 4.0) - 1.0
    tl.store(out_ptr + r, y)

@triton.jit
def _k10_elementwise_chain_depth6(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    y = tl.maximum(0.0, ((x * 1.5 + 0.5) * x - 2.0) + 1.0)
    tl.store(out_ptr + r, y)


# --------------------------------------------------------------------------- #
# 6. Data-dependent control flow
# --------------------------------------------------------------------------- #

@triton.jit
def _k11_branch_where(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    cond = x > 0.0
    y = tl.where(cond, x * 2.0, x * 0.5)
    tl.store(out_ptr + r, y)

@triton.jit
def _k12_masked_threshold(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    mask = x >= 0.0
    tl.store(out_ptr + r, x * 3.0, mask=mask)


# --------------------------------------------------------------------------- #
# 7. Tiled 2D with non-unit strides
# --------------------------------------------------------------------------- #

@triton.jit
def _k13_tiled_loop_nonunit_stride(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    # Stride of 128 for rows, stride of 2 for columns
    offs = rm[:, None] * 128 + rn[None, :] * 2
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x * 1.5)

@triton.jit
def _k14_tiled_column_major(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    # Column major layout: stride along M is 1, stride along N is M
    offs = rm[:, None] + rn[None, :] * M
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x + 2.0)


# --------------------------------------------------------------------------- #
# 8. Mixed dtypes
# --------------------------------------------------------------------------- #

@triton.jit
def _k15_mixed_dtypes_f16_f32(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    x_f32 = x.to(tl.float32)
    y = x_f32 * 2.5 + 1.0
    tl.store(out_ptr + r, y)

@triton.jit
def _k16_mixed_dtypes_i32_f32(x_ptr, scale_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    scale = tl.load(scale_ptr + r)
    y = x + scale.to(tl.float32)
    tl.store(out_ptr + r, y)


# --------------------------------------------------------------------------- #
# 9. Fused statistics and reductions
# --------------------------------------------------------------------------- #

@triton.jit
def _k17_fused_online_max(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    m = tl.max(x, axis=0)
    tl.store(out_ptr + r, x - m)

@triton.jit
def _k18_layer_norm_1d(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    norm = diff / tl.sqrt(var + 1e-5)
    tl.store(out_ptr + r, norm)

@triton.jit
def _k19_vector_dot(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    y = tl.load(y_ptr + r)
    dot = tl.sum(x * y, axis=0)
    tl.store(out_ptr, dot)


# --------------------------------------------------------------------------- #
# 10. Fused activation & residual
# --------------------------------------------------------------------------- #

@triton.jit
def _k20_fused_relu_scaled(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    y = tl.maximum(0.0, x) * 0.5
    tl.store(out_ptr + r, y)

@triton.jit
def _k21_conv1d_sliding(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x0 = tl.load(x_ptr + r)
    x1 = tl.load(x_ptr + r + 1)
    x2 = tl.load(x_ptr + r + 2)
    y = x0 * 0.25 + x1 * 0.5 + x2 * 0.25
    tl.store(out_ptr + r, y)

@triton.jit
def _k22_residual_add_relu(x_ptr, res_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    res = tl.load(res_ptr + r)
    y = tl.maximum(0.0, x + res)
    tl.store(out_ptr + r, y)


# --------------------------------------------------------------------------- #
# Corpus Registry
# --------------------------------------------------------------------------- #

def get_corpus() -> list[CorpusKernel]:
    rng = np.random.default_rng(9412)
    return [
        CorpusKernel(
            name="k00_dynamic_add_64",
            category="elementwise_dynamic",
            description="Fresh @triton.jit elementwise add compiled AOT without GPU (lowered to ISA)",
            fn=_k00_dynamic_add,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
            constexprs={"BLOCK": 64},
            env={"n": 64, "%n": 64, "M": 64, "N": 1, "%__flat_width__": 64, "%__block__": 64, "%__tile_m__": 64, "%__tile_n__": 1, "%__tile_k__": 1},
            make_inputs=lambda: {
                "%x_ptr": rng.standard_normal(64).astype(np.float32),
                "%y_ptr": rng.standard_normal(64).astype(np.float32),
                "%out_ptr": np.zeros(64, dtype=np.float32),
                "%n": 64,
            },
            reference=lambda inp: {"out_ptr": inp["%x_ptr"] + inp["%y_ptr"]},
        ),

        CorpusKernel(
            name="k01_reduction_sum_1d",
            category="reduction",
            description="1D axis=0 reduction (tl.sum)",
            fn=_k01_reduction_sum_1d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(1, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.sum(inp["%x_ptr"], keepdims=True)},
        ),
        CorpusKernel(
            name="k02_reduction_axis1_2d",
            category="reduction",
            description="2D axis=1 row-wise reduction",
            fn=_k02_reduction_axis1_2d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 16, "N": 32},
            env={"M": 16, "N": 32, "%M": 16, "%N": 32},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal((16, 32)).astype(np.float32), "%out_ptr": np.zeros(16, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.sum(inp["%x_ptr"], axis=1)},
        ),
        CorpusKernel(
            name="k03_reduction_axis0_2d",
            category="reduction",
            description="2D axis=0 column-wise reduction",
            fn=_k03_reduction_axis0_2d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 32, "N": 16},
            env={"M": 32, "N": 16, "%M": 32, "%N": 16},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal((32, 16)).astype(np.float32), "%out_ptr": np.zeros(16, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.sum(inp["%x_ptr"], axis=0)},
        ),
        CorpusKernel(
            name="k04_broadcast_1d_to_2d",
            category="broadcast",
            description="1D vector broadcast along axis 1 to 2D matrix",
            fn=_k04_broadcast_1d_to_2d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 16, "N": 32},
            env={"M": 16, "N": 32, "%M": 16, "%N": 32},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(16).astype(np.float32), "%out_ptr": np.zeros((16, 32), dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.broadcast_to(inp["%x_ptr"][:, None], (16, 32)).copy()},
        ),
        CorpusKernel(
            name="k05_broadcast_scalar_to_2d",
            category="broadcast",
            description="Scalar constant broadcast into 2D tensor (tl.full)",
            fn=_k05_broadcast_scalar_to_2d,
            signature={"s_val": "fp32", "out_ptr": "*fp32"},
            constexprs={"M": 16, "N": 16},
            env={"s_val": 3.1415, "M": 16, "N": 16, "%M": 16, "%N": 16},
            make_inputs=lambda: {"s_val": 3.1415, "%out_ptr": np.zeros((16, 16), dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.full((16, 16), 3.1415, dtype=np.float32)},
        ),
        CorpusKernel(
            name="k06_transpose_2d",
            category="transpose",
            description="2D matrix transposition store (tl.trans)",
            fn=_k06_transpose_2d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 16, "N": 32},
            env={"M": 16, "N": 32, "%M": 16, "%N": 32},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal((16, 32)).astype(np.float32), "%out_ptr": np.zeros((32, 16), dtype=np.float32)},
            reference=lambda inp: {"out_ptr": inp["%x_ptr"].T.copy()},
        ),
        CorpusKernel(
            name="k07_indirect_gather_1d",
            category="gather",
            description="Indirect lookup gather via integer index array",
            fn=_k07_indirect_gather_1d,
            signature={"idx_ptr": "*i32", "tbl_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 32},
            env={"N": 32, "%N": 32},
            make_inputs=lambda: {
                "%idx_ptr": rng.integers(0, 64, size=32, dtype=np.int32),
                "%tbl_ptr": rng.standard_normal(64).astype(np.float32),
                "%out_ptr": np.zeros(32, dtype=np.float32),
            },
            reference=lambda inp: {"out_ptr": inp["%tbl_ptr"][inp["%idx_ptr"]]},
        ),
        CorpusKernel(
            name="k08_indirect_scatter_1d",
            category="scatter",
            description="Indirect indexed scatter store into output array",
            fn=_k08_indirect_scatter_1d,
            signature={"idx_ptr": "*i32", "val_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 32},
            env={"N": 32, "%N": 32},
            make_inputs=lambda: {
                "%idx_ptr": rng.choice(64, size=32, replace=False).astype(np.int32),
                "%val_ptr": rng.standard_normal(32).astype(np.float32),
                "%out_ptr": np.zeros(64, dtype=np.float32),
            },
            reference=lambda inp: {
                "out_ptr": (lambda out: (out.__setitem__(inp["%idx_ptr"], inp["%val_ptr"]), out)[1])(np.zeros(64, dtype=np.float32))
            },
        ),
        CorpusKernel(
            name="k09_elementwise_chain_depth4",
            category="elementwise_deep",
            description="4-stage arithmetic chain ((x*2+3)/4)-1",
            fn=_k09_elementwise_chain_depth4,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": ((inp["%x_ptr"] * 2.0 + 3.0) / 4.0) - 1.0},
        ),
        CorpusKernel(
            name="k10_elementwise_chain_depth6",
            category="elementwise_deep",
            description="6-stage nonlinear arithmetic chain with relu",
            fn=_k10_elementwise_chain_depth6,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.maximum(0.0, ((inp["%x_ptr"] * 1.5 + 0.5) * inp["%x_ptr"] - 2.0) + 1.0)},
        ),
        CorpusKernel(
            name="k11_branch_where",
            category="control_flow",
            description="Data-dependent selection via tl.where",
            fn=_k11_branch_where,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.where(inp["%x_ptr"] > 0.0, inp["%x_ptr"] * 2.0, inp["%x_ptr"] * 0.5)},
        ),
        CorpusKernel(
            name="k12_masked_threshold",
            category="control_flow",
            description="Conditional store with boolean predicate mask",
            fn=_k12_masked_threshold,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.where(inp["%x_ptr"] >= 0.0, inp["%x_ptr"] * 3.0, 0.0)},
        ),
        CorpusKernel(
            name="k13_tiled_loop_nonunit_stride",
            category="strided_tile",
            description="2D tiled access with non-unit strides (row: 128, col: 2)",
            fn=_k13_tiled_loop_nonunit_stride,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 8, "N": 16},
            env={"M": 8, "N": 16, "%M": 8, "%N": 16},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(2048).astype(np.float32), "%out_ptr": np.zeros(2048, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": inp["%x_ptr"] * 1.5},
        ),
        CorpusKernel(
            name="k14_tiled_column_major",
            category="strided_tile",
            description="2D tiled access in Fortran column-major order",
            fn=_k14_tiled_column_major,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"M": 16, "N": 16},
            env={"M": 16, "N": 16, "%M": 16, "%N": 16},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(256).astype(np.float32), "%out_ptr": np.zeros(256, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": inp["%x_ptr"] + 2.0},
        ),
        CorpusKernel(
            name="k15_mixed_dtypes_f16_f32",
            category="mixed_dtypes",
            description="Loads fp16, casts to fp32, arithmetic, stores fp32",
            fn=_k15_mixed_dtypes_f16_f32,
            signature={"x_ptr": "*fp16", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float16), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": inp["%x_ptr"].astype(np.float32) * 2.5 + 1.0},
        ),
        CorpusKernel(
            name="k16_mixed_dtypes_i32_f32",
            category="mixed_dtypes",
            description="Combines i32 integer array with fp32 data",
            fn=_k16_mixed_dtypes_i32_f32,
            signature={"x_ptr": "*fp32", "scale_ptr": "*i32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {
                "%x_ptr": rng.standard_normal(64).astype(np.float32),
                "%scale_ptr": rng.integers(-5, 5, size=64, dtype=np.int32),
                "%out_ptr": np.zeros(64, dtype=np.float32),
            },
            reference=lambda inp: {"out_ptr": inp["%x_ptr"] + inp["%scale_ptr"].astype(np.float32)},
        ),
        CorpusKernel(
            name="k17_fused_online_max",
            category="statistics",
            description="Online maximum subtraction (tl.max)",
            fn=_k17_fused_online_max,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": inp["%x_ptr"] - np.max(inp["%x_ptr"])},
        ),
        CorpusKernel(
            name="k18_layer_norm_1d",
            category="statistics",
            description="Layer normalization 1D (mean, variance, normalize)",
            fn=_k18_layer_norm_1d,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {
                "out_ptr": (inp["%x_ptr"] - np.mean(inp["%x_ptr"])) / np.sqrt(np.var(inp["%x_ptr"]) + 1e-5)
            },
        ),
        CorpusKernel(
            name="k19_vector_dot",
            category="reduction",
            description="Vector dot product reduction tl.sum(x * y)",
            fn=_k19_vector_dot,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {
                "%x_ptr": rng.standard_normal(64).astype(np.float32),
                "%y_ptr": rng.standard_normal(64).astype(np.float32),
                "%out_ptr": np.zeros(1, dtype=np.float32),
            },
            reference=lambda inp: {"out_ptr": np.array([np.dot(inp["%x_ptr"], inp["%y_ptr"])], dtype=np.float32)},
        ),
        CorpusKernel(
            name="k20_fused_relu_scaled",
            category="activation",
            description="Fused ReLU with constant scaling",
            fn=_k20_fused_relu_scaled,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {"out_ptr": np.maximum(0.0, inp["%x_ptr"]) * 0.5},
        ),
        CorpusKernel(
            name="k21_conv1d_sliding",
            category="stencil",
            description="1D sliding window filter [0.25, 0.5, 0.25]",
            fn=_k21_conv1d_sliding,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {"%x_ptr": rng.standard_normal(64).astype(np.float32), "%out_ptr": np.zeros(64, dtype=np.float32)},
            reference=lambda inp: {
                "out_ptr": (
                    inp["%x_ptr"][:64] * 0.25 + inp["%x_ptr"][1:65] * 0.5 + inp["%x_ptr"][2:66] * 0.25
                )
            },
        ),
        CorpusKernel(
            name="k22_residual_add_relu",
            category="activation",
            description="Residual add followed by ReLU activation",
            fn=_k22_residual_add_relu,
            signature={"x_ptr": "*fp32", "res_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 64},
            env={"N": 64, "%N": 64},
            make_inputs=lambda: {
                "%x_ptr": rng.standard_normal(64).astype(np.float32),
                "%res_ptr": rng.standard_normal(64).astype(np.float32),
                "%out_ptr": np.zeros(64, dtype=np.float32),
            },
            reference=lambda inp: {"out_ptr": np.maximum(0.0, inp["%x_ptr"] + inp["%res_ptr"])},
        ),
    ]
