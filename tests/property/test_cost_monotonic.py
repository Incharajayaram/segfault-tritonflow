"""Property-based tests for cost model monotonicity and totality (spec §6 Gate G2)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tritonflow.isa.cost import (
    CostQuery,
    evaluate_cost,
    load_machine_by_name,
)
from tritonflow.isa.schema import load_builtin


class PropertyDescriptor:
    def __init__(
        self,
        sizes: tuple[int, ...] = (64,),
        strides: tuple[int, ...] = (1,),
        offsets: tuple[int, ...] = (0,),
        shape: tuple[int, ...] = (64,),
        dtype: str = "f32",
    ) -> None:
        self.sizes = sizes
        self.strides = strides
        self.offsets = offsets
        self.shape = shape
        self.dtype = dtype
        self.base_num = 0
        self.base = "%ptr_0"


@settings(max_examples=50)
@given(
    w1=st.integers(min_value=1, max_value=512),
    w2=st.integers(min_value=1, max_value=512),
)
def test_elements_cost_monotonic(w1: int, w2: int) -> None:
    """Monotonicity: ↑elements => cost never decreases."""
    schema = load_builtin("tritonflow1")
    mach = load_machine_by_name("tritonflow1")
    dma1d = schema.instruction("DMA1D")

    smaller = min(w1, w2)
    larger = max(w1, w2)

    q_small = CostQuery(
        instruction=dma1d,
        access=PropertyDescriptor(sizes=(smaller,)),
        env={"words": smaller},
        machine=mach,
    )
    q_large = CostQuery(
        instruction=dma1d,
        access=PropertyDescriptor(sizes=(larger,)),
        env={"words": larger},
        machine=mach,
    )

    c_small = evaluate_cost(q_small, mach).select_cost
    c_large = evaluate_cost(q_large, mach).select_cost

    assert c_small <= c_large, f"Expected {c_small} <= {c_large} for elements {smaller} vs {larger}"


@settings(max_examples=50)
@given(
    stride=st.integers(min_value=2, max_value=64),
    dim_n=st.integers(min_value=4, max_value=64),
)
def test_coalescing_stride_monotonic(stride: int, dim_n: int) -> None:
    """Coalescing: stride=1 has <= transactions than stride > 1."""
    schema = load_builtin("vortex_rvgpu")
    mach = load_machine_by_name("vortex_rvgpu")
    ldg = schema.instruction("LDG")

    # Contiguous inner dimension
    q_contig = CostQuery(
        instruction=ldg,
        access=PropertyDescriptor(sizes=(dim_n,), strides=(1,)),
        tile=(dim_n,),
        machine=mach,
    )
    # Strided inner dimension
    q_strided = CostQuery(
        instruction=ldg,
        access=PropertyDescriptor(sizes=(dim_n,), strides=(stride,)),
        tile=(dim_n,),
        machine=mach,
    )

    res_contig = evaluate_cost(q_contig, mach)
    res_strided = evaluate_cost(q_strided, mach)

    assert res_contig.resources.transactions <= res_strided.resources.transactions
    # And total select_cost for contiguous access <= strided access
    assert res_contig.select_cost <= res_strided.select_cost


@settings(max_examples=30)
@given(
    m=st.integers(min_value=1, max_value=64),
    n=st.integers(min_value=1, max_value=64),
    k=st.integers(min_value=1, max_value=64),
)
def test_mac_cost_monotonic(m: int, n: int, k: int) -> None:
    """Compute: ↑tile dimensions => mac_ops and cost never decrease."""
    schema = load_builtin("tritonflow1")
    mach = load_machine_by_name("tritonflow1")
    mac = schema.instruction("MAC16")

    tile_small = (m, n, k)
    tile_large = (m * 2, n * 2, k * 2)

    q_small = CostQuery(instruction=mac, tile=tile_small, machine=mach)
    q_large = CostQuery(instruction=mac, tile=tile_large, machine=mach)

    res_small = evaluate_cost(q_small, mach)
    res_large = evaluate_cost(q_large, mach)

    assert res_small.select_cost <= res_large.select_cost
    assert res_small.resources.mac_ops <= res_large.resources.mac_ops
