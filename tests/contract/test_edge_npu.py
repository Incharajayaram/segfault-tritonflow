from tritonflow.pipeline import compile_fixture

def test_edge_npu_refuses_fp32_matmul():
    """Verify the Edge NPU schema correctly refuses FP32 math based on dtype constraints."""
    r = compile_fixture("t1_matmul", "edge_npu")
    assert not r.fully_lowered
    
    # Extract the string reason (or object with __str__)
    reason = str(r.unsupported[0])
    assert "tt.dot: no mac instruction is admissible" in reason
    assert "dtype_bits == 8" in reason
