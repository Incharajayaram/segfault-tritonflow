import sys; sys.path.insert(0, "src")
from tritonflow.lower import lower_fixture
ctx = lower_fixture("t2_matmul_relu", isa_name="vortex_rvgpu")
print("execution_error:", ctx.execution_error)
print("parity_max_rel_err:", ctx.parity_max_rel_err)
