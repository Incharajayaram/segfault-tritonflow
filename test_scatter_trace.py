import sys; sys.path.insert(0, "src")
import numpy as np
import tritonflow.emu.exec as exec_mod

orig_scatter = exec_mod.MachineState.scatter
def traced_scatter(self, indices, values, mask):
    idx_arr = np.asarray(indices)
    val_arr = np.asarray(values)
    print(f"SCATTER: indices shape={idx_arr.shape} values shape={val_arr.shape} nonzero_vals={np.count_nonzero(val_arr)}")
    return orig_scatter(self, indices, values, mask)
exec_mod.MachineState.scatter = traced_scatter

from tritonflow.lower import lower_fixture
ctx = lower_fixture("t1_matmul", isa_name="vortex_rvgpu")
print("parity:", ctx.parity_max_rel_err)
print("exec_err:", ctx.execution_error)
