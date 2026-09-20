import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod

orig_scatter = exec_mod.MachineState.scatter
def traced_scatter(self, indices, values, mask):
    idx_arr = np.asarray(indices)
    val_arr = np.asarray(values)
    nz = np.count_nonzero(val_arr)
    print(f"SCATTER: indices shape={idx_arr.shape} values shape={val_arr.shape} nonzero={nz} idx_min={idx_arr.min()} idx_max={idx_arr.max()}")
    return orig_scatter(self, indices, values, mask)
exec_mod.MachineState.scatter = traced_scatter

from tritonflow.torch_backend.compiler import tritonflow_backend
import torch._dynamo as dynamo

A = torch.randn(8, 64)
B = torch.randn(64, 32)

def fn(a, b):
    return torch.mm(a, b)

compiled = torch.compile(fn, backend=tritonflow_backend)
result = compiled(A, B)
expected = fn(A, B)
print("rel_err:", (result - expected).abs().max().item() / expected.abs().max().item())
