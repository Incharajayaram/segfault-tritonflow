import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend

orig_scatter = exec_mod.MachineState.scatter
def traced_scatter(self, indices, values, mask):
    idx = np.asarray(indices)
    val = np.asarray(values)
    print(f"SCATTER: idx.shape={idx.shape} val.shape={val.shape} idx_min={idx.min()} idx_max={idx.max()} val_nz={np.count_nonzero(val)}")
    return orig_scatter(self, indices, values, mask)
exec_mod.MachineState.scatter = traced_scatter

torch.manual_seed(0)
A = torch.randn(128, 64)
B = torch.randn(64, 128)

def fn(a, b):
    return torch.mm(a, b)

graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(A, B)
call = tritonflow_backend(graph, (A, B))
result = call(A, B)
if isinstance(result, (list, tuple)):
    result = result[0]
r = result.numpy()
print("result nonzero:", np.count_nonzero(r), "of", r.size)
# Where are the nonzero values?
nz = np.argwhere(np.count_nonzero(r, axis=1) > 0)
print("nonzero rows:", nz[:5].flatten() if len(nz) else "none", "...", nz[-5:].flatten() if len(nz) else "")
nz2 = np.argwhere(np.count_nonzero(r, axis=0) > 0)
print("nonzero cols:", nz2[:5].flatten() if len(nz2) else "none")
