import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod

orig_mac = exec_mod._apply_mac
def traced_mac(instr, state, policy):
    a = np.asarray(state.resolve(instr.operand("a")))
    b = np.asarray(state.resolve(instr.operand("b")))
    print(f"MAC: a.shape={a.shape} b.shape={b.shape} a[0,:4]={a[0,:4]} b[0,:4]={b[0,:4]}")
    return orig_mac(instr, state, policy)
exec_mod._apply_mac = traced_mac

orig_bind = exec_mod.MachineState.bind
def traced_bind(self, name, value):
    if isinstance(value, np.ndarray) and value.ndim > 0 and name.startswith("%a_ptrs_34") or name.startswith("%b_ptrs_35"):
        print(f"BIND {name}: shape={value.shape} flat[:4]={np.asarray(value).reshape(-1)[:4]}")
    return orig_bind(self, name, value)
exec_mod.MachineState.bind = traced_bind

from tritonflow.torch_backend.compiler import tritonflow_backend

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
expected = fn(A, B)
print("rel_err:", (result - expected).abs().max().item() / expected.abs().max().item())
print("result nonzero:", np.count_nonzero(result.numpy()), "of", result.numel())
