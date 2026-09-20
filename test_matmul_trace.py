import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod

orig_mac = exec_mod._apply_mac
def traced_mac(instr, state, policy):
    a = np.asarray(state.resolve(instr.operand("a")))
    b = np.asarray(state.resolve(instr.operand("b")))
    print(f"MAC {instr.name}: a.shape={a.shape} b.shape={b.shape} a_nonzero={np.count_nonzero(a)} b_nonzero={np.count_nonzero(b)}")
    return orig_mac(instr, state, policy)
exec_mod._apply_mac = traced_mac

orig_bind = exec_mod.MachineState.bind
def traced_bind(self, name, value):
    if isinstance(value, np.ndarray) and value.ndim > 0:
        print(f"BIND {name}: shape={value.shape} nonzero={np.count_nonzero(value)}")
    return orig_bind(self, name, value)
exec_mod.MachineState.bind = traced_bind

from tritonflow.torch_backend.compiler import tritonflow_backend
import torch._dynamo as dynamo

A = torch.randn(8, 64)
B = torch.randn(64, 32)

def fn(a, b):
    return torch.mm(a, b)

graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(A, B)
call = tritonflow_backend(graph, (A, B))
result = call(A, B)
expected = fn(A, B)
print("result shape:", result.shape, "expected:", expected.shape)
print("rel_err:", (result - expected).abs().max().item() / expected.abs().max().item())
print("result nonzero:", np.count_nonzero(result.numpy()))
