import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend

# Trace all binds of ptr operands
orig_bind = exec_mod.MachineState.bind
def traced_bind(self, name, value):
    if name in ("%a_ptr", "%b_ptr", "%c_ptr"):
        print(f"BIND {name} = {value} (storages: {list(self.storages.keys())})")
        for s in self.storages.values():
            if s.name == name or s.base == value:
                print(f"   storage {s.name}: base={s.base} end={s.end} shape={s.shape}")
    return orig_bind(self, name, value)
exec_mod.MachineState.bind = traced_bind

# Trace resolve of MemRef operands
orig_resolve = exec_mod.MachineState.resolve
def traced_resolve(self, operand):
    r = orig_resolve(self, operand)
    if hasattr(operand, 'base') and operand.base in ("%a_ptr", "%b_ptr", "%c_ptr"):
        print(f"RESOLVE {operand.base} -> {np.asarray(r).flat[:4] if hasattr(r,'__len__') else r}")
    return r
exec_mod.MachineState.resolve = traced_resolve

torch.manual_seed(0)
A = torch.randn(128, 64)
B = torch.randn(64, 128)

def fn(a, b):
    return torch.mm(a, b)

graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(A, B)
call = tritonflow_backend(graph, (A, B))
result = call(A, B)
