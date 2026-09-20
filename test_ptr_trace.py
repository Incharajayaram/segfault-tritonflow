import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend

# Trace addptr operands
orig_addptr = exec_mod._addptr
def traced_addptr(instr, state, shape):
    base, offset = exec_mod._operands(instr, state)
    b = np.asarray(base).astype(np.int64)
    o = np.asarray(offset).astype(np.int64)
    if b.size > 0 and int(b.flat[0]) in (8192, 8256, 0, 4096, 12288, 16384, 16384):
        print(f"ADDPTR {instr.name}: base_flat[0]={int(b.flat[0])} offset_flat[0]={int(o.flat[0])} -> {int(b.flat[0])+int(o.flat[0])}")
    return orig_addptr(instr, state, shape)
exec_mod._addptr = traced_addptr

# Trace splat of b_ptr
orig_splat = exec_mod._splat
def traced_splat(instr, state, shape):
    val = state.resolve(instr.operand("v"))
    if isinstance(val, int) and val in (8192, 16384):
        print(f"SPLAT {instr.name}: v={val} shape={shape}")
    return orig_splat(instr, state, shape)
exec_mod._splat = traced_splat

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
print("done, rel_err:", (result - fn(A,B)).abs().max().item() / fn(A,B).abs().max().item())
