import sys; sys.path.insert(0, "src")
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend
import torch

orig = exec_mod.MachineState._materialize_descriptor
def patched(self, memref, base):
    print(f"grid: {self.grid}")
    import numpy as np
    res = orig(self, memref, base)
    print(f"MATERIALIZE {memref} -> min={res.min()} max={res.max()}")
    return res
exec_mod.MachineState._materialize_descriptor = patched

A = torch.randn(128, 64)
B = torch.randn(64, 128)
def fn(a, b): return a @ b
compiled = torch.compile(fn, backend=tritonflow_backend)
compiled(A, B)
