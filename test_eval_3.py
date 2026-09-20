import sys; sys.path.insert(0, "src")
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend
import torch

orig = exec_mod.MachineState._materialize_descriptor
def patched(self, memref, base):
    print(f"grid: {self.grid}")
    import numpy as np
    local_vars = {}
    if len(self.grid) > 0: local_vars["pid_x"] = self.grid[0]
    if len(self.grid) > 1: local_vars["pid_y"] = self.grid[1]
    if len(self.grid) > 2: local_vars["pid_z"] = self.grid[2]
    if len(self.grid) > 0: local_vars["pid"] = self.grid[0]
    for k, v in self.values.items():
        if k.startswith("%"):
            try:
                local_vars[k[1:]] = int(np.asarray(v).item())
            except Exception:
                pass
    print(f"local_vars: {local_vars}")
    return orig(self, memref, base)
exec_mod.MachineState._materialize_descriptor = patched

A = torch.randn(128, 64)
B = torch.randn(64, 128)
def fn(a, b): return a @ b
compiled = torch.compile(fn, backend=tritonflow_backend)
compiled(A, B)
