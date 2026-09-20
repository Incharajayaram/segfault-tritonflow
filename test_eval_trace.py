import sys; sys.path.insert(0, "src")
import tritonflow.emu.exec as exec_mod
orig_materialize = exec_mod.MachineState._materialize_descriptor
def traced_materialize(self, memref, base):
    print(f"MATERIALIZE: memref={memref} base={base} grid={self.grid}")
    import tritonflow.emu.exec as ex
    fields = ex._descriptor_fields(memref.access_key)
    print(f"FIELDS: {fields}")
    res = orig_materialize(self, memref, base)
    print(f"RESULT shape: {res.shape} min={res.min()} max={res.max()}")
    return res
exec_mod.MachineState._materialize_descriptor = traced_materialize

from tritonflow.torch_backend.compiler import tritonflow_backend
import torch

A = torch.randn(128, 64)
B = torch.randn(64, 128)
def fn(a, b): return a @ b
compiled = torch.compile(fn, backend=tritonflow_backend)
compiled(A, B)
