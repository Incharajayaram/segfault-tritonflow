import sys; sys.path.insert(0, "src")
import tritonflow.emu.exec as exec_mod
from tritonflow.torch_backend.compiler import tritonflow_backend
import torch

def patched_materialize(self, memref, base):
    from tritonflow.emu.exec import _descriptor_fields
    import numpy as np
    fields = _descriptor_fields(memref.access_key)
    raw_sizes = fields.get("sizes", "[]").strip("[]").strip()
    sizes = tuple(int(s) for s in raw_sizes.split(",") if s.strip()) if raw_sizes else ()

    raw_strides = fields.get("strides", "[]").strip("[]").strip()
    strides = []
    for s in (raw_strides.split(",") if raw_strides else []):
        s = s.strip()
        if not s: continue
        try:
            strides.append(int(s))
        except ValueError:
            if s in self.values:
                strides.append(int(np.asarray(self.values[s]).item()))
            else:
                strides.append(1)
    strides = tuple(strides)

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

    raw_offsets = fields.get("offsets", "[]").strip("[]").strip()
    offsets = []
    for o in (raw_offsets.split(",") if raw_offsets else []):
        o = o.strip()
        if not o: continue
        try:
            offsets.append(int(o))
        except ValueError:
            expr = o.replace("%", "")
            try:
                offsets.append(int(eval(expr, {}, local_vars)))
            except Exception as e:
                print(f"EVAL ERROR: {expr} -> {e}")
                offsets.append(0)

    is_loop_carried = fields.get("loop_carried", "False") == "True"
    raw_inc = fields.get("increment", "0")
    inc = 0
    try:
        inc = int(raw_inc)
    except ValueError:
        if raw_inc in self.values:
            inc = int(np.asarray(self.values[raw_inc]).item())
        else:
            expr = raw_inc.replace("%", "")
            try:
                inc = int(eval(expr, {}, local_vars))
            except Exception:
                inc = 0
            
    stride_for_inc = 1
    for i, size in enumerate(sizes):
        if size == inc and i < len(strides):
            stride_for_inc = strides[i]
            break
            
    loop_offset = self.loop_iteration * inc * stride_for_inc if is_loop_carried else 0

    if not sizes: return np.array([base], dtype=np.int64)

    coords = np.indices(sizes, dtype=np.int64)
    addresses = np.full(sizes, base + loop_offset, dtype=np.int64)
    for dim in range(len(sizes)):
        off = offsets[dim] if dim < len(offsets) else 0
        stride = strides[dim] if dim < len(strides) else 1
        addresses += coords[dim] * stride + off
    return addresses

exec_mod.MachineState._materialize_descriptor = patched_materialize

A = torch.randn(128, 64)
B = torch.randn(64, 128)
def fn(a, b): return a @ b
compiled = torch.compile(fn, backend=tritonflow_backend)
compiled(A, B)
