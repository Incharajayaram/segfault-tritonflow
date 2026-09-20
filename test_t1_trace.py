import sys
sys.path.insert(0, "src")
import numpy as np
from tritonflow.lower import make_inputs
from tritonflow.emu.exec import emulate, MachineState, _declared_shape
from tritonflow.emu.ir import Program, Instr, MemRef, SsaRef
from tritonflow.emit.ir import PrecisionPolicy

# Replicate what lower_fixture does for t1_matmul with vortex_rvgpu
inputs = make_inputs("t1_matmul")
print("inputs keys:", list(inputs.keys()))
for k, v in inputs.items():
    print(f"  {k}: shape={v.shape}")

# Build the same instrs as lower.py does
emu_inputs = {"%a": inputs["a"], "%b": inputs["b"], "%c": np.zeros((64, 64), dtype=np.float32)}

# Trace what _resolve_memref does
from tritonflow.lower import lower_fixture
import tritonflow.emu.exec as emu_exec

orig_bind = emu_exec.MachineState.bind
def trace_bind(self, name, value):
    shape = getattr(value, 'shape', 'scalar')
    print(f"BIND {name}: shape={shape}")
    return orig_bind(self, name, value)
emu_exec.MachineState.bind = trace_bind

orig_gather = emu_exec.MachineState.gather
def trace_gather(self, indices, mask):
    print(f"GATHER: indices shape={np.asarray(indices).shape}")
    result = orig_gather(self, indices, mask)
    print(f"GATHER result: shape={result.shape}")
    return result
emu_exec.MachineState.gather = trace_gather

try:
    ctx = lower_fixture("t1_matmul", isa_name="vortex_rvgpu")
    print("parity_max_rel_err:", ctx.parity_max_rel_err)
except Exception as e:
    import traceback
    traceback.print_exc()
