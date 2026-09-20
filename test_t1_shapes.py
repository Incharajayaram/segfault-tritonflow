import sys; sys.path.insert(0, "src")
import numpy as np
from tritonflow.lower import make_inputs

inputs = make_inputs("t1_matmul")
print("A shape:", inputs["a"].shape)    # (64, 32)
print("B shape:", inputs["b"].shape)    # (32, 64)

# So storage for %a => shape=(64,32), size=2048
# _declared_shape for %a => constrained_on="sizes=[64, 32]" => (64, 32)
# arange(0, 2048).reshape((64,32)) => 2048 indices => gather 2048 elems => reshape OK

# Why does it fail then? Let's check constrained_on on LDG instr
from tritonflow.lower import lower_fixture
import tritonflow.emu.exec as emu_exec

orig_apply_memory = emu_exec._apply_memory
def traced_apply_memory(instr, state):
    print(f"\n--- {instr.name} ---")
    print(f"  constrained_on: {instr.constrained_on!r}")
    shape = emu_exec._declared_shape(instr)
    print(f"  _declared_shape: {shape}")
    src = instr.operand("src")
    if src is not None:
        resolved = state.resolve(src)
        arr = np.asarray(resolved)
        print(f"  resolved indices shape: {arr.shape}, size: {arr.size}")
        print(f"  declared product: {int(np.prod(shape)) if shape else 'N/A'}")
    return orig_apply_memory(instr, state)
emu_exec._apply_memory = traced_apply_memory

try:
    ctx = lower_fixture("t1_matmul", isa_name="vortex_rvgpu")
    print("parity_max_rel_err:", ctx.parity_max_rel_err)
except Exception as e:
    import traceback; traceback.print_exc()
