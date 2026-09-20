from tritonflow.lower import lower_fixture
from tritonflow.emu.exec import emulate, PrecisionPolicy
import numpy as np
ctx = lower_fixture("t0_vecadd", isa_name="vortex_rvgpu")
rng = np.random.default_rng(123)
x = rng.standard_normal(1024).astype(np.float32)
y = rng.standard_normal(1024).astype(np.float32)
out = emulate(ctx.program, {"%x": x, "%y": y, "%out": np.zeros_like(x), "%n": 1024}, policy=PrecisionPolicy(input_precision="ieee"))
print("emulate output sum:", out["%out"].sum())
print("expected sum:", (x + y).sum())
print("error:", np.max(np.abs(out["%out"] - (x + y))))
print(out["%out"][:10])
