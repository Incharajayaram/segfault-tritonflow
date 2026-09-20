import sys
sys.path.insert(0, "src")
import torch
from tests.integration.test_dynamic_extraction import run_backend

def debug_scatter(self, indices, values, mask):
    import numpy as np
    flat = np.asarray(indices).reshape(-1).astype(np.int64)
    print(f"SCATTER indices min={flat.min()} max={flat.max()}")
    if mask is not None:
        keep = np.asarray(mask).reshape(-1)
        print(f"SCATTER mask keeps {keep.sum()} elements. First 5 keeps: {keep[:5]}")
        print(f"SCATTER values shape {values.shape}, first 5 kept values: {values.reshape(-1)[keep][:5]}")
    else:
        print(f"SCATTER no mask")

import tritonflow.emu.exec as emu_exec
# Patch the scatter method temporarily
orig_scatter = emu_exec.MachineState.scatter
emu_exec.MachineState.scatter = debug_scatter

n = 100
x, y = torch.randn(n), torch.randn(n)
call, result = run_backend(lambda p, q: p + q, x, y)
