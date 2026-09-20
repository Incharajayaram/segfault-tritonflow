import sys
sys.path.insert(0, "src")
import torch
from tests.integration.test_dynamic_extraction import run_backend

def debug_bind(self, name, value):
    print(f"BIND {name} = {value}")
    self.values[name] = value

import tritonflow.emu.exec as emu_exec
emu_exec.MachineState.bind = debug_bind

def debug_program_id(instr, state, shape):
    axis = int(state.resolve(emu_exec._require(instr, "value")))
    val = state.grid[axis]
    print(f"GET_PROGRAM_ID axis={axis} grid={state.grid} val={val}")
    return val

emu_exec._program_id = debug_program_id

n = 100
x, y = torch.randn(n), torch.randn(n)
try:
    call, result = run_backend(lambda p, q: p + q, x, y)
except Exception as e:
    print(e)
