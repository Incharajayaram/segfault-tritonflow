import sys
sys.path.insert(0, "src")
import torch
import tritonflow.torch_backend.compiler as compiler

orig_emulate = compiler.emulate
def debug_emulate(program, *args, **kwargs):
    print("PROGRAM INSTRUCTIONS:")
    for instr in program.instructions():
        print(instr.name)
    return orig_emulate(program, *args, **kwargs)
compiler.emulate = debug_emulate

from tests.integration.test_dynamic_extraction import run_backend

n = 100
x, y = torch.randn(n), torch.randn(n)
try:
    run_backend(lambda p, q: p + q, x, y)
except Exception as e:
    pass
