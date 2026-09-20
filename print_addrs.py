import torch
from tritonflow.extract.dynamic_extract import extract_matmul
import tritonflow.torch_backend.compiler as c
from tritonflow.emu.exec import MachineState

ex = extract_matmul(128, 64, 128)
k = c.prepare(ex.name, ex.ttir, ex.env, provenance='dynamic')
for instr in k.program.instructions():
    if instr.name.startswith('DMA') and 'src' in instr.roles:
        memref = instr.operands['src']
        print(memref.access_key)
