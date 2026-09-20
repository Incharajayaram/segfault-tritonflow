import sys

with open("src/tritonflow/emu/exec.py", "r") as f:
    content = f.read()

content = content.replace("    def _materialize_descriptor(self, memref: \"MemRef\", base: int) -> np.ndarray:",
                          "    def _materialize_descriptor(self, memref: \"MemRef\", base: int) -> np.ndarray:\n        import numpy as np")

with open("src/tritonflow/emu/exec.py", "w") as f:
    f.write(content)
