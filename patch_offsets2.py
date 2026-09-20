import sys
import re

with open("src/tritonflow/emu/exec.py", "r") as f:
    content = f.read()

old_offsets = """        local_vars = {}
        if len(self.grid) > 0: local_vars["pid_x"] = self.grid[0]
        if len(self.grid) > 1: local_vars["pid_y"] = self.grid[1]
        if len(self.grid) > 2: local_vars["pid_z"] = self.grid[2]
        # Also alias pid to grid[0] just in case
        if len(self.grid) > 0: local_vars["pid"] = self.grid[0]
        
        offsets = []"""

new_offsets = """        local_vars = {}
        if len(self.grid) > 0: local_vars["pid_x"] = self.grid[0]
        if len(self.grid) > 1: local_vars["pid_y"] = self.grid[1]
        if len(self.grid) > 2: local_vars["pid_z"] = self.grid[2]
        if len(self.grid) > 0: local_vars["pid"] = self.grid[0]
        import numpy as np
        for k, v in self.values.items():
            if k.startswith("%"):
                try:
                    local_vars[k[1:]] = int(np.asarray(v).item())
                except Exception:
                    pass
        
        offsets = []"""

if old_offsets in content:
    content = content.replace(old_offsets, new_offsets)
    with open("src/tritonflow/emu/exec.py", "w") as f:
        f.write(content)
    print("PATCHED")
else:
    print("NOT FOUND")
