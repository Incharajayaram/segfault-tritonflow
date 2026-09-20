import sys
import re

with open("src/tritonflow/emu/exec.py", "r") as f:
    content = f.read()

# Fix offsets parsing in _materialize_descriptor
old_offsets = """        raw_offsets = fields.get("offsets", "[]").strip("[]").strip()
        offsets = []
        for o in (raw_offsets.split(",") if raw_offsets else []):
            o = o.strip()
            try:
                offsets.append(int(o))
            except ValueError:
                offsets.append(0)

        is_loop_carried = fields.get("loop_carried", "False") == "True"
        raw_inc = fields.get("increment", "0")
        inc = 0
        try:
            inc = int(raw_inc)
        except ValueError:
            if raw_inc in self.values:
                inc = int(np.asarray(self.values[raw_inc]).item())
                
        loop_offset = self.loop_iteration * inc if is_loop_carried else 0"""

new_offsets = """        raw_offsets = fields.get("offsets", "[]").strip("[]").strip()
        
        local_vars = {}
        if len(self.grid) > 0: local_vars["pid_x"] = self.grid[0]
        if len(self.grid) > 1: local_vars["pid_y"] = self.grid[1]
        if len(self.grid) > 2: local_vars["pid_z"] = self.grid[2]
        # Also alias pid to grid[0] just in case
        if len(self.grid) > 0: local_vars["pid"] = self.grid[0]
        
        offsets = []
        for o in (raw_offsets.split(",") if raw_offsets else []):
            o = o.strip()
            if not o: continue
            try:
                offsets.append(int(o))
            except ValueError:
                # evaluate symbolic offsets like 64*pid_x
                expr = o.replace("%", "")
                try:
                    offsets.append(int(eval(expr, {}, local_vars)))
                except Exception:
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
                
        # loop_offset stride lookup: stride_for_inc = strides[i] where sizes[i] == inc
        stride_for_inc = 1
        for i, size in enumerate(sizes):
            if size == inc and i < len(strides):
                stride_for_inc = strides[i]
                break
                
        loop_offset = self.loop_iteration * inc * stride_for_inc if is_loop_carried else 0"""

if old_offsets in content:
    content = content.replace(old_offsets, new_offsets)
else:
    print("Could not find old offsets code!")

with open("src/tritonflow/emu/exec.py", "w") as f:
    f.write(content)
