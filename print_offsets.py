import numpy as np

class DummyState:
    def __init__(self):
        self.values = {
            "%pid_m": 1, "%pid_n": 2,
            "%sam": 128, "%sak": 1,
            "%sbk": 128, "%sbn": 1,
            "%scm": 128, "%scn": 1
        }
    def calc_offsets(self, raw_offsets):
        offsets = []
        for o in (raw_offsets.split(",") if raw_offsets else []):
            o = o.strip()
            try:
                offsets.append(int(o))
            except ValueError:
                try:
                    expr = o.replace('%', '')
                    local_vars = {}
                    for k, v in self.values.items():
                        if isinstance(k, str):
                            try:
                                local_vars[k.replace('%', '')] = int(np.asarray(v).item())
                            except Exception:
                                pass
                    offsets.append(int(eval(expr, {"__builtins__": {}}, local_vars)))
                except Exception as e:
                    print("Error:", e)
                    offsets.append(0)
        return offsets

state = DummyState()
print(state.calc_offsets("64*%pid_m*%sam, 0"))
print(state.calc_offsets("64*%pid_n*%sbn, 0"))
print(state.calc_offsets("64*%pid_m*%scm + 64*%pid_n*%scn, 0"))
