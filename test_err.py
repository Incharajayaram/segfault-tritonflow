import sys; sys.path.insert(0, "src")
import tritonflow.emu.exec as exec_mod
orig_apply_memory = exec_mod._apply_memory
def traced_apply_memory(instr, state):
    try:
        orig_apply_memory(instr, state)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise e
exec_mod._apply_memory = traced_apply_memory

from tritonflow.lower import lower_fixture
lower_fixture("t0_vecadd", isa_name="vortex_rvgpu")
