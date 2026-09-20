import sys
sys.path.insert(0, "src")
import tritonflow.emu.exec as emu_exec
orig = emu_exec.emulate
def mock_emulate(*args, **kwargs):
    try:
        return orig(*args, **kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
emu_exec.emulate = mock_emulate
from tests.contract.test_end_to_end import TestEndToEndContract

t = TestEndToEndContract()
t.test_t1_matmul_end_to_end()
