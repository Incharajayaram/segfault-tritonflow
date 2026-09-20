import sys
sys.path.insert(0, "src")
import torch
from tests.integration.test_dynamic_extraction import run_backend

a, b = torch.randn(128, 64), torch.randn(64, 128)
call, result = run_backend(lambda x, y: x @ y, a, b)
for f in call.tritonflow_plan.fallbacks:
    print(f.reason, f.detail)
