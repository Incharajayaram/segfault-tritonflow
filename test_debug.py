import sys
sys.path.insert(0, "src")
import torch
from tests.integration.test_dynamic_extraction import run_backend

n = 100
x, y = torch.randn(n), torch.randn(n)
call, result = run_backend(lambda p, q: p + q, x, y)
print("Result shape:", result.shape)
print("Result first 70:", result[:70])
print("Result last 30:", result[-30:])
