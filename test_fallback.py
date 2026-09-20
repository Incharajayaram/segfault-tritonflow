import torch
from tests.integration.test_dynamic_extraction import run_backend
M, K, N = 128, 64, 128
a, b = torch.randn(M, K), torch.randn(K, N)
call, result = run_backend(lambda x, y: x @ y, a, b)
print(vars(call.tritonflow_plan))
