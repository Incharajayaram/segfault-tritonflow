import torch, torch.nn as nn; import tritonflow.torch_backend.compiler; torch._dynamo.reset(); 
def matmul(a, b):
    return a @ b
M, K, N = 128, 64, 128
a, b = torch.randn(M, K), torch.randn(K, N)
opt = torch.compile(matmul, backend='tritonflow')
out = opt(a, b)
