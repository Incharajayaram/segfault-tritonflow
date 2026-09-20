import sys; sys.path.insert(0, "src")
import numpy as np
import torch
from tritonflow.torch_backend.compiler import tritonflow_backend

A = torch.randn(8, 64)
B = torch.randn(64, 32)

def fn(a, b):
    return torch.mm(a, b)

graph, _ = torch._dynamo.export(fn, tracing_mode="real", aten_graph=False)(A, B)
call = tritonflow_backend(graph, (A, B))
result = call(A, B)
if isinstance(result, (list, tuple)):
    result = result[0]
expected = fn(A, B)
print("result shape:", tuple(result.shape), "expected:", tuple(expected.shape))
print("rel_err:", (result - expected).abs().max().item() / expected.abs().max().item())
print("result[:4,:4]:\n", result[:4,:4])
print("expected[:4,:4]:\n", expected[:4,:4])
print("result nonzero:", np.count_nonzero(result.numpy()), "of", result.numel())
