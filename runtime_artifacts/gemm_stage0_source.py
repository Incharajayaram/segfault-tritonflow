# PyTorch / Triton High-Level Specification
import torch

# Problem dimensions: M=128, N=128, K=64 | Tiled Block: (64, 64, 32)
A = torch.randn(128, 64, dtype=torch.float32)
B = torch.randn(64, 128, dtype=torch.float32)
C = torch.matmul(A, B)  # (128x64) @ (64x128) -> (128x128)
