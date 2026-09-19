import torch
import torch.nn as nn


class MatMulBiasReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64))
        self.bias   = nn.Parameter(torch.randn(64))

    def forward(self, x):
        out = x @ self.weight
        out = out + self.bias
        out = torch.relu(out)
        return out


model = MatMulBiasReLU().eval()
compiled_model = torch.compile(model, backend="tritonflow")

x      = torch.randn(64, 64)
output = compiled_model(x)
