# Deep Learning Network Definition
class FusedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(64, 32))  # Layer 1 Weight
        self.w2 = nn.Parameter(torch.randn(16, 64))  # Layer 2 Weight
    def forward(self, x):
        h = F.linear(x, self.w1)                     # 1st Matrix Multiply
        a = F.relu(h)                                # Pointwise Epilogue
        return F.linear(a, self.w2)                  # 2nd Matrix Multiply
