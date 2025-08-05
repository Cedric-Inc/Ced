import torch
import torch.nn as nn
import math
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        res = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return res
    def forward(self, x: torch.Tensor):
        res = self._norm(x.float()).type_as(x)
        return res * self.weight


# norm = RMSNorm(768, 1e-6)
# x = torch.randn(1, 50, 768)
# output = norm(x)
# print(output.shape)
