import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, feature: int, eps=1e-6):
        super().__init__()

        self.a = nn.Parameter(torch.ones(feature))
        self.b = nn.Parameter(torch.zeros(feature))

        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        output = (x - mean) / (std + self.eps)

        output = self.a * output + self.b

        return output


# a = torch.tensor([[1,2,3,4],
#                   [2,2,3,4],
#                   [5,6,7,8]], dtype=torch.float32)
#
# print(a.shape)
#
# norm = LayerNorm(feature=a.shape[-1])
# z = norm.forward(a)
#
# print(z)