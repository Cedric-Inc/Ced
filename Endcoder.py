import torch.nn as nn
import torch
from dataclasses import dataclass
import torch.nn.functional as F
import math
from multi_head_attn import Multi_Head_att

@dataclass
class ModelArgs:
    n_heads: int = 8
    n_embd: int = 128
    n_dropout: float = 0.1
    dim: int = 128
    is_casual: bool = True
    head_dim: int = 16
    max_seq_len: int = 13


# 本章新内容
# 前馈神经网络
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.w0 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)

        self.w1 = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.dropout(self.w1(F.relu(self.w0(x))))
    

class Layer_Norm(nn.Module):
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
    

class EncoderLayer(nn.Module):
    def __init__(self, feature: int):
        super().__init__()

        args = ModelArgs()

        self.multi_head = Multi_Head_att(args)

        self.norm1 = Layer_Norm(feature=feature)
        self.norm2 = Layer_Norm(feature=feature)
                               
        self.mlp = MLP(input_dim=feature, hidden_dim=feature*4, dropout=args.n_dropout)

    def forward(self, x):
        attn_score = self.multi_head(x, x, x)
        attn_score = self.norm1(x + attn_score)

        x = self.mlp(attn_score)
        x = self.norm2(attn_score + x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        args = ModelArgs()
        self.Model_l = nn.ModuleList([EncoderLayer(args.dim) for _ in range(6)])
        self.norm = Layer_Norm(feature=args.dim)

    def forward(self, x):
        for encoder in self.Model_l:
            x = encoder(x)

        x = self.norm(x)
        return x

if __name__ == "__main__":
    batch_size = 4
    seq_len = 13
    dim = 128

    x = torch.randn(batch_size, seq_len, dim)

    encoder = Encoder()
    output = encoder(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)  # 应为 [4, 13, 128]
