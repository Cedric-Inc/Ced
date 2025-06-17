import torch.nn as nn
import torch
from dataclasses import dataclass
import torch.nn.functional as F
import math


@dataclass
class ModelArgs:
    n_heads: int = 8
    n_embd: int = 128
    n_dropout: float = 0.1
    dim: int = 128
    is_casual: bool = True
    head_dim: int = 16
    max_seq_len: int = 13

class Multi_Head_att(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.head_dim == args.n_embd / args.n_heads

        model_paralel_size = 1

        self.n_local_heads = args.n_heads // model_paralel_size

        self.head_dim = args.head_dim

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)

        self.wo = nn.Linear(args.dim, args.dim, bias=False) 

        self.dropout = nn.Dropout(args.n_dropout)

        self.resid_dropout = nn.Dropout(args.n_dropout)

        self.is_casual = args.is_casual
        if self.is_casual:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        batch_size, seq_len, n_dim = q.size()

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算Attention
        scores = torch.matmul(q, k.transpose(2, 3))/ math.sqrt(self.head_dim)

        if self.is_casual:
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seq_len, :seq_len]

        scores = scores.softmax(-1) 
        # scores = torch.softmax(scores.float(), dim=-1).type_as(xq)  # 更健壮写法，检查了数值类型，可用于混合精度训练

        scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        output = output.transpose(1,2)
        output = output.reshape(batch_size, seq_len, n_dim) 
        # output = output.contiguous().view(batch_size, seq_len, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


batch_size = 5
seq_len = 13
n_dim = 128


q = torch.randn(batch_size, seq_len, n_dim)
k = torch.randn(batch_size, seq_len, n_dim)
v = torch.randn(batch_size, seq_len, n_dim)

args = ModelArgs()
mth_attn = Multi_Head_att(args=args)
z = mth_attn.forward(q, k, v)

print(z)