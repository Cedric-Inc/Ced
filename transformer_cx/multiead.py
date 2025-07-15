import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_cx.utils import ModelArg, args


class MultiHead(nn.Module):
    def __init__(self, args: ModelArg, is_casual: bool):
        super().__init__()

        feature_dim = args.dim
        self.n_heads = args.n_head
        self.head_dim = args.head_dim

        self.wq = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)
        self.wk = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)
        self.wv = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

        self.is_casual = is_casual
        seq_len = args.seq_len
        if is_casual:
            mask = torch.full((1, 1, seq_len, seq_len), -float('inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        batch_size, seq_len, dim = q.size()

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)


        # iscasual
        if self.is_casual:
            assert hasattr(self, 'mask')
            # print(self.mask)
            score = score+self.mask[:, :, :seq_len, :seq_len]

        score = score.softmax(-1)

        score = self.dropout(score)

        out = torch.matmul(score, v)

        out = out.transpose(1,2)
        out = out.reshape(batch_size, seq_len, dim)

        out = self.resid_dropout(out)
        out = self.wo(out)
        return out



# batch_size = 8
# seq_len = 24
# n_dim = 256
#
#
# x = torch.randn(batch_size, seq_len, n_dim)
#
# mth_attn = MultiHead(args=args, is_casual=True)
# z = mth_attn.forward(x)
#
# print(z)
