import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_cx.utils import args, ModelArg
import math

# def positional_encoding(args: ModelArg):
#     seq_len = args.seq_len
#     dim = args.dim
#
#     pos = torch.zeros(seq_len, dim)
#     position = torch.arange(0, seq_len).unsqueeze(1)
#
#     en_core = torch.exp(torch.arange(0, seq_len, 2)*(-2)/dim * math.log(10000))
#
#     pos[:, 0::2] = torch.sin(position * en_core)
#     pos[:, 1::2] = torch.cos(position * en_core)
#
#     return pos.unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        seq_len = args.seq_len
        dim = args.dim

        pos = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len).unsqueeze(1)

        en_core = torch.exp(torch.arange(0, dim, 2) * (-2) / dim * math.log(10000))

        pos[:, 0::2] = torch.sin(position * en_core)
        pos[:, 1::2] = torch.cos(position * en_core)

        self.register_buffer('pos', pos)

    def forward(self, x: torch.Tensor):
        pos_embed = self.pos[:x.size(1), :].unsqueeze(0)  # [1, seq_len, dim]
        x = x + pos_embed
        return self.dropout(x)

# batch_size = 8
# seq_len = 24
# n_dim = 256
#
#
# x = torch.randn(batch_size, seq_len, n_dim)
# pos = PositionalEncoding(args)
# x = pos(x)
# print(x.shape)