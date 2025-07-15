import torch
import torch.nn as nn
from utils import ModelArg, args
from multiead import MultiHead
from LayerNorm import LayerNorm
from MLP import MLP
from positional import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.multi_head = MultiHead(args=args, is_casual=False)

        dim = args.dim
        self.norm1 = LayerNorm(feature=dim)
        self.norm2 = LayerNorm(feature=dim)

        self.mlp = MLP(dim, dim*4, args.dropout)

    def forward(self, x: torch.Tensor):

        attn = self.multi_head(x, x, x)
        x = self.norm1(x + attn)

        ffn_x = self.mlp(x)
        x = self.norm2(ffn_x + x)
        return x


class Encoder(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.model_list = nn.ModuleList([EncoderLayer(args) for _ in range(8)])
        self.norm = LayerNorm(feature=args.dim)
        self.pos_enco = PositionalEncoding(args=args)
    def forward(self, x):

        x = self.pos_enco(x)
        for module in self.model_list:
            x = module(x)
        x = self.norm(x)
        return x