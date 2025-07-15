import torch
import torch.nn as nn
import torch.nn.functional as F
from multiead import MultiHead
from utils import ModelArg, args
from LayerNorm import LayerNorm
from MLP import MLP
from positional import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()


        self.masked_attn = MultiHead(args=args, is_casual=True)
        self.norm1 = LayerNorm(feature=args.dim)

        self.multihead = MultiHead(args=args, is_casual=False)
        self.norm2 = LayerNorm(feature=args.dim)

        self.ffn = MLP(input_dim=args.dim, hidden_dim=args.dim*4, dropout=args.dropout)
        self.norm3 = LayerNorm(feature=args.dim)


    def forward(self, x, enc_out):
        mask_attn = self.masked_attn(x, x, x)
        x = self.norm1(x+mask_attn)

        cross = self.multihead(enc_out, enc_out, x)
        x = self.norm2(x + cross)

        fnn_x = self.ffn(x)
        x = self.norm3(x+fnn_x)
        return x


class Decoder(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.pos_enco = PositionalEncoding(args=args)
        self.module_list = nn.ModuleList([DecoderLayer(args=args) for _ in range(6)])
        self.norm = LayerNorm(feature=args.dim)

    def forward(self, x, enc_out):
        x = self.pos_enco(x)
        for module in self.module_list:
            x = module(x, enc_out)
        return self.norm(x)