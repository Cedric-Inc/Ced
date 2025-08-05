from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer的层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 50, # 最大序列长度
            dropout: float = 0.1, # dropout概率
            flash_attn: bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

args = ModelConfig()


def repeat_kv(args: ModelConfig, kv: torch.Tensor):
    n_repeat = int(args.n_heads / args.n_kv_heads)
    n_batch, seq_len, n_kv_head, kv_dim = kv.shape

    if n_repeat == 1:
        return kv

    kv = kv[:, :, :, None,:]
    kv = kv.expand(n_batch, seq_len, n_kv_head, n_repeat, kv_dim)
    kv = kv.reshape(n_batch, seq_len, n_kv_head * n_repeat, kv_dim)

    return kv


def RoPE_core(seq_len: int, dim: int):
    freqs = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    t = torch.arange(seq_len)

    core = torch.outer(t, freqs)

    res_2k = torch.cos(core)
    res_2k1 = torch.sin(core)

    return res_2k, res_2k1


def reshape4broadcast(x: torch.Tensor, freq: torch.Tensor):
    shape = [freq.shape[0] if i == 1 else 1 for i in range(x.ndim)]
    shape[-1] = freq.shape[-1]
    return freq.view(shape)


def apply_RoPE(x: torch.Tensor, f_sin: torch.Tensor, f_cos: torch.Tensor):
    real, imag = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

    f_sin = reshape4broadcast(x=real, freq=f_sin)
    f_cos = reshape4broadcast(x=real, freq=f_cos)

    real_ = real * f_sin - imag * f_cos
    imag_ = real * f_cos + imag * f_sin

    y = torch.stack([real_, imag_], dim=-1).flatten(3)

    return y


class llama_attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        self.head_dim = args.dim//args.n_heads
        self.args = args
        self.wq = nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)
        self.wk = nn.Linear(in_features=args.dim, out_features=self.head_dim * args.n_kv_heads, bias=False)
        self.wv = nn.Linear(in_features=args.dim, out_features=self.head_dim * args.n_kv_heads, bias=False)

        self.w0 = nn.Linear(in_features=args.dim, out_features=args.dim, bias=False)

        sin_freq, cos_freq = RoPE_core(args.max_seq_len, self.head_dim)

        self.register_buffer("sin_freq", sin_freq, persistent=False)
        self.register_buffer("cos_freq", cos_freq, persistent=False)

        self.flash_attn = args.flash_attn
        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)


    def forward(self, x: torch.Tensor):

        q = self.wq(x)  # (8, 13, 256)
        k = self.wk(x)  # (8, 13, 128)
        v = self.wv(x)  # (8, 13, 128)
        n_batch, seq_len, q_dim = q.size()

        q = q.reshape(n_batch, seq_len, self.args.n_heads, self.head_dim)  # (8, 13, 8, 32)
        k = k.reshape(n_batch, seq_len, self.args.n_kv_heads, self.head_dim)  # (8, 13, 4, 32)
        v = v.reshape(n_batch, seq_len, self.args.n_kv_heads, self.head_dim)

        q = apply_RoPE(q, self.sin_freq, self.cos_freq)  # (8, 13, 8, 32)
        k = apply_RoPE(k, self.sin_freq, self.cos_freq)  # (8, 13, 4, 32)

        k = repeat_kv(args=args, kv=k)  # (8, 13, 8, 32)
        v = repeat_kv(args=args, kv=v)  # (8, 13, 8, 32)

        # 交换 seq & head
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        output = output.transpose(1, 2).contiguous().view(n_batch, seq_len, -1)

        output = self.w0(output)
        output = self.resid_dropout(output)

        return output


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

from rmsnorm import RMSNorm

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads

        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads

        self.attention = llama_attention(args)

        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        self.layer_id = layer_id

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)



    def forward(self, x):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
import math

