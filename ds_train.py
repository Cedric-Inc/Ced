import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PretrainedConfig, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from transformers.modeling_outputs import CausalLMOutput

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 4,
            n_heads: int = 8,
            n_kv_heads: int = 4,
            vocab_size: int = 50257,
            hidden_dim: int = 1024,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            flash_attn: bool = True,
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


# 模型组件
class RMSNorm(nn.Module):
    def __init__(self, dim: int = args.dim, ep: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.ep = ep

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.ep)
        return (x / rms) * self.g


class SwiGLUMLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        hidden_dim = int((2 * args.dim * 4) / 3)
        hidden_dim = (hidden_dim + args.multiple_of - 1) // args.multiple_of * args.multiple_of

        self.w0 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w1 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x):
        return self.w1(self.w0(x) * F.silu(self.w_gate(x)))


def repeat_kv(args: ModelConfig, kv: torch.Tensor):
    n_repeat = int(args.n_heads / args.n_kv_heads)
    n_batch, seq_len, n_kv_head, kv_dim = kv.shape

    if n_repeat == 1:
        return kv

    kv = kv[:, :, :, None, :]
    kv = kv.expand(n_batch, seq_len, n_kv_head, n_repeat, kv_dim)
    return kv.reshape(n_batch, seq_len, n_kv_head * n_repeat, kv_dim)


def RoPE_core(seq_len: int, dim: int):
    freqs = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    core = torch.outer(t, freqs)
    return torch.cos(core), torch.sin(core)


def reshape4broadcast(x: torch.Tensor, freq: torch.Tensor):
    shape = [1] * x.ndim
    shape[1] = freq.shape[0]
    shape[-1] = freq.shape[-1]
    return freq.view(shape)


def apply_RoPE(x: torch.Tensor, f_cos: torch.Tensor, f_sin: torch.Tensor):
    real, imag = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

    f_cos = reshape4broadcast(x=real, freq=f_cos)
    f_sin = reshape4broadcast(x=real, freq=f_sin)

    real_ = real * f_cos - imag * f_sin
    imag_ = real * f_sin + imag * f_cos

    return torch.stack([real_, imag_], dim=-1).flatten(3)


class Cedric_attention(nn.Module):
    def __init__(self, args: ModelConfig, is_casual=True):
        super().__init__()
        self.is_casual = is_casual
        self.head_dim = args.dim // args.n_heads
        self.args = args
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * args.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * args.n_kv_heads, bias=False)
        self.w0 = nn.Linear(args.dim, args.dim, bias=False)

        sin_freq, cos_freq = RoPE_core(args.max_seq_len, self.head_dim)
        self.register_buffer("sin_freq", sin_freq, persistent=False)
        self.register_buffer("cos_freq", cos_freq, persistent=False)

        self.flash_attn = args.flash_attn
        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        n_batch, seq_len, _ = q.size()

        q = q.reshape(n_batch, seq_len, self.args.n_heads, self.head_dim)
        k = k.reshape(n_batch, seq_len, self.args.n_kv_heads, self.head_dim)
        v = v.reshape(n_batch, seq_len, self.args.n_kv_heads, self.head_dim)

        q = apply_RoPE(q, self.cos_freq[:seq_len], self.sin_freq[:seq_len])
        k = apply_RoPE(k, self.cos_freq[:seq_len], self.sin_freq[:seq_len])

        k = repeat_kv(args=self.args, kv=k)
        v = repeat_kv(args=self.args, kv=v)

        q = q.transpose(1, 2)  # (B, n_heads, L, d)
        k = k.transpose(1, 2)  # (B, n_heads, L, d)
        v = v.transpose(1, 2)

        if self.flash_attn:
            if attention_mask is not None:
                attn_mask = attention_mask[:, None, None, :].to(q.dtype)  # broadcast
            else:
                attn_mask = None

            attn_scores = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_casual
            )
        else:
            score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                # (B, 1, 1, L) 方便 broadcast
                mask = attention_mask[:, None, None, :].to(dtype=score.dtype)
                score = score.masked_fill(mask == 0, float("-inf"))

            attn_scores = F.softmax(score, dim=-1)
            attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)
            attn_scores = torch.matmul(attn_scores, v)

        output = attn_scores.transpose(1, 2).contiguous().view(n_batch, seq_len, -1)
        output = self.w0(output)
        return self.resid_dropout(output)


class Transformer_Block(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(args.dim)
        self.mask_attention = Cedric_attention(args, is_casual=True)
        self.norm2 = RMSNorm(args.dim)
        self.mlp = SwiGLUMLP(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, attention_mask=None):

        x = x + self.mask_attention(self.norm1(x), attention_mask=attention_mask)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Cedric_Base(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim)
        # self.w = nn.Linear(args.dim, args.vocab_size, bias=False)
        # self.w.weight = self.token_emb.weight
        self.module_list = nn.ModuleList([Transformer_Block(args) for _ in range(args.n_layers)])

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.token_emb(input_ids)
        for module in self.module_list:
            x = module(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = F.linear(x, self.token_emb.weight)

        loss = None
        if labels is not None:
          # print('label is not None')
          shift_logits = logits[:, :-1, :].contiguous()
          shift_labels = labels[:, 1:].contiguous()
          loss = F.cross_entropy(
              shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1),
              ignore_index=-100
          )

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("dhruveshpatel/tiny_roc_stories", split="train[:10000]")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=args.max_seq_len,
    )


tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text", "source"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

model = Cedric_Base(args)

training_args = TrainingArguments(
    # output_dir="/content/drive/MyDrive/LLM/outputs",
    output_dir="/outputs",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

import time

t0 = time.time()

trainer.train()
print('cost:', time.time()-t0, 's')

trainer.save_model("/outputs/Ced_33M")
tokenizer.save_pretrained("/outputs/Ced_33M")

# from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub import login
login("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")

model.push_to_hub("UserCedric/Ced_33M")
tokenizer.push_to_hub("UserCedric/Ced_33M")
