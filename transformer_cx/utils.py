from dataclasses import dataclass

'''
设计：
序列长度：24
向量维度：256

batch数：8

head数：16
head_dim: 16

dropout: 0.1

'''


@dataclass
class ModelArg:
    dim: int = 256
    seq_len: int = 24
    n_batch: int = 8

    dropout: float = 0.1
    is_casual: bool = True

    n_head: int = 16
    head_dim: int = 16

    vocab_size = 50257


args = ModelArg()