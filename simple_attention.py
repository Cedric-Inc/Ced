import torch
import torch.nn.functional as F
import math

# 设置随机种子确保可复现
torch.manual_seed(42)

# 假设 batch_size=1，序列长度 seq_len=4，embedding_dim=6
batch_size = 1
seq_len = 4
d_k = 6  # Q, K 的维度

# 随机生成 Q, K, V 向量
Q = torch.randn(batch_size, seq_len, d_k)  # [1, 4, 6]
K = torch.randn(batch_size, seq_len, d_k)  # [1, 4, 6]
V = torch.randn(batch_size, seq_len, d_k)  # [1, 4, 6]

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

def attention_cal(query, key, value, dropout=None):
    k_v = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_v)
    
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    att = torch.matmul(p_attn, value)

    return att, p_attn

dropout = torch.nn.Dropout(p=0.1)

attn_score = attention_cal(Q, K, V)

print('x')