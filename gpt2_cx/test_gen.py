import torch
import torch.nn.functional as F
from transformer_cx.utils import ModelArg, args
from gpt2_cx.gpt2 import GPT2


device = 'cuda' if torch.cuda.is_available() else 'cpu'
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    从 input_ids 开始生成新 token，直到达到 max_new_tokens。
    - input_ids: [1, T] 的 token 序列（通常由 tokenizer.encode 返回）
    - 返回: 生成后的完整序列 [1, T+N]
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 只保留最近 args.seq_len 个 token（防止超长）
        input_crop = input_ids[:, -args.seq_len:]

        # 前向传播
        logits = model(input_crop)  # [1, T, vocab]
        logits = logits[:, -1, :]   # 取最后一个 token 的输出 [1, vocab]
        logits = logits / temperature

        if top_k is not None:
            # top-k 筛选（截断低概率）
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_val, torch.full_like(logits, -float('Inf')), logits)

        probs = F.softmax(logits, dim=-1)  # [1, vocab]
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # 拼接到输入序列
        input_ids = torch.cat((input_ids, next_token), dim=1)  # [1, T+1]

    return input_ids

from transformers import GPT2Tokenizer

# 加载 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 准备 prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 加载模型
model = GPT2(args).to(device)
model.load_state_dict(torch.load("gpt2_trained.pth"))

# 生成
output_ids = generate(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=50)

# 解码
print(tokenizer.decode(output_ids[0]))
