import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_cx.multiead import MultiHead
from transformer_cx.utils import ModelArg, args
from transformer_cx.LayerNorm import LayerNorm
from transformer_cx.positional import PositionalEncoding

class mlp4gpt2(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.w0 = nn.Linear(args.dim, args.dim*4, bias=False)
        self.act = nn.GELU()
        self.w1 = nn.Linear(args.dim*4, args.dim, bias=False)

    def forward(self, x):
        x = self.w1(self.act(self.w0(x)))
        return x


class Transformer_Block(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.norm1 = LayerNorm(args.dim)
        self.mask_attention = MultiHead(args, is_casual=True)
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm2 = LayerNorm(args.dim)
        self.mlp = mlp4gpt2(args)
        self.dropout2 = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        attn = self.dropout1(self.mask_attention(x, x, x))

        x = self.norm2(x+attn)
        x_ffn = self.dropout2(self.mlp(x))

        return x+x_ffn

class GPT2(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.token_emb = nn.Embedding(args.vocab_size, args.dim)
        self.positional = PositionalEncoding(args)
        self.norm = LayerNorm(args.dim)
        self.w = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.w.weight = self.token_emb.weight

        self.module_list = nn.ModuleList([Transformer_Block(args) for _ in range(6)])


    def forward(self, idx):

        x = self.token_emb(idx)
        x = self.positional(x)
        for module in self.module_list:
            x = module(x)
        x = self.norm(x)
        logits = self.w(x)
        return logits

# ========================
# Training
# ========================
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=args.seq_len):
        self.data = torch.randint(0, args.vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # input, target


def train():
    model = GPT2(args).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    dataloader = DataLoader(DummyDataset(), batch_size=8, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")
        print(f"[Epoch {epoch}] Average loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "gpt2_trained.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()