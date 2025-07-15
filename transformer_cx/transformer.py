import torch
import torch.nn as nn
import torch.nn.functional as F
from Decoder import Decoder
from Encoder import Encoder
from utils import ModelArg, args
from torch.utils.data import DataLoader, TensorDataset


class transformer(nn.Module):
    def __init__(self, args: ModelArg):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.linear = nn.Linear(in_features=args.dim, out_features=args.vocab_size, bias=False)
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, src_idx, tgt_idx=None, target=None):

        x = self.embedding(src_idx)
        enc_out = self.encoder(x)

        if tgt_idx is None:
            # 若不传 tgt_idx，就默认预测下一个（语言建模）
            tgt_idx = src_idx

        tgt_emb = self.embedding(tgt_idx)
        deco_out = self.decoder(x=tgt_emb,enc_out=enc_out)
        logits = self.linear(deco_out)
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
            return logits, loss
        else:
            probs = F.softmax(logits)
            return probs

if __name__ == '__main__':
    # src = torch.randint(0, args.vocab_size, (args.n_batch, args.seq_len))
    # tgt = torch.randint(0, args.vocab_size, (args.n_batch, args.seq_len))
    #
    # tf = transformer(args)
    # logits, loss = tf(src, tgt, targets=tgt)
    #
    # print("logits:", logits.shape)  # [batch, seq_len, vocab_size]
    # print("loss:", loss)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformer(args).to(device)

    # ===== Dummy 数据构造 =====
    # 假设你只是测试能不能训练通
    num_samples = 100
    x_data = torch.randint(0, args.vocab_size, (num_samples, args.seq_len))
    y_data = x_data.clone()  # 自回归语言模型：预测下一个

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=args.n_batch, shuffle=True)

    # ===== 优化器 & 损失 =====
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # ===== 正式训练 =====
    for epoch in range(3):
        total_loss = 0
        model.train()
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            logits, loss = model(src_idx=src, tgt_idx=src, target=tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/3 | Loss: {avg_loss:.4f}")