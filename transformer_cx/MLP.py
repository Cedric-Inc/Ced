import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ModelArg


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.w0 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)

        self.w1 = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.dropout(self.w1(F.relu(self.w0(x))))


