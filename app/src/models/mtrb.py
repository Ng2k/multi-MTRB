import torch
import torch.nn as nn
import math
from typing import Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Seq, Dim]
        pe_tensor = getattr(self, 'pe')
        return x + pe_tensor[:, :x.size(1), :]

class RethinkingBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.layer_norm = nn.LayerNorm(input_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.layer_norm(x)
        attn_logits = self.attention(x_norm)
        weights = torch.softmax(attn_logits / self.temperature, dim=1)
        bag_repr = torch.sum(weights * x, dim=1)
        return bag_repr, weights

class MultiMTRBClassifier(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, temperature: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        self.mtrb = RethinkingBlock(input_dim, hidden_dim, temperature)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pos_encoder(x)
        bag_repr, weights = self.mtrb(x)
        logits = self.classifier(bag_repr)
        return logits, weights, bag_repr

