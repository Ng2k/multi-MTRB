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
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, temperature: float = 0.5, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.pos_encoder = PositionalEncoding(input_dim)

        # Multi-Head initialization
        self.heads = nn.ModuleList([
            RethinkingBlock(input_dim, hidden_dim, temperature) 
            for _ in range(n_heads)
        ])

        self.output_projection = nn.Linear(input_dim * n_heads, input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) 
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pos_encoder(x)
        head_outputs, head_weights = [], []

        for head in self.heads:
            bag_repr, weights = head(x)
            head_outputs.append(bag_repr)
            head_weights.append(weights)

        combined_repr = torch.cat(head_outputs, dim=-1)
        bag_repr = self.output_projection(combined_repr)
        logits = self.classifier(bag_repr)
        avg_weights = torch.mean(torch.stack(head_weights), dim=0)

        return logits, avg_weights, bag_repr

