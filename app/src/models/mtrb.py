import torch
import torch.nn as nn
import math
from typing import Dict, Tuple

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
        weights = torch.softmax(attn_logits / (self.temperature + 1e-8), dim=1)
        bag_repr = torch.sum(weights * x, dim=1)
        return bag_repr, weights

class MultiMTRBClassifier(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, temperature: float = 0.5, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.pos_encoder = PositionalEncoding(input_dim)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.pos_encoder(x)
        head_outputs, head_weights = [], []

        for head in self.heads:
            repr_h, weight_h = head(x)
            head_outputs.append(repr_h)
            head_weights.append(weight_h)

        # Calculate Entropy Loss: -sum(p * log(p))
        # head_weights shape is [n_heads, batch, seq_len, 1]
        all_weights = torch.stack(head_weights) 
        # Average across heads for a single distribution, or calculate per head
        avg_weights = all_weights.mean(dim=0).squeeze(-1) # [batch, seq_len]

        # Entropy calculation (add epsilon to avoid log(0))
        entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8), dim=1)

        combined_repr = torch.cat(head_outputs, dim=-1)
        projected = self.output_projection(combined_repr)
        logits = self.classifier(projected)

        return logits, avg_weights, {"entropy": entropy.mean(), "gate": torch.tensor(0.0).to(x.device)}

