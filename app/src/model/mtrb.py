import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.model.positional_encoding import PositionalEncoding
from src.model.rethinking_block import RethinkingBlock


class MultiMTRBClassifier(nn.Module):
    def __init__(self,
                 input_dim: int = 1280,
                 hidden_dim: int = 256,
                 temperature: float = 0.5,
                 n_heads: int = 4):
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

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor|None = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.pos_encoder(x)
        head_outputs, head_weights = [], []

        for head in self.heads:
            repr_h, weight_h = head(x, mask)
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

