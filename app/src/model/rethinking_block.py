from typing import Tuple, Optional
import torch
import torch.nn as nn


class RethinkingBlock(nn.Module):
    """Attention Head parameterized for search-driven sharpness."""
    def __init__(self, input_dim: int, attn_dim: int, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))
        self.layer_norm = nn.LayerNorm(input_dim)

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.layer_norm(x)
        attn_logits = self.attention(x_norm)

        temp = torch.clamp(self.temperature, min=0.01)
        scaled_logits = attn_logits / temp

        if mask is not None:
            # Mask should be [Batch, Seq, 1]
            scaled_logits = scaled_logits.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scaled_logits, dim=1)
        bag_repr = torch.sum(weights * x, dim=1)
        return bag_repr, weights

