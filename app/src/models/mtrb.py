"""
"""

import torch
import torch.nn as nn


class RethinkingBlock(nn.Module):
    """MTRB: Attention-based Multiple Instance Pooling."""
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq, 1280]
        weights = self.attention(x) # [Batch, Seq, 1]
        weights = torch.softmax(weights, dim=1)
 
        bag_repr = torch.sum(weights * x, dim=1) 
        return bag_repr, weights


class MultiMTRBClassifier(nn.Module):
    """Multi-MTRB Classifier with Logits output."""
    def __init__(self, input_dim=1280):
        super().__init__()
        self.mtrb = RethinkingBlock(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout for better generalization
            nn.Linear(256, 1) # Removed Sigmoid: we use BCEWithLogitsLoss
        )

    def forward(self, x):
        bag_repr, weights = self.mtrb(x)
        logits = self.classifier(bag_repr)
        return logits, weights, bag_repr

