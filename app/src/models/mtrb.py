import torch
import torch.nn as nn

class RethinkingBlock(nn.Module):
    """
    MTRB: Attention-based Multiple Instance Pooling.
    With Temperature Scaling to prevent 'Flat Attention' over 300+ sentences.
    """
    def __init__(self, input_dim=1280, hidden_dim=256, temperature=0.5):
        super().__init__()
        self.temperature = temperature
 
        # LayerNorm to fuse RoBERTa and mT5 features better. They have different scales
        self.layer_norm = nn.LayerNorm(input_dim)
 
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq, 1280]
        x = self.layer_norm(x)
 
        attn_logits = self.attention(x) # [Batch, Seq, 1]
 
        # Apply Temperature Scaling: 
        weights = torch.softmax(attn_logits / self.temperature, dim=1)
 
        # Weighted sum of instances
        bag_repr = torch.sum(weights * x, dim=1)
        return bag_repr, weights


class MultiMTRBClassifier(nn.Module):
    """
    Multi-MTRB Classifier.
    Adds a hidden layer and increased dropout to combat the 'False Alarm' bias.
    """
    def __init__(self, input_dim=1280):
        super().__init__()
        self.mtrb = RethinkingBlock(input_dim, temperature=0.5)
 
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),

            nn.LayerNorm(512), 
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)        )

    def forward(self, x):
        # bag_repr: [Batch, 1280], weights: [Batch, Seq, 1]
        bag_repr, weights = self.mtrb(x)
        logits = self.classifier(bag_repr)
        return logits, weights, bag_repr

