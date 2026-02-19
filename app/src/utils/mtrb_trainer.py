import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Any, Optional
from pytorch_metric_learning import losses 

from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import get_logger

logger = get_logger().bind(module="mtrb_trainer")


class FocalLoss(nn.Module):
    """
    Focal Loss helps the model focus on 'hard' samples (Phase 3).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class MTRBTrainer:
    def __init__(self, device: str, features_dir: Path, output_dir: Path, overrides: Optional[Dict[str, Any]] = None):
        self.device = torch.device(device)
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.overrides = overrides or {}

        # Supervised Contrastive Loss to separate Healthy vs Depressed clusters
        self.cont_loss_func = losses.SupConLoss(temperature=0.1)

    def _get_model(self) -> MultiMTRBClassifier:
        """Initializes the model with Phase 2/4 Multi-Head support."""
        return MultiMTRBClassifier(
            input_dim=Config.INPUT_DIM,
            hidden_dim=int(self.overrides.get("hidden_dim", 256)),
            temperature=float(self.overrides.get("temperature", 0.5)),
            n_heads=int(self.overrides.get("n_heads", 4))
        ).to(self.device)

    def _compute_entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Prevents attention collapse by encouraging diversity in focus."""
        p = weights.squeeze(-1) + 1e-8
        entropy = -torch.sum(p * torch.log(p), dim=1)
        return -torch.mean(entropy)

    def run_fold(self, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, save_model: bool = True) -> float:
        tmp_train, tmp_val = self.output_dir / f"t_{fold_idx}.csv", self.output_dir / f"v_{fold_idx}.csv"
        train_df.to_csv(tmp_train, index=False)
        val_df.to_csv(tmp_val, index=False)

        train_ds = MultiMTRBDataset(self.features_dir, tmp_train, max_seq=Config.MAX_SEQ_LEN)
        val_ds = MultiMTRBDataset(self.features_dir, tmp_val, max_seq=Config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = self._get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.overrides.get("learning_rate", 1e-4)))

        # Focal Loss
        criterion = FocalLoss(
            alpha=float(self.overrides.get("alpha", 0.25)),
            gamma=float(self.overrides.get("gamma", 2.0))
        )

        # Dynamic Lambda Injection from Random Search
        entropy_lambda = float(self.overrides.get("entropy_lambda", 0.015))
        contrastive_lambda = float(self.overrides.get("contrastive_lambda", 0.05))

        best_auprc = 0.0

        for epoch in range(Config.MAX_EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device).view(-1, 1)
                optimizer.zero_grad()

                logits, weights, bag_repr = model(x)

                class_loss = criterion(logits, y)
                ent_loss = self._compute_entropy_loss(weights)
                cont_loss = self.cont_loss_func(bag_repr, y.flatten())

                sharp_loss = torch.norm(weights, p=1) 

                sharpness_lambda = float(self.overrides.get("sharpness_lambda", 0.005))

                loss = class_loss + (entropy_lambda * ent_loss) + \
                       (contrastive_lambda * cont_loss) + (sharpness_lambda * sharp_loss)

                loss.backward()
                optimizer.step()

            model.eval()
            all_probs, targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    logits, _, _ = model(x.to(self.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
                    all_probs.extend(probs)
                    targets.extend(y.numpy().astype(int).flatten().tolist())

            # Monitoring: Optimize for AUPRC
            precisions, recalls, _ = precision_recall_curve(targets, all_probs)
            current_auprc = auc(recalls, precisions)

            if current_auprc > best_auprc:
                best_auprc = current_auprc
                if save_model: 
                    torch.save(model.state_dict(), self.output_dir / f"mtrb_fold_{fold_idx}.pt")

        # Cleanup temporary files
        tmp_train.unlink()
        tmp_val.unlink()
        return float(best_auprc)

