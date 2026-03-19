from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve, auc
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from src.model import MultiMTRBClassifier
from src.data import MultiMTRBDataset
from src.utils import get_logger, settings

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class Trainer:
    def __init__(self, device: str, features_dir: Path, output_dir: Path, overrides: Optional[Dict[str, Any]] = None):
        self.logger = get_logger().bind(module="mtrb_trainer")
        self.device = torch.device(device)
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.overrides = overrides or {}

    def get_balanced_loader(self, dataset: MultiMTRBDataset, labels: np.ndarray):
        """Creates a DataLoader with WeightedRandomSampler for class balance."""
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in labels])

        weights_list = samples_weight.tolist() 
 
        sampler = WeightedRandomSampler(weights_list, len(weights_list))

        return DataLoader(
            dataset,
            batch_size=settings.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )

    def run_fold(self, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, save_model: bool = False) -> float:
        train_ds = MultiMTRBDataset(self.features_dir, train_df, max_seq=settings.token_size) # type: ignore
        val_ds = MultiMTRBDataset(self.features_dir, val_df, max_seq=settings.token_size)     # type: ignore

        label_col = next(c for c in train_df.columns if 'phq' in c.lower() and 'binary' in c.lower())

        train_labels = np.asarray(train_df[label_col].values, dtype=np.int64)

        train_loader = self.get_balanced_loader(train_ds, train_labels)
        val_loader = DataLoader(val_ds, batch_size=settings.batch_size, shuffle=False)

        model = MultiMTRBClassifier(
            input_dim=settings.input_dim,
            hidden_dim=int(self.overrides.get("hidden_dim", 256)),
            temperature=float(self.overrides.get("temperature", 0.07)),
            n_heads=int(self.overrides.get("n_heads", 8))
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.overrides.get("learning_rate", 1e-4)),
            weight_decay=0.01
        )

        criterion = FocalLoss(alpha=0.75) 
        scaler = GradScaler(enabled=(self.device.type == 'cuda'))

        best_auprc = 0.0

        for epoch in range(1, settings.epochs + 1):
            model.train()
            total_loss = 0

            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).unsqueeze(1)

                optimizer.zero_grad()
                with autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                    logits, _, expert_losses = model(x)
                    ce_loss = criterion(logits, y)
                    e_lambda = float(self.overrides.get("entropy_lambda", 0.02))
                    loss = ce_loss + e_lambda * expert_losses["entropy"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            # Validation
            model.eval()
            all_probs, targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    with autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                        logits, _, _ = model(x.to(self.device, non_blocking=True))
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
                    targets.extend(y.numpy().astype(int).flatten().tolist())

            prec, rec, _ = precision_recall_curve(targets, all_probs)
            cur_auprc = auc(rec, prec)

            if cur_auprc > best_auprc:
                best_auprc = cur_auprc
                if save_model:
                    torch.save(model.state_dict(), self.output_dir / f"mtrb_fold_{fold_idx}.pt")

            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    f"Fold {fold_idx} | Epoch {epoch:03d}", 
                    loss=round(total_loss/len(train_loader), 4), 
                    auprc=round(cur_auprc, 4), 
                    best=round(best_auprc, 4)
                )

        return best_auprc

