import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Any, Optional
from pytorch_metric_learning import losses 

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import get_logger


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class MTRBTrainer:
    def __init__(self, device: str, features_dir: Path, output_dir: Path, overrides: Optional[Dict[str, Any]] = None):
        self.logger = get_logger().bind(module="mtrb_trainer")
        self.device = torch.device(device)
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.overrides = overrides or {}
        self.cont_loss_func = losses.SupConLoss(temperature=0.1)

        self.scaler = GradScaler(device='cuda') if self.device.type == 'cuda' else None

    def _get_model(self) -> MultiMTRBClassifier:
        return MultiMTRBClassifier(
            input_dim=Config.INPUT_DIM,
            hidden_dim=int(self.overrides.get("hidden_dim", 256)),
            temperature=float(self.overrides.get("temperature", 0.1)),
            n_heads=int(self.overrides.get("n_heads", 4))
        ).to(self.device)

    def _compute_entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        p = weights.squeeze(-1) + 1e-8
        entropy = -torch.sum(p * torch.log(p), dim=1)
        return -torch.mean(entropy)

    def run_fold(self, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, save_model: bool = True) -> float:
        t_id = self.overrides.get('trial_idx', '0')
        tmp_train = self.output_dir / f"trial_{t_id}_fold_{fold_idx}_train.csv"
        tmp_val = self.output_dir / f"trial_{t_id}_fold_{fold_idx}_val.csv"

        train_df.to_csv(tmp_train, index=False)
        val_df.to_csv(tmp_val, index=False)

        train_ds = MultiMTRBDataset(self.features_dir, tmp_train, max_seq=Config.MAX_SEQ_LEN)
        val_ds = MultiMTRBDataset(self.features_dir, tmp_val, max_seq=Config.MAX_SEQ_LEN)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True 
        )
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True)

        model = self._get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.overrides.get("learning_rate", 1e-4)))
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        ent_lam = float(self.overrides.get("entropy_lambda", 0.015))
        cont_lam = float(self.overrides.get("contrastive_lambda", 0.05))
        sharp_lam = float(self.overrides.get("sharpness_lambda", 0.005))
        gate_lam = float(self.overrides.get("gate_lambda", 0.02))

        best_auprc = 0.0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).view(-1, 1)
                optimizer.zero_grad(set_to_none=True) 

                with autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                    logits, weights, bag_repr = model(x)
                    class_loss = criterion(logits, y)
                    e_loss = self._compute_entropy_loss(weights)
                    c_loss = self.cont_loss_func(bag_repr, y.flatten())
                    s_loss = torch.mean(torch.norm(weights, p=1, dim=1))
                    peak_loss = torch.mean(torch.max(weights, dim=1)[0])

                    loss = class_loss + (ent_lam * e_loss) + (cont_lam * c_loss) + \
                           (sharp_lam * s_loss) + (gate_lam * peak_loss)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

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
            epoch_duration = time.time() - epoch_start

            if cur_auprc > best_auprc:
                best_auprc = cur_auprc
                if save_model: torch.save(model.state_dict(), self.output_dir / f"mtrb_fold_{fold_idx}.pt")

            # Periodic logging to avoid overhead
            if epoch % 10 == 0 or epoch == Config.MAX_EPOCHS or epoch == 1:
                self.logger.info(
                    f"Trial {self.overrides['trial_idx']} | Fold {fold_idx} | Epoch {epoch:03d}", 
                    loss=f"{total_loss/len(train_loader):.4f}", 
                    auprc=f"{cur_auprc:.4f}",
                    best=f"{best_auprc:.4f}",
                    sec_per_epoch=f"{epoch_duration:.3f}"
                )

        if tmp_train.exists(): tmp_train.unlink()
        if tmp_val.exists(): tmp_val.unlink()

        return float(best_auprc)

