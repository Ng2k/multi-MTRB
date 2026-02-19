import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Dict, Any, Optional
from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import get_logger

logger = get_logger().bind(module="mtrb_trainer")

class MTRBTrainer:
    def __init__(self, device: str, features_dir: Path, output_dir: Path, overrides: Optional[Dict[str, Any]] = None):
        self.device = torch.device(device)
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.overrides = overrides or {}


    def _get_model(self) -> MultiMTRBClassifier:
        return MultiMTRBClassifier(
            input_dim=Config.INPUT_DIM,
            hidden_dim=int(self.overrides.get("hidden_dim", 256)),
            temperature=float(self.overrides.get("temperature", 0.5))
        ).to(self.device)


    def run_fold(self, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, save_model: bool = True) -> float:
        tmp_train, tmp_val = self.output_dir / f"t_{fold_idx}.csv", self.output_dir / f"v_{fold_idx}.csv"
        train_df.to_csv(tmp_train, index=False); val_df.to_csv(tmp_val, index=False)

        train_ds = MultiMTRBDataset(self.features_dir, tmp_train, max_seq=Config.MAX_SEQ_LEN)
        val_ds = MultiMTRBDataset(self.features_dir, tmp_val, max_seq=Config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = self._get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.overrides.get("learning_rate", 1e-4)))
        p_weight = torch.tensor([float(self.overrides.get("pos_weight", 1.0))]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight)

        best_f1 = 0.0
        for epoch in range(Config.MAX_EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device).view(-1, 1)
                optimizer.zero_grad()
                logits, _, _ = model(x)
                loss = criterion(logits, y)
                loss.backward(); optimizer.step()

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    logits, _, _ = model(x.to(self.device))
                    preds.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy().flatten().tolist())
                    targets.extend(y.numpy().astype(int).flatten().tolist())

            f1 = float(f1_score(targets, preds, zero_division=0.0)) # type: ignore
            if f1 > best_f1:
                best_f1 = f1
                if save_model: torch.save(model.state_dict(), self.output_dir / f"mtrb_fold_{fold_idx}.pt")

        tmp_train.unlink(); tmp_val.unlink()
        return best_f1

