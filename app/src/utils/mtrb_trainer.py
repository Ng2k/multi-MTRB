import os
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from src.utils.logger import configure_logger, get_logger
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train_pipeline")


class MTRBTrainer:
    def __init__(self, device, feat_dir, model_dir):
        self.device = device
        self.feat_dir = feat_dir
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_criterion(self):
        # Weight slightly higher for the positive class (Depressed)
        pos_weight = torch.tensor([2.0]).to(self.device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def validate(self, model, loader, criterion):
        model.eval()
        val_loss, preds, labels = 0, [], []
 
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device).unsqueeze(1)
                logits, _, _ = model(x)
 
                val_loss += criterion(logits, y).item()
                preds.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
                labels.extend(y.int().cpu().numpy())
 
        return val_loss / len(loader), f1_score(labels, preds)

    def run_fold(self, fold_idx, train_df, val_df):
        logger.info(f"Starting Fold {fold_idx}", train_size=len(train_df), val_size=len(val_df))
 
        tmp_train, tmp_val = Path(f"tmp_tr_{fold_idx}.csv"), Path(f"tmp_vl_{fold_idx}.csv")
        train_df.to_csv(tmp_train, index=False); val_df.to_csv(tmp_val, index=False)

        try:
            train_loader = DataLoader(MultiMTRBDataset(self.feat_dir, tmp_train), batch_size=4, shuffle=True)
            val_loader = DataLoader(MultiMTRBDataset(self.feat_dir, tmp_val), batch_size=1, shuffle=False)

            model = MultiMTRBClassifier().to(self.device)
            criterion = self._get_criterion()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            best_f1, patience_counter = 0, 0
            fold_path = self.model_dir / f"mtrb_fold_{fold_idx}.pt"

            for epoch in range(1, 101):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device).unsqueeze(1)
                    logits, _, _ = model(bx)
                    loss = criterion(logits, by)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                v_loss, v_f1 = self.validate(model, val_loader, criterion)
                scheduler.step(v_loss)

                if v_f1 > best_f1:
                    best_f1 = v_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), fold_path)
                else:
                    patience_counter += 1
 
                if patience_counter >= 25: break

            return best_f1

        finally:
            # Ensures cleanup even if training crashes
            if tmp_train.exists(): tmp_train.unlink()
            if tmp_val.exists(): tmp_val.unlink()

