import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import configure_logger, get_logger
from src.cleaning.parallel_processor import ParallelCleaner
from src.features.manager import FeatureManager
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train")


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits, _, _ = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            preds_list.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
            labels_list.extend(y.int().cpu().numpy())
    return val_loss / len(loader), f1_score(labels_list, preds_list)


def train_fold(fold_idx, train_df, val_df, feat_dir, model_path, device):
    # Save temporary CSVs for the dataset loader
    tmp_train = Path(f"../dataset/tmp_train_fold_{fold_idx}.csv")
    tmp_val = Path(f"../dataset/tmp_val_fold_{fold_idx}.csv")
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val, index=False)

    train_ds = MultiMTRBDataset(feat_dir, tmp_train)
    val_ds = MultiMTRBDataset(feat_dir, tmp_val)

    # SANITY CHECK:
    logger.info(f"Fold {fold_idx} - Training on {len(train_ds)} samples, Validating on {len(val_ds)} samples")
    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty! Check your FEAT_DIR and CSV paths.")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = MultiMTRBClassifier().to(device)
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_fold_f1, patience_counter = 0, 0

    for epoch in range(1, 101):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            logits, _, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        v_loss, v_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        if v_f1 > best_fold_f1:
            best_fold_f1 = v_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= 25: break

    tmp_train.unlink(); tmp_val.unlink()
    return best_fold_f1


if __name__ == "__main__":
    # 1. PATHS & CONFIGS
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RAW_DIR = Path(os.getenv("DATASET_RAW_DIR", "../dataset/raw"))
    CLEAN_DIR = Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean"))
    FEAT_DIR = Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features"))
    TRAIN_CSV = Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv"))
    DEV_CSV = Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv"))
    MODEL_DIR = Path(os.getenv("MODEL_DIR", "../outputs"))

    # 2. RUN PIPELINE (CLEANING & EXTRACTION)
    cleaner = ParallelCleaner()
    cleaner.run(RAW_DIR, CLEAN_DIR)
 
    manager = FeatureManager(CLEAN_DIR, FEAT_DIR)
    manager.process_all()

    # 3. K-FOLD SETUP
    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.strip() for c in df.columns]
    y = df['PHQ8_Binary'].values # For Stratification

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    logger.info("Starting 5-Fold Stratified Cross-Validation")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        logger.info(f"--- FOLD {fold+1}/5 ---")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        fold_model_path = MODEL_DIR / f"mtrb_fold_{fold+1}.pt"

        f1 = train_fold(fold, train_df, val_df, FEAT_DIR, fold_model_path, DEVICE)
        fold_results.append(f1)
        logger.info(f"Fold {fold+1} Result", f1=round(f1, 4))

    # 4. FINALIZING
    avg_f1 = np.mean(fold_results)
    logger.info("CV Finished", avg_f1=round(avg_f1, 4), f1_list=fold_results)

    # Save the absolute best fold for the Evaluation script to use
    best_fold_idx = np.argmax(fold_results)
    best_model_data = torch.load(MODEL_DIR / f"mtrb_fold_{best_fold_idx+1}.pt")
    torch.save(best_model_data, "../outputs/mtrb_model.pt")
    logger.info(f"Production model saved using weights from Fold {best_fold_idx+1}")

