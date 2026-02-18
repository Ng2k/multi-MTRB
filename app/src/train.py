import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.utils.logger import configure_logger, get_logger
from src.cleaning.parallel_processor import ParallelCleaner
from src.features.manager import FeatureManager
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train")

def validate(model, loader, criterion, device):
    """Runs a validation pass."""
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

if __name__ == "__main__":
    # 1. PATHS
    RAW_DIR = Path(os.getenv("DATASET_RAW_DIR", "../data/raw"))
    CLEAN_DIR = Path(os.getenv("DATASET_CLEAN_DIR", "../data/clean"))
    FEAT_DIR = Path(os.getenv("DATASET_FEATURES_DIR", "../data/features"))
    TRAIN_CSV = Path(os.getenv("TRAIN_SPLIT", "../data/train_split_Depression_AVEC2017.csv"))
    DEV_CSV = Path(os.getenv("DEV_SPLIT", "../data/dev_split_Depression_AVEC2017.csv"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "../outputs/mtrb_model.pt"))

    # 2. PRE-PROCESSING and FEATURE EXTRACTION
    cleaner = ParallelCleaner()
    cleaner.run(RAW_DIR, CLEAN_DIR)
 
    manager = FeatureManager(CLEAN_DIR, FEAT_DIR)
    manager.process_all()

    # 3. DATA LOADERS
    train_ds = MultiMTRBDataset(FEAT_DIR, TRAIN_CSV)
    dev_ds = MultiMTRBDataset(FEAT_DIR, DEV_CSV)
 
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False)

    # 4. MODEL & OPTIMIZER
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiMTRBClassifier().to(device)
 
    # Handle Class Imbalance: weight the positive class (Depressed) 3x more
    pos_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 5. TRAINING LOOP
    best_f1 = 0
    logger.info("Starting Multi-MTRB Training with Validation")

    for epoch in range(1, 101):
        model.train()
        train_loss = 0
 
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
 
            logits, _, _ = model(batch_x)
            loss = criterion(logits, batch_y)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        v_loss, v_f1 = validate(model, dev_loader, criterion, device)
 
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"New Best Model", epoch=epoch, f1=round(v_f1, 4))

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}", t_loss=round(train_loss/len(train_loader), 4), v_f1=round(v_f1, 4))

    logger.info("Training complete. Best Validation F1:", best_f1=round(best_f1, 4))

