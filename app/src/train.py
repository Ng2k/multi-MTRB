import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import configure_logger, get_logger
from src.utils.mtrb_trainer import MTRBTrainer
from src.cleaning.parallel_processor import ParallelCleaner
from src.features.manager import FeatureManager

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train_pipeline")


def run_full_pipeline():
    # 1. Resource Initialization
    paths = {
        "raw": Path(os.getenv("DATASET_RAW_DIR", "../dataset/raw")),
        "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "train_csv": Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs"))
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Preprocessing
    ParallelCleaner().run(paths["raw"], paths["clean"])
    FeatureManager(paths["clean"], paths["feat"]).process_all()

    # 3. K-Fold Preparation
    df = pd.read_csv(paths["train_csv"])
    df.columns = [c.strip() for c in df.columns]
    y = df['PHQ8_Binary'].values 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trainer = MTRBTrainer(device, paths["feat"], paths["outputs"])
    fold_results = []

    # 4. Cross-Validation Loop
    for fold, (t_idx, v_idx) in enumerate(skf.split(df, y), 1):
        f1 = trainer.run_fold(fold, df.iloc[t_idx], df.iloc[v_idx])
        fold_results.append(f1)
        logger.info(f"Fold {fold} Complete", f1=round(f1, 4))

    # 5. Finalize Best Model
    avg_f1 = np.mean(fold_results)
    best_idx = np.argmax(fold_results) + 1
 
    # Copy best weights to production path
    best_weights = torch.load(paths["outputs"] / f"mtrb_fold_{best_idx}.pt", weights_only=True)
    torch.save(best_weights, paths["outputs"] / "mtrb_model.pt")
 
    logger.info("CV Summary", average_f1=round(avg_f1, 4), best_fold=best_idx)

if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        raise

