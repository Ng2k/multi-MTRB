import os
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

from src.config import Config
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
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Preprocessing
    logger.info("Starting Data Preprocessing")
    ParallelCleaner().run(paths["raw"], paths["clean"])
    FeatureManager(paths["clean"], paths["feat"]).process_all()

    # 3. Load Hyperparameters
    best_params_path = paths["outputs"] / "best_params.json"
    overrides = None
 
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            overrides = json.load(f)
        logger.info("Using optimized hyperparameters from RandomSearch", **overrides)
    else:
        logger.info("No optimized params found. Using default Config settings.")

    # 4. K-Fold Preparation
    df = pd.read_csv(paths["train_csv"])
    df.columns = [c.strip() for c in df.columns]
    y = df['PHQ8_Binary'].values

    # We use Config.SEED to ensure the K-Fold split is identical to the one used in search
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
 
    # Initialize trainer with potential overrides
    trainer = MTRBTrainer(device, paths["feat"], paths["outputs"], overrides=overrides)
    fold_results = []

    # 5. Cross-Validation Loop
    logger.info("Starting 5-Fold Cross-Validation")
    for fold, (t_idx, v_idx) in enumerate(skf.split(df, y), 1):
        f1 = trainer.run_fold(fold, df.iloc[t_idx], df.iloc[v_idx], save_model=True)
        fold_results.append(f1)
        logger.info(f"Fold {fold} Complete", f1=round(f1, 4))

    # 6. Finalize Best Model
    avg_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    best_fold_idx = np.argmax(fold_results) + 1
 
    # Copy weights of the best performing fold to a generic production file
    best_weights_path = paths["outputs"] / f"mtrb_fold_{best_fold_idx}.pt"
    production_path = paths["outputs"] / "mtrb_model.pt"
 
    if best_weights_path.exists():
        best_weights = torch.load(best_weights_path, weights_only=True)
        torch.save(best_weights, production_path)
        logger.info("Best fold weights promoted to production model", best_fold=best_fold_idx)
 
    logger.info("Pipeline Finished Successfully", 
                avg_f1=round(avg_f1, 4), 
                std_f1=round(std_f1, 4),
                best_fold=best_fold_idx)

if __name__ == "__main__":
    Config.seed_everything()
 
    try:
        run_full_pipeline()
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        raise

