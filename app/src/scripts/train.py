import os
import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils.logger import configure_logger, get_logger
from src.utils.mtrb_trainer import MTRBTrainer
from src.data.dataset import MultiMTRBDataset

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train_pipeline")

def main():
    # 1. Resource Initialization
    paths = {
        "raw": Path(os.getenv("DATASET_RAW_DIR", "../dataset/raw")),
        "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "train_csv": Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv")),
        "models": Path(os.getenv("MODEL_DIR", "../outputs/models")),
        "outputs": Path(os.getenv("OUTPUTS", "../outputs"))
    }
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Start tracking total training time
    start_time = time.time()

    # 2. Hardware Optimization & Seeding
    Config.seed_everything() 

    # 3. Load Best Hyperparameters from Random Search
    best_params_path = paths["outputs"] / "best_params.json"
    overrides = {}
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            overrides = json.load(f)
        logger.info("Loaded best parameters from random search", params=overrides)
    else:
        logger.warning("best_params.json not found, using default Config settings")

    overrides['trial_idx'] = "FINAL" 

    # 4. Dataset Preparation
    full_dataset = MultiMTRBDataset(
        paths["feat"], 
        paths["train_csv"], 
        clean_dir=paths["clean"], 
        max_seq=Config.MAX_SEQ_LEN
    )

    valid_pids = [f.name.split("_")[0] for f in full_dataset.file_list]
    full_df = pd.read_csv(paths["train_csv"])
    full_df.columns = [c.strip() for c in full_df.columns]
    id_col = next(c for c in full_df.columns if c.lower() in ['participant_id', 'id'])
    filtered_df = full_df[full_df[id_col].astype(str).isin(valid_pids)].reset_index(drop=True)
    label_col = next(c for c in filtered_df.columns if c.lower() in ['phq8_binary', 'phq_binary'])
    y_stratify = filtered_df[label_col].values

    # 5. K-Fold Execution
    # Ensure random_state matches Config.SEED for consistency with the search trials
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)

    trainer = MTRBTrainer(device, paths["feat"], paths["models"], overrides=overrides)
    fold_results = []

    logger.info("Starting Final 5-Fold Cross-Validation")
    for fold, (t_idx, v_idx) in enumerate(skf.split(filtered_df, y_stratify), 1):
        train_df_fold = filtered_df.iloc[t_idx]
        val_df_fold = filtered_df.iloc[v_idx]

        auprc = trainer.run_fold(fold, train_df_fold, val_df_fold, save_model=True)
        fold_results.append(auprc)
        logger.info(f"Fold {fold} Complete", auprc=round(auprc, 4))

    # 6. Finalize Reporting
    total_duration = (time.time() - start_time) / 60
    avg_auprc = np.mean(fold_results).item()
    std_auprc = np.std(fold_results).item()
    best_fold_idx = (np.argmax(fold_results) + 1).item()

    best_fold = paths["models"] / f"mtrb_fold_{best_fold_idx}.pt"
    production_path = paths["models"] / "mtrb_model.pt"

    if best_fold.exists():
        best_weights = torch.load(best_fold, map_location=device, weights_only=True)
        torch.save(best_weights, production_path)

    logger.info(
        "FINAL TRAINING PIPELINE COMPLETE", 
        average_auprc=round(avg_auprc, 4), 
        best_fold=best_fold_idx,
        standard_deviation=round(std_auprc, 4),
        total_training_time_min=round(total_duration, 2)
    )

if __name__ == "__main__":
    main()

