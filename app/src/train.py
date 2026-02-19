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
from src.data.dataset import MultiMTRBDataset

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

    # 4. Dataset Preparation with Speaker Filtering
    full_dataset = MultiMTRBDataset(
        paths["feat"], 
        paths["train_csv"], 
        clean_dir=paths["clean"], 
        max_seq=Config.MAX_SEQ_LEN
    )

    # Stratification Setup
    valid_pids = [f.name.split("_")[0] for f in full_dataset.file_list]
    full_df = pd.read_csv(paths["train_csv"])
    full_df.columns = [c.strip() for c in full_df.columns]
    id_col = next(c for c in full_df.columns if c.lower() in ['participant_id', 'id'])
    filtered_df = full_df[full_df[id_col].astype(str).isin(valid_pids)].reset_index(drop=True)
    label_col = next(c for c in filtered_df.columns if c.lower() in ['phq8_binary', 'phq_binary'])
    y_stratify = filtered_df[label_col].values

    # 5. K-Fold Execution
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
    trainer = MTRBTrainer(device, paths["feat"], paths["outputs"], overrides=overrides)
    fold_results = []

    logger.info("Starting 5-Fold Cross-Validation")
    for fold, (t_idx, v_idx) in enumerate(skf.split(filtered_df, y_stratify), 1):
        train_df_fold = filtered_df.iloc[t_idx]
        val_df_fold = filtered_df.iloc[v_idx]
        f1 = trainer.run_fold(fold, train_df_fold, val_df_fold, save_model=True)
        fold_results.append(f1)
        logger.info(f"Fold {fold} Complete", f1=round(f1, 4))

    # 6. Finalize Best Model
    best_fold_idx = np.argmax(fold_results) + 1
    best_weights_path = paths["outputs"] / f"mtrb_fold_{best_fold_idx}.pt"
    production_path = paths["outputs"] / "mtrb_model.pt"
    if best_weights_path.exists():
        best_weights = torch.load(best_weights_path, map_location=device, weights_only=True)
        torch.save(best_weights, production_path)

    logger.info("Pipeline Finished", avg_f1=round(np.mean(fold_results), 4), best_fold=best_fold_idx)

if __name__ == "__main__":
    Config.seed_everything()
    run_full_pipeline()

