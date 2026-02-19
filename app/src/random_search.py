import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils.mtrb_trainer import MTRBTrainer
from src.utils.logger import get_logger, configure_logger

load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="random_search")


def run_random_search():
    Config.seed_everything()

    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "train_csv": Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs"))
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(paths["train_csv"])
    y = df['PHQ8_Binary'].values 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    best_overall_auprc = 0
    best_params = {}
    history = []

    for trial in range(Config.NUM_SEARCH_TRIALS):
        params = Config.get_search_trial(trial)
        logger.info(f"--- TRIAL {trial+1}/{Config.NUM_SEARCH_TRIALS} ---", **params)
 
        trainer = MTRBTrainer(device, paths["feat"], paths["outputs"], overrides=params)
        fold_auprcs = []

        for fold, (t_idx, v_idx) in enumerate(skf.split(df, y), 1):
            auprc = trainer.run_fold(fold, df.iloc[t_idx], df.iloc[v_idx], save_model=False)
            fold_auprcs.append(auprc)
 
        avg_auprc = np.mean(fold_auprcs)
        logger.info(f"Trial Result", avg_auprc=round(avg_auprc, 4))
 
        history.append({"trial": trial, "params": params, "auprc": avg_auprc})

        if avg_auprc > best_overall_auprc:
            best_overall_auprc = avg_auprc
            best_params = params
            logger.info("New Best Leader found (AUPRC)!")

    # Save the parameters that yielded the highest AUPRC
    with open(paths["outputs"] / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    with open(paths["outputs"] / "search_history.json", "w") as f:
        json.dump(history, f, indent=4)
 
    logger.info("Search Finished", best_auprc=round(best_overall_auprc, 4))


if __name__ == "__main__":
    run_random_search()

