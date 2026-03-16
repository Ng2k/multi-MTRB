import os
import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import Config
from src.utils.mtrb_trainer import MTRBTrainer
from src.utils.logger import get_logger, configure_logger

load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="parallel_random_search")

def evaluate_trial(trial_idx, params, paths, device):
    """
    Worker function to run a single trial (5 folds) in a separate process.
    Optimized for multi-core CPU and GPU parallelization.
    """
    trial_start = time.time()

    params['trial_idx'] = trial_idx

    # Isolated seeding for the process to ensure diverse search
    np.random.seed(Config.SEED + trial_idx)
    torch.manual_seed(Config.SEED + trial_idx)

    df = pd.read_csv(paths["train_csv"])
    y = df['PHQ8_Binary'].values 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    trainer = MTRBTrainer(device, paths["feat"], paths["outputs"], overrides=params)
    fold_auprcs = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(df, y), 1):
        auprc = trainer.run_fold(fold, df.iloc[t_idx], df.iloc[v_idx], save_model=False)
        fold_auprcs.append(auprc)
 
    avg_auprc = np.mean(fold_auprcs)
    duration = time.time() - trial_start

    return {
        "trial_idx": trial_idx,
        "params": params,
        "avg_auprc": avg_auprc,
        "std_auprc": np.std(fold_auprcs),
        "duration": duration
    }

def run_random_search():
    start_time = time.time()
    Config.seed_everything()

    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "train_csv": Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs"))
    }
    paths["outputs"].mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HARDWARE OPTIMIZATION: 
    # Local GPU: 4060 Ti 8GB VRAM, 2-3 parallel trials.
    max_parallel_trials = 4

    logger.info(
        "Initializing Parallel Search", 
        device=device, 
        max_workers=max_parallel_trials,
        total_trials=Config.NUM_SEARCH_TRIALS
    )

    best_overall_auprc = 0
    best_params = {}
    history = []

    with ProcessPoolExecutor(max_workers=max_parallel_trials) as executor:
        futures = []
        for i in range(Config.NUM_SEARCH_TRIALS):
            params = Config.get_search_trial(i)
            futures.append(executor.submit(evaluate_trial, i+1, params, paths, device))

        for future in as_completed(futures):
            result = future.result()
            t_idx = result["trial_idx"]
            avg_auprc = result["avg_auprc"]
            params = result["params"]

            logger.info(f"TRIAL {t_idx} COMPLETE", 
                        auprc=round(avg_auprc, 4), 
                        duration_sec=round(result["duration"], 2),
                        temp=params.get('temperature'))

            history.append({"trial": t_idx, "params": params, "auprc": avg_auprc})

            # Track and Log New Best Parameters
            if avg_auprc > best_overall_auprc:
                improvement = avg_auprc - best_overall_auprc
                best_overall_auprc = avg_auprc
                best_params = params

                logger.info(
                    "NEW BEST FOUND", 
                    auprc=round(best_overall_auprc, 4), 
                    improvement=round(improvement, 4)
                )

                with open(paths["outputs"] / "best_params.json", "w") as f:
                    json.dump(best_params, f, indent=4)

    # Final Reporting
    total_duration = (time.time() - start_time) / 60
    logger.info(
        "RANDOM SEARCH FINISHED", 
        total_time_min=round(total_duration, 2),
        final_best_auprc=round(best_overall_auprc, 4)
    )

    # Save complete history for analysis
    history_df = pd.DataFrame([
        {**h['params'], 'auprc': h['auprc'], 'trial': h['trial']} for h in history
    ])
    history_df.to_csv(paths["outputs"] / "search_history.csv", index=False)

if __name__ == "__main__":
    run_random_search()

