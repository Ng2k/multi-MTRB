import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    precision_recall_curve, 
    auc, 
    confusion_matrix
)

from src.config import Config
from src.utils.logger import configure_logger, get_logger
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="evaluation")

def run_evaluation():
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
        "test_csv": Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
        "model_weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt"
    }

    # 1. Load optimized architecture params
    best_params_path = paths["outputs"] / "best_params.json"
    overrides = {}
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            overrides = json.load(f)

    # 2. Setup Data (with Speaker Filtering)
    test_ds = MultiMTRBDataset(paths["feat"], paths["test_csv"], clean_dir=paths["clean"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Initialize Model
    model = MultiMTRBClassifier(
        hidden_dim=overrides.get("hidden_dim", 512),
        temperature=overrides.get("temperature", 0.33)
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(paths["model_weights"], map_location=Config.DEVICE, weights_only=True))
    model.eval()

    # 4. Inference
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits, _, _ = model(x.to(Config.DEVICE))
            all_probs.append(torch.sigmoid(logits).item())
            all_labels.append(y.item())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # 5. Clinical Optimization (Targeting 75% Recall)
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
    clinical_idx = np.where(recall_vals >= 0.75)[0][-1]
    clinical_threshold = thresholds[min(clinical_idx, len(thresholds)-1)]

    preds = (probs >= clinical_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # 6. Save Clinical Config for XAI
    with open(paths["outputs"] / "clinical_config.json", "w") as f:
        json.dump({"threshold": float(clinical_threshold)}, f)

    # 7. Print Final Report
    print("\n" + "="*60)
    print("        MULTI-MTRB CLINICAL EVALUATION (75% RECALL MODE)")
    print("="*60)
    print(f"{'Clinical Threshold:':<25} {clinical_threshold:.4f}")
    print(f"{'Recall (Sensitivity):':<25} {tp/(tp+fn):.1%}")
    print(f"{'Precision:':<25} {tp/(tp+fp):.1%}")
    print(f"{'AUPRC:':<25} {auc(recall_vals, precision_vals):.4f}")
    print("-" * 60)
    print(f"{'True Positives:':<25} {tp} (Hits)")
    print(f"{'False Negatives:':<25} {fn} (Misses)")
    print(f"{'False Positives:':<25} {fp} (False Alarms)")
    print("-" * 60)
    print("\nDetailed Clinical Classification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("="*60)

if __name__ == "__main__":
    run_evaluation()

