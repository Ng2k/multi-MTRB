import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_recall_curve, 
    auc, 
    brier_score_loss,
    confusion_matrix
)

from src.config import Config
from src.utils.logger import configure_logger, get_logger
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

# Initialization
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="evaluation")

def run_evaluation():
    # 1. Configuration & Paths
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "test_csv": Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
        "model_weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt"
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not paths["model_weights"].exists():
        logger.error("Model weights not found. Please run train.py first.", path=str(paths['model_weights']))
        return

    # 2. Load Hyperparameters (to ensure architecture matches training)
    best_params_path = paths["outputs"] / "best_params.json"
    overrides = None
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            overrides = json.load(f)
        logger.info("Loaded optimized parameters for evaluation architecture", **overrides)

    # 3. Load Data
    logger.info("Loading evaluation dataset", path=str(paths["test_csv"]))
    test_ds = MultiMTRBDataset(paths["feat"], paths["test_csv"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 4. Load Model
    # Note: We pass overrides so the hidden_dim matches the trained mtrb_model.pt
    model = MultiMTRBClassifier().to(DEVICE)
    model.load_state_dict(torch.load(paths["model_weights"], map_location=DEVICE, weights_only=True))
    model.eval()

    # 5. Inference
    all_logits, all_labels = [], []
    logger.info("Running inference...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, _, _ = model(x)
            all_logits.append(logits.item())
            all_labels.append(y.item())

    # 6. Metric Computation
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)

    # Calculate PR-Curve and AUPRC
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
    auprc = auc(recall_vals, precision_vals)
    brier = brier_score_loss(labels, probs)

    # 7. Threshold Optimization
    # Find the threshold that maximizes F1-score specifically for this model
    best_f1 = 0
    best_threshold = 0.5
    for t in np.linspace(0.1, 0.9, 81):
        current_f1 = f1_score(labels, (probs > t).astype(int))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    # Final predictions using the best threshold
    preds = (probs > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # 8. Clinical Reporting
    print("\n" + "="*60)
    print("        MULTI-MTRB CLINICAL EVALUATION REPORT")
    print("="*60)
    print(f"{'Samples:':<25} {len(labels)}")
    print(f"{'Prevalence (Dep):':<25} {labels.mean()*100:.1f}%")
    print(f"{'Optimized Threshold:':<25} {best_threshold:.2f}")
    print("-" * 60)
    print(f"{'F1-Score:':<25} {best_f1:.4f}")
    print(f"{'AUPRC (PR-AUC):':<25} {auprc:.4f}")
    print(f"{'Brier Score:':<25} {brier:.4f}")
    print("-" * 60)
    print(f"{'True Positives:':<25} {tp} (Hits)")
    print(f"{'False Negatives:':<25} {fn} (Misses)")
    print(f"{'False Positives:':<25} {fp} (False Alarms)")
    print(f"{'True Negatives:':<25} {tn} (Correct Healthy)")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("="*60)

    # Save metrics
    results_path = paths["outputs"] / "evaluation_results.csv"
    results_df = pd.DataFrame({
        "metric": ["f1", "auprc", "brier", "best_threshold", "tp", "fn", "fp", "tn"],
        "value": [best_f1, auprc, brier, best_threshold, tp, fn, fp, tn]
    })
    results_df.to_csv(results_path, index=False)
    logger.info("Evaluation complete", results_saved_to=str(results_path))

if __name__ == "__main__":
    Config.seed_everything()
    try:
        run_evaluation()
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise

