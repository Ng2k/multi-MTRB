"""Module for rigorous evaluation of the Multi-MTRB model.

Calculates enterprise-standard metrics including F1, Precision, Recall, 
AUPRC, and Brier Score to assess clinical diagnostic performance.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_recall_curve, 
    auc, 
    brier_score_loss,
    confusion_matrix
)

from src.utils.logger import configure_logger, get_logger
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

# Initialization
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train")


def run_evaluation():
    # 1. Configuration & Paths
    FEAT_DIR = Path(os.getenv("DATASET_FEATURES_DIR", "../data/features"))
    TEST_CSV = Path(os.getenv("DEV_SPLIT", "../data/dev_split_Depression_AVEC2017.csv"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "../outputs/mtrb_model.pt"))
    EVAL_METRICS_PATH = Path(os.getenv("EVAL_METRICS_PATH", "../../outputs/evaluation_results.csv"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not MODEL_PATH.exists():
        logger.error("Model weights not found. Please train the model first.")
        return

    # 2. Load Data
    logger.info("Loading evaluation dataset", path=str(TEST_CSV))
    test_ds = MultiMTRBDataset(FEAT_DIR, TEST_CSV)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Load Model
    model = MultiMTRBClassifier(input_dim=1280).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4. Inference
    all_logits = []
    all_labels = []
 
    logger.info("Running inference on test split...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            # Forward pass: note we catch the attention weights but don't use them yet
            logits, _, _ = model(x)
 
            all_logits.append(logits.item())
            all_labels.append(y.item())

    # 5. Metric Computation
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(all_labels)

    # Precision-Recall Curve & AUC
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = auc(recall, precision)
 
    # Brier Score (Calibration: 0 is perfect, 1 is total failure)
    brier = brier_score_loss(labels, probs)
 
    # F1 & Confusion Matrix
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # 6. Reporting
    print("\n" + "="*50)
    print("      MULTI-MTRB CLINICAL EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples:      {len(labels)}")
    print(f"Positive Class %:   {labels.mean()*100:.1f}%")
    print("-" * 50)
    print(f"F1-Score:           {f1:.4f}")
    print(f"AUPRC (PR-AUC):     {auprc:.4f}")
    print(f"Brier Score:        {brier:.4f}")
    print("-" * 50)
    print(f"True Positives:     {tp} (Correctly caught Depression)")
    print(f"False Negatives:    {fn} (Missed Depression)")
    print(f"False Positives:    {fp} (False Alarms)")
    print(f"True Negatives:     {tn} (Correctly identified Healthy)")
    print("-" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("="*50)

    # Save results to CSV for record keeping
    results_df = pd.DataFrame({
        "metric": ["f1", "auprc", "brier", "tp", "fn", "fp", "tn"],
        "value": [f1, auprc, brier, tp, fn, fp, tn]
    })
    results_df.to_csv(EVAL_METRICS_PATH, index=False)
    logger.info("Evaluation results saved to ../data/evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()

