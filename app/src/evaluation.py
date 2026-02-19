import os
import json
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import classification_report, precision_recall_curve, auc

from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import configure_logger, get_logger

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="evaluation")

def run_evaluation():
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "test_csv": Path(os.getenv("TEST_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
        "model_weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt",
        "best_params": Path(os.getenv("MODEL_DIR", "../outputs")) / "best_params.json"
    }

    # 1. Load Hyperparameters to prevent architecture mismatch (Phase 4 fix)
    overrides = {}
    if paths["best_params"].exists():
        with open(paths["best_params"], "r") as f:
            overrides = json.load(f)
        logger.info("Loaded hyperparameters from random search", **overrides)
    else:
        logger.warning("best_params.json not found, using default architecture")

    # 2. Initialize Model with the correct dimensions and heads
    model = MultiMTRBClassifier(
        input_dim=Config.INPUT_DIM,
        hidden_dim=int(overrides.get("hidden_dim", 256)),
        temperature=float(overrides.get("temperature", 0.5)),
        n_heads=int(overrides.get("n_heads", 4)) # Matches saved weight shapes
    ).to(Config.DEVICE)

    # 3. Load State Dict
    if not paths["model_weights"].exists():
        logger.error(f"Model weights not found at {paths['model_weights']}")
        return

    model.load_state_dict(torch.load(paths["model_weights"], map_location=Config.DEVICE, weights_only=True))
    model.eval()

    # 4. Prepare Evaluation Dataset
    test_ds = MultiMTRBDataset(paths["feat"], paths["test_csv"], max_seq=Config.MAX_SEQ_LEN)

    all_probs = []
    all_targets = []

    logger.info("Starting Clinical Inference")
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            x = x.unsqueeze(0).to(Config.DEVICE)

            logits, _, _ = model(x)
            prob = torch.sigmoid(logits).item()

            all_probs.append(prob)
            all_targets.append(int(y.item()))

    # 5. Clinical Thresholding (Targeting 75% Recall for Phase 3/4)
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)

    # Find the threshold that yields at least 75% recall
    target_recall = 0.75
    idx = np.where(recalls >= target_recall)[0][-1]
    clinical_threshold = thresholds[idx]

    final_preds = [1 if p >= clinical_threshold else 0 for p in all_probs]
    current_auprc = auc(recalls, precisions)

    # 6. Generate Clinical Report
    report = classification_report(
        all_targets, 
        final_preds, 
        target_names=["Healthy", "Depressed"]
    )

    print("\n" + "="*60)
    print(f"{'MULTI-MTRB CLINICAL EVALUATION (75% RECALL MODE)':^60}")
    print("="*60)
    print(f"Clinical Threshold:        {clinical_threshold:.4f}")
    print(f"Recall (Sensitivity):      {recalls[idx]*100:.1f}%")
    print(f"Precision:                 {precisions[idx]*100:.1f}%")
    print(f"AUPRC:                     {current_auprc:.4f}")
    print("-" * 60)

    tp = sum((p == 1 and t == 1) for p, t in zip(final_preds, all_targets))
    fn = sum((p == 0 and t == 1) for p, t in zip(final_preds, all_targets))
    fp = sum((p == 1 and t == 0) for p, t in zip(final_preds, all_targets))

    print(f"True Positives:            {tp} (Hits)")
    print(f"False Negatives:           {fn} (Misses)")
    print(f"False Positives:           {fp} (False Alarms)")
    print("-" * 60)
    print("\nDetailed Clinical Classification Report:")
    print(report)
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()

