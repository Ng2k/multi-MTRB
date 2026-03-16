import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import configure_logger, get_logger

# --- Configuration & Setup ---
load_dotenv("../.env", override=True)
configure_logger(enable_json=False, log_level="INFO")
logger = get_logger().bind(module="dataset_study")

def analyze_split(name, csv_path, feat_dir):
    """Performs deep analysis on a specific dataset split."""
    if not csv_path.exists():
        logger.error(f"{name} CSV not found", path=str(csv_path))
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Identify key columns
    id_col = next(c for c in df.columns if c.lower() in ['participant_id', 'id'])
    label_col = next(c for c in df.columns if 'phq' in c.lower() and 'binary' in c.lower())
    score_col = next((c for c in df.columns if 'phq' in c.lower() and 'score' in c.lower()), None)

    stats = {
        "total_participants": len(df),
        "class_distribution": df[label_col].value_counts(normalize=True).to_dict(),
        "raw_counts": df[label_col].value_counts().to_dict(),
    }

    if score_col:
        stats["avg_phq_score"] = round(df[score_col].mean(), 2)

    # Feature Analysis (sequence length and scaling)
    seq_lengths = []
    all_means = []
    all_stds = []
    missing_files = 0

    for pid in df[id_col].astype(str):
        # DAIC files are usually named [PID]_P.pt or similar
        feat_file = feat_dir / f"{pid}_P.pt"
        if not feat_file.exists():
            # Try alternative naming convention if needed
            feat_file = next(feat_dir.glob(f"{pid}_*.pt"), None)

        if feat_file and feat_file.exists():
            features = torch.load(feat_file, map_location='cpu', weights_only=True)
            seq_lengths.append(features.shape[0])
            all_means.append(features.mean().item())
            all_stds.append(features.std().item())
        else:
            missing_files += 1

    stats["feature_stats"] = {
        "missing_files": missing_files,
        "avg_seq_len": int(np.mean(seq_lengths)) if seq_lengths else 0,
        "max_seq_len": int(np.max(seq_lengths)) if seq_lengths else 0,
        "min_seq_len": int(np.min(seq_lengths)) if seq_lengths else 0,
        "global_feat_mean": round(float(np.mean(all_means)), 4) if all_means else 0,
        "global_feat_std": round(float(np.mean(all_stds)), 4) if all_stds else 0
    }

    return stats

def run_full_study():
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "train_csv": Path(os.getenv("TRAIN_SPLIT", "../dataset/train_split_Depression_AVEC2017.csv")),
        "dev_csv": Path(os.getenv("TEST_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv"))
    }

    logger.info("Starting Dataset In-Depth Analysis")

    train_report = analyze_split("Train", paths["train_csv"], paths["feat"])
    dev_report = analyze_split("Dev/Test", paths["dev_csv"], paths["feat"])

    # --- Print Comparison Report ---
    print("\n" + "="*60)
    print(f"{'DAIC-WOZ DATASET IMBALANCE & FEATURE REPORT':^60}")
    print("="*60)

    for name, report in [("TRAIN SET", train_report), ("DEV SET", dev_report)]:
        if not report: continue
        print(f"\n[{name}]")
        print(f"  Participants:   {report['total_participants']}")
        print(f"  Healthy (0):    {report['raw_counts'].get(0, 0)} ({report['class_distribution'].get(0, 0)*100:.1f}%)")
        print(f"  Depressed (1):  {report['raw_counts'].get(1, 0)} ({report['class_distribution'].get(1, 0)*100:.1f}%)")

        f_stats = report['feature_stats']
        print(f"  Avg Seq Length: {f_stats['avg_seq_len']} frames")
        print(f"  Feat Mean/Std:  {f_stats['global_feat_mean']} / {f_stats['global_feat_std']}")
        if f_stats['missing_files'] > 0:
            print(f"  !! WARNING: {f_stats['missing_files']} missing feature files !!")

    print("-" * 60)

    # Check for distribution shift
    train_ratio = train_report['class_distribution'].get(1, 0)
    dev_ratio = dev_report['class_distribution'].get(1, 0)
    diff = abs(train_ratio - dev_ratio)

    print(f"Class Distribution Shift: {diff*100:.2f}%")
    if diff > 0.05:
        print("ALERT: Significant distribution shift detected between Train and Dev.")

    print("="*60 + "\n")

if __name__ == "__main__":
    run_full_study()

