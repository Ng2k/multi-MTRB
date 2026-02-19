from pathlib import Path
import pandas as pd

def load_labels(labels_csv_path: Path):
    """Robustly loads DAIC-WOZ labels and returns a mapping {pid: binary_label}."""
    df = pd.read_csv(labels_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
 
    pid_col = next((c for c in df.columns if c in ['participant_id', 'id']), None)
    label_col = next((c for c in df.columns if c in ['phq8_binary', 'phq_binary']), None)

    if pid_col is None:
        raise KeyError(f"Could not find ID column in {labels_csv_path}")

    if label_col is None:
        return {str(pid): -1 for pid in df[pid_col]}
 
    return dict(zip(df[pid_col].astype(str), df[label_col]))

