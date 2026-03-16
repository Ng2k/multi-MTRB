from pathlib import Path
from typing import Union
import pandas as pd

def load_labels(labels_input: Union[Path, str, pd.DataFrame]):
    """Robustly loads DAIC-WOZ labels from a CSV path OR uses a provided DataFrame."""
    if isinstance(labels_input, pd.DataFrame):
        df = labels_input.copy()
    else:
        df = pd.read_csv(labels_input)

    df.columns = [c.strip().lower() for c in df.columns]
 
    pid_col = next((c for c in df.columns if c in ['participant_id', 'id']), None)
    label_col = next((c for c in df.columns if c in ['phq8_binary', 'phq_binary']), None)

    if pid_col is None:
        raise KeyError(f"Could not find ID column in {labels_input}")

    if label_col is None:
        return {str(pid): -1 for pid in df[pid_col]}
 
    return dict(zip(df[pid_col].astype(str), df[label_col]))

