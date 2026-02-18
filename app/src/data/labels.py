"""
"""

from pathlib import Path
import pandas as pd

def load_labels(labels_csv_path: Path):
    """Loads DAIC-WOZ labels and returns a mapping {participant_id: binary_label}.

    Args:
        labels_csv_path: path of the csv file whith the references to the datasets

    Returns:
        mapping of labels and participant
    """
    df = pd.read_csv(labels_csv_path)
    return dict(zip(df['Participant_ID'].astype(str), df['PHQ_Binary']))

