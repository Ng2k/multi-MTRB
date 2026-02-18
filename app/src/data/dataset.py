"""
Multi-MTRB Dataset module for handling Multiple Instance Learning bags.
"""

from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset

class MultiMTRBDataset(Dataset):
    def __init__(self, features_dir: Path, labels_csv: Path, max_seq=300):
        self.features_dir = features_dir
        self.max_seq = max_seq
 
        df = pd.read_csv(labels_csv)
        df.columns = [c.strip() for c in df.columns]
 
        pid_col = next((c for c in df.columns if c.lower() in ['participant_id', 'id']), None)
        label_col = next((c for c in df.columns if c.lower() in ['phq8_binary', 'phq_binary']), None)

        if pid_col is None:
            raise KeyError(f"Could not find ID column in {labels_csv}")

        if label_col is None:
            print(f"INFO: No labels found in {labels_csv.name}. Evaluation metrics will not be possible.")
            self.labels = {str(pid): -1 for pid in df[pid_col]}
        else:
            self.labels = dict(zip(df[pid_col].astype(str), df[label_col]))
 
        self.file_list = [
            f for f in features_dir.glob("*.pt") 
            if f.name.split("_")[0] in self.labels
        ]

    def __len__(self):
        """Required by DataLoader to know the dataset size."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Loads a single session bag, applies padding/truncation, and returns label."""
        file_path = self.file_list[idx]
        participant_id = file_path.name.split("_")[0]
        label = self.labels[participant_id]

        # Load the [N, 1280] tensor
        # weights_only=True is a security best practice for loading .pt files
        features = torch.load(file_path, weights_only=True)

        # Handle sequence length (Multiple Instance Learning Bag size)
        n_instances = features.size(0)
 
        if n_instances > self.max_seq:
            # Truncate longer interviews
            features = features[:self.max_seq]
        elif n_instances < self.max_seq:
            # Pad shorter interviews with zeros
            padding = torch.zeros((self.max_seq - n_instances, 1280))
            features = torch.cat([features, padding], dim=0)

        return features, torch.tensor(label, dtype=torch.float32)
