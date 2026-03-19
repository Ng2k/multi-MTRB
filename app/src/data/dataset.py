"""
Multi-MTRB Dataset module for handling Multiple Instance Learning bags.
"""

from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset

from src.data.labels import load_labels

class MultiMTRBDataset(Dataset):
    def __init__(self, features_dir: Path, labels_csv: Path, clean_dir: Path|None = None, max_seq=200):
        """
        Args:
            features_dir: Path to directory containing .pt feature files.
            labels_csv: Path to the labels CSV file.
            clean_dir: Optional path to directory containing speaker-labeled CSVs (for cleaning).
            max_seq: Maximum number of instances (segments) per bag. Reduced to 200 to decrease noise.
        """
        self.features_dir = features_dir
        self.clean_dir = clean_dir
        self.max_seq = max_seq
        self.labels = load_labels(labels_csv)
        self.file_list = [
            f for f in features_dir.glob("*.pt") 
            if f.name.split("_")[0] in self.labels
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        participant_id = file_path.name.split("_")[0]
        label = self.labels[participant_id]

        features = torch.load(file_path, weights_only=True)

        # 1. Speaker Cleaning Logic
        if self.clean_dir is not None:
            clean_path = self.clean_dir / f"{participant_id}_CLEAN.csv"
            if clean_path.exists():
                df = pd.read_csv(clean_path)
                spk_col = next((c for c in ['speaker', 'personId'] if c in df.columns), None)
                if spk_col:
                    # Keep only Participant rows (exclude Interviewer)
                    mask = df[spk_col].astype(str).str.lower().str.contains(f'participant|p|{participant_id}')
                    valid_idx = mask[mask].index.tolist()
                    # Boundary check to ensure mask indices don't exceed feature size
                    valid_idx = [i for i in valid_idx if i < features.size(0)]
                    features = features[valid_idx]

        n_instances = features.size(0)

        # 2. Sequence Length Handling (Middle-Crop Strategy)
        if n_instances > self.max_seq:
            start = (n_instances - self.max_seq) // 2
            features = features[start : start + self.max_seq]

        elif n_instances < self.max_seq:
            # If the sequence is too short, we pad with zeros
            padding = torch.zeros((self.max_seq - n_instances, features.size(1)))
            features = torch.cat([features, padding], dim=0)

        return features, torch.tensor(label, dtype=torch.float32)

