import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import configure_logger, get_logger

load_dotenv(dotenv_path="../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="xai_suite")


class UnifiedXAISuite:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.paths = {
            "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
            "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
            "dev_csv": Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
            "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
            "weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt",
            "params": Path(os.getenv("MODEL_DIR", "../outputs")) / "best_params.json",
            "clinical": Path(os.getenv("MODEL_DIR", "../outputs")) / "clinical_config.json"
        }
        self.threshold = self._load_clinical_threshold()
        self.model = self._load_model()


    def _load_clinical_threshold(self):
        if self.paths["clinical"].exists():
            with open(self.paths["clinical"], "r") as f:
                return json.load(f).get("threshold", 0.4970)
        return 0.4970


    def _load_model(self):
        with open(self.paths["params"], "r") as f:
            overrides = json.load(f)
        model = MultiMTRBClassifier(
            hidden_dim=overrides.get("hidden_dim", 512),
            temperature=overrides.get("temperature", 0.33)
        ).to(self.device)
        model.load_state_dict(torch.load(self.paths["weights"], map_location=self.device, weights_only=True))
        model.eval()
        return model


    def run_local_analysis(self, participant_id: str):
        feat_path = self.paths["feat"] / f"{participant_id}_FEATURES.pt"
        text_path = self.paths["clean"] / f"{participant_id}_CLEAN.csv"
        if not feat_path.exists() or not text_path.exists(): return

        features = torch.load(feat_path, weights_only=True).unsqueeze(0).to(self.device)
        df_text = pd.read_csv(text_path)

        with torch.no_grad():
            logits, weights, _ = self.model(features)
            prob = torch.sigmoid(logits).item()
            importance = weights.squeeze().cpu().numpy()

        prediction = "DEPRESSED" if prob >= self.threshold else "HEALTHY"

        plt.figure(figsize=(14, 6))
        sns.set_theme(style="whitegrid")
        smoothed = pd.Series(importance).rolling(window=5, min_periods=1, center=True).mean()
        plt.plot(range(len(importance)), smoothed, color='#2c3e50', linewidth=2)
        plt.fill_between(range(len(importance)), smoothed, color='#e74c3c' if prediction=="DEPRESSED" else '#3498db', alpha=0.3)
        plt.title(f"Clinical Attention Profile: {participant_id} | Result: {prediction} (Prob: {prob:.2%})")
        plt.savefig(self.paths["outputs"] / f"xai_local_{participant_id}.png", dpi=300)
        plt.close()

        print(f"\n[CLINICAL REPORT: {participant_id}] | PREDICTION: {prediction}")
        print("-" * 80)
        text_col = next(c for c in ['value', 'text', 'utterance'] if c in df_text.columns)
        results = []
        for i in range(min(len(importance), len(df_text))):
            results.append((importance[i], df_text.iloc[i][text_col]))

        for score, txt in sorted(results, key=lambda x: x[0], reverse=True)[:7]:
            print(f"[{score:.4f}] | {txt}")


    def run_global_analysis(self, min_freq=5):
        dev_ds = MultiMTRBDataset(self.paths["feat"], self.paths["dev_csv"], clean_dir=self.paths["clean"])
        word_hits = Counter()
        noise = {"the", "and", "that", "was", "for", "you", "with", "this", "laughter", "have", "just", "like", "well", "really", "know", "don't", "think", "it's", "they", "there"}

        for i in range(len(dev_ds)):
            feat, _ = dev_ds[i]
            pid = dev_ds.file_list[i].name.split("_")[0]
            df_text = pd.read_csv(self.paths["clean"] / f"{pid}_CLEAN.csv")
            text_col = next(c for c in ['value', 'text', 'utterance'] if c in df_text.columns)
            with torch.no_grad():
                _, weights, _ = self.model(feat.unsqueeze(0).to(self.device))
                importance = weights.squeeze().cpu().numpy()

            threshold = np.percentile(importance, 90)
            for idx in range(min(len(importance), len(df_text))):
                if importance[idx] >= threshold:
                    for w in [w.strip(".,!?;:").lower() for w in str(df_text.iloc[idx][text_col]).split() if len(w) > 3]:
                        if w not in noise:
                            word_hits[w] += 1

        stats = sorted([{'word': w, 'hits': h} for w, h in word_hits.items()], key=lambda x: x['hits'], reverse=True)[:20]
        plt.figure(figsize=(12, 8))
        words = [x['word'] for x in stats]
        hits = [x['hits'] for x in stats]
        # Fixed palette and hue assignment to stop FutureWarning
        sns.barplot(x=hits, y=words, hue=words, palette="rocket", legend=False)
        plt.title("Top Global Clinical Markers (Attention-Weighted)")
        plt.tight_layout()
        plt.savefig(self.paths["outputs"] / "xai_global_markers.png")


if __name__ == "__main__":
    suite = UnifiedXAISuite()
    suite.run_local_analysis("402")
    suite.run_global_analysis()

