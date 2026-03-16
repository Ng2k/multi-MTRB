import os, json, torch, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.config import Config
from src.models.mtrb import MultiMTRBClassifier

load_dotenv("../.env", override=True)

class UnifiedXAISuite:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.paths = {
            "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
            "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
            "dev_csv": Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
            "weights": Path(os.getenv("MODEL_DIR", "../outputs/models")) / "mtrb_model.pt",
            "params": Path(os.getenv("OUTPUTS_DIR", "../outputs")) / "best_params.json"
        }
        self.model = self._load_model()

    def _load_model(self):
        overrides = {}
        if self.paths["params"].exists():
            with open(self.paths["params"], "r") as f: overrides = json.load(f)
        model = MultiMTRBClassifier(
            input_dim=Config.INPUT_DIM,
            hidden_dim=int(overrides.get("hidden_dim", 256)),
            temperature=float(overrides.get("temperature", 0.1)),
            n_heads=int(overrides.get("n_heads", 4))
        ).to(self.device)
        model.load_state_dict(torch.load(self.paths["weights"], map_location=self.device, weights_only=True))
        return model.eval()

    def run_clinical_explanation(self, participant_id: str):
        feat_path = self.paths["feat"] / f"{participant_id}_FEATURES.pt"
        text_path = self.paths["clean"] / f"{participant_id}_CLEAN.csv"

        feat = torch.load(feat_path).to(self.device)
        df_text = pd.read_csv(text_path)

        with torch.no_grad():
            logits, weights, _ = self.model(feat.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
            importance = weights.squeeze().cpu().numpy()

        sorted_idx = np.argsort(importance)[::-1]
        print(f"\n[REPORT: {participant_id}] | SCORE: {prob:.4f}")
        print("-" * 50)

        noise = {"um", "uh", "mm", "laughter", "sil", "okay", "yeah"}
        count = 0
        for idx in sorted_idx:
            text = str(df_text.iloc[idx]['value']).lower().strip()
            if text in noise: continue
            print(f"[{importance[idx]:.4f}] | {text}")
            count += 1
            if count >= 7: break

if __name__ == "__main__":
    suite = UnifiedXAISuite()
    for pid in ["402", "300"]: suite.run_clinical_explanation(pid)

