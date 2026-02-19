import os
import json
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.utils.logger import configure_logger, get_logger

# --- Setup ---
load_dotenv(dotenv_path="../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="xai_local")

def explain_session(participant_id):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
    # 1. Paths
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")) / f"{participant_id}_FEATURES.pt",
        "text": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")) / f"{participant_id}_CLEAN.csv",
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
        "model_weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt",
        "best_params": Path(os.getenv("MODEL_DIR", "../outputs")) / "best_params.json"
    }

    # 2. Validation
    if not paths["feat"].exists() or not paths["text"].exists():
        logger.error(f"Data for {participant_id} not found. Check features/clean folders.")
        return
    if not paths["model_weights"].exists():
        logger.error("Trained model weights not found. Run train.py first.")
        return

    # 3. Load Hyperparameters
    overrides = None
    if paths["best_params"].exists():
        with open(paths["best_params"], "r") as f:
            overrides = json.load(f)
        logger.info(f"Using optimized architecture for XAI", **overrides)

    # 4. Load Model & Data
    # MultiMTRBClassifier should handle the 'overrides' to set its internal hidden_dim
    model = MultiMTRBClassifier().to(DEVICE)
    model.load_state_dict(torch.load(paths["model_weights"], map_location=DEVICE, weights_only=True))
    model.eval()

    features = torch.load(paths["feat"], weights_only=True).unsqueeze(0).to(DEVICE)
    df_text = pd.read_csv(paths["text"])

    # 5. Column Resolver
    def get_col(options):
        for opt in options:
            if opt in df_text.columns: return opt
        return None

    text_col = get_col(['value', 'text', 'utterance'])
    spk_col = get_col(['speaker', 'personId', 'participant_id'])

    # 6. Inference & Attention Extraction
    with torch.no_grad():
        # model returns (logits, attention_weights, latent_vector)
        logits, weights, _ = model(features)
        probs = torch.sigmoid(logits).item()
        # weights shape: [1, Seq, 1] -> [Seq]
        importance = weights.squeeze().cpu().numpy()

    # 7. Mapping logic
    num_sentences = min(len(importance), len(df_text))
    results = []
    for i in range(num_sentences):
        results.append({
            "text": df_text.iloc[i][text_col] if text_col else "N/A",
            "speaker": df_text.iloc[i][spk_col] if spk_col else "Participant",
            "score": float(importance[i])
        })

    # 8. Visualization Output
    print(f"\n" + "="*70)
    print(f"XAI ANALYSIS: Participant {participant_id}")
    print(f"{'Diagnosis:':<20} {'DEPRESSED' if probs > 0.48 else 'HEALTHY'} (p={probs:.4f})")
    print(f"{'Architecture:':<20} Hidden_Dim={overrides.get('hidden_dim', 512) if overrides else 512}")
    print("="*70)
    print(f"\nTOP 7 INFLUENTIAL UTTERANCES (MTRB Attention Weights):")
 
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, res in enumerate(sorted_results[:7]):
        # Adding a visual indicator (bar) for the score
        bar = "█" * int(res['score'] * 20)
        print(f"{i+1}. [{res['score']:.4f}] {bar:<21} {res['speaker']}: \"{res['text']}\"")
 
    print("-" * 70)

if __name__ == "__main__":
    # Ensure reproducibility
    Config.seed_everything()
 
    # Set the ID you want to investigate
    TARGET_ID = "302"
    explain_session(TARGET_ID)

