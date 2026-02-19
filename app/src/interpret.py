from os import getenv
from pathlib import Path
import torch
import pandas as pd
from dotenv import load_dotenv

from src.models.mtrb import MultiMTRBClassifier
from src.utils.logger import configure_logger, get_logger

load_dotenv(dotenv_path="../.env", override=True)
configure_logger(getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="train")


def explain_session(participant_id):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEAT_PATH = Path(f"{getenv("DATASET_FEATURES_DIR", "../dataset/features")}/{participant_id}_FEATURES.pt")
    TEXT_PATH = Path(f"{getenv("DATASET_CLEAN_DIR", "../dataset/clean")}/{participant_id}_CLEAN.csv")
    MODEL_PATH = Path(getenv("MODEL_PATH", "../outputs/mtrb_model.pt"))

    if not FEAT_PATH.exists() or not TEXT_PATH.exists():
        logger.error(f"Files for {participant_id} not found. Check your data/features and data/clean folders.")
        return

    # Load Model & Data
    model = MultiMTRBClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    features = torch.load(FEAT_PATH, weights_only=True).unsqueeze(0).to(DEVICE) # Add batch dim
    df_text = pd.read_csv(TEXT_PATH)

    # Helper to find existing columns
    def get_col(options, default="N/A"):
        for opt in options:
            if opt in df_text.columns: return opt
        return None

    text_col = get_col(['value', 'text', 'utterance'])
    spk_col = get_col(['speaker', 'personId', 'participant_id'])

    # Inference & Extract Attention
    with torch.no_grad():
        logits, weights, _ = model(features)
        probs = torch.sigmoid(logits).item()
        # weights shape: [1, Seq, 1] -> squeeze to [Seq]
        importance = weights.squeeze().cpu().numpy()

    # Map back to text
    num_sentences = min(len(importance), len(df_text))
    results = []
    for i in range(num_sentences):
        line_text = df_text.iloc[i][text_col] if text_col else "Unknown Text"
        line_spk = df_text.iloc[i][spk_col] if spk_col else "Participant"
 
        results.append({
            "text": line_text,
            "speaker": line_spk,
            "score": importance[i]
        })

    # Show Results
    print(f"\n--- Analysis for Participant {participant_id} ---")
    print(f"Depression Probability: {probs:.4f}")
    print("\nTop 5 Most Influential Sentences:")
 
    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, res in enumerate(sorted_results[:7]):
        print(f"{i+1}. [{res['score']:.4f}] {res['speaker']}: \"{res['text']}\"")

    print("="*60)


if __name__ == "__main__":
    # Pick one of your False Positives from the Evaluation
    # (Check your data/features folder for IDs)
    explain_session("302")
