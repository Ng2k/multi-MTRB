import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

from src.config import Config
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset
from src.utils.logger import configure_logger, get_logger

# --- Setup ---
load_dotenv(dotenv_path="../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="xai_global")

def get_global_importance():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
    # 1. Configuration & Paths
    paths = {
        "feat": Path(os.getenv("DATASET_FEATURES_DIR", "../dataset/features")),
        "dev_csv": Path(os.getenv("DEV_SPLIT", "../dataset/dev_split_Depression_AVEC2017.csv")),
        "clean": Path(os.getenv("DATASET_CLEAN_DIR", "../dataset/clean")),
        "outputs": Path(os.getenv("MODEL_DIR", "../outputs")),
        "model_weights": Path(os.getenv("MODEL_DIR", "../outputs")) / "mtrb_model.pt",
        "best_params": Path(os.getenv("MODEL_DIR", "../outputs")) / "best_params.json"
    }

    # 2. Load Optimized Architecture
    overrides = None
    if paths["best_params"].exists():
        with open(paths["best_params"], "r") as f:
            overrides = json.load(f)
        logger.info("Using optimized architecture for global analysis", **overrides)

    # 3. Load Model
    model = MultiMTRBClassifier().to(DEVICE)
    if paths["model_weights"].exists():
        model.load_state_dict(torch.load(paths["model_weights"], map_location=DEVICE, weights_only=True))
    model.eval()

    # 4. Initialize Dataset and Counters
    dev_ds = MultiMTRBDataset(paths["feat"], paths["dev_csv"])
    word_hit_counter = Counter()
    word_freq_counter = Counter()
 
    # Expanded noise filter for cleaner clinical reporting
    noise = {
        "the", "a", "an", "and",
        "or", "but", "if", "then",
        "he", "she", "it", "they",
        "i", "my", "me", "you",
        "we", "was", "were", "is",
        "am", "are", "to", "of",
        "in", "for", "with", "on",
        "at", "by", "this", "that",
        "um", "uh", "laughter", "nan",
        "scrubbedentry", "synch"
    }

    print(f"\nAnalyzing clinical drivers across {len(dev_ds)} sessions...")

    # 5. Process each session
    for i in range(len(dev_ds)):
        features, _ = dev_ds[i]
        pid = dev_ds.file_list[i].name.split("_")[0]
        csv_path = paths["clean"] / f"{pid}_CLEAN.csv"
 
        if not csv_path.exists():
            continue
 
        df_text = pd.read_csv(csv_path)
        # Resolve text column (handles 'value', 'text', or 'utterance')
        text_col = next((c for c in ['value', 'text', 'utterance'] if c in df_text.columns), None)
        if not text_col: continue

        with torch.no_grad():
            _, weights, _ = model(features.unsqueeze(0).to(DEVICE))
            importance = weights.squeeze().cpu().numpy()

        if len(importance) == 0: continue
 
        # Define Top 10% importance threshold for this specific session
        threshold = np.percentile(importance, 90)

        num_to_map = min(len(importance), len(df_text))
        for idx in range(num_to_map):
            sentence = str(df_text.iloc[idx][text_col]).lower()
            attn_score = importance[idx]
 
            # Tokenize sentence into words to find key clinical markers
            words = [w.strip(".,!?;:\"") for w in sentence.split()]
 
            for w in words:
                if len(w) > 2 and w not in noise:
                    word_freq_counter[w] += 1
                    if attn_score >= threshold:
                        word_hit_counter[w] += 1

    # 6. Calculate Significance Stats
    stats = []
    for word, hits in word_hit_counter.items():
        freq = word_freq_counter[word]
        if freq >= 3: # Ignore words that rarely appear
            stats.append({
                'word': word,
                'hits': hits,
                'freq': freq,
                'relevance': hits / freq
            })

    # Sort by total hits (impact)
    top_features = sorted(stats, key=lambda x: x['hits'], reverse=True)

    # 7. Final Clinical Report
    print("\n" + "="*75)
    print("      GLOBAL CLINICAL FEATURE IMPORTANCE (WORD-LEVEL VOTING)")
    print("="*75)
    print(f"{'RANK':<5} | {'WORD/MARKER':<25} | {'TOP 10% HITS':<15} | {'FREQ':<8} | {'RATIO'}")
    print("-" * 75)
 
    for i, item in enumerate(top_features[:25]):
        print(f"{i+1:<5} | {item['word']:<25} | {item['hits']:<15} | {item['freq']:<8} | {item['relevance']:.3f}")
    print("="*75)
    print("Interpretation: 'Hits' = how many times this word appeared in a sentence")
    print("ranked in the top 10% of attention for that session.")

if __name__ == "__main__":
    Config.seed_everything()
    get_global_importance()

