import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from src.models.mtrb import MultiMTRBClassifier
from src.data.dataset import MultiMTRBDataset

def get_global_importance():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEAT_DIR = Path("../dataset/features")
    DEV_CSV = Path("../dataset/dev_split_Depression_AVEC2017.csv")
    MODEL_PATH = Path("../outputs/mtrb_model.pt")
    CLEAN_DIR = Path("../dataset/clean")

    # 1. Load Model
    model = MultiMTRBClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 2. Load Dev Set (Test Set)
    dev_ds = MultiMTRBDataset(FEAT_DIR, DEV_CSV)
 
    # Voting counters
    top_hit_counter = Counter()  # How many times word was in Top 10%
    total_occurrence_counter = Counter()
 
    # Generic noise to ignore in the report
    noise_filter = {"synch", "scrubbedentry", "laughter", "um", "uh", "and", "the", "to", "of", "a", "i"}

    print(f"Aggregating clinical insights from {len(dev_ds)} test sessions...")

    # 3. Process each session
    for i in range(len(dev_ds)):
        features, _ = dev_ds[i]
        file_path = dev_ds.file_list[i]
        pid = file_path.name.split("_")[0]
 
        df_text = pd.read_csv(CLEAN_DIR / f"{pid}_CLEAN.csv")
 
        with torch.no_grad():
            _, weights, _ = model(features.unsqueeze(0).to(DEVICE))
            importance = weights.squeeze().cpu().numpy()

        if len(importance) == 0: continue

        # Determine threshold for Top 10% most important sentences in this session
        threshold = np.percentile(importance, 90)

        # Map attention back to words
        num_to_map = min(len(importance), len(df_text))
        for idx in range(num_to_map):
            word = str(df_text.iloc[idx]['value']).lower().strip()
            word = word.replace('"', '').replace('.', '').replace(',', '')
 
            total_occurrence_counter[word] += 1
            if importance[idx] >= threshold and word not in noise_filter:
                top_hit_counter[word] += 1

    # 4. Calculate "Clinical Significance Score"
    # Logic: Score = (How many times it was a Top 10% feature)
    stats = []
    for word, hits in top_hit_counter.items():
        if total_occurrence_counter[word] >= 3: # Only words appearing in 3+ contexts
            stats.append({
                'word': word,
                'hits': hits,
                'frequency': total_occurrence_counter[word]
            })

    # Sort by hits (total times it was highly influential)
    top_features = sorted(stats, key=lambda x: x['hits'], reverse=True)

    # 5. Final Report
    print("\n" + "="*60)
    print("      GLOBAL CLINICAL FEATURE IMPORTANCE (VOTING LOGIC)")
    print("="*60)
    print(f"{'RANK':<5} | {'WORD/PHRASE':<25} | {'TOP 10% HITS':<12} | {'FREQ'}")
    print("-" * 60)
 
    for i, item in enumerate(top_features[:25]):
        print(f"{i+1:<5} | {item['word']:<25} | {item['hits']:<12} | {item['frequency']}")
    print("="*60)
    print("Interpretation: 'Hits' represent how many times this specific")
    print("phrase was a top-tier driver for a diagnosis across the test set.")

if __name__ == "__main__":
    get_global_importance()

