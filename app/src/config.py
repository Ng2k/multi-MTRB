import numpy as np
import torch
import random

class Config:
    SEED = 42 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 1280
    MAX_SEQ_LEN = 300
    MAX_EPOCHS = 100
    NUM_SEARCH_TRIALS = 40 
    N_SPLITS = 5
    BATCH_SIZE = 64

    @staticmethod
    def seed_everything():
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    SEARCH_SPACE = {
        "learning_rate": lambda: 10**np.random.uniform(-4.0, -3.5),
        "temperature": lambda: round(np.random.uniform(0.05, 0.15), 2),
        "hidden_dim": lambda: int(np.random.choice([256, 512])),
        "n_heads": lambda: 8, 
        "entropy_lambda": lambda: round(np.random.uniform(0.01, 0.015), 4),
        "contrastive_lambda": lambda: round(np.random.uniform(0.045, 0.055), 4),
        "sharpness_lambda": lambda: round(np.random.uniform(0.004, 0.006), 4),
        "gate_lambda": lambda: round(np.random.uniform(0.015, 0.020), 4)
    }

    @staticmethod
    def get_search_trial(trial_idx: int):
        np.random.seed(Config.SEED + trial_idx)
        return {k: v() for k, v in Config.SEARCH_SPACE.items()}

