import numpy as np
import torch
import random

class Config:
    # --- GLOBAL SEED ---
    SEED = 42 

    # --- DEFAULTS ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 1280
    MAX_SEQ_LEN = 300
    MAX_EPOCHS = 100
    NUM_SEARCH_TRIALS = 20
    N_SPLITS = 5
    BATCH_SIZE = 32

    # --- REPRODUCIBILITY UTILITY ---
    @staticmethod
    def seed_everything():
        """Locks all random number generators to Config.SEED."""
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- SEARCH SPACE ---
    SEARCH_SPACE = {
        "learning_rate": lambda: 10**np.random.uniform(-5, -3),
        "pos_weight": lambda: round(np.random.uniform(1.0, 4.0), 2),
        "temperature": lambda: round(np.random.uniform(0.3, 0.7), 2),
        "hidden_dim": lambda: int(np.random.choice([256, 512, 768])),
        "n_heads": lambda: int(np.random.choice([2, 4, 8])),
        "gamma": lambda: round(np.random.uniform(1.0, 3.0), 1),
        "alpha": lambda: round(np.random.uniform(0.2, 0.8), 2),
        "entropy_lambda": lambda: round(np.random.uniform(0.01, 0.03), 4),
        "contrastive_lambda": lambda: round(np.random.uniform(0.05, 0.2), 2),
        "sharpness_lambda": lambda: round(np.random.uniform(0.001, 0.01), 4),
    }

    @staticmethod
    def get_search_trial(trial_idx):
        """
        Generates the params for a specific trial. 
        By re-seeding with (SEED + trial_idx), we ensure that if someone 
        only runs Trial #5, they get the exact same params as your Trial #5.
        """
        np.random.seed(Config.SEED + trial_idx)
        return {param: fn() for param, fn in Config.SEARCH_SPACE.items()}

