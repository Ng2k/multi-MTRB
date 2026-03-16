import numpy as np
import torch
import random

class Config:
    SEED = 42 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 1280
    MAX_SEQ_LEN = 200
    MAX_EPOCHS = 100
    NUM_SEARCH_TRIALS = 20
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
        "learning_rate": lambda: 10**np.random.uniform(-4.5, -3.5),
        "temperature": lambda: round(np.random.uniform(0.05, 0.20), 2),
        "hidden_dim": lambda: int(np.random.choice([256, 512])),
        "n_heads": lambda: int(np.random.choice([4, 8])),
        "entropy_lambda": lambda: round(np.random.uniform(0.015, 0.025), 4),
        "contrastive_lambda": lambda: round(np.random.uniform(0.04, 0.06), 4),
        "sharpness_lambda": lambda: round(np.random.uniform(0.004, 0.007), 4),
        "gate_lambda": lambda: round(np.random.uniform(0.015, 0.025), 4)
    }

