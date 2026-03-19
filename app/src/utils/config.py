import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import numpy as np
import random

class Config:
    """Centralized configuration orchestrator."""
    def __init__(self, load_env_file: bool = True):
        if load_env_file:
            load_dotenv(override=True)
 
        self.raw_data = Path(os.getenv("RAW_DATA_DIR", ""))
        self.clean_data = Path(os.getenv("CLEAN_DATA_DIR", ""))
        self.features = Path(os.getenv("FEATURES_DIR", ""))
        self.train_csv = Path(os.getenv("TRAIN_SPLIT", ""))
        self.dev_csv = Path(os.getenv("DEV_SPLIT", ""))
        self.model_path = Path(os.getenv("ARTIFACTS_DIR", "")) / "models"
        self.artifacts = Path(os.getenv("ARTIFACTS_DIR", ""))

        self.max_workers = int(os.getenv("MAX_WORKERS", 4))

        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_dim = 1280
        self.token_size = 200
        self.epochs = 100
        self.n_split = 5
        self.batch_size = 64


    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# Global settings instance
settings = Config(load_env_file=True)

