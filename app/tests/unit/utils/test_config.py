import os
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
import random
from src.utils.config import settings, Config

def test_config_default_values():
    """Verifies that the Config class initializes with expected default values."""
    config = Config(load_env_file=False)

    assert config.seed == 42
    assert config.input_dim == 1280
    assert config.token_size == 200
    assert config.epochs == 100
    assert config.n_split == 5
    assert config.batch_size == 64
    assert isinstance(config.raw_data, Path)
    assert isinstance(config.artifacts, Path)

def test_config_device_assignment():
    """Verifies that the device is correctly assigned based on CUDA availability."""
    config = Config(load_env_file=False)
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert config.device == expected_device

@patch.dict(os.environ, {
    "RAW_DATA_DIR": "/tmp/raw",
    "CLEAN_DATA_DIR": "/tmp/clean",
    "FEATURES_DIR": "/tmp/features",
    "TRAIN_SPLIT": "/tmp/train.csv",
    "ARTIFACTS_DIR": "/tmp/artifacts",
    "MAX_WORKERS": "8"
})
def test_config_env_loading():
    """Verifies that Config correctly prioritizes environment variables."""
    config = Config(load_env_file=False)

    assert config.raw_data.as_posix() == "/tmp/raw"
    assert config.clean_data.as_posix() == "/tmp/clean"
    assert config.features.as_posix() == "/tmp/features"
    assert config.train_csv.as_posix() == "/tmp/train.csv"
    assert config.artifacts.as_posix() == "/tmp/artifacts"
    assert config.model_path.as_posix() == "/tmp/artifacts/models"
    assert config.max_workers == 8

def test_seed_everything():
    """Ensures seed_everything correctly sets deterministic states for all libraries."""
    config = Config(load_env_file=False)
    config.seed = 123
    config.seed_everything()

    # Verify python random
    val1 = random.random()
    random.seed(123)
    val2 = random.random()
    assert val1 == val2

    # Verify numpy
    np_val1 = np.random.rand()
    np.random.seed(123)
    np_val2 = np.random.rand()
    assert np_val1 == np_val2

    # Verify torch
    torch_val1 = torch.rand(1)
    torch.manual_seed(123)
    torch_val2 = torch.rand(1)
    assert torch.equal(torch_val1, torch_val2)

def test_global_settings_instance():
    """Ensures the global settings instance is correctly exported and initialized."""
    assert isinstance(settings, Config)
    assert settings.device in ["cuda", "cpu"]

@patch("src.utils.config.load_dotenv")
def test_config_load_env_logic(mock_load_dotenv):
    """Tests the conditional loading of .env files."""
    Config(load_env_file=True)
    mock_load_dotenv.assert_called_once_with(override=True)

