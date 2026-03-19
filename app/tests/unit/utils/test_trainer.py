import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import torch.nn as nn
from src.utils.trainer import Trainer, FocalLoss

# --- Tests for FocalLoss ---

def test_focal_loss_forward():
    """Verifies FocalLoss computes a scalar tensor and handles shapes correctly."""
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    inputs = torch.randn(4, 1, requires_grad=True)
    targets = torch.randint(0, 2, (4, 1)).float()
    
    loss = criterion(inputs, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Should be a scalar
    assert loss.item() >= 0

# --- Tests for Trainer ---

@pytest.fixture
def mock_trainer(tmp_path):
    """Provides a Trainer instance with mocked dependencies."""
    with patch("src.utils.trainer.get_logger"), \
         patch("src.utils.trainer.settings"):
        
        trainer = Trainer(
            device="cpu",
            features_dir=tmp_path / "features",
            output_dir=tmp_path / "output",
            overrides={"entropy_lambda": 0.05}
        )
        return trainer

def test_get_balanced_loader(mock_trainer):
    """Tests that the balanced loader correctly initializes a WeightedRandomSampler."""
    mock_dataset = MagicMock()
    # 3 negative samples, 1 positive sample to test weighting logic
    labels = np.array([0, 0, 0, 1]) 
    
    with patch("src.utils.trainer.DataLoader") as mock_dl:
        mock_trainer.get_balanced_loader(mock_dataset, labels)
        
        # Verify DataLoader was called with a sampler
        args, kwargs = mock_dl.call_args
        assert "sampler" in kwargs
        assert kwargs["sampler"].num_samples == 4


def test_trainer_device_initialization(tmp_path):
    """Ensures string device is converted to torch.device object."""
    trainer = Trainer("cpu", tmp_path, tmp_path)
    assert isinstance(trainer.device, torch.device)

