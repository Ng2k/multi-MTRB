import torch
import torch.nn as nn
import pytest
from src.model.mtrb import MultiMTRBClassifier

def test_classifier_initialization_defaults():
    """Verifies that the classifier initializes with the correct default structure."""
    input_dim = 1280
    n_heads = 4
    model = MultiMTRBClassifier(input_dim=input_dim, n_heads=n_heads)
    
    # Verify the number of attention heads
    assert len(model.heads) == n_heads
    
    # Verify the output projection dimensions
    assert model.output_projection.in_features == input_dim * n_heads
    assert model.output_projection.out_features == input_dim
    
    # Verify the fixed classifier structure
    # Sequential: Linear(0), LayerNorm(1), ReLU(2), Dropout(3), Linear(4)
    assert isinstance(model.classifier[2], nn.ReLU)
    assert model.classifier[4].out_features == 1

def test_classifier_forward_pass():
    """Verifies the forward pass output shapes and metadata content."""
    batch_size = 4
    seq_len = 20
    input_dim = 1280
    model = MultiMTRBClassifier(input_dim=input_dim, n_heads=2)
    x = torch.randn(batch_size, seq_len, input_dim)

    logits, avg_weights, info = model(x)

    # Check output shapes
    assert logits.shape == (batch_size, 1)
    assert avg_weights.shape == (batch_size, seq_len)
    
    # Check returned metadata
    assert "entropy" in info
    assert "gate" in info
    assert info["entropy"].dim() == 0  # It is returned as .mean()

def test_entropy_calculation():
    """Ensures entropy is calculated and is a positive value."""
    model = MultiMTRBClassifier()
    x = torch.randn(2, 10, 1280)
    _, _, info = model(x)
    
    # Entropy should be >= 0
    assert info["entropy"] >= 0

def test_device_moving():
    """Verifies the model and its metadata tensors move to the correct device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda")
    model = MultiMTRBClassifier().to(device)
    x = torch.randn(1, 5, 1280).to(device)
    
    logits, _, info = model(x)
    
    assert logits.device.type == "cuda"
    assert info["gate"].device.type == "cuda"
