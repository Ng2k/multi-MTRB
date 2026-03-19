import torch
from src.model.mtrb import PositionalEncoding


def test_positional_encoding_shape():
    """Verify that positional encoding doesn't change input shape."""
    d_model = 1280
    seq_len = 100
    batch_size = 2
    pe = PositionalEncoding(d_model=d_model, max_len=2000)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pe(x)
    assert output.shape == x.shape

def test_positional_encoding_consistency():
    """Ensure the encoding is additive and deterministic."""
    pe = PositionalEncoding(d_model=128, max_len=100)
    x = torch.zeros(1, 10, 128)
    output1 = pe(x)
    output2 = pe(x)
    assert torch.equal(output1, output2)
    assert not torch.equal(output1, x)
