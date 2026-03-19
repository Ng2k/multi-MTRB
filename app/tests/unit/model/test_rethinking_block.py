import torch
from src.model.mtrb import RethinkingBlock


def test_rethinking_block_forward():
    """Test the basic attention aggregation logic."""
    input_dim = 1280
    attn_dim = 256
    block = RethinkingBlock(input_dim=input_dim, attn_dim=attn_dim)
    x = torch.randn(2, 50, input_dim) # [Batch, Seq, Dim]

    bag_repr, weights = block(x)

    assert bag_repr.shape == (2, input_dim)
    assert weights.shape == (2, 50, 1)
    # Check if weights are a valid probability distribution
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1))

def test_rethinking_block_with_mask():
    """Ensure masked values receive zero weight (effectively -inf in logits)."""
    input_dim = 64
    block = RethinkingBlock(input_dim=input_dim, attn_dim=32)
    x = torch.randn(1, 10, input_dim)
    mask = torch.ones(1, 10, 1)
    mask[:, 5:, :] = 0  # Mask the second half of the sequence

    _, weights = block(x, mask=mask)

    assert torch.all(weights[:, 5:, :] == 0)
    assert torch.allclose(weights[:, :5, :].sum(), torch.tensor(1.0))
