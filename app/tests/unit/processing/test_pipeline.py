import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from src.processing.pipeline import TranscriptPipeline


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'speaker': ['Ellie', 'Participant', 'Participant'],
        'value': ['How are you?', 'I am [laughter] okay.', '  ']
    })


def test_process_file_success(tmp_path, mock_df):
    """Verifies filtering, cleaning, and file I/O."""
    input_file = tmp_path / "101_TRANSCRIPT.csv"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Mock read_csv to provide our test data
    with patch("pandas.read_csv", return_value=mock_df):
        pipeline = TranscriptPipeline()
        result = pipeline.process_file(input_file, output_dir)

    assert result["status"] == "success"
    assert result["initial"] == 2
    assert result["final"] == 1

    output_file = output_dir / "101_CLEAN.csv"
    assert output_file.exists()

    # Read the actual file from disk to verify
    saved_df = pd.read_csv(output_file)
    assert len(saved_df) == 1

    # Verify the specific transformation
    cleaned_val = saved_df.iloc[0]['value']
    assert "daic_laughter" in cleaned_val
    assert "how are you" not in cleaned_val


def test_process_file_error(tmp_path):
    """Tests the pipeline's resilience to missing files."""
    pipeline = TranscriptPipeline()
    result = pipeline.process_file(Path("missing.csv"), tmp_path)
    assert result["status"] == "error"

