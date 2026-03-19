import pytest
import torch
import pandas as pd
from unittest.mock import patch
from src.features.manager import FeatureManager

@pytest.fixture
def mock_settings():
    """Mocks global settings to avoid dependency on actual config files."""
    with patch("src.features.text_extractor.settings") as mock:
        mock.batch_size = 32
        mock.device = "cpu"
        yield mock


def test_process_single_session_skip(tmp_path, mock_settings):
    """Verifies idempotency check (skipping existing files)."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Create an "already processed" feature file
    (output_dir / "101_FEATURES.pt").touch()

    with patch("src.features.manager.MultiMTRBExtractor"):
        manager = FeatureManager(input_dir, output_dir)
        result = manager._process_single_session(input_dir / "101_CLEAN.csv")

    assert result["status"] == "skipped"
    assert result["id"] == "101"


def test_process_all_orchestration_with_error(tmp_path, mock_settings):
    """
    Verifies the batch processing loop and ensures line 68 (error logging) 
    is covered by simulating a failed session.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Create two mock files: one will succeed, one will fail
    file_success = input_dir / "101_CLEAN.csv"
    file_error = input_dir / "102_CLEAN.csv"

    pd.DataFrame({"value": ["good"]}).to_csv(file_success, index=False)
    pd.DataFrame({"value": ["bad"]}).to_csv(file_error, index=False)

    with patch("src.features.manager.MultiMTRBExtractor") as mock_ext_cls:
        mock_ext = mock_ext_cls.return_value

        # Side effect: first call succeeds, second call raises an exception
        # which triggers the internal try-except in _process_single_session
        mock_ext.extract_session.side_effect = [
            torch.randn(1, 1280),
            Exception("Simulated Extraction Failure")
        ]

        with patch("src.features.manager.tqdm", side_effect=lambda x, **kwargs: x):
            manager = FeatureManager(input_dir, output_dir)
            # Use max_workers=1 to ensure deterministic order for side_effect
            results = manager.process_all(max_workers=1)

    # Verify we have both results
    assert len(results) == 2

    statuses = [r["status"] for r in results]
    assert "success" in statuses
    assert "error" in statuses

    # Check that the error message was captured (covering line 68 in the manager)
    error_res = [r for r in results if r["status"] == "error"][0]
    assert "Simulated Extraction Failure" in error_res["error"]


def test_process_single_session_success(tmp_path, mock_settings):
    """Directly tests the success branch of the single session processor."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    csv_path = input_dir / "201_CLEAN.csv"
    pd.DataFrame({"value": ["hello"]}).to_csv(csv_path, index=False)

    with patch("src.features.manager.MultiMTRBExtractor") as mock_ext_cls:
        mock_ext = mock_ext_cls.return_value
        mock_ext.extract_session.return_value = torch.randn(1, 1280)

        manager = FeatureManager(input_dir, output_dir)
        result = manager._process_single_session(csv_path)

    assert result["status"] == "success"
    assert (output_dir / "201_FEATURES.pt").exists()

