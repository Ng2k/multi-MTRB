import pytest
from unittest.mock import MagicMock, patch
from src.processing.service import DatasetProcessingService


@pytest.fixture
def mock_settings(tmp_path):
    """Provides mock settings with temporary directories."""
    with patch("src.processing.service.settings") as mock:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Create two mock files to test multiple results
        (raw_dir / "101_TRANSCRIPT.csv").touch()
        (raw_dir / "102_TRANSCRIPT.csv").touch()

        mock.raw_data = raw_dir
        mock.clean_data = tmp_path / "clean"
        mock.max_workers = 2
        yield mock

def test_service_run_full_coverage(mock_settings):
    """
    Tests the parallel execution engine with both success and error branches
    to achieve 100% code coverage.
    """
    service = DatasetProcessingService()

    # Define results: one success, one error (to hit line 39)
    res_success = {"id": "101", "status": "success", "final": 10}
    res_error = {"id": "102", "status": "error", "error": "Mock Failure"}

    # We mock the pipeline's process_file to simulate different outcomes
    with patch.object(service.pipeline, "process_file") as mock_process:
        # side_effect allows us to return different values for consecutive calls
        mock_process.side_effect = [res_success, res_error]

        # Mock tqdm to avoid terminal interference
        with patch("src.processing.service.tqdm", side_effect=lambda x, **kwargs: x):
            # Mock as_completed to yield our simulated futures
            with patch("src.processing.service.as_completed") as mock_as_completed:
                future_success = MagicMock()
                future_success.result.return_value = res_success

                future_error = MagicMock()
                future_error.result.return_value = res_error

                mock_as_completed.return_value = [future_success, future_error]

                # Execute the service
                results = service.run()

    # Assertions for 100% coverage verification
    assert len(results) == 2

    # Verify Success Branch
    success_results = [r for r in results if r["status"] == "success"]
    assert len(success_results) == 1
    assert success_results[0]["id"] == "101"

    # Verify Error Branch (Line 39)
    error_results = [r for r in results if r["status"] == "error"]
    assert len(error_results) == 1
    assert error_results[0]["id"] == "102"
    assert error_results[0]["error"] == "Mock Failure"


def test_service_init():
    """Verifies service initialization and setting assignment."""
    with patch("src.processing.service.settings") as mock_s:
        mock_s.max_workers = 8
        service = DatasetProcessingService()
        assert service.max_workers == 8
        assert service.pipeline is not None

