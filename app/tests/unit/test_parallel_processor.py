"""
Unit tests for the ParallelCleaner and worker functions.
Uses mocking to simulate process execution without real multiprocessing overhead.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.cleaning.parallel_processor import ParallelCleaner, _process_single_file

class TestParallelProcessor:

    @pytest.fixture
    def mock_dirs(self, tmp_path):
        """Creates dummy input and output directories."""
        in_dir = tmp_path / "raw"
        out_dir = tmp_path / "clean"
        in_dir.mkdir()

        (in_dir / "300_TRANSCRIPT.csv").write_text("dummy content")
        # Create a dummy hidden system file to test the filter
        (in_dir / "._300_TRANSCRIPT.csv").write_text("hidden")

        return in_dir, out_dir

    def test_process_single_file_error(self, mock_dirs, monkeypatch):
        """Verifies the worker catches exceptions and returns an error dict."""
        in_dir, out_dir = mock_dirs
        file_path = in_dir / "300_TRANSCRIPT.csv"

        # Force an exception during loader initialization
        def mock_init_fail():
            raise ValueError("Simulated Load Failure")
        monkeypatch.setattr("src.cleaning.parallel_processor.ScriptLoader", mock_init_fail)

        result = _process_single_file(file_path, out_dir)

        assert result["status"] == "error"
        assert "Simulated Load Failure" in result["error"]

    def test_process_single_file_success(self, mock_dirs, monkeypatch):
        """Verifies the worker correctly loads, cleans, and saves a file."""
        in_dir, out_dir = mock_dirs
        # IMPORTANT: Create the output directory because the worker expects it to exist
        out_dir.mkdir(parents=True, exist_ok=True) 

        file_path = in_dir / "300_TRANSCRIPT.csv"

        # Mock ScriptLoader
        mock_loader = MagicMock()
        mock_loader.load_and_clean.return_value = pd.DataFrame({"value": ["line1", "line2"]})
        monkeypatch.setattr("src.cleaning.parallel_processor.ScriptLoader", lambda: mock_loader)

        result = _process_single_file(file_path, out_dir)

        # Debugging: if it still fails, print the error captured in the result dict
        if result["status"] == "error":
             pytest.fail(f"Worker failed unexpectedly with: {result.get('error')}")

        assert result["status"] == "success"
        assert result["id"] == "300"
        assert (out_dir / "300_CLEAN.csv").exists()

    # --- Test Orchestrator (ParallelCleaner) ---
    def test_cleaner_initialization(self):
        """Ensures worker count defaults correctly."""
        cleaner = ParallelCleaner(max_workers=4)
        assert cleaner.max_workers == 4
 
        cleaner_default = ParallelCleaner()
        assert cleaner_default.max_workers is not None # Should be cpu_count

    def test_run_orchestration(self, mock_dirs):
        """
        Uses mocking to verify that the ProcessPoolExecutor is called correctly
        without actually spawning real processes.
        """
        in_dir, out_dir = mock_dirs
        cleaner = ParallelCleaner(max_workers=1)

        # We mock the entire ProcessPoolExecutor to avoid real subprocesses
        with patch("src.cleaning.parallel_processor.ProcessPoolExecutor") as mock_executor:
            # Setup the mock to return a completed future
            mock_future = MagicMock()
            mock_future.result.return_value = {"id": "300", "status": "success", "utterances": 2}
 
            # This simulates the context manager and the as_completed generator
            executor_instance = mock_executor.return_value.__enter__.return_value
            executor_instance.submit.return_value = mock_future
 
            with patch("src.cleaning.parallel_processor.as_completed", return_value=[mock_future]):
                results = cleaner.run(in_dir, out_dir)

        assert len(results) == 1
        assert results[0]["id"] == "300"
        assert out_dir.exists() # Verifies mkdir was called

    def test_run_logs_error_on_failure(self, mock_dirs):
        """Ensures the orchestrator correctly identifies and logs failures."""
        in_dir, out_dir = mock_dirs
        cleaner = ParallelCleaner(max_workers=1)

        with patch("src.cleaning.parallel_processor.ProcessPoolExecutor") as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = {"id": "300_TRANSCRIPT.csv", "status": "error", "error": "fail"}
 
            executor_instance = mock_executor.return_value.__enter__.return_value
            executor_instance.submit.return_value = mock_future
 
            with patch("src.cleaning.parallel_processor.as_completed", return_value=[mock_future]):
                results = cleaner.run(in_dir, out_dir)

        assert results[0]["status"] == "error"

