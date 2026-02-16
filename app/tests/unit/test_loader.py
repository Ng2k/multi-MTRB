"""
Unit tests for the ScriptLoader class.
Covers I/O handling, column validation, and pipeline orchestration.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.cleaning.loader import ScriptLoader

class TestScriptLoader:
    """Suite to verify DAIC-WOZ transcript loading and cleaning orchestration."""

    @pytest.fixture
    def mock_csv(self, tmp_path):
        """Fixture to create a temporary valid DAIC-WOZ tab-separated file."""
        file_path = tmp_path / "300_TRANSCRIPT.csv"
        content = (
            "start_time\tstop_time\tspeaker\tvalue\n"
            "0.1\t0.5\tEllie\tHi [noise]\n"
            "0.6\t1.0\tParticipant\tI am [laughter] happy\n"
            "1.1\t1.5\tParticipant\t\n"  # Empty utterance
            "1.6\t2.0\tEllie\tHow are you?\n"
            "2.1\t2.5\tParticipant\tI'm GOOD.\n"
        )
        file_path.write_text(content)
        return file_path

    def test_init_default_pipeline(self):
        """Ensures loader initializes with the correct number of default steps."""
        loader = ScriptLoader()
        # Default pipeline has 5 steps defined in the code
        assert len(loader.pipeline) == 5
        assert loader.pipeline[0].__name__ == "remove_daic_tags"

    def test_init_custom_pipeline(self):
        """Ensures loader accepts a custom list of transformation functions."""
        mock_transform = MagicMock()
        loader = ScriptLoader(pipeline=[mock_transform])
        assert loader.pipeline == [mock_transform]

    def test_load_and_clean_happy_path(self, mock_csv):
        """
        GIVEN a valid transcript file
        WHEN load_and_clean is called
        THEN it should only return cleaned 'Participant' utterances.
        """
        loader = ScriptLoader()
        df = loader.load_and_clean(mock_csv)

        # Original had 3 participant lines, but 1 was empty -> should result in 2
        assert len(df) == 2
        assert list(df.columns) == ["value"]
        assert df.iloc[0]["value"] == "i am happy"
        assert df.iloc[1]["value"] == "i'm good."

    def test_load_and_clean_file_not_found(self, tmp_path):
        """Ensures an empty DataFrame is returned if the file doesn't exist."""
        loader = ScriptLoader()
        non_existent = tmp_path / "ghost.csv"
        df = loader.load_and_clean(non_existent)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_and_clean_malformed_columns(self, tmp_path):
        """Ensures empty DataFrame is returned if required columns are missing."""
        file_path = tmp_path / "bad_cols.csv"
        # Missing 'speaker' column
        file_path.write_text("start_time\tvalue\n0.1\tHello\n")
 
        loader = ScriptLoader()
        df = loader.load_and_clean(file_path)
        assert df.empty

    def test_load_and_clean_runtime_error(self, mock_csv, monkeypatch):
        """Ensures a RuntimeError is raised if pandas fails unexpectedly."""
        loader = ScriptLoader()

        # Simulate a crash during pd.read_csv
        def mock_read_fail(*args, **kwargs):
            raise Exception("Critical Disk Failure")
 
        monkeypatch.setattr(pd, "read_csv", mock_read_fail)

        with pytest.raises(RuntimeError, match="Error processing"):
            loader.load_and_clean(mock_csv)

    def test_apply_pipeline_logic(self):
        """Directly tests the internal _apply_pipeline method."""
        # Create a mock pipeline that just appends '!'
        mock_step = lambda x: x + "!"
        loader = ScriptLoader(pipeline=[mock_step, mock_step])
 
        result = loader._apply_pipeline("test")
        assert result == "test!!"

    def test_load_and_clean_not_a_dataframe(self, mock_csv, monkeypatch):
        """
        Forces the 'not isinstance(df_raw, pd.DataFrame)' branch.
        We return a MagicMock that handles .fillna() but isn't a DataFrame.
        """
        loader = ScriptLoader()
 
        # Create an object that has a fillna method
        mock_obj = MagicMock()
        mock_obj.fillna.return_value = mock_obj # So .fillna("").something works
 
        # Mock read_csv to return our fake object
        monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: mock_obj)

        # This will now pass line 85 and trigger the 'if not isinstance' block
        df = loader.load_and_clean(mock_csv)
 
        assert isinstance(df, pd.DataFrame)
        assert df.empty

