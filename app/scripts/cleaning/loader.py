""" Script Loader

Module for handling the loading and cleaning of DAIC-WOZ transcript datasets.

Usage example:

"""

import pandas as pd
from pathlib import Path
from typing import List, Callable, Optional, cast

from logger import get_logger
from scripts.cleaning.strategies import CleaningStrategies

logger = get_logger().bind(module="scripts.cleaning.loader")

class ScriptLoader:
    """Orchestrator for loading and cleaning DAIC-WOZ transcripts.

    This class manages the extraction of participant dialogue from raw CSV files
    and applies a sequence of transformation strategies to prepare text for
    feature extraction.

    Attributes:
        pipeline (List[Callable[[str], str]]): A sequence of functions applied 
            to each utterance.
    """

    def __init__(self, pipeline: Optional[List[Callable[[str], str]]] = None) -> None:
        """Initializes the ScriptLoader with a cleaning pipeline.

        Args:
            pipeline: An optional list of cleaning functions. If None, uses
                the default research-standard pipeline including tag removal,
                lowercasing, and whitespace normalization.
        """
        self.pipeline = pipeline or [
            CleaningStrategies.remove_daic_tags,
            CleaningStrategies.lowercase,
            CleaningStrategies.remove_special_chars,
            CleaningStrategies.collapse_whitespace,
            CleaningStrategies.strip_edges
        ]
        logger.info("ScriptLoader initialized", pipeline_steps=len(self.pipeline))


    def _apply_pipeline(self, text: str) -> str:
        """Applies all pipeline transformations to a single string.

        Args:
            text: The raw input string.

        Returns:
            The fully transformed and cleaned string.
        """
        for transform in self.pipeline:
            text = transform(text)
        return text


    def load_and_clean(self, file_path: Path) -> pd.DataFrame:
        """Loads a transcript file and executes the cleaning pipeline.

        Args:
            file_path: Absolute or relative path to the DAIC-WOZ _TRANSCRIPT.csv.

        Returns:
            A pandas DataFrame containing a single 'value' column with cleaned 
            participant utterances.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ValueError: If the file format is incorrect or columns are missing.
            RuntimeError: If an unexpected error occurs during processing.
        """
        local_logger = logger.bind(file_path=str(file_path))

        if not file_path.exists():
            local_logger.error("File check failed: transcript not found")
            raise FileNotFoundError(f"Transcript not found: {file_path}")

        try:
            # DAIC-WOZ is tab-separated (\t)
            df_raw = pd.read_csv(file_path, sep='\t').fillna("")

            if not isinstance(df_raw, pd.DataFrame):
                local_logger.error("IO Error: loaded object is not a DataFrame")
                raise ValueError("Loaded object is not a DataFrame")

            if 'speaker' not in df_raw.columns or 'value' not in df_raw.columns:
                local_logger.error("Validation Error: malformed CSV columns", 
                                   columns=list(df_raw.columns))
                raise ValueError(f"Malformed CSV at {file_path}")

            df_participant = df_raw[df_raw['speaker'] == 'Participant'].copy()
            df_participant = cast(pd.DataFrame, df_participant)

            initial_count = len(df_participant)

            df_participant['value'] = (
                df_participant['value']
                .astype(str)
                .map(self._apply_pipeline)
            )

            mask = df_participant['value'].str.len() > 0
            df_final = df_participant.loc[mask, ['value']]
            df_final = cast(pd.DataFrame, df_final)

            final_count = len(df_final)

            local_logger.info("Processing successful", 
                              raw_utterances=initial_count, 
                              clean_utterances=final_count,
                              reduction_ratio=round(1 - (final_count/initial_count if initial_count > 0 else 0), 2))

            return df_final.reset_index(drop=True)

        except Exception as e:
            local_logger.exception("Pipeline execution failed", error=str(e))
            raise RuntimeError(f"Error processing {file_path}") from e

