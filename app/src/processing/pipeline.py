import pandas as pd
from pathlib import Path
from typing import Dict, Any, cast
from .text_cleaner import TextCleaner

class TranscriptPipeline:
    """Orchestrates the cleaning flow for a single DAIC-WOZ transcript."""

    def process_file(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        participant_id: str = input_path.name.split("_")[0]
        try:
            # DAIC-WOZ transcripts are tab-separated
            df_raw: pd.DataFrame = pd.read_csv(input_path, sep='\t').fillna("")

            # Filter specifically for Participant dialogue
            df_participant = cast(pd.DataFrame, df_raw[df_raw['speaker'] == 'Participant'].copy())
            initial_count: int = len(df_participant)

            # Apply domain-specific cleaning
            df_participant['value'] = (
                df_participant['value']
                .astype(str)
                .apply(TextCleaner.apply_clinical_tags)
                .apply(TextCleaner.sanitize)
            )

            # Drop empty rows and isolate the text column
            mask = df_participant['value'].str.len() > 0
            df_final = cast(pd.DataFrame, df_participant.loc[mask, ['value']])

            output_path: Path = output_dir / f"{participant_id}_CLEAN.csv"
            df_final.to_csv(output_path, index=False)

            return {
                "id": participant_id,
                "status": "success",
                "initial": initial_count,
                "final": len(df_final)
            }
        except Exception as e:
            return {"id": participant_id, "status": "error", "error": str(e)}

