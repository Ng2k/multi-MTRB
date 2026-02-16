"""
Module for parallelized cleaning of the DAIC-WOZ dataset.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger
from src.cleaning.loader import ScriptLoader

logger = get_logger().bind(module="scripts.cleaning.parallel_processor")

def _process_single_file(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Worker function executed in separate processes.

    Args:
        file_path: Path to the raw transcript.
        output_dir: Directory to save the cleaned CSV.

    Returns:
        A dictionary containing processing metadata.
    """
    try:
        loader = ScriptLoader()
        participant_id = file_path.name.split("_")[0]

        df_clean = loader.load_and_clean(file_path)

        output_path = output_dir / f"{participant_id}_CLEAN.csv"
        df_clean.to_csv(output_path, index=False)

        return {
            "id": participant_id,
            "status": "success",
            "utterances": len(df_clean)
        }
    except Exception as e:
        return {
            "id": file_path.name,
            "status": "error",
            "error": str(e)
        }


class ParallelCleaner:
    """Orchestrates multi-process cleaning of transcript datasets."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Args:
            max_workers: Number of processes. Defaults to number of CPU cores.
        """
        self.max_workers = max_workers or os.cpu_count()

    def run(self, input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Executes the cleaning pipeline in parallel.

        Args:
            input_dir: Directory containing raw transcripts.
            output_dir: Directory where cleaned CSVs will be stored.

        Returns:
            A list of result dictionaries for each file processed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript_files = [
            f for f in input_dir.glob("**/*_TRANSCRIPT.csv") 
            if not f.name.startswith("._")
        ]
        total_files = len(transcript_files)

        logger.info("Starting parallel cleaning", 
                    total_files=total_files, 
                    workers=self.max_workers)

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Schedule the worker function for each file
            futures = {
                executor.submit(_process_single_file, f, output_dir): f 
                for f in transcript_files
            }

            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if res["status"] == "error":
                    logger.error("File processing failed", file=res["id"], error=res["error"])

        logger.info("Parallel cleaning complete", 
                    success_count=len([r for r in results if r["status"] == "success"]),
                    error_count=len([r for r in results if r["status"] == "error"]))

        return results

