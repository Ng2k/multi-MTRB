"""Module for management of Multi-MTRB feature extraction.

This module provides a multi-threaded orchestrator that maximizes throughput on 
modern GPUs (like the RTX 4060Ti) by overlapping I/O operations with 
Transformer inference.

Typical usage example:
    manager = FeatureManager(input_dir=Path("data/clean"), output_dir=Path("data/features"))
    results = manager.process_all(max_workers=4)
"""

import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch

from src.features.text_extractor import MultiMTRBExtractor
from src.utils.logger import get_logger

logger = get_logger().bind(module="features.manager")


class FeatureManager:
    """Orchestrates high-speed bulk extraction of features using multi-threading.

    This manager utilizes a thread pool to handle file I/O and pre-processing
    concurrently while the MultiMTRBExtractor handles GPU-bound inference.

    Attributes:
        input_dir: Directory containing cleaned CSV files.
        output_dir: Directory to save '.pt' feature tensors.
        extractor: The dual-stream MultiMTRBExtractor instance.
    """

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        """Initializes the FeatureManager and creates output directories.

        Args:
            input_dir: Path to directory with '*_CLEAN.csv' files.
            output_dir: Path to directory for '*_FEATURES.pt' files.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
 
        self.extractor = MultiMTRBExtractor()
 
        logger.info(
            "FeatureManager initialized for parallel execution", 
            input=str(input_dir), 
            output=str(output_dir)
        )

    def _process_single_session(self, file_path: Path) -> Dict[str, Any]:
        """Worker task to process a single participant session.

        This handles the I/O -> Inference -> Serialization pipeline for one file.

        Args:
            file_path: Path to the cleaned CSV file.

        Returns:
            A dictionary containing the status and metadata of the operation.
        """
        participant_id = file_path.name.split("_")[0]
        output_path = self.output_dir / f"{participant_id}_FEATURES.pt"

        # Idempotency check
        if output_path.exists():
            return {"id": participant_id, "status": "skipped"}

        try:
            df = pd.read_csv(file_path)
            feature_tensor = self.extractor.extract_session(df)
            torch.save(feature_tensor, output_path)

            return {
                "id": participant_id, 
                "status": "success", 
                "shape": list(feature_tensor.shape)
            }

        except Exception as e:
            logger.exception("Task failed", id=participant_id, error=str(e))
            return {"id": participant_id, "status": "error", "error": str(e)}

    def process_all(self, max_workers: int = 2) -> List[Dict[str, Any]]:
        """Executes the extraction pipeline across all files using a thread pool.

        Args:
            max_workers: Number of concurrent sessions to prepare. 
                Recommended 2-4 for an 8GB VRAM card to avoid OOM.

        Returns:
            List of result dictionaries for all processed files.
        """
        files = sorted(list(self.input_dir.glob("*_CLEAN.csv")))
        total_files = len(files)
        results = []

        logger.info(
            "Starting parallel feature extraction", 
            total_files=total_files,
            concurrency=max_workers
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the worker function across all files
            future_to_file = {
                executor.submit(self._process_single_session, f): f for f in files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                res = future.result()
                results.append(res)
 
                # Feedback on progress
                if res["status"] != "skipped":
                    logger.info(
                        "Session processed", 
                        id=res["id"], 
                        status=res["status"],
                        progress=f"{len(results)}/{total_files}"
                    )

        success_count = len([r for r in results if r["status"] == "success"])
        skipped_count = len([r for r in results if r["status"] == "skipped"])
 
        logger.info(
            "Bulk parallel extraction complete", 
            success=success_count,
            skipped=skipped_count,
            errors=total_files - success_count - skipped_count
        )

        return results

