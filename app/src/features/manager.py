"""Module for management of Multi-MTRB feature extraction."""

import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from tqdm import tqdm

from src.features.text_extractor import MultiMTRBExtractor
from src.utils import get_logger, settings

class FeatureManager:
    """Orchestrates parallel session extraction with progress tracking."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.logger = get_logger().bind(module="features.manager")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
 
        # Batch size can be tuned in config for your RTX 4060Ti
        self.extractor = MultiMTRBExtractor(batch_size=settings.batch_size)


    def _process_single_session(self, file_path: Path) -> Dict[str, Any]:
        participant_id = file_path.name.split("_")[0]
        output_path = self.output_dir / f"{participant_id}_FEATURES.pt"

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
            return {"id": participant_id, "status": "error", "error": str(e)}


    def process_all(self, max_workers: int = 2) -> List[Dict[str, Any]]:
        """Processes all clean transcripts with a progress bar."""
        files = sorted(list(self.input_dir.glob("*_CLEAN.csv")))
        total_files = len(files)
        results = []

        self.logger.info("Starting feature extraction", total=total_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_session, f): f for f in files
            }

            # Progress bar for the entire dataset
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                              total=total_files, 
                              desc="Extracting Features"):
                res = future.result()
                results.append(res)

                if res["status"] == "error":
                    self.logger.error("Session failed", id=res["id"], error=res.get("error"))

        return results

