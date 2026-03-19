from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

from src.utils import get_logger, settings
from src.processing.pipeline import TranscriptPipeline

class DatasetProcessingService:
    """Enterprise-level parallel processing service with progress tracking."""

    def __init__(self):
        self.logger = get_logger().bind(module="processing.service")
        self.max_workers = settings.max_workers
        self.pipeline = TranscriptPipeline()

    def run(self) -> List[Dict[str, Any]]:
        """Processes all transcripts from raw_data to clean_data directory with a progress bar."""
        input_dir = settings.raw_data
        output_dir = settings.clean_data
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("**/*_TRANSCRIPT.csv"))
        total_files = len(files)

        self.logger.info("Starting dataset cleaning", file_count=total_files)

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.pipeline.process_file, f, output_dir): f 
                for f in files
            }

            for future in tqdm(as_completed(futures), total=total_files, desc="Cleaning Transcripts"):
                res = future.result()
                results.append(res)

                if res["status"] == "error":
                    self.logger.error("Processing failed", id=res["id"], error=res["error"])
                else:
                    pass 

        self.logger.info("Dataset cleaning complete", 
                         successful=len([r for r in results if r["status"] == "success"]),
                         failed=len([r for r in results if r["status"] == "error"]))

        return results

