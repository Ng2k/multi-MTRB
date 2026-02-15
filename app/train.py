"""
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from logger import configure_logger, get_logger
from scripts.cleaning.parallel_processor import ParallelCleaner


# --- LOGGING CONFIGURATION ---
load_dotenv("../.env", override=True)
IS_PRODUCTION = os.getenv("PYTHON_ENV") == "production"
configure_logger(IS_PRODUCTION, log_level="DEBUG")

from scripts.cleaning.loader import ScriptLoader


logger = get_logger();

if __name__ == "__main__":
    logger.info("Starting training script")
    loader = ScriptLoader()

    # Configuration from ENV or defaults
    RAW_DATA_DIR = Path(os.getenv("DATASET_RAW_DIR", "../data/raw"))
    CLEAN_DATA_DIR = Path("../data/clean")

    cleaner = ParallelCleaner()
    stats = cleaner.run(RAW_DATA_DIR, CLEAN_DATA_DIR)

    # Optional: Quick audit summary
    total_utterances = sum(s.get("utterances", 0) for s in stats if s["status"] == "success")
    logger.info("Pipeline finished", total_extracted_utterances=total_utterances)
