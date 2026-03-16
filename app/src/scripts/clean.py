import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import configure_logger, get_logger
from src.cleaning.parallel_processor import ParallelCleaner

# 1. Environment and Logger Setup
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="cleaning_pipeline")

def main():
    start_time = time.time()

    # 2. Define Paths
    raw_dir = Path(os.getenv("DATASET_RAW_DIR", "../../../dataset/raw"))
    output_dir = Path(os.getenv("DATASET_CLEAN_DIR", "../../../dataset/clean"))

    logger.info(
        "Initializing Cleaning Pipeline", 
        input_source=str(raw_dir), 
        output_destination=str(output_dir)
    )

    # 3. Validation
    if not raw_dir.exists():
        logger.error("Input directory does not exist. Check your paths.", path=str(raw_dir))
        return

    # 4. Execute Parallel Cleaning
    cleaner = ParallelCleaner(max_workers=None)

    results = cleaner.run(raw_dir, output_dir)

    # 5. Reporting
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    duration = (time.time() - start_time) / 60

    logger.info("Cleaning Task Completed",
                total_processed=len(results),
                successful=len(successes),
                failed=len(errors),
                duration_minutes=round(duration, 2))

    if errors:
        logger.warning("Some files failed to process", failed_count=len(errors))
        for err in errors[:5]:  # Show first 5 errors
            logger.debug("Failure details", file=err["id"], error=err["error"])


if __name__ == "__main__":
    main()

