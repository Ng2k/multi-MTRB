import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import configure_logger, get_logger
from src.features.manager import FeatureManager

# 1. Setup Environment and Logger
load_dotenv("../.env", override=True)
configure_logger(os.getenv("PYTHON_ENV") == "production", log_level="INFO")
logger = get_logger().bind(module="cleaning_pipeline")

def main():
    start_time = time.time()

    # 2. Define Paths from .env or Defaults
    input_dir = Path(os.getenv("DATASET_CLEAN_DIR", "../../../dataset/clean"))
    output_dir = Path(os.getenv("DATASET_FEATURES_DIR", "../../../dataset/features"))

    logger.info("Initializing Feature Extraction Pipeline", 
                input_source=str(input_dir), 
                output_destination=str(output_dir))

    # 3. Validation
    if not input_dir.exists():
        logger.error(
            "Cleaned data directory not found. Did you run the cleaning script first?", 
            path=str(input_dir)
        )
        return

    # 4. Initialize Manager and Run Extraction
    try:
        manager = FeatureManager(input_dir=input_dir, output_dir=output_dir)
        results = manager.process_all(max_workers=4)

        # 5. Summary Reporting
        success_count = len([r for r in results if r["status"] == "success"])
        skipped_count = len([r for r in results if r["status"] == "skipped"])
        error_count = len(results) - success_count - skipped_count

        duration = (time.time() - start_time) / 60

        logger.info("Feature Extraction Task Completed",
                    total_files=len(results),
                    successful=success_count,
                    skipped=skipped_count,
                    failed=error_count,
                    duration_minutes=round(duration, 2))

    except Exception as e:
        logger.exception("Critical failure in Feature Manager", error=str(e))


if __name__ == "__main__":
    main()

