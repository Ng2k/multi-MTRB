import os

from src.utils import configure_logger, get_logger, settings
from src.processing import DatasetProcessingService

def main():
    """
    Entry point for the DAIC-WOZ transcript cleaning pipeline.

    This script initializes the structured logging system and executes the 
    parallelized DatasetProcessingService to transform raw transcripts into 
    cleaned CSVs for feature extraction.
    """
    env = os.getenv("PYTHON_ENV", "development")
    configure_logger(enable_json=False, log_level="INFO" if env == "production" else "DEBUG")
    logger = get_logger().bind(module="clean_entry")

    logger.info("cleaning_pipeline_started")

    try:
        service = DatasetProcessingService()
        results = service.run()

        success_count = len([r for r in results if r["status"] == "success"])
        error_count = len([r for r in results if r["status"] == "error"])

        logger.info(
            "cleaning_pipeline_finished",
            total_processed=len(results),
            successful=success_count,
            failed=error_count
        )

    except Exception as e:
        logger.exception("cleaning_pipeline_critical_failure", error=str(e))
        exit(1)

if __name__ == "__main__":
    main()

