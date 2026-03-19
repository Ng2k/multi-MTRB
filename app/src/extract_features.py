"""
Entry point for the Multi-MTRB Feature Extraction pipeline.

This script transforms cleaned transcripts into high-dimensional tensors (1280-dim)
using a dual-stream Transformer approach (RoBERTa + mT5), optimized for 
Multiple Instance Learning (MIL) tasks.
"""

import sys
from src.utils import configure_logger, get_logger, settings
from src.features import FeatureManager

def main():
    configure_logger(enable_json=False)
    logger = get_logger().bind(module="feature_extraction_entry")

    logger.info("feature_extraction_pipeline_started")

    try:
        input_dir = settings.clean_data
        output_dir = settings.features

        if not input_dir.exists():
            logger.error("input_directory_not_found", path=str(input_dir))
            sys.exit(1)

        manager = FeatureManager(
            input_dir=input_dir, 
            output_dir=output_dir
        )

        results = manager.process_all(max_workers=settings.max_workers)

        success_count = len([r for r in results if r["status"] == "success"])
        skipped_count = len([r for r in results if r["status"] == "skipped"])
        error_count = len([r for r in results if r["status"] == "error"])

        logger.info(
            "feature_extraction_pipeline_complete",
            total=len(results),
            successful=success_count,
            skipped=skipped_count,
            failed=error_count
        )

    except Exception as e:
        logger.exception("feature_extraction_critical_failure", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

