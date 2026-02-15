import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from remotezip import RemoteZip

from logger import configure_logger, get_logger

load_dotenv("../.env", override=True)

# --- LOGGING CONFIGURATION ---
IS_PRODUCTION = os.getenv("PYTHON_ENV") == "production"
configure_logger(IS_PRODUCTION, log_level="DEBUG")
logger = get_logger();

def validate_env():
    """Check if all required environment variables are set and paths exist."""
    required_vars = ["DATASET_URL", "TRAIN_SPLIT", "DEV_SPLIT", "TEST_SPLIT", "DATASET_RAW_DIR"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        raise EnvironmentError("Please check your .env file.")

    # Check if split CSV files exist
    for var in ["TRAIN_SPLIT", "DEV_SPLIT", "TEST_SPLIT"]:
        path = os.getenv(var)
        if not os.path.exists(path):
            logger.error(f"Split file not found at: {path}")
            raise FileNotFoundError(f"Check the path for {var}")

    logger.info("Environment validation successful.")

def fast_extract_transcript(p_id, base_url, target_dir):
    """Downloads ONLY the transcript file from the remote ZIP."""
    zip_url = f"{base_url.rstrip('/')}/{p_id}_P.zip"
    try:
        with RemoteZip(zip_url) as rz:
            # Find the transcript file
            transcript_file = next((f for f in rz.namelist() if f.endswith('_TRANSCRIPT.csv')), None)

            if transcript_file:
                # Check if file already exists to skip download
                target_path = Path(target_dir) / transcript_file
                if target_path.exists():
                    return f"Skipped {p_id}: Already exists"

                rz.extract(transcript_file, target_dir)
                return f"Success {p_id}"
            else:
                return f"Warning {p_id}: No _TRANSCRIPT.csv found in ZIP"
    except Exception as e:
        return f"Error {p_id}: {str(e)}"

def run_setup(base_url, split_files, target_dir, max_workers=20):
    """Main execution logic for streaming transcripts."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Collect and normalize Participant IDs
    ids = set()
    for f in split_files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]

        id_col = "participant_id"
        if id_col in df.columns:
            ids.update(df[id_col].dropna().astype(int).tolist())
        else:
            logger.warning(f"Column '{id_col}' not found in {f}. Available: {df.columns.tolist()}")

    participant_ids = sorted(list(ids))
    total_ids = len(participant_ids)
    logger.info(f"Found {total_ids} unique participants across all splits.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fast_extract_transcript, p_id, base_url, target_dir): p_id 
            for p_id in participant_ids
        }

        # Progress bar
        for future in tqdm(as_completed(futures), total=total_ids, desc="Streaming Transcripts"):
            results.append(future.result())

    # --- VERBOSE LOGGING OF RESULTS ---
    successes = [r for r in results if "Success" in r]
    skipped = [r for r in results if "Skipped" in r]
    warnings = [r for r in results if "Warning" in r]
    errors = [r for r in results if "Error" in r]

    logger.info("--- DOWNLOAD SUMMARY ---")
    logger.info(f"Total processed: {total_ids}")
    logger.info(f"Newly downloaded: {len(successes)}")
    logger.info(f"Already existed:  {len(skipped)}")

    if warnings:
        logger.warning(f"Missing transcripts: {len(warnings)}")
        for w in warnings: logger.debug(w)

    if errors:
        logger.error(f"Failed downloads: {len(errors)}")
        for e in errors: logger.error(e)

    logger.info(f"Data directory: {target_path.absolute()}")


if __name__ == "__main__":
    try:
        validate_env()

        run_setup(
            base_url=os.getenv("DATASET_URL"),
            split_files=[
                os.getenv("TRAIN_SPLIT"),
                os.getenv("DEV_SPLIT"),
                os.getenv("TEST_SPLIT")
            ],
            target_dir=os.getenv("DATASET_RAW_DIR"),
            max_workers=25
        )
    except Exception as e:
        logger.critical(f"Script aborted: {e}")
