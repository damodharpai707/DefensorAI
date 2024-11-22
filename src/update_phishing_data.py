import logging
import argparse
from src.detection.yara_rules.phishing_rules_loader import update_phishing_lists, download_yara_rules, compile_yara_rules

# Configure logging
log_level = "INFO"
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

def update_phishing_data(dry_run=False):
    """
    Update phishing datasets and YARA rules.
    Args:
        dry_run (bool): If True, only simulate the update process without making changes.
    Returns:
        bool: True if the process completes successfully, False otherwise.
    """
    success = True
    logging.info("Starting the update process for phishing data and YARA rules...")

    if dry_run:
        logging.info("Dry-run mode enabled. No files will be downloaded or modified.")
        return True

    # Step 1: Update phishing domain and URL lists
    try:
        logging.info("Updating phishing domain and URL lists...")
        update_phishing_lists()
        logging.info("Phishing domain and URL lists updated successfully.")
    except FileNotFoundError as e:
        success = False
        logging.error(f"File not found during update: {e}")
    except Exception as e:
        success = False
        logging.error(f"Unexpected error updating phishing datasets: {e}")

    # Step 2: Download and update YARA rules
    try:
        logging.info("Updating YARA rules...")
        download_yara_rules()
        logging.info("YARA rules downloaded successfully.")
    except Exception as e:
        success = False
        logging.error(f"Error updating YARA rules: {e}")

    # Step 3: Compile YARA rules
    try:
        logging.info("Compiling YARA rules...")
        compiled_rules = compile_yara_rules()
        if compiled_rules:
            logging.info("YARA rules compiled successfully and ready for use.")
        else:
            logging.warning("No YARA rules compiled.")
    except Exception as e:
        success = False
        logging.error(f"Error compiling YARA rules: {e}")

    logging.info("Phishing data update process completed.")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update phishing data and YARA rules.")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode.")
    args = parser.parse_args()

    if update_phishing_data(dry_run=args.dry_run):
        logging.info("Update process completed successfully.")
    else:
        logging.error("Update process encountered errors.")
