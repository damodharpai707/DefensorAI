import logging
from src.detection.yara_rules.phishing_rules_loader import update_phishing_lists, download_yara_rules, compile_yara_rules

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def update_phishing_data():
    """
    Update phishing datasets and YARA rules.
    """
    logging.info("Starting the update process for phishing data and YARA rules...")

    # Step 1: Update phishing domain and URL lists
    try:
        logging.info("Updating phishing domain and URL lists...")
        update_phishing_lists()
        logging.info("Phishing domain and URL lists updated successfully.")
    except Exception as e:
        logging.error(f"Error updating phishing datasets: {e}")

    # Step 2: Download and update YARA rules
    try:
        logging.info("Updating YARA rules...")
        download_yara_rules()
        logging.info("YARA rules downloaded successfully.")
    except Exception as e:
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
        logging.error(f"Error compiling YARA rules: {e}")

    logging.info("Phishing data update process completed.")

if __name__ == "__main__":
    update_phishing_data()
