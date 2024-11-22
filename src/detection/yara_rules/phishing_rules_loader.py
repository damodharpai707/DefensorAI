import os
import requests
import tarfile
import yara
import logging
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from update_phishing_data import update_phishing_lists

# Paths and URLs
YARA_RULES_FOLDER = "src/detection/yara_rules/phishing_rules/"
DATA_FOLDER = "data/phishing/"
DOMAINS_FILE = os.path.join(DATA_FOLDER, "domains.lst")
URLS_FILE = os.path.join(DATA_FOLDER, "urls.lst")
INDEX_FILE = os.path.join(YARA_RULES_FOLDER, "index.txt")

DOMAINS_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-domains.tar.gz"
URLS_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-links.tar.gz"
GITHUB_YARA_REPO = "https://raw.githubusercontent.com/t4d/PhishingKit-Yara-Rules/main/"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_file(url, destination):
    """
    Download a file with a progress bar.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                pbar.update(len(chunk))

        logging.info(f"Downloaded: {destination}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")


def extract_tar_gz(tar_path, output_folder):
    """
    Extract a tar.gz file to a specified folder.
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_folder)
        logging.info(f"Extracted: {tar_path}")
    except Exception as e:
        logging.error(f"Failed to extract {tar_path}: {e}")


def file_hash(filepath):
    """
    Calculate SHA-256 hash of a file.
    """
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {filepath}: {e}")
        return None


def fetch_yara_rule_files():
    """
    Fetch the list of YARA rule files from the local index.txt file.
    Returns a list of filenames.
    """
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        logging.warning(f"{INDEX_FILE} not found. Creating a new one.")
        open(INDEX_FILE, "w").close()
        return []


def update_index_file():
    """
    Update the index.txt file with all `.yar` files in the YARA_RULES_FOLDER.
    Only adds new files to the index.
    """
    try:
        # Get current `.yar` files in the folder
        current_files = [f for f in os.listdir(YARA_RULES_FOLDER) if f.endswith(".yar")]

        # Get files already listed in index.txt
        indexed_files = fetch_yara_rule_files()

        # Find new files to add
        new_files = [f for f in current_files if f not in indexed_files]

        # Update index.txt
        if new_files:
            with open(INDEX_FILE, "a", encoding="utf-8") as file:
                for new_file in new_files:
                    file.write(f"{new_file}\n")
            logging.info(f"Updated {INDEX_FILE} with {len(new_files)} new files.")
        else:
            logging.info(f"No new files to add to {INDEX_FILE}.")
    except Exception as e:
        logging.error(f"Error updating {INDEX_FILE}: {e}")


def is_file_updated(rule_file, destination):
    """
    Check if the file needs to be downloaded by comparing the hash of the file.
    """
    try:
        response = requests.get(GITHUB_YARA_REPO + rule_file, stream=True, timeout=10)
        response.raise_for_status()

        remote_hash = hashlib.sha256(response.content).hexdigest()
        local_hash = file_hash(destination)

        return local_hash != remote_hash
    except Exception as e:
        logging.error(f"Error checking update status for {rule_file}: {e}")
        return True  # Assume update is needed


def download_yara_rules():
    """
    Download new or updated YARA rules listed in the index.txt file using concurrency.
    """
    os.makedirs(YARA_RULES_FOLDER, exist_ok=True)
    rule_files = fetch_yara_rule_files()

    if not rule_files:
        logging.warning("No rule files found in index.txt.")
        return

    def download_rule(rule_file):
        destination = os.path.join(YARA_RULES_FOLDER, rule_file)
        if not os.path.exists(destination) or is_file_updated(rule_file, destination):
            try:
                url = GITHUB_YARA_REPO + rule_file
                download_file(url, destination)
            except Exception as e:
                logging.error(f"Failed to download {rule_file}: {e}")
        else:
            logging.info(f"Skipped (already up-to-date): {rule_file}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_rule, rule_files)


def compile_yara_rules():
    """
    Compile all downloaded YARA rules into a single ruleset.
    """
    try:
        rule_files = [
            os.path.join(YARA_RULES_FOLDER, f)
            for f in os.listdir(YARA_RULES_FOLDER)
            if f.endswith(".yar")
        ]
        if not rule_files:
            logging.warning("No YARA rules found to compile.")
            return None

        rules = yara.compile(filepaths={os.path.basename(f): f for f in rule_files})
        logging.info(f"Compiled {len(rule_files)} YARA rules.")
        return rules
    except yara.SyntaxError as e:
        logging.error(f"Error compiling YARA rules: {e}")
        return None


if __name__ == "__main__":
    logging.info("Updating phishing domain and URL lists...")
    update_phishing_lists()

    logging.info("\nUpdating YARA rules...")
    update_index_file()
    download_yara_rules()

    logging.info("\nCompiling YARA rules...")
    compiled_rules = compile_yara_rules()

    if compiled_rules:
        logging.info("\nYARA rules successfully compiled and ready for use!")
    else:
        logging.warning("\nNo YARA rules compiled.")
