# phishing_rules_loader.py
import logging
import os
import requests
import hashlib
import concurrent.futures
import time
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Optional, Dict
from datetime import datetime

@dataclass(frozen=True)
class YaraConfig:
    base_dir: Path
    yara_folder: Path
    yara_rules_dir: Path
    index_file: Path
    github_api: str
    max_retries: int = 3
    timeout: int = 30
    chunk_size: int = 8192
    max_workers: int = 5

class YaraRulesManager:
    def __init__(self, config: YaraConfig):
        self.config = config
        self.session = self._create_session()
        self._setup_logging()
        self._setup_directories()
        self._metrics: Dict[str, float] = {}

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'YaraRulesManager/1.0',
            'Accept': 'application/json'
        })
        return session

    def _setup_logging(self) -> None:
        log_file = self.config.base_dir / "yara_updates.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("YaraRulesManager")

    def _setup_directories(self) -> None:
        for directory in [self.config.yara_folder, self.config.yara_rules_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_remote_yara_rules(self) -> Set[str]:
        try:
            response = self.session.get(self.config.github_api)
            response.raise_for_status()
            rules = {item['name'] for item in response.json() if item['name'].endswith('.yar')}
            print(f"\nFound {len(rules)} YARA rules in repository")
            return rules
        except Exception as e:
            self.logger.error(f"Failed to fetch remote YARA rules: {e}")
            return set()

    def _get_current_yara_rules(self) -> Set[str]:
        try:
            if self.config.index_file.exists():
                rules = set(self.config.index_file.read_text(encoding="utf-8").splitlines())
                print(f"Current local YARA rules: {len(rules)}")
                return rules
            return set()
        except Exception as e:
            self.logger.error(f"Error reading index file: {e}")
            return set()

    def update_yara_rules(self) -> bool:
        try:
            current_rules = self._get_current_yara_rules()
            remote_rules = self._get_remote_yara_rules()
            new_rules = remote_rules - current_rules
            
            if new_rules:
                print(f"\nFound {len(new_rules)} new YARA rules to download!")
                with tqdm(desc="Updating index.txt", total=len(new_rules), unit="rule") as pbar:
                    with open(self.config.index_file, "a", encoding="utf-8") as f:
                        for rule in new_rules:
                            f.write(f"{rule}\n")
                            pbar.update(1)
                
                with tqdm(desc="Downloading YARA rules", total=len(new_rules), unit="rule") as pbar:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                        futures = []
                        for rule in new_rules:
                            futures.append(executor.submit(self._download_yara_rule, rule))
                        
                        for future in concurrent.futures.as_completed(futures):
                            pbar.update(1)
            else:
                print("\nYARA rules are up to date!")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update YARA rules: {e}")
            return False

    def _download_yara_rule(self, rule_name: str) -> bool:
        try:
            destination = self.config.yara_rules_dir / rule_name
            url = f"https://raw.githubusercontent.com/t4d/PhishingKit-Yara-Rules/master/{rule_name}"
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            with open(destination, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to download YARA rule {rule_name}: {e}")
            return False