import logging
import requests
import tarfile
import concurrent.futures
import time
import pandas as pd
import humanize
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
from phishing_rules_loader import YaraRulesManager, YaraConfig

class DownloadProgress:
    def __init__(self, filename):
        self.root = tk.Tk()
        self.root.title(f"Downloading {filename}")
        self.root.geometry('400x150')
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width/2) - (400/2)
        y = (screen_height/2) - (150/2)
        self.root.geometry(f'+{int(x)}+{int(y)}')

        self.label = ttk.Label(self.root, text="Starting download...")
        self.label.pack(pady=20)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("custom.Horizontal.TProgressbar",
                       troughcolor='#E0E0E0',
                       background='#00A86B',
                       thickness=15)

        self.progress = ttk.Progressbar(
            self.root,
            style="custom.Horizontal.TProgressbar",
            length=350,
            mode='determinate'
        )
        self.progress.pack(pady=10)

    def update(self, current, total):
        percentage = (current / total) * 100
        self.progress['value'] = percentage
        size = humanize.naturalsize(current)
        total_size = humanize.naturalsize(total)
        self.label['text'] = f"Downloaded: {size} / {total_size} ({percentage:.1f}%)"
        self.root.update()

    def close(self):
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None
            import gc
            gc.collect()

@dataclass(frozen=True)
class PhishingConfig:
    base_dir: Path
    data_folder: Path
    domains_file: Path
    urls_file: Path
    domains_url: str
    urls_url: str
    max_retries: int = 3
    timeout: int = 30
    chunk_size: int = 8192
    max_workers: int = 5

class PhishingDataManager:
    def __init__(self, config: PhishingConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_logging()
        self._setup_directories()
        self._metrics: Dict[str, float] = {}

    def _setup_logging(self) -> None:
        log_file = self.config.base_dir / "phishing_updates.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PhishingDataManager")

    def _setup_directories(self) -> None:
        self.config.data_folder.mkdir(parents=True, exist_ok=True)

    def _check_updates_available(self) -> Tuple[bool, bool]:
        try:
            domains_response = self.session.head(self.config.domains_url)
            urls_response = self.session.head(self.config.urls_url)
            
            domains_new = not self.config.domains_file.exists() or \
                int(domains_response.headers.get('content-length', 0)) != self.config.domains_file.stat().st_size
            urls_new = not self.config.urls_file.exists() or \
                int(urls_response.headers.get('content-length', 0)) != self.config.urls_file.stat().st_size
                
            return domains_new, urls_new
        except Exception:
            return True, True

    def _download_with_retry(self, url: str, destination: Path) -> bool:
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, stream=True, timeout=self.config.timeout)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                progress = DownloadProgress(destination.name)
                downloaded = 0

                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            file.write(chunk)
                            downloaded += len(chunk)
                            progress.update(downloaded, total_size)
                
                progress.close()
                return True
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return False
                time.sleep(2 ** attempt)
        return False

    def convert_lst_to_csv(self) -> bool:
        try:
            anomaly_dataset_csv = self.config.data_folder / "anomaly_dataset.csv"
            
            if anomaly_dataset_csv.exists():
                existing_df = pd.read_csv(anomaly_dataset_csv)
                existing_urls = set(existing_df['url'].values)
            else:
                existing_df = pd.DataFrame(columns=['url', 'label'])
                existing_urls = set()

            new_entries = []
            
            if self.config.domains_file.exists():
                with open(self.config.domains_file, 'r', encoding='utf-8', errors='ignore') as f:
                    domains = [line.strip() for line in f if line.strip()]
                    for domain in domains:
                        if domain not in existing_urls:
                            new_entries.append({'url': domain, 'label': 0})
                self.config.domains_file.unlink()

            if self.config.urls_file.exists():
                with open(self.config.urls_file, 'r', encoding='utf-8', errors='ignore') as f:
                    urls = [line.strip() for line in f if line.strip()]
                    for url in urls:
                        if url not in existing_urls:
                            new_entries.append({'url': url, 'label': 0})
                self.config.urls_file.unlink()

            if new_entries:
                new_df = pd.DataFrame(new_entries)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(anomaly_dataset_csv, index=False, encoding='utf-8')
                print(f"\nAdded {len(new_entries)} new entries to anomaly_dataset.csv")
                print(f"Total size of anomaly_dataset.csv: {humanize.naturalsize(anomaly_dataset_csv.stat().st_size)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV conversion failed: {e}")
            return False

    def update_phishing_data(self) -> bool:
        start_time = time.time()
        success = True

        try:
            domains_update, urls_update = self._check_updates_available()
            
            if domains_update or urls_update:
                print("\nPhishing data updates available!")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    domains_tar = self.config.data_folder / "domains.tar.gz"
                    urls_tar = self.config.data_folder / "urls.tar.gz"

                    if domains_update:
                        futures.append(
                            executor.submit(self._download_with_retry, 
                                         self.config.domains_url, 
                                         domains_tar)
                        )
                    if urls_update:
                        futures.append(
                            executor.submit(self._download_with_retry, 
                                         self.config.urls_url, 
                                         urls_tar)
                        )

                    concurrent.futures.wait(futures)
                    if all(future.result() for future in futures):
                        if domains_update:
                            success &= self._extract_tar_gz(domains_tar, 
                                                          self.config.domains_file)
                        if urls_update:
                            success &= self._extract_tar_gz(urls_tar, 
                                                          self.config.urls_file)
                        if success:
                            success &= self.convert_lst_to_csv()

            elapsed_time = time.time() - start_time
            print(f"\nUpdate completed in {elapsed_time:.2f} seconds")
            return success

        except Exception as e:
            self.logger.error(f"Update process failed: {e}")
            return False

    def _extract_tar_gz(self, tar_path: Path, output_file: Path) -> bool:
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                members = tar.getmembers()
                if not members:
                    raise ValueError("Empty archive")
                    
                content_file = members[0]
                with tar.extractfile(content_file) as source:
                    if not source:
                        raise ValueError("Cannot read source file")
                    content = source.read()
                    
                    with open(output_file, 'wb') as target:
                        target.write(content)
                        
            tar_path.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {tar_path}: {e}")
            return False

def main():
    base_dir = Path(__file__).parent.parent
    phishing_config = PhishingConfig(
        base_dir=base_dir,
        data_folder=base_dir / "data" / "phishing",
        domains_file=base_dir / "data" / "phishing" / "domains.lst",
        urls_file=base_dir / "data" / "phishing" / "urls.lst",
        domains_url="https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-domains.tar.gz",
        urls_url="https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-links.tar.gz"
    )

    yara_config = YaraConfig(
        base_dir=base_dir,
        yara_folder=base_dir / "detection" / "yara_rules" / "phishing_rules",
        yara_rules_dir=base_dir / "detection" / "yara_rules" / "phishing_rules" / "yara_rules_phishing",
        index_file=base_dir / "detection" / "yara_rules" / "phishing_rules" / "index.txt",
        github_api="https://api.github.com/repos/t4d/PhishingKit-Yara-Rules/contents/"
    )

    yara_manager = YaraRulesManager(yara_config)
    yara_success = yara_manager.update_yara_rules()
    
    phishing_manager = PhishingDataManager(phishing_config)
    phishing_success = phishing_manager.update_phishing_data()
    
    if yara_success and phishing_success:
        print("\nAll updates completed successfully!")
        return 0
    else:
        print("\nUpdate process encountered errors")
        return 1

if __name__ == "__main__":
    exit(main())