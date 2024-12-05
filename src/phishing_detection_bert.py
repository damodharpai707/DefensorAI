import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
import warnings
warnings.filterwarnings('ignore')

console = Console()

@dataclass
class PhishingConfig:
    """Configuration for sophisticated phishing detection"""
    base_dir: Path = Path(__file__).parent.parent.parent  # Go up to project root
    data_path: Path = Path("/data/Phishing Detection Dataset/Phishing_Email.xlsx")
    model_dir: Path = Path("DefensorAI/models")
    log_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")
    
    # Model parameters
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Training parameters
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    num_workers: int = 4
    
    def __post_init__(self):
        """Create necessary directories"""
        for directory in [self.model_dir, self.log_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class EmailDataset(Dataset):
    """Custom Dataset for email phishing detection"""
    
    def __init__(self, texts: List[str], labels: List[str], 
                 tokenizer: DistilBertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == "Safe Email" else 0
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MetricsTracker:
    """Track and display training metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.best_metrics = {}
        
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def display_progress(self):
        table = Table(title="Training Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="magenta")
        table.add_column("Best", style="green")
        
        for key in self.metrics:
            table.add_row(
                key,
                f"{self.metrics[key][-1]:.4f}",
                f"{self.best_metrics[key]:.4f}"
            )
        
        console.print(table)

class PhishingDetector:
    """Sophisticated phishing email detection system"""
    
    def __init__(self, config: PhishingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.metrics_tracker = MetricsTracker()
        
        console.print(f"[bold green]Using device: {self.device}[/bold green]")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
            cache_dir=config.cache_dir
        ).to(self.device)
    
    def setup_logging(self):
        """Initialize logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.config.log_dir / f'phishing_detection_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare training data with progress tracking"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn()
        ) as progress:
            load_task = progress.add_task("[cyan]Loading dataset...", total=100)
            
            df = pd.read_excel(self.config.base_dir / self.config.data_path)
            progress.update(load_task, advance=50)
            
            # Display dataset statistics
            total_samples = len(df)
            safe_emails = sum(df['Email Type'] == "Safe Email")
            phishing_emails = sum(df['Email Type'] == "Phishing Email")
            
            console.print(Panel(f"""
            Dataset Statistics:
            Total Emails: {total_samples:,}
            Safe Emails: {safe_emails:,}
            Phishing Emails: {phishing_emails:,}
            """))
            
            X_train, X_val, y_train, y_val = train_test_split(
                df['Email Text'].values,
                df['Email Type'].values,
                test_size=self.config.validation_split,
                stratify=df['Email Type'],
                random_state=42
            )
            
            progress.update(load_task, advance=50)
            
            train_dataset = EmailDataset(X_train, y_train, self.tokenizer, self.config.max_length)
            val_dataset = EmailDataset(X_val, y_val, self.tokenizer, self.config.max_length)
            
            return train_dataset, val_dataset
    
    def train(self):
        """Train the model with comprehensive progress tracking"""
        train_dataset, val_dataset = self.load_data()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        num_training_steps = len(train_loader) * self.config.epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        best_val_loss = float('inf')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn()
        ) as progress:
            epoch_task = progress.add_task(
                "[cyan]Training Progress...",
                total=self.config.epochs
            )
            
            for epoch in range(self.config.epochs):
                self.model.train()
                train_loss = self._train_epoch(
                    train_loader,
                    optimizer,
                    scheduler,
                    progress
                )
                
                val_loss, metrics = self._validate(val_loader)
                
                progress.update(epoch_task, advance=1)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model()
                
                self.metrics_tracker.update({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **metrics
                })
                self.metrics_tracker.display_progress()
    
    def _train_epoch(self, train_loader, optimizer, scheduler, progress):
        """Train for one epoch with detailed progress tracking"""
        total_loss = 0
        batch_task = progress.add_task(
            "[green]Processing batches...",
            total=len(train_loader)
        )
        
        for batch in train_loader:
            loss = self._process_batch(batch, optimizer, scheduler)
            total_loss += loss
            progress.update(batch_task, advance=1)
        
        progress.remove_task(batch_task)
        return total_loss / len(train_loader)
    
    def _process_batch(self, batch, optimizer, scheduler):
        """Process a single batch with gradient accumulation"""
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        optimizer.step()
        scheduler.step()
        
        return loss.item()
    
    def _validate(self, val_loader):
        """Validate the model and compute metrics"""
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        metrics = classification_report(
            true_labels,
            predictions,
            target_names=['Phishing', 'Safe'],
            output_dict=True
        )
        
        return val_loss / len(val_loader), {
            'accuracy': metrics['accuracy'],
            'precision': metrics['weighted avg']['precision'],
            'recall': metrics['weighted avg']['recall'],
            'f1_score': metrics['weighted avg']['f1-score']
        }
    
    def _save_model(self):
        """Save model with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.config.model_dir / f'bert_phishing_detector_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'tokenizer': self.tokenizer
        }, model_path)
        
        console.print(f"[bold green]Model saved to {model_path}[/bold green]")

def main():
    """Main execution function"""
    try:
        console.print("[bold magenta]Starting Email Phishing Detection System Training[/bold magenta]")
        
        config = PhishingConfig()
        detector = PhishingDetector(config)
        detector.train()
        
        console.print("\n[bold green]Training completed successfully![/bold green]")
    
    except Exception as e:
        logging.error("Error in main execution")
        logging.error(str(e))
        raise

if __name__ == "__main__":
    main()