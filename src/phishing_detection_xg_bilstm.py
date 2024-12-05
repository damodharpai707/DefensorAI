import logging
import re
import random
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.callbacks import ( # type: ignore
    ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
)
keras = tf.keras
from tensorflow.keras.layers import ( # type: ignore
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout, 
    concatenate, BatchNormalization
)
from tensorflow.keras import Model # type: ignore
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PhishingConfig:
    base_dir: Path = Path(__file__).parent.parent.parent
    data_file: Path = Path("DefensorAI/data/phishing/anomaly_dataset.csv")
    model_dir: Path = Path("DefensorAI/models")
    bilstm_path: str = "bilstm_model.keras"
    xgb_path: str = "xgboost_model.json"
    log_dir: Path = Path("DefensorAI/logs")
    max_url_length: int = 100
    embedding_dim: int = 128
    lstm_units: int = 256
    dropout_rate: float = 0.4
    batch_size: int = 64
    epochs: int = 20
    validation_split: float = 0.2
    early_stopping_patience: int = 2
    reduce_lr_patience: int = 2

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

def f1_score(y_true, y_pred):
    """Custom F1 score metric with global calculation"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)  # Round predictions to 0 or 1
    
    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))
    
    precision = true_positives / (true_positives + false_positives + K.epsilon())
    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

class TrainingMonitor(keras.callbacks.Callback):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.epoch_times = []
        self.training_logs = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        self.epoch_times.append(epoch_time)
        logs = logs or {}
        logs['epoch_time'] = epoch_time
        self.training_logs.append(logs)
        print(f"\n{self.model_name} - Epoch {epoch+1}")
        print(f"Time taken: {epoch_time:.2f}s")
        for metric, value in logs.items():
            if metric != 'epoch_time':
                print(f"{metric}: {value:.4f}")

class PhishingDetector:
    def __init__(self, config: PhishingConfig):
        self.config = config
        self.setup_logging()
        self.initialize_models()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_dir / 'phishing_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_models(self):
        self.bilstm_model = self._create_bilstm_model()
        self.xgb_model = None
        self.logger.info("Models initialized successfully")

    def _create_bilstm_model(self):
        url_input = Input(shape=(self.config.max_url_length,))
        
        embedding = Embedding(128, self.config.embedding_dim, mask_zero=True)(url_input)
        embedding = BatchNormalization()(embedding)

        lstm1 = Bidirectional(LSTM(self.config.lstm_units, return_sequences=True))(embedding)
        lstm1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(self.config.dropout_rate)(lstm1)

        lstm2 = Bidirectional(LSTM(self.config.lstm_units // 2, return_sequences=True))(dropout1)
        lstm2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(self.config.dropout_rate)(lstm2)

        lstm3 = Bidirectional(LSTM(self.config.lstm_units // 4))(dropout2)
        lstm3 = BatchNormalization()(lstm3)
        dropout3 = Dropout(self.config.dropout_rate)(lstm3)

        dense1 = Dense(256, activation='relu')(dropout3)
        dense1 = BatchNormalization()(dense1)
        dropout4 = Dropout(self.config.dropout_rate)(dense1)

        dense2 = Dense(128, activation='relu')(dropout4)
        dense2 = BatchNormalization()(dense2)
        dense2_residual = concatenate([dropout3, dense2])

        dense3 = Dense(64, activation='relu')(dense2_residual)
        dense3 = BatchNormalization()(dense3)
        
        output = Dense(1, activation='sigmoid')(dense3)

        model = Model(inputs=url_input, outputs=output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', f1_score, tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract comprehensive features from URL"""
        try:
            return {
                'length': len(url),
                'num_dots': url.count('.'),
                'num_digits': sum(c.isdigit() for c in url),
                'num_special': sum(not c.isalnum() for c in url),
                'has_https': int(url.startswith('https')),
                'num_directories': len(url.split('/')) - 3 if len(url.split('/')) > 3 else 0,
                'domain_length': len(url.split('/')[2]) if len(url.split('/')) > 2 else 0,
                'num_subdomains': len(url.split('/')[2].split('.')) - 2 if len(url.split('/')) > 2 else 0,
                'num_queries': len(url.split('?')[1].split('&')) if '?' in url else 0,
                'num_fragments': url.count('#'),
                'num_params': url.count('='),
                'entropy': self._calculate_entropy(url),
                'longest_word_length': max(len(word) for word in url.split('/')),
                'avg_word_length': np.mean([len(word) for word in url.split('/')]),
                'unique_chars_ratio': len(set(url)) / len(url) if url else 0,
                'suspicious_tld': int(any(tld in url.lower() for tld in ['.xyz', '.top', '.work', '.click'])),
                'has_ip_pattern': int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))),
                'has_suspicious_chars': int(bool(re.search(r'[<>{}|\[\]]', url))),
                'num_suspicious_words': sum(1 for word in ['login', 'bank', 'account', 'verify', 'secure'] if word in url.lower()),
                'has_excessive_delimiters': int(bool(re.search(r'[./-]{3,}', url)))
            }
        except Exception as e:
            self.logger.error(f"Error extracting features from URL: {url}")
            self.logger.error(str(e))
            raise

    def _calculate_entropy(self, url: str) -> float:
        """Calculate Shannon entropy of URL"""
        chars = {}
        for c in url:
            chars[c] = chars.get(c, 0) + 1
        length = len(url)
        return -sum(count/length * np.log2(count/length) for count in chars.values())

    def _encode_url(self, url: str) -> np.ndarray:
        """Convert URL to numeric sequence"""
        encoded = np.zeros(self.config.max_url_length)
        for i, c in enumerate(url[:self.config.max_url_length]):
            encoded[i] = ord(c) % 128
        return encoded

    def prepare_data(self):
        """Prepare data for training"""
        try:
            df = pd.read_csv(self.config.data_file)
            urls = df['url'].values
            labels = df['label'].values

            # Prepare URL sequences for BiLSTM
            X_urls = np.zeros((len(urls), self.config.max_url_length))
            for i, url in enumerate(urls):
                X_urls[i] = self._encode_url(url)

            # Extract features for XGBoost
            features = []
            for url in tqdm(urls, desc="Extracting features"):
                features.append(self.extract_features(url))
            X_features = pd.DataFrame(features)

            return X_urls, X_features, labels

        except Exception as e:
            self.logger.error("Error preparing data")
            self.logger.error(str(e))
            raise

    def train(self):
        try:
            X_urls, X_features, y = self.prepare_data()
            
            print("\nTraining BiLSTM model:")
            history = self.bilstm_model.fit(
                X_urls, 
                y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=[
                    ModelCheckpoint(
                        str(self.config.model_dir / 'bilstm_weights.keras'),
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min',
                        verbose=1
                    ),
                    EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        patience=2,
                        min_delta=0.001,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    EarlyStopping(
                        monitor='val_accuracy',
                        mode='max',
                        patience=2,
                        min_delta=0.001,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        mode='min',
                        patience=2,
                        factor=0.5,
                        min_lr=1e-6,
                        verbose=1
                    ),
                    TrainingMonitor("BiLSTM"),
                    TensorBoard(log_dir=str(self.config.log_dir / 'bilstm'))
                ],
                verbose=1
            )

            print("\nTraining XGBoost model:")
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y, 
                test_size=self.config.validation_split,
                stratify=y
            )

            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric=['logloss', 'auc']
            )

            with tqdm(total=200, desc="XGBoost Training Progress") as pbar:
                self.xgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False,
                    callbacks=[
                        xgb.callback.TrainingCallback(
                            lambda env: pbar.update(1)
                        )
                    ]
                )

            print("\nSaving models...")
            self.xgb_model.save_model(
                str(self.config.model_dir / 'xgboost_model.json')
            )
            print(f"Models saved in {self.config.model_dir}")
            
            return history

        except Exception as e:
            self.logger.error("Error during training")
            self.logger.error(str(e))
            raise

    def predict(self, url: str) -> Tuple[float, Dict]:
        """Make prediction for a given URL"""
        try:
            url_encoded = self._encode_url(url)
            features = self.extract_features(url)
            
            bilstm_pred = self.bilstm_model.predict(
                np.array([url_encoded]), 
                verbose=0
            )[0][0]
            
            if self.xgb_model:
                feature_vector = pd.DataFrame([features])
                xgb_pred = self.xgb_model.predict_proba(feature_vector)[:, 1][0]
                final_score = 0.4 * bilstm_pred + 0.6 * xgb_pred
            else:
                final_score = bilstm_pred
                xgb_pred = None
                
            return final_score, {
                'detection_method': 'ml_ensemble',
                'bilstm_score': float(bilstm_pred),
                'xgb_score': float(xgb_pred) if self.xgb_model else None,
                'features': features,
                'confidence': float(final_score)
            }
        except Exception as e:
            self.logger.error(f"Error predicting URL: {url}")
            self.logger.error(str(e))
            raise

    def generate_adversarial_samples(self, urls, features, labels, num_samples=1000):
        """Generate adversarial samples for robust training"""
        adversarial_samples = np.zeros((num_samples, self.config.max_url_length))
        adversarial_features = []
        adversarial_labels = []
    
        progress_bar = tqdm(
            range(num_samples),
            desc="Generating adversarial samples",
            unit="samples",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        for i in progress_bar:
            url = urls[i]
            feature = features[i]
            label = labels[i]
        
            mutated_url = self._mutate_url(url)
            adversarial_samples[i] = self._encode_url(mutated_url)
            adversarial_features.append(self._mutate_features(feature))
            adversarial_labels.append(label)
        
            if i % 50 == 0:
                progress_bar.set_description(f"Generated {i}/{num_samples} samples")
            
        return adversarial_samples, pd.DataFrame(adversarial_features), np.array(adversarial_labels)
    
    def _mutate_url(self, url: str) -> str:
        """Apply various mutations to URL"""
        mutations = [
            lambda x: x.replace('www.', 'ww1.'),
            lambda x: x + '?' + ''.join(random.choices(string.ascii_letters, k=5)),
            lambda x: x.replace('http://', 'https://'),
            lambda x: x + '#' + ''.join(random.choices(string.ascii_letters, k=5)),
            lambda x: x.replace('.com', '.com.' + ''.join(random.choices(string.ascii_letters, k=3))),
            lambda x: x.replace('https://', 'http://'),
            lambda x: x + '/' + ''.join(random.choices(string.ascii_letters, k=4)),
            lambda x: x.replace('-', '_')
            ]
        return random.choice(mutations)(url)
    def _mutate_features(self, features: Dict) -> Dict:
        """Apply mutations to URL features"""
        mutated = features.copy()
        mutations = [
            lambda x: {**x, 'num_dots': x['num_dots'] + 1},
            lambda x: {**x, 'num_digits': x['num_digits'] + random.randint(1, 3)},
            lambda x: {**x, 'num_special': x['num_special'] + random.randint(1, 2)},
            lambda x: {**x, 'num_directories': x['num_directories'] + 1},
            lambda x: {**x, 'num_subdomains': x['num_subdomains'] + 1},
            lambda x: {**x, 'num_queries': x['num_queries'] + random.randint(1, 2)},
            lambda x: {**x, 'num_fragments': x['num_fragments'] + 1},
            lambda x: {**x, 'num_params': x['num_params'] + random.randint(1, 2)}
        ]
        return random.choice(mutations)(mutated)

    def prepare_data(self):
        """Prepare and preprocess training data"""
        try:
            print("\nLoading dataset...")
            dataset_path = self.config.base_dir / self.config.data_file
            df = pd.read_csv(dataset_path)
        
            # Display dataset statistics
            total_samples = len(df)
            legitimate_samples = sum(df['label'] == 1)
            phishing_samples = sum(df['label'] == 0)
        
            print(f"\nDataset Statistics:")
            print(f"Total URLs: {total_samples:,}")
            print(f"Legitimate URLs (1): {legitimate_samples:,}")
            print(f"Phishing URLs (0): {phishing_samples:,}")
        
            # Prepare URL sequences
            print("\nEncoding URLs...")
            X_urls = np.array([
                self._encode_url(url) 
                for url in tqdm(df['url'], desc="URL Encoding Progress")
            ])
            # Extract features
            print("\nExtracting features...")
            X_features = pd.DataFrame([
                self.extract_features(url) 
                for url in tqdm(df['url'], desc="Feature Extraction Progress")
            ])
        
            y = df['label'].values
        
            # Generate adversarial samples
            print("\nGenerating adversarial samples...")
            X_urls_adv, X_features_adv, y_adv = self.generate_adversarial_samples(
                df['url'].values,
                [self.extract_features(url) for url in df['url'].values],
                df['label'].values
            )
        
            # Combine original and adversarial data
            X_urls = np.concatenate([X_urls, X_urls_adv])
            X_features = pd.concat([X_features, X_features_adv])
            y = np.concatenate([y, y_adv])
        
            return X_urls, X_features, y
        
        except Exception as e:
            self.logger.error("Error preparing data")
            self.logger.error(str(e))
            raise
    
def main():
    """Main execution function"""
    try:
        # Initialize configuration and detector
        config = PhishingConfig()
        detector = PhishingDetector(config)
        
        # Training phase
        print("\nStarting model training...")
        history = detector.train()
        
        # Print training results
        print("\nTraining completed!")
        print("\nBiLSTM Final Metrics:")
        for metric, value in history.history.items():
            if not metric.startswith('val_'):
                print(f"{metric}: {value[-1]:.4f}")
            
        # Prediction phase
        while True:
            url = input("\nEnter URL to analyze (or 'quit' to exit): ")
            if url.lower() == 'quit':
                break
                
            score, details = detector.predict(url)
            print(f"\nPhishing Detection Results:")
            print(f"URL: {url}")
            print(f"Classification: {'Legitimate' if score > 0.5 else 'Phishing'}")
            print(f"Confidence Score: {details['confidence']:.4f}")
            print(f"BiLSTM Score: {details['bilstm_score']:.4f}")
            
            if details['xgb_score'] is not None:
                print(f"XGBoost Score: {details['xgb_score']:.4f}")
            
            # Print key features
            print("\nKey Features:")
            for feature, value in details['features'].items():
                if isinstance(value, (int, float)) and value > 0:
                    print(f"{feature}: {value}")
                    
    except Exception as e:
        logging.error("Error in main execution")
        logging.error(str(e))
        raise

if __name__ == "__main__":
    main()