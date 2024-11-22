import pandas as pd
import re
import logging
from scipy.stats import entropy
from urllib.parse import urlparse
from idna import decode as decode_punycode  # For punycode detection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_entropy(string):
    """
    Calculate Shannon entropy of a string.
    """
    if not string:
        return 0
    probabilities = [string.count(char) / len(string) for char in set(string)]
    return entropy(probabilities, base=2)


def is_punycode(domain):
    """
    Check if a domain is encoded in punycode.
    """
    try:
        decode_punycode(domain)
        return True
    except Exception:
        return False


def load_phishing_data(domain_file, url_file):
    """
    Load domains and URLs from .lst files and return DataFrames.
    """
    try:
        with open(domain_file, 'r') as df, open(url_file, 'r') as uf:
            domains = [line.strip() for line in df if line.strip()]
            urls = [line.strip() for line in uf if line.strip()]
        logging.info(f"Loaded {len(domains)} domains and {len(urls)} URLs.")
        return pd.DataFrame({"domain": domains}), pd.DataFrame({"url": urls})
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()


def preprocess_domain_data(df):
    """
    Extract features for domains to prepare data for ML and AI models.
    """
    if df.empty:
        logging.warning("Domain DataFrame is empty. Skipping preprocessing.")
        return df

    logging.info("Preprocessing domain data...")

    # Add features
    df['length'] = df['domain'].apply(len)
    df['num_dots'] = df['domain'].apply(lambda x: x.count('.'))
    df['entropy'] = df['domain'].apply(calculate_entropy)
    df['is_punycode'] = df['domain'].apply(is_punycode)
    df['has_suspicious_keywords'] = df['domain'].apply(
        lambda x: any(keyword in x.lower() for keyword in ["login", "secure", "verify", "bank", "password", "signin", "admin", "update", "confirm", "account", "pay"])
    )
    df['contains_ip'] = df['domain'].apply(
        lambda x: bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", x))
    )
    df['has_uncommon_tld'] = df['domain'].apply(
        lambda x: x.split('.')[-1] not in {"com", "org", "net", "edu", "gov"}
    )
    return df


def preprocess_url_data(df):
    """
    Extract features for URLs to prepare data for ML and AI models.
    """
    if df.empty:
        logging.warning("URL DataFrame is empty. Skipping preprocessing.")
        return df

    logging.info("Preprocessing URL data...")

    # Add features
    df['length'] = df['url'].apply(len)
    df['entropy'] = df['url'].apply(calculate_entropy)
    df['num_special_chars'] = df['url'].apply(lambda x: len(re.findall(r"[@?&=%]", x)))
    df['has_encoded_chars'] = df['url'].apply(lambda x: '%' in x)
    df['path_length'] = df['url'].apply(lambda x: len(urlparse(x).path))
    df['query_length'] = df['url'].apply(lambda x: len(urlparse(x).query))
    df['contains_sensitive_extension'] = df['url'].apply(
        lambda x: any(ext in urlparse(x).path for ext in [".php", ".asp", ".exe", ".jsp", ".html"])
    )
    df['has_suspicious_keywords'] = df['url'].apply(
        lambda x: any(keyword in x.lower() for keyword in ["login", "secure", "verify", "bank", "password", "signin", "admin", "update", "confirm", "account", "pay"])
    )
    df['num_subdomains'] = df['url'].apply(lambda x: urlparse(x).netloc.count('.') - 1)
    df['has_uncommon_tld'] = df['url'].apply(
        lambda x: urlparse(x).netloc.split('.')[-1] not in {"com", "org", "net", "edu", "gov"}
    )
    df['is_shortened'] = df['url'].apply(
        lambda x: any(service in urlparse(x).netloc for service in ["bit.ly", "tinyurl.com", "t.co", "goo.gl"])
    )
    df['has_malformed_url'] = df['url'].apply(
        lambda x: bool(re.search(r"https?:\/\/.+\s", x))  # Checks for space after the URL schema
    )
    return df
