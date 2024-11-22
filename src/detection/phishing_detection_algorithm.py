import re
import requests
import whois
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline

# Load AI-based phishing detector (fine-tuned transformer model)
ai_model_pipeline = pipeline("text-classification", model="your-transformer-model-path", tokenizer="your-transformer-model-path")

def detect_phishing(entry, entry_type, ml_model=None):
    """
    Detect phishing using enhanced feature extraction and AI/ML models.
    """
    if entry_type == "domain":
        features = extract_domain_features(entry)
    elif entry_type == "url":
        features = extract_url_features(entry)
    else:
        raise ValueError("Invalid entry_type. Must be 'domain' or 'url'.")

    phishing_score = (
        (features.get("length", 0) > 50) * 0.3 +
        (features.get("num_dots", 0) > 3) * 0.2 +
        features.get("has_suspicious_keywords", False) * 0.4 +
        features.get("contains_ip", False) * 0.1 +
        (features.get("is_recently_registered", False) * 0.2 if "is_recently_registered" in features else 0) +
        (features.get("is_untrusted_ssl", False) * 0.2 if "is_untrusted_ssl" in features else 0)
    )

    if entry_type == "url":
        phishing_score += (
            (features.get("num_special_chars", 0) > 5) * 0.2 +
            features.get("has_encoded_chars", False) * 0.2 +
            features.get("has_uncommon_tld", False) * 0.1
        )

    # Use AI model for textual analysis
    ai_prediction = ai_model_pipeline(entry)[0]
    ai_score = ai_prediction["score"] if ai_prediction["label"] == "PHISHING" else 1 - ai_prediction["score"]
    phishing_score += ai_score * 0.3  # Weight AI model prediction

    # Use ML model for feature-based prediction
    if ml_model:
        ml_features = [[
            features.get("length", 0),
            features.get("num_dots", 0),
            features.get("has_suspicious_keywords", 0),
            features.get("contains_ip", 0),
            features.get("is_recently_registered", 0),
            features.get("is_untrusted_ssl", 0),
            features.get("num_special_chars", 0),
            features.get("has_encoded_chars", 0),
            features.get("has_uncommon_tld", 0)
        ]]
        ml_score = ml_model.predict_proba(ml_features)[0][1]
        phishing_score += ml_score * 0.4  # Weight ML model prediction

    return {
        "status": "Phishing" if phishing_score >= 0.5 else "Legitimate",
        "score": phishing_score,
        "features": features
    }


def extract_domain_features(domain):
    """
    Extract features for domains with additional WHOIS and heuristic checks.
    """
    suspicious_keywords = ["login", "secure", "verify", "bank", "update", "confirm", "account"]
    features = {
        "length": len(domain),
        "num_dots": domain.count('.'),
        "has_suspicious_keywords": any(keyword in domain.lower() for keyword in suspicious_keywords),
        "contains_ip": bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)),
        "has_uncommon_tld": domain.split('.')[-1] not in {"com", "org", "net", "edu", "gov"}
    }

    # WHOIS-based feature
    try:
        whois_data = whois.whois(domain)
        if whois_data.creation_date:
            creation_date = whois_data.creation_date[0] if isinstance(whois_data.creation_date, list) else whois_data.creation_date
            features["is_recently_registered"] = (datetime.now() - creation_date).days < 180
    except Exception:
        features["is_recently_registered"] = False

    return features


def extract_url_features(url):
    """
    Extract features for URLs including SSL validation, response status, and content.
    """
    suspicious_keywords = ["login", "secure", "verify", "bank", "password", "signin", "admin"]
    features = {
        "length": len(url),
        "num_special_chars": len(re.findall(r"[@?&=%]", url)),
        "has_encoded_chars": '%' in url,
        "has_suspicious_keywords": any(keyword in url.lower() for keyword in suspicious_keywords),
        "has_uncommon_tld": url.split('.')[-1] not in {"com", "org", "net", "edu", "gov"},
        "num_subdomains": url.count('.') - 1,
        "has_malformed_url": bool(re.search(r"https?:\/\/.+\s", url))
    }

    # SSL and HTTP status-based feature
    try:
        response = requests.get(url, timeout=5, verify=True)
        features["is_untrusted_ssl"] = not response.url.startswith("https")
        features["http_status_code"] = response.status_code
    except Exception:
        features["is_untrusted_ssl"] = True  # Assume untrusted if SSL validation fails
        features["http_status_code"] = 0

    return features


def train_phishing_detection_model(X, y):
    """
    Train a phishing detection model using RandomForestClassifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    # Example usage
    sample_domain = "secure-login.bank.com"
    result = detect_phishing(sample_domain, entry_type="domain")
    print(f"Detection Result for {sample_domain}: {result}")

    sample_url = "http://example.com/login?verify=true"
    result = detect_phishing(sample_url, entry_type="url")
    print(f"Detection Result for {sample_url}: {result}")
