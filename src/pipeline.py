import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from src.detection.yara_rules.phishing_rules_loader import update_phishing_lists, compile_yara_rules
from src.detection.phishing_ml_model import train_ml_model, evaluate_ml_model
from src.detection.phishing_ai_model import build_ai_model, plot_training_history, train_ai_model, evaluate_ai_model
from src.preprocessing.phishing_preprocessing import load_phishing_data, preprocess_domain_data, preprocess_url_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_pipeline():
    """
    Main pipeline to update data, preprocess, train ML and AI models, 
    and evaluate YARA rules.
    """
    logging.info("Starting the phishing detection pipeline...")

    # Step 1: Update phishing data and YARA rules
    logging.info("Updating phishing datasets and YARA rules...")
    try:
        update_phishing_lists()
    except Exception as e:
        logging.error(f"Error updating phishing datasets: {e}")
        return

    # Step 2: Load updated data
    domain_file = "data/phishing/domains.lst"
    url_file = "data/phishing/urls.lst"
    try:
        domains, urls = load_phishing_data(domain_file, url_file)
        if domains.empty or urls.empty:
            logging.error("No data loaded. Ensure the domain and URL files are updated and non-empty.")
            return
    except Exception as e:
        logging.error(f"Error loading phishing datasets: {e}")
        return

    # Step 3: Preprocess data
    logging.info("Preprocessing domain and URL data...")
    try:
        domains = preprocess_domain_data(domains)
        urls = preprocess_url_data(urls)
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return

    # Combine datasets for a holistic ML approach
    combined_features = domains.append(urls, ignore_index=True)
    combined_features = shuffle(combined_features, random_state=42)  # Shuffle data for better training

    # Step 4: Prepare features and labels for ML/AI models
    feature_columns = [
        "length", 
        "num_dots", 
        "has_suspicious_keywords", 
        "contains_ip", 
        "num_special_chars", 
        "has_encoded_chars", 
        "has_uncommon_tld",
        "entropy",  # Added entropy as a feature
    ]
    
    if any(col not in combined_features.columns for col in feature_columns):
        logging.error("Some required feature columns are missing in the dataset.")
        return

    X = combined_features[feature_columns]
    y = combined_features.get("label", [1] * len(X))  # Dynamically assign labels if available

    # Normalize features for AI model
    logging.info("Normalizing feature data for AI model...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logging.info("Splitting data into training and testing sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        logging.error(f"Error during data split: {e}")
        return

    # Step 5: Train and evaluate the ML model
    logging.info("Training the ML model...")
    try:
        ml_model = train_ml_model(X_train, y_train)
        logging.info("Evaluating the ML model...")
        evaluate_ml_model(ml_model, X_test, y_test)
    except Exception as e:
        logging.error(f"Error in ML model training/evaluation: {e}")

    # Step 6: Train and evaluate the AI model
    logging.info("Training the AI model...")
    try:
        ai_model = build_ai_model(input_dim=X_train.shape[1])
        history = train_ai_model(ai_model, X_train, y_train, X_val=X_test, y_val=y_test)
        logging.info("Evaluating the AI model...")
        evaluate_ai_model(ai_model, X_test, y_test)

        # Optional: Plot training history
        logging.info("Plotting training history...")
        plot_training_history(history)
    except Exception as e:
        logging.error(f"Error in AI model training/evaluation: {e}")

    # Step 7: Compile and test YARA rules
    logging.info("Compiling and evaluating YARA rules...")
    try:
        yara_rules = compile_yara_rules()
        sample_domain = "secure-login.bank.com"  # Example data; replace with dynamic input as needed.
        matches = yara_rules.match(data=sample_domain)
        logging.info(f"YARA Matches for '{sample_domain}': {matches}")
    except Exception as e:
        logging.error(f"Error in YARA rule evaluation: {e}")

    logging.info("Phishing detection pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
