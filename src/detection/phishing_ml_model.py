from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import optuna


def train_ml_model(X, y, save_path="phishing_xgb_model.pkl"):
    """
    Train an XGBoost model with hyperparameter optimization using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "random_state": 42,
            "use_label_encoder": False,
            "n_jobs": -1,
            "eval_metric": "auc"
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            y_val_pred = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_val_pred))

        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print(f"Best Parameters: {study.best_params}")

    # Train the best model
    best_model = XGBClassifier(**study.best_params, use_label_encoder=False, random_state=42, eval_metric="auc")
    best_model.fit(X, y)
    joblib.dump(best_model, save_path)
    print(f"Model saved to {save_path}")

    return best_model


def evaluate_ml_model(model, X_test, y_test):
    """
    Evaluate the ML model on the test set and visualize results.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, y_pred))
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC-ROC Score: {auc_roc}")

    # Precision-Recall Curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    plot_confusion_matrix(cm)

    # Precision-Recall Curve
    plot_precision_recall_curve(precision, recall)

    # Feature importance using SHAP
    explain_shap_importance(model, X_test)

    return y_pred


def explain_shap_importance(model, X):
    """
    Explain model predictions using SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.title("Feature Importance (SHAP)")
    plt.show()


def plot_confusion_matrix(cm, labels=["Legitimate", "Phishing"]):
    """
    Plot a confusion matrix with enhanced visualization.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = "d"
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plot_precision_recall_curve(precision, recall):
    """
    Plot Precision-Recall curve.
    """
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
