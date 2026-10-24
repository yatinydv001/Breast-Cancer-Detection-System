"""Train, compare and evaluate models, then save the best one.

Usage:
    python -m src.train
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, works without a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import RANDOM_STATE, load_data, split_data
from .models import get_models

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"


def compare_models(X_train, y_train, X_test, y_test):
    """Cross-validate and test every candidate model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    for name, model in get_models().items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "CV Accuracy": cv_acc.mean(),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, proba),
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results, fitted


def plot_results(results, best_model, X_test, y_test, best_name):
    """Save comparison bar chart, confusion matrix and feature importance."""
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Model comparison
    ax = results.set_index("Model")[["Test Accuracy", "F1", "ROC AUC"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0.9, 1.0), rot=15
    )
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test,
        display_labels=["Malignant", "Benign"], cmap="Blues", ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 3. Feature importance (only for tree-based models)
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=X_test.columns).nlargest(10)
        plt.figure(figsize=(7, 5))
        sns.barplot(x=imp.values, y=imp.index, color="steelblue")
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    results, fitted = compare_models(X_train, y_train, X_test, y_test)
    print("\n=== Model comparison ===")
    print(results.round(4).to_string(index=False))

    best_name = results.loc[0, "Model"]
    best_model = fitted[best_name]
    print(f"\nBest model (by F1): {best_name}\n")
    print(classification_report(
        y_test, best_model.predict(X_test), target_names=["Malignant", "Benign"]
    ))

    plot_results(results, best_model, X_test, y_test, best_name)
    results.round(4).to_csv(REPORTS_DIR / "results.csv", index=False)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": list(X.columns), "name": best_name}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved plots and results.csv to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
