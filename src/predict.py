"""Make predictions with the trained model.

Usage:
    python -m src.predict                     # predicts 5 samples from the dataset
    python -m src.predict --csv my_data.csv   # CSV must contain the 30 feature columns
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
LABELS = {0: "Malignant", 1: "Benign"}


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_df(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """Return predictions and benign-probability for each row of df."""
    bundle = bundle or load_model()
    X = df[bundle["features"]]
    model = bundle["model"]
    return pd.DataFrame({
        "prediction": [LABELS[p] for p in model.predict(X)],
        "benign_probability": model.predict_proba(X)[:, 1].round(4),
    }, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Breast cancer prediction")
    parser.add_argument("--csv", type=str, help="Path to CSV with feature columns")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        X, y = load_data()
        df = X.sample(5, random_state=1)
        print("Actual labels:", [LABELS[v] for v in y.loc[df.index]])

    print(predict_df(df))


if __name__ == "__main__":
    main()
