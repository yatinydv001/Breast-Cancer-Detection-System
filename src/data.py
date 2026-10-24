"""Data loading and splitting utilities."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset (built into scikit-learn).

    Target: 0 = malignant, 1 = benign.
    """
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified train/test split so class ratios are preserved."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
