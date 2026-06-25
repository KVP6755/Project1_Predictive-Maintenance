# ============================================================
# IMPORTS
# ============================================================
# pandas/numpy → data handling
# sklearn.model_selection → StratifiedKFold split logic
# pathlib → clean file path handling across OS

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# ============================================================
# CONSTANTS
# ============================================================
# These are fixed values used throughout the pipeline.
# Centralizing them here makes the pipeline easy to reconfigure.

N_SPLITS = 5          # number of CV folds
RANDOM_STATE = 42     # ensures reproducibility across runs
DATA_PATH = Path('data/engineered_iot_weather_dataset.csv')

# Columns to EXCLUDE from features
# UDI/Product ID/Type/date = identifiers, not sensor signals
# Machine failure = target variable
# TWF/HDF/PWF/OSF/RNF = failure type labels (would cause leakage)
EXCLUDE_COLS = [
    'UDI', 'Product ID', 'Type', 'date',
    'Machine failure',
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

TARGET_COL = 'Machine failure'

# ============================================================
# FUNCTION 1: load_and_prepare_data
# ============================================================

def load_and_prepare_data(data_path=DATA_PATH):
    """
    Load the engineered IoT weather dataset and separate
    features (X) from the target variable (y).

    Steps:
        1. Load CSV from given path
        2. Strip column name whitespace
        3. Convert target column to integer
        4. Separate X (30 rolling features) and y (Machine failure)

    Args:
        data_path (str or Path): path to the CSV file

    Returns:
        X (pd.DataFrame): 9996 rows × 30 feature columns
        y (pd.Series)   : 9996 binary labels (0=normal, 1=failure)
    """
    # Load dataset
    df = pd.read_csv(data_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert target to integer (prevents string/int mismatch)
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL], errors='coerce'
    ).astype(int)

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    print("=" * 50)
    print("DATA LOADED SUCCESSFULLY")
    print("=" * 50)
    print(f"Total rows     : {len(df)}")
    print(f"Feature columns: {X.shape[1]}")
    print(f"Target column  : {TARGET_COL}")
    print(f"Failures (1)   : {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"Normal   (0)   : {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
    print("=" * 50)

    return X, y


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_and_prepare_data()
# ============================================================
# FUNCTION 2: validate_class_distribution
# ============================================================

def validate_class_distribution(y_train, y_val, fold_num):
    """
    Verify that each fold maintains the same failure ratio
    as the full dataset (~3.39%).

    This is the key proof that StratifiedKFold is working
    correctly — if folds were random, failure rates would
    vary significantly between folds.

    Args:
        y_train  (pd.Series): training fold target labels
        y_val    (pd.Series): validation fold target labels
        fold_num (int)      : fold number (1-5) for display

    Returns:
        dict: failure rates for train and validation splits
    """
    train_failure_rate = y_train.mean() * 100
    val_failure_rate   = y_val.mean() * 100

    print(f"  Fold {fold_num} | "
          f"Train failures: {y_train.sum()} "
          f"({train_failure_rate:.2f}%) | "
          f"Val failures: {y_val.sum()} "
          f"({val_failure_rate:.2f}%)")

    return {
        'fold': fold_num,
        'train_failure_rate': train_failure_rate,
        'val_failure_rate': val_failure_rate
    }
