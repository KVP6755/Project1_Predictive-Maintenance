"""
================================================================================
pipeline_builder.py
================================================================================
Author      : Subhashree Behera
Week        : Week 3 — Imbalanced Classification & LightGBM Modeling
Role        : Pipeline Architect

--------------------------------------------------------------------------------
Responsibility:
    Build the core skeletal framework for the 5-fold Stratified Cross-Validation
    pipeline. This module handles ONLY the data splitting logic — it is completely
    independent of any specific model or balancing algorithm.

    Other modules (smote_balancer.py, lgbm_model.py) will plug INTO this
    pipeline structure. This design prevents data leakage by ensuring SMOTE
    is applied only inside training folds, never touching validation data.

--------------------------------------------------------------------------------
Why Stratified K-Fold?
    The dataset has severe class imbalance — only 3.39% failures (339/9996).
    Standard KFold would create folds with uneven failure distributions.
    StratifiedKFold guarantees each fold maintains the same 3.39% failure
    ratio, making evaluation reliable and reproducible.

--------------------------------------------------------------------------------
Fold Structure (verified on engineered_iot_weather_dataset.csv):
    Fold 1: Train=7996 (failures=271) | Val=2000 (failures=68)
    Fold 2: Train=7997 (failures=272) | Val=1999 (failures=67)
    Fold 3: Train=7997 (failures=271) | Val=1999 (failures=68)
    Fold 4: Train=7997 (failures=271) | Val=1999 (failures=68)
    Fold 5: Train=7997 (failures=271) | Val=1999 (failures=68)

--------------------------------------------------------------------------------
Functions:
    load_and_prepare_data()     → loads dataset, separates X and y
    validate_class_distribution() → checks failure rate per fold
    build_cv_pipeline()         → creates StratifiedKFold split structure
    get_fold()                  → returns one specific fold by index
    summarize_pipeline()        → prints full pipeline summary report

--------------------------------------------------------------------------------
Dataset:
    engineered_iot_weather_dataset.csv
    9,996 rows × 30 feature columns + 1 target column (Machine failure)

--------------------------------------------------------------------------------
Usage:
    from pipeline_builder import build_cv_pipeline, load_and_prepare_data
    X, y = load_and_prepare_data('data/engineered_iot_weather_dataset.csv')
    folds = build_cv_pipeline(X, y)
================================================================================
"""
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
# ============================================================
# FUNCTION 3: build_cv_pipeline  (CORE FUNCTION)
# ============================================================

def build_cv_pipeline(X, y, n_splits=N_SPLITS):
    """
    Build a 5-fold Stratified Cross-Validation split structure.

    This is the CORE function of pipeline_builder.py.
    It creates all 5 train/validation splits while maintaining
    class balance in each fold.

    IMPORTANT — What this function does NOT do:
        - Does NOT apply SMOTE (that is smote_balancer.py)
        - Does NOT train any model (that is lgbm_model.py)
        - Does NOT evaluate metrics (that is tuner_evaluation.py)

    This clean separation prevents data leakage — SMOTE will
    only be applied to X_train inside each fold, never to X_val.

    Args:
        X        (pd.DataFrame): feature matrix (30 columns)
        y        (pd.Series)   : binary target (0/1)
        n_splits (int)         : number of folds (default=5)

    Returns:
        folds (list of dicts): each dict contains:
            {
                'fold'   : fold number (1-5),
                'X_train': training features,
                'X_val'  : validation features,
                'y_train': training labels,
                'y_val'  : validation labels
            }
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = []

    print("=" * 50)
    print(f"BUILDING {n_splits}-FOLD CV PIPELINE")
    print("=" * 50)

    for fold_num, (train_idx, val_idx) in enumerate(
        skf.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val   = X.iloc[val_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val   = y.iloc[val_idx].reset_index(drop=True)

        # Validate class balance per fold
        stats = validate_class_distribution(
            y_train, y_val, fold_num
        )

        folds.append({
            'fold'   : fold_num,
            'X_train': X_train,
            'X_val'  : X_val,
            'y_train': y_train,
            'y_val'  : y_val,
            'stats'  : stats
        })

    print("=" * 50)
    print(f"Pipeline built: {len(folds)} folds ready")
    print("=" * 50)

    return folds

# ============================================================
# FUNCTION 4: get_fold
# ============================================================

def get_fold(folds, fold_num):
    """
    Retrieve a specific fold by number from the folds list.

    This helper allows Member 2 (SMOTE) and Member 3 (LightGBM)
    to easily access individual folds without indexing manually.

    Args:
        folds    (list): output from build_cv_pipeline()
        fold_num (int) : fold to retrieve (1-5)

    Returns:
        dict: single fold with X_train, X_val, y_train, y_val

    Example:
        fold_1 = get_fold(folds, 1)
        X_train = fold_1['X_train']
        y_train = fold_1['y_train']
    """
    if fold_num < 1 or fold_num > len(folds):
        raise ValueError(
            f"fold_num must be between 1 and {len(folds)}, "
            f"got {fold_num}"
        )

    fold = folds[fold_num - 1]

    print(f"Fold {fold_num} retrieved:")
    print(f"  X_train shape: {fold['X_train'].shape}")
    print(f"  X_val shape  : {fold['X_val'].shape}")
    print(f"  y_train failures: {fold['y_train'].sum()}")
    print(f"  y_val failures  : {fold['y_val'].sum()}")

    return fold

# ============================================================
# FUNCTION 5: summarize_pipeline
# ============================================================

def summarize_pipeline(folds):
    """
    Print a complete summary report of the CV pipeline.

    Shows fold sizes, failure rates, and confirms stratification
    is working correctly. Use this to document pipeline output
    in the team report.

    Args:
        folds (list): output from build_cv_pipeline()
    """
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY REPORT")
    print("=" * 60)
    print(f"{'Fold':<6} {'Train Size':<12} {'Val Size':<10} "
          f"{'Train Fail%':<13} {'Val Fail%':<10}")
    print("-" * 60)

    for fold in folds:
        stats = fold['stats']
        print(f"{stats['fold']:<6} "
              f"{len(fold['X_train']):<12} "
              f"{len(fold['X_val']):<10} "
              f"{stats['train_failure_rate']:<13.2f} "
              f"{stats['val_failure_rate']:<10.2f}")

    print("=" * 60)
    print(f"Total folds    : {len(folds)}")
    print(f"Features/fold  : {folds[0]['X_train'].shape[1]}")
    print(f"Strategy       : StratifiedKFold (shuffle=True, "
          f"random_state=42)")
    print(f"Leakage safe   : SMOTE NOT applied here")
    print("=" * 60)

# ============================================================
# MAIN — Full Pipeline Test
# ============================================================

if __name__ == "__main__":

    print("\n>>> RUNNING FULL PIPELINE TEST\n")

    # Step 1: Load data
    X, y = load_and_prepare_data()

    # Step 2: Build CV pipeline
    folds = build_cv_pipeline(X, y)

    # Step 3: Print summary
    summarize_pipeline(folds)

    # Step 4: Test get_fold helper
    print("\n>>> TESTING get_fold() HELPER")
    fold_1 = get_fold(folds, 1)

    # Step 5: Confirm pipeline is ready for SMOTE + LightGBM
    print("\n>>> PIPELINE READY")
    print("Member 2 (SMOTE) can now call:")
    print("  fold = get_fold(folds, n)")
    print("  X_resampled, y_resampled = smote_balancer(fold['X_train'],")
    print("                                            fold['y_train'])")
    print("\nMember 3 (LightGBM) can now call:")
    print("  model = lgbm_model.train(X_resampled, y_resampled)")
    print("  preds = model.predict(fold['X_val'])")

    # Step 6: Test with different fold counts
    print("\n>>> TESTING WITH 3 FOLDS")
    folds_3 = build_cv_pipeline(X, y, n_splits=3)
    summarize_pipeline(folds_3)
    print("Configurable n_splits working correctly ✓")

# ============================================================
# FUNCTION 6: get_feature_names
# ============================================================

def get_feature_names(data_path=DATA_PATH):
    """
    Return the list of feature column names used in the pipeline.

    This is a utility for Member 3 (LightGBM) to access feature
    names when plotting feature importance charts.

    Returns:
        list: 30 feature column names
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    print(f"Feature names exported: {len(feature_cols)} columns")
    return feature_cols