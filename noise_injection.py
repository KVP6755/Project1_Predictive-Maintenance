"""
================================================================================
noise_injection.py
================================================================================
Project     : Predictive Maintenance
Author      : Subhashree Behera
Week        : Week 4 — Noise Sensitivity Analysis & Threshold Tuning
Role        : Noise Injection

--------------------------------------------------------------------------------
Responsibility:
    Inject synthetic noise into the test dataset at multiple intensity
    levels (5%, 10%, 15%) to simulate real-world sensor degradation —
    things like electrical interference, calibration drift, or faulty
    wiring that occur in actual factory/IoT deployments.

    This module produces 3 noisy versions of the test set, which
    Member 2 (Robustness Evaluation) will run the trained model
    against to measure how much performance degrades under noise.

--------------------------------------------------------------------------------
Why This Matters (Business Context):
    A model that only works on perfectly clean lab data is not
    deployment-ready. Real IoT sensors experience drift, electrical
    noise, and measurement error. If the model's F1 score collapses
    under 10-15% noise, it cannot be trusted on a factory floor.

    This step answers the question: "Will this model still catch
    machine failures when the sensors are slightly unreliable?"

--------------------------------------------------------------------------------
Method:
    Gaussian noise is added to each feature, scaled to that feature's
    own standard deviation. This matters because the 30 features have
    wildly different scales (e.g., Air temperature mean=300, std=2.0
    vs Air temperature_roll_var mean=0.0, std=0.001) — using a single
    fixed noise value would be meaningless across such different scales.

    noisy_value = original_value + N(0, noise_level * feature_std)

--------------------------------------------------------------------------------
Functions:
    load_test_set()              → loads dataset, recreates the same
                                    train/test split used by Member 3/4
    inject_gaussian_noise()      → adds noise to one feature column
    create_noisy_dataset()       → applies noise across all 30 features
    generate_all_noise_levels()  → produces 5%, 10%, 15% noisy datasets
    save_noisy_datasets()        → exports datasets for Member 2 to use
    summarize_noise_impact()     → prints before/after stats per level

--------------------------------------------------------------------------------
Dataset:
    engineered_iot_weather_dataset.csv
    Test split: 2000 rows (20%), stratified, random_state=42
    Test failures: 68 (3.40%) — matches overall 3.39% failure rate

--------------------------------------------------------------------------------
Output Files (saved to data/noisy/):
    test_noisy_5pct.csv
    test_noisy_10pct.csv
    test_noisy_15pct.csv

--------------------------------------------------------------------------------
Usage:
    from noise_injection import generate_all_noise_levels
    noisy_datasets = generate_all_noise_levels()
    # Member 2 then loads these for model evaluation
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONSTANTS
# ============================================================
RANDOM_STATE = 42        # MUST match Week 3 pipeline_builder.py
                          # so we test on the SAME test set the
                          # model was evaluated on — not a new split
TEST_SIZE = 0.2

NOISE_LEVELS = [0.05, 0.10, 0.15]   # 5%, 10%, 15%

DATA_PATH = Path('data/engineered_iot_weather_dataset.csv')
OUTPUT_DIR = Path('data/noisy')

EXCLUDE_COLS = [
    'UDI', 'Product ID', 'Type', 'date',
    'Machine failure',
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]
TARGET_COL = 'Machine failure'

# ============================================================
# FUNCTION 1: load_test_set
# ============================================================

def load_test_set(data_path=DATA_PATH):
    """
    Load the dataset and recreate the exact same test split
    used during model training (Week 3).

    Using the same random_state=42 and test_size=0.2 ensures
    we inject noise into the SAME 2000 rows the model was
    evaluated on, not a different random sample.

    Returns:
        X_test (pd.DataFrame): 2000 rows x 30 features (clean)
        y_test (pd.Series)   : 2000 binary labels
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL], errors='coerce'
    ).astype(int)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("=" * 50)
    print("TEST SET LOADED")
    print("=" * 50)
    print(f"Test set size  : {X_test.shape}")
    print(f"Test failures  : {y_test.sum()} "
          f"({y_test.mean()*100:.2f}%)")
    print("=" * 50)

    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


if __name__ == "__main__":
    X_test, y_test = load_test_set()


# ============================================================
# FUNCTION 2: inject_gaussian_noise
# ============================================================

def inject_gaussian_noise(series, noise_level, random_state=RANDOM_STATE):
    """
    Add Gaussian noise to a single feature column, scaled to
    that feature's own standard deviation.

    Why scale by std?
        Our 30 features have wildly different ranges — e.g.
        Air temperature_roll_mean (~300) vs
        Air temperature_roll_var (~0.001). A fixed noise value
        like +/-5 would be meaningless for the variance column
        but invisible for the temperature column. Scaling by
        each feature's own std makes noise proportional and fair.

    Formula:
        noisy_value = original_value + N(0, noise_level * std)

    Args:
        series       (pd.Series): original feature values
        noise_level  (float)    : 0.05, 0.10, or 0.15
        random_state (int)      : reproducibility seed

    Returns:
        pd.Series: noisy version of the input feature
    """
    rng = np.random.default_rng(random_state)
    feature_std = series.std()
    noise = rng.normal(
        loc=0,
        scale=noise_level * feature_std,
        size=len(series)
    )
    return series + noise

# ============================================================
# FUNCTION 3: create_noisy_dataset
# ============================================================

def create_noisy_dataset(X_test, noise_level):
    """
    Apply Gaussian noise across all 30 feature columns at once.

    Each column gets its own random seed offset (column index)
    so noise patterns differ across features — avoiding the
    unrealistic case where every feature gets identical noise.

    Args:
        X_test      (pd.DataFrame): clean test features
        noise_level (float)       : 0.05, 0.10, or 0.15

    Returns:
        pd.DataFrame: noisy version of X_test, same shape
    """
    X_noisy = X_test.copy()

    for i, col in enumerate(X_test.columns):
        X_noisy[col] = inject_gaussian_noise(
            X_test[col],
            noise_level,
            random_state=RANDOM_STATE + i  # vary seed per column
        )

    print(f"Applied {int(noise_level*100)}% noise to "
          f"{X_test.shape[1]} feature columns")

    return X_noisy

# Quick test in __main__

if __name__ == "__main__":
    X_test, y_test = load_test_set()

    # Test single noise level first
    X_noisy_5pct = create_noisy_dataset(X_test, noise_level=0.05)

    print("\nSample comparison (first feature, first 5 rows):")
    col = X_test.columns[0]
    print("Original:", X_test[col].head().values.round(2))
    print("Noisy 5%:", X_noisy_5pct[col].head().values.round(2))

# ============================================================
# FUNCTION 4: generate_all_noise_levels
# ============================================================

def generate_all_noise_levels(X_test):
    """
    Generate noisy versions of the test set at all 3 required
    levels: 5%, 10%, 15%.

    Args:
        X_test (pd.DataFrame): clean test features

    Returns:
        dict: {
            0.05: noisy_df_5pct,
            0.10: noisy_df_10pct,
            0.15: noisy_df_15pct
        }
    """
    noisy_datasets = {}

    print("=" * 50)
    print("GENERATING NOISY DATASETS")
    print("=" * 50)

    for level in NOISE_LEVELS:
        noisy_datasets[level] = create_noisy_dataset(X_test, level)

    print("=" * 50)
    print(f"Generated {len(noisy_datasets)} noisy datasets")
    print("=" * 50)

    return noisy_datasets
