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

    

