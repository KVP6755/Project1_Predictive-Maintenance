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