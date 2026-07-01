# Predictive Maintenance Project

## Overview
This project uses the **AI4I 2020 Predictive Maintenance Dataset** (fused with simulated weather data) to engineer time-series features from machine sensor readings, supporting failure prediction in Week 3.

---

## Week 1: IoT Telemetry Ingestion & Signal Processing
**Contributor:** Subhashree Behera

### What Was Done
| Step | Description |
|------|-------------|
| 1 | Loaded fused dataset (`ai4i_fused.csv`) — 10,000 rows × 21 columns |
| 2 | Cleaned column names, verified zero null values |
| 3 | Sorted chronologically by `date` for valid time-series analysis |
| 4 | Applied rolling window (size=5) on 5 sensor columns |
| 5 | Generated rolling mean, std, and variance for each sensor |

### Sensors Processed
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Result
| Metric | Value |
|--------|-------|
| Original columns | 21 |
| New features added | 15 (5 sensors × 3 stats) |
| Final columns | 36 |
| Rows | 10,000 (no rows dropped — `min_periods=1` used) |

### Key Files
- `data/ai4i_fused.csv` — input dataset
- `data/ai4i_rolling_features.csv` — output with rolling features
- `notebooks/week1_rolling_window.ipynb` — feature engineering notebook

### Why This Matters
Rolling window statistics capture **short-term trends and instability** in sensor readings — a key early signal that a machine is degrading before it fully fails. These features feed directly into the LightGBM classifier in Week 3.

# Week 2 

#### Contributions — Subhashree Behera

## Track 2: EDA & Visualization

### Tasks Completed
| Step | Description |
|------|-------------|
| 1 | Loaded engineered IoT weather dataset (9,996 rows × 40 columns) |
| 2 | Dataset overview — shape, failure rate (3.39%), null check |
| 3 | Analyzed failure type breakdown (TWF, HDF, PWF, OSF, RNF) |
| 4 | Selected IoT + weather columns for correlation analysis |
| 5 | Generated IoT vs weather correlation heatmap |
| 6 | Identified top features correlated with Machine failure |
| 7 | Visualized class imbalance (96.61% normal vs 3.39% failure) |
| 8 | Plotted failure type distribution bar chart |
| 9 | Created tool wear lifecycle degradation chart |
| 10 | Compared Torque rolling std: Normal(9.23) vs Failure(11.74) |
| 11 | EDA summary with key findings documented |

### Key Findings
- Class imbalance ratio: 28.5:1 (normal:failure)
- HDF is most common failure type (115 cases)
- Torque rolling std increases by 2.51 before failure
- Tool wear shows clear degradation pattern before failure points

### Files
- `notebooks/week2_Advanced_feature_engineering.ipynb`
- `plots/correlation_heatmap_iot_weather.png`
- `plots/class_imbalance.png`
- `plots/failure_type_distribution.png`
- `plots/tool_wear_lifecycle.png`
- `plots/torque_std_by_failure.png`

---

## Week 3 Contributions — Pipeline Architect

### File Owned
`pipeline_builder.py`

### Responsibility
Built the core 5-fold Stratified Cross-Validation pipeline
structure. This is the skeletal framework all other Week 3
modules (SMOTE, LightGBM, Evaluation) plug into.

### Functions Built
| Function | Purpose |
|---|---|
| `load_and_prepare_data()` | Loads dataset, separates X (30 features) and y |
| `validate_class_distribution()` | Verifies failure rate per fold |
| `build_cv_pipeline()` | Core — creates all 5 stratified folds |
| `get_fold()` | Helper to retrieve individual folds |
| `summarize_pipeline()` | Prints full pipeline report |
| `get_feature_names()` | Exports feature list for LightGBM member |

### Pipeline Output (verified)
| Fold | Train Size | Val Size | Train Fail% | Val Fail% |
|------|-----------|---------|------------|---------|
| 1 | 7996 | 2000 | 3.39% | 3.40% |
| 2 | 7997 | 1999 | 3.40% | 3.35% |
| 3 | 7997 | 1999 | 3.39% | 3.40% |
| 4 | 7997 | 1999 | 3.39% | 3.40% |
| 5 | 7997 | 1999 | 3.39% | 3.40% |

### Why Stratified?
Dataset has severe imbalance (3.39% failures). Standard KFold
would create uneven fold distributions. StratifiedKFold
guarantees each fold maintains the same failure ratio.

### Leakage Prevention
SMOTE is intentionally NOT applied in this file. It will be
applied by Member 2 (smote_balancer.py) ONLY inside training
folds — never touching validation data.