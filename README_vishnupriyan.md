# Individual Project Contributions - [VISHNUPRIYAN K]

## Table of Contents
1. [Week 1 - Feature Engineering & Data Preprocessing](#week-1---feature-engineering--data-preprocessing)
2. [Week 2 - Predictive Model Training, Hyperparameter Tuning & Evaluation](#week-2---predictive-model-training-hyperparameter-tuning--evaluation)

# Week 1 - Feature Engineering & Data Preprocessing

## Project Overview
During Week 1, the primary focus was on cleaning the core predictive maintenance dataset (`ai4i_fused.csv`) and engineering robust time-series features. By fusing continuous machine sensor logs with environmental weather metrics, we established a robust feature matrix optimized for predictive maintenance classification algorithms.

## Preprocessing & Engineering Pipeline

1. **Data Cleaning & Column Standardization**:
   * Removed all leading and trailing whitespaces from the dataset columns using `str.strip()` to eliminate lookup errors.
   * Chronologically sorted the dataset by `date` and `UDI` columns to maintain strict time-series alignment for subsequent window calculations.

2. **Telemetry & Weather Feature Selection**:
   * **Machine Diagnostics**: Air temperature, Process temperature, Rotational speed, Torque, and Tool wear.
   * **Ambient Weather Context**: Average/minimum/maximum temperature, precipitation levels, average wind speed, and sea-level atmospheric pressure.

3. **Rolling Window Statistical Feature Generation**:
   * Implemented a moving operational window size of **5 logs** (`window_size = 5`).
   * Generated three dynamic continuous features for every primary telemetry metrics:
     * **Rolling Mean** (captures baseline operational shifts)
     * **Rolling Standard Deviation** (tracks signal instability)
     * **Rolling Variance** (highlights volatility anomalies)

4. **Warmup Buffer Filtering & Metadata Merging**:
   * Merged engineered window components back with explicit failure mode attributes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
   * Dropped early `NaN` artifacts naturally generated during the initial 5-log rolling calculation warmup phase to yield a clean dataset.

## Outputs & Specifications
* **Source Dataset Path**: `D:\DS115-VIS\.venv\Project1_Predictive-Maintenance\ai4i_fused.csv`
* **Generated Output File**: `engineered_iot_weather_dataset.csv`
* **Target Integrity**: Retained exact tracking for primary machine failure status indicators.

# Week 2 - Predictive Model Training, Hyperparameter Tuning & Evaluation

## Project Overview
The focus for Week 2 shifted toward building robust predictive pipelines to classify machine failures. Due to the high class imbalance common in predictive maintenance data, the training pipeline incorporated balanced class weighting, rigorous cross-validation grid searches optimizing for F1-score, and validation using Receiver Operating Characteristic (ROC) curves.

## Modeling Pipeline & Methods Implemented

1. **Scikit-Learn Pipeline Construction**:
   * Encapsulated data scaling and model execution into unified `Pipeline` architectures to prevent data leakage during training and validation.
   * **Logistic Regression (LR)**: Configured with standard scaling (`StandardScaler`) and a maximum iteration cap of 1000 to guarantee convergence.
   * **Random Forest Classifier (RF)**: Initialized with a fixed random seed (`random_state=42`) for reproducibility.

2. **Class Imbalance Handling**:
   * Configured both classifiers with `class_weight='balanced'`. This penalizes minority class misclassifications more heavily, preventing the models from biasing toward dominant operational metrics.

3. **Hyperparameter Tuning via GridSearchCV**:
   * Executed an exhaustive multi-core parallelized hyperparameter sweep using 3-Fold Cross-Validation (`cv=3`).
   * **LR Tuning Parameter**: Inverse regularization strength `C` optimized across `[0.01, 0.1, 1, 10]`.
   * **RF Tuning Parameters**: Max tree depth (`max_depth`) and estimator count (`n_estimators`) combinations optimized to establish the best fitting limits.
   * **Target Metric**: The tuning function was configured to optimize specifically for **F1-Score** (`scoring='f1'`) to ensure an optimal balance between precision and recall.

4. **Performance Evaluation & Validation**:
   * Generated complete `classification_report` output tracking Precision, Recall, and overall binary F1-score on holdout test partitions.
   * Plotted a consolidated **ROC (Receiver Operating Characteristic) Curve** comparison map.
   * Extracted individual Area Under the Curve (AUC) scores to evaluate the True Positive Rate (Sensitivity) against the False Positive Rate across various probability thresholds.

## Key Outputs
* **Optimized Models Saved**: Best-performing iterations for both Linear (LR) and Ensemble (RF) models preserved in the execution runtime.
* **Evaluation Outputs**: Test set classification performance report logs and an explicit ROC comparison plot visual asset.
