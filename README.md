# Project1_Predictive-Maintenance: Machine Failure Pipeline

An end-to-end machine learning system designed to predict equipment failures using rolling telemetry data fused with historical environmental conditions. This repository contains the data ingestion, exploratory analysis, time-series feature engineering, and data imbalance handling components.

---

## 📊 Project Status & Progress Tracker

| Milestone | Focus | Key Output | Status |
| :--- | :--- | :--- | :--- |
| **Week 1** | Ingestion, EDA & Feature Engineering | Fused dataset with 15 rolling window features | ✅ Complete |
| **Week 2** | Handling Severe Data Imbalance | Stratified sampling, SMOTE pipelines & evaluation plots | ✅ Complete |
| **Week 3** | Predictive Modeling | Classifiers, optimization & ensemble methods | 🚀 Up Next |

---

## 🕒 Week 1: IoT Telemetry Ingestion, EDA & Feature Engineering

### 1. Exploratory Data Analysis (EDA)
* Inspecting dataset structural patterns and summary statistics.
* Analyzing the highly skewed baseline distribution of the target variable (`Machine failure`).
* Creating distribution visualizations for all five primary core operational telemetry markers.
* Generating a Pearson correlation heatmap to establish colinear relationships and minimize multi-collinearity during feature selections.

### 2. Time-Series Feature Engineering
The pipeline processes the chronologically ordered raw dataset to convert standard telemetry into historical, trend-based features. Rolling statistical windows extract indicators of degradation or mechanical friction over time.

* **Dataset Scale:** Loaded fused dataset (`ai4i_fused.csv`) containing 10,000 baseline observations.
* **Rolling Configurations:** Applied a rolling window size of **5 cycles** (`min_periods=1` enforced to preserve total data shape).
* **Statistical Extrapolations:** Extracted rolling **mean**, **standard deviation**, and **variance** for all primary machine sensors.

### ⚙️ Sensor & Contextual Metrics Tracked
* **Machine Telemetry (Rolling Stats Applied):** Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
* **Contextual Weather Features Fused:** Average Temperature, Minimum Temperature, Maximum Temperature, Precipitation, Wind Speed, Sea Level Pressure.

### Data Dimensions Summary
* **Original Columns:** 21
* **New Rolling Features Generated:** 15 (5 core sensors × 3 historical metrics)
* **Final Transformed Dataset:** 36 columns, 10,000 rows (zero rows dropped)

---

## ⚖️ Week 2: Handling Data Imbalance (Pipeline Setup)

Predictive maintenance records are intrinsically highly imbalanced. In this framework, severe target skewness is observed, with a **~3.4% machine failure rate** against ~96.6% normal operations. Standard accuracy scoring is an unviable metric. Week 2 addresses this structural risk via parallel data balancing strategies.

### 1. Pipeline Protections & Data Splits
* Implemented a clean **Stratified Train/Test Split ($80/20$)** mapping via `stratify=y`. This locks identical failure/non-failure ratios across both internal partitions prior to any transformations.
* Removed target leakages by isolating sub-failure types (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and index strings (`UDI`, `Product ID`, `date`) away from features $X$.

### 2. Imbalance Remediation Strategies Setup
* **Synthetic Sampling Engine:** Configured SMOTE (Synthetic Minority Over-sampling Technique) to automatically scale the active minority class to an exact $50/50$ ratio. *Enforced strictly inside the training set partition only to prevent out-of-sample data leakage.*
* **Algorithmic Weighting Alternative:** Built an integrated alternative configuration using an enterprise Scikit-Learn `RandomForestClassifier(class_weight='balanced')` trained directly on un-resampled distributions for benchmark contrasts.

### 📈 Baseline Evaluation Utilities & Results
To correctly assess minority prediction performance, specialized visual curves are integrated, defaulting to **Average Precision (AP)** and **Area Under ROC** over raw accuracy metrics.

* **Precision-Recall Curve Average Precision (AP):** `0.708`
* **Area Under the ROC Curve (ROC-AUC):** `0.960`

---

## 📂 Repository File Guide

* `data/ai4i_fused.csv` — Primary input dataset containing raw telemetry fused with environmental metrics.
* `data/ai4i_rolling_features.csv` — Feature engineering output ledger containing the final 36-column transformed telemetry.
* `notebooks/week1_rolling_window.ipynb` — Core interactive notebook containing the exploratory plotting analysis and rolling window design.
* `src/imbalance_handling.py` — Production pipeline containing data split logic, SMOTE generation, and curve export functions.

---

## 🛠️ Infrastructure Setup & Dependencies
Ensure your Python environment contains the necessary scientific packages before deploying models:
```bash
pip install pandas scikit-learn imbalanced-learn matplotlib
