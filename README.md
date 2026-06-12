# Predictive Maintenance Project

## Overview

This project focuses on predictive maintenance using the AI4I Industrial Predictive Maintenance Dataset. The objective is to engineer meaningful time-series features from machine sensor data that can later be used for failure prediction and machine health monitoring.

---

## Week 1: IoT Telemetry Ingestion and Signal Processing

### Contributor

**Subhashree Behera**

### Tasks Completed

* Loaded and explored the fused industrial dataset (`ai4i_fused.csv`).

* Sorted records chronologically using the `date` column.

* Selected key machine sensor signals:

  * Air temperature [K]
  * Process temperature [K]
  * Rotational speed [rpm]
  * Torque [Nm]
  * Tool wear [min]

* Applied a rolling window of **5 observations** on each sensor.

* Generated the following statistical features:

  * Rolling Mean
  * Rolling Standard Deviation
  * Rolling Variance

### Feature Engineering Output

For each sensor, three new features were created:

Example:

* Torque [Nm]_rolling_mean
* Torque [Nm]_rolling_std
* Torque [Nm]_rolling_var

Total new features created:

5 sensors × 3 statistics = **15 engineered features**

### Dataset Expansion

* Original dataset: **21 columns**
* Feature-engineered dataset: **36 columns**

### Files

* `data/ai4i_fused.csv` — Original fused dataset
* `data/ai4i_rolling_features.csv` — Dataset with rolling window features
* `notebooks/week1_rolling_window_subha.ipynb` — Feature engineering notebook

### Purpose

Rolling window statistics capture short-term machine behavior and trends, allowing future machine learning models to detect anomalies and predict equipment failures more effectively.
