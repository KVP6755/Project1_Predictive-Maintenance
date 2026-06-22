## Week 1 — Contextual Data Fusion Pipeline

**Script:** dataset.py
**Input:**  data/ai4i2020.csv, data/daily_weather.parquet, data/cities.csv
**Output:** data/ai4i_fused.csv

### What I did
- Loaded the AI4I 2020 Predictive Maintenance Dataset (10,000 rows, 14 columns)
- Loaded the Kaggle Global Daily Climate dataset (weather + city lookup)
- Assigned synthetic hourly timestamps to AI4I, since it has no real
  timestamps (one reading per hour, starting 2020-01-01)
- Filtered the weather dataset to Bremen, Germany for 2020-2021
  (Chennai was preferred but had no weather readings in the dataset)
- Merged AI4I sensor data with weather data on the date column
- Filled missing values (forward fill, then column mean)

### Result
- Final shape: (10,000, 20)
- 0 missing values
- Columns: original 14 AI4I sensor/failure columns + 6 weather columns
  (avg_temp_c, min_temp_c, max_temp_c, precipitation_mm,
  avg_wind_speed_kmh, avg_sea_level_pres_hpa)

  

## WEEK 2 - Handling Data Imbalance

**Script:** imbalanced_data_handling.py
**Author:** Varnika Valliammai V
**Input:**  data/ai4i_rolling_features.csv 
**Output:** Reusable SMOTE pipeline, class_weight model setup, evaluation plotting functions

### Objective
Set up tools to handle the severe class imbalance in machine failure
prediction (failures are only ~3.4% of the data), and prepare proper
evaluation metrics since accuracy is misleading on imbalanced data.

### What I did

**1. Data preparation**
- Loaded the rolling-features dataset (sensors + weather + rolling stats)
- Dropped leakage columns (TWF, HDF, PWF, OSF, RNF — these are
  components of the target itself, not real predictive features)
- One-hot encoded the `Type` column (L/M/H)
- Split into train/test sets BEFORE any oversampling, to prevent
  synthetic data leakage into the test set

**2. SMOTE oversampling**
- Applied SMOTE (Synthetic Minority Over-sampling Technique) on the
  training data only
- Creates synthetic "failure" examples by interpolating between
  existing ones, balancing the ~28:1 class ratio

**3. class_weight='balanced' alternative**
- Set up a Random Forest model using `class_weight='balanced'` as an
  alternative to SMOTE
- Instead of altering the data, this penalizes misclassifying the
  rare failure class more heavily during training
- Added a comparison helper to view before/after class distributions

**4. Precision-Recall curve**
- Built a reusable `plot_precision_recall()` function
- Accuracy is misleading here — a model predicting "no failure"
  always would score ~96% accuracy while being useless
- This curve shows the tradeoff between catching real failures
  (recall) and avoiding false alarms (precision)

**5. ROC-AUC curve**
- Built a reusable `plot_roc_auc()` function
- Plots True Positive Rate vs False Positive Rate across thresholds
- AUC summarizes overall model separability (0.5 = random, 1.0 = perfect)

### Result
- `X_train_smote`, `y_train_smote` — balanced training data ready for modeling
- `model_balanced` — trained reference Random Forest using class weighting
- `plot_precision_recall(model, X_test, y_test, label)` — reusable evaluation function
- `plot_roc_auc(model, X_test, y_test, label)` — reusable evaluation function

### How to run
