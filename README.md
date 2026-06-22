
# Project1_Predictive-Maintenance
# Predictive Maintenance Project

## Overview
This project uses the **AI4I 2020 Predictive Maintenance Dataset** (fused with simulated weather data) to engineer time-series features from machine sensor readings, supporting failure prediction in Week 3.


---

## Week 1: IoT Telemetry Ingestion & Signal Processing
**Contributor:
-Subhashree Behera
-Vishnupriyan(Team lead)
-Shilpa
-Varnika


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

## Sensors Processed
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## Result
- Original Columns: 21
- New Features Added: 15
- Final Columns: 36

## Contextual Data Fusion
This project combines machine telemetry data with external weather data.

### Weather Features
- Average Temperature
- Minimum Temperature
- Maximum Temperature
- Precipitation
- Wind Speed
- Sea Level Pressure

## Week 1 – Exploratory Data Analysis (EDA)

### Work Completed

* Loaded and explored the AI4I Predictive Maintenance dataset.
* Performed dataset inspection using data overview and summary statistics.
* Analyzed machine failure distribution.
* Created distribution plots for:

  * Air Temperature
  * Process Temperature
  * Rotational Speed
  * Torque
  * Tool Wear
* Generated a correlation heatmap to study relationships among sensor variables.
* Documented observations and insights from each visualization.

### Outcome

The analysis provided an understanding of machine operating conditions, sensor behavior, feature distributions, and relationships among variables. These insights support feature engineering and predictive maintenance modeling in later stages of the project.
=======








# Week 2: Contextual Data Fusion and Feature Engineering (Exploratory Data Analysis (EDA))

## Objective

The objective of Week 2 was to explore and analyze the engineered IoT-weather dataset created by integrating machine telemetry data with external environmental context. The focus was on understanding how contextual weather features may contribute to predictive maintenance applications.

---

## Work Completed

### 1. Engineered Dataset Exploration

* Loaded and inspected the engineered IoT-weather dataset.
* Verified dataset dimensions, feature inventory, and data quality.
* Checked for missing values and validated feature engineering outputs.

### 2. External Context Feature Analysis

Analyzed engineered weather-related features including:

* Average Temperature (Rolling Mean, Standard Deviation, Variance)
* Precipitation (Rolling Mean, Standard Deviation, Variance)
* Sea Level Pressure (Rolling Mean, Standard Deviation, Variance)

Generated distribution plots to understand environmental variability and contextual trends.

### 3. Machine Failure Distribution Analysis

* Examined the distribution of machine failure records.
* Identified class imbalance between normal and failure conditions.
* Visualized failure frequencies using bar charts.

### 4. Engineered Sensor Correlation Analysis

Generated correlation heatmaps for:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

This helped identify relationships among machine telemetry features.

### 5. Sensor–Weather Relationship Analysis

Investigated correlations between machine telemetry and contextual weather variables to understand how environmental conditions may influence machine behavior.

### 6. Failure Mode Analysis

Analyzed the distribution of failure categories:

* TWF (Tool Wear Failure)
* HDF (Heat Dissipation Failure)
* PWF (Power Failure)
* OSF (Overstrain Failure)
* RNF (Random Failure)

Visualized failure mode frequencies and compared their occurrence.

### 7. Tool Wear Lifecycle Analysis

Studied degradation patterns using the engineered Tool Wear rolling mean feature and visualized operational wear trends over time.

---

## Key Findings

* Engineered weather features successfully captured environmental trends and variability.
* Machine failure events were significantly less frequent than normal operations, indicating class imbalance.
* Strong relationships were observed among certain machine telemetry features.
* Contextual weather variables provided additional information beyond machine sensor readings.
* Tool wear trends demonstrated progressive degradation patterns useful for predictive maintenance.

---

## Generated Outputs

All Week 2 visualizations were saved under:

```text
plots/week 2/
```

including:

* Weather feature distribution plots
* Machine failure distribution chart
* Engineered sensor correlation heatmap
* Sensor–weather correlation heatmap
* Failure mode distribution chart
* Tool wear lifecycle analysis plot

---

## Conclusion

Week 2 focused on contextual feature exploration and visualization. The analysis provided evidence that environmental context and engineered rolling statistics may offer valuable information for predictive maintenance systems and future machine learning model development.
