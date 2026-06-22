<<<<<<< HEAD
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

=======

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
>>>>>>> ec1ed952d7002125b285a6ad5d25e4f2857a5d6a
