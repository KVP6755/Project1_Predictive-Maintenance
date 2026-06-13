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