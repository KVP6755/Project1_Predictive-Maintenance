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

