# Data

Dataset files are not included in this repository due to size limits.

## Download Instructions

### AI4I 2020 (Machine sensor data)
- URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Save as: `data/ai4i2020.csv`

### Global Daily Climate (Weather data)
- URL: https://www.kaggle.com/datasets/guillemservera/global-daily-climate-data
- Save as: `data/daily_weather.parquet`
- Save as: `data/cities.csv`

## Run
After downloading all files run:
python dataset.py
This generates `data/ai4i_fused.csv` automatically.
