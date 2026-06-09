"""
Contextual Data Fusion Pipeline

Merges internal IoT telemetry (AI4I 2020 dataset) with external
weather data (Kaggle Global Daily Climate) using date-based alignment.

Author: Varnika Valliammai V
Branch: feature/data-fusion-varnika
"""


import pandas as pd
import numpy as np
import os

# 1:LOAD DATASETS

# AI4I Predictive Maintenance Dataset
ai4i = pd.read_csv(r'C:\Users\varni\OneDrive\Desktop\ai4i2020.csv')

# Daily weather dataset
weather = pd.read_parquet(r'C:\Users\varni\OneDrive\Desktop\daily_weather.parquet')

# City-to-weather-station mapping
cities = pd.read_csv(r'C:\Users\varni\OneDrive\Desktop\cities.csv')


# 2: BASIC DATA INSPECTION

print(ai4i.shape)
print("Columns:", ai4i.columns.tolist())
print(ai4i.head(3))

print("weather.shape: ",weather.shape)
print("weather.columns: ",(weather.columns.tolist()))
print(weather.head(3))

print("Cities shape:", cities.shape)
print("Cities columns:", cities.columns.tolist())
print(cities.head(5))

#CHECK WHETHER CHENNAI CITY EXISTS
print("Cities columns:", cities.columns.tolist())
print()
print(cities.head(5))

print(cities['city_name'].head(30).tolist())
print(cities[cities['city_name'].str.contains('Chennai',case=False)])

#3: FIND WEATHER STATION

city_name='Chennai'

# Search for Chennai in city mapping table
city_row=cities[cities['city_name'].str.contains(city_name,case=False)]
print(city_row)

# Extract weather station ID
station_id=city_row['station_id'].values[0]
print("station_id", station_id)

# 4: FILTER WEATHER DATA

weather_city=weather[weather['station_id']==station_id].copy()
print("Filtered weather shape: ",weather_city.shape)
print(weather_city.head(3))

# Keep only useful weather features
weather_city['date']=pd.to_datetime(weather_city['date']).dt.date
weather_slim=weather_city[[
    'date',
    'avg_temp_c',
    'min_temp_c',
    'max_temp_c',
    'precipitation_mm',
    'avg_wind_speed_kmh',
    'avg_sea_level_pres_hpa',
    'sunshine_total_min'
]].copy()

print("weather slim shape: ",weather_slim.shape)
print(weather_slim.head(3))

# 5: FILTER WEATHER DATA TO MATCH AI4I PERIOD

weather_slim['date']=pd.to_datetime(weather_slim['date'])

weather_slim=weather_slim[
    (weather_slim['date'].dt.year >= 2020) &
    (weather_slim['date'].dt.year <= 2021)
].copy()

weather_slim['date']=weather_slim['date'].dt.date

print("Bremen 2020-2021 shape:", weather_slim.shape)

# 6: HANDLE MISSING WEATHER VALUES

print("Missing values:\n", weather_slim.isnull().sum())
print(weather_slim.head(5))

weather_slim=weather_slim.drop(columns=['sunshine_total_min'])
weather_slim['precipitation_mm'] = weather_slim['precipitation_mm'].fillna(0)

print("Missing values after fix:\n", weather_slim.isnull().sum())
print(weather_slim.head(5))

# 7: CREATE TIMESTAMPS FOR AI4I DATA

ai4i['datetime']=pd.date_range(
    start='2020-01-01',
    periods=len(ai4i),
    freq='h'
)
ai4i['date']=ai4i['datetime'].dt.date

print("AI4I date range:",ai4i['date'].min(),'to',ai4i['date'].max())

# 8: DATA FUSION

# Merge machine telemetry with weather context
fused=pd.merge(ai4i,weather_slim,on='date',how='left')

print("Fused shape:", fused.shape)

print("Missing values:\n", fused.isnull().sum())

#9: HANDLE POST-MERGE MISSING VALUES

fused=pd.merge(ai4i,weather_slim,on='date',how='left')

print("fused shape:",fused.shape)
print("missing values:\n", fused.isnull().sum())

#REMOVE TEMPORARY COLUMNS
fused = fused.drop(columns=['datetime'])

#Forward fill weather observations
fused = fused.ffill()

# Fill remaining numeric nulls using column means
fused = fused.fillna(fused.mean(numeric_only=True))
print("missing after fill:",fused.isnull().sum().sum())

#11: SAVE FUSED DATASET

output_path=r'C:\Users\varni\PycharmProjects\varnika\Predictive Maintenance\ai4i_fused.csv'
fused.to_csv(output_path,index=False)

print("Saved successfully!")
print("Final shape:", fused.shape)
print("Final columns:", fused.columns.tolist())
