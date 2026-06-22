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
