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


----------------------------------------------------------------------------------------------------------


# Week 3: Imbalanced Classification and SMOTE Balancing

## Objective

The objective of Week 3 was to address the severe class imbalance in the predictive maintenance dataset. Since machine failures represent only a small percentage of the observations, a reusable oversampling module was developed using SMOTE techniques. This module is designed to be integrated safely within a 5-Fold Stratified Cross Validation pipeline without introducing data leakage.

## Tasks Completed

- Developed a reusable SMOTE balancing module (smote_balancer.py).
- Implemented Standard SMOTE for minority class oversampling.
- Implemented Borderline-SMOTE for generating synthetic samples near the decision boundary.
- Designed functions to operate only on training data to prevent data leakage.
- Created reusable helper functions for:
    - Standard SMOTE balancing
    - Borderline-SMOTE balancing
    - Class distribution comparison
    - Class count retrieval
- Verified the balancing functions using a separate Jupyter Notebook (week3_smote_testing.ipynb).
- Confirmed balanced class distributions after oversampling.

## Files Added

    Week 3/
    ├── smote_balancer.py
    └── week3_smote_testing.ipynb

## Functions Implemented

1.apply_smote()

    Performs standard SMOTE oversampling on the training dataset.

    Input

    X_train
    y_train

    Output

    X_resampled
    y_resampled

2.apply_borderline_smote()

    Applies Borderline-SMOTE to generate synthetic samples near difficult decision boundaries.

    Input

    X_train
    y_train

    Output

    X_resampled
    y_resampled

3.compare_class_distribution()

   Compares the target class distribution before and after balancing.

4.get_class_counts()

   Returns the number of samples belonging to each class.

## Validation Results

The implemented balancing methods successfully increased the minority class samples to match the majority class.


## Outcome

The Week 3 deliverables provide a reusable imbalance handling module that can be directly integrated into the team's LightGBM training pipeline. By applying SMOTE only within each training fold of Stratified Cross Validation, the implementation prevents data leakage while improving the model's ability to learn from rare machine failure cases.