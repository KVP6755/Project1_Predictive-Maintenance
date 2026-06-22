# Week 2 

### Contributions — Subhashree Behera

## Track 2: EDA & Visualization

### Tasks Completed
| Step | Description |
|------|-------------|
| 1 | Loaded engineered IoT weather dataset (9,996 rows × 40 columns) |
| 2 | Dataset overview — shape, failure rate (3.39%), null check |
| 3 | Analyzed failure type breakdown (TWF, HDF, PWF, OSF, RNF) |
| 4 | Selected IoT + weather columns for correlation analysis |
| 5 | Generated IoT vs weather correlation heatmap |
| 6 | Identified top features correlated with Machine failure |
| 7 | Visualized class imbalance (96.61% normal vs 3.39% failure) |
| 8 | Plotted failure type distribution bar chart |
| 9 | Created tool wear lifecycle degradation chart |
| 10 | Compared Torque rolling std: Normal(9.23) vs Failure(11.74) |
| 11 | EDA summary with key findings documented |

### Key Findings
- Class imbalance ratio: 28.5:1 (normal:failure)
- HDF is most common failure type (115 cases)
- Torque rolling std increases by 2.51 before failure
- Tool wear shows clear degradation pattern before failure points

### Files
- `notebooks/week2_Advanced_feature_engineering.ipynb`
- `plots/correlation_heatmap_iot_weather.png`
- `plots/class_imbalance.png`
- `plots/failure_type_distribution.png`
- `plots/tool_wear_lifecycle.png`
- `plots/torque_std_by_failure.png`