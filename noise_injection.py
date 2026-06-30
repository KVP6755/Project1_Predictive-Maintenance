"""
================================================================================
noise_injection.py
================================================================================
Project     : Predictive Maintenance
Author      : Subhashree Behera
Week        : Week 4 — Noise Sensitivity Analysis & Threshold Tuning
Role        : Noise Injection

--------------------------------------------------------------------------------
Responsibility:
    Inject synthetic noise into the test dataset at multiple intensity
    levels (5%, 10%, 15%) to simulate real-world sensor degradation —
    things like electrical interference, calibration drift, or faulty
    wiring that occur in actual factory/IoT deployments.

    This module produces 3 noisy versions of the test set, which
    Member 2 (Robustness Evaluation) will run the trained model
    against to measure how much performance degrades under noise.

--------------------------------------------------------------------------------
Why This Matters (Business Context):
    A model that only works on perfectly clean lab data is not
    deployment-ready. Real IoT sensors experience drift, electrical
    noise, and measurement error. If the model's F1 score collapses
    under 10-15% noise, it cannot be trusted on a factory floor.

    This step answers the question: "Will this model still catch
    machine failures when the sensors are slightly unreliable?"

--------------------------------------------------------------------------------
Method:
    Gaussian noise is added to each feature, scaled to that feature's
    own standard deviation. This matters because the 30 features have
    wildly different scales (e.g., Air temperature mean=300, std=2.0
    vs Air temperature_roll_var mean=0.0, std=0.001) — using a single
    fixed noise value would be meaningless across such different scales.

    noisy_value = original_value + N(0, noise_level * feature_std)

--------------------------------------------------------------------------------
Functions:
    load_test_set()              → loads dataset, recreates the same
                                    train/test split used by Member 3/4
    inject_gaussian_noise()      → adds noise to one feature column
    create_noisy_dataset()       → applies noise across all 30 features
    generate_all_noise_levels()  → produces 5%, 10%, 15% noisy datasets
    save_noisy_datasets()        → exports datasets for Member 2 to use
    summarize_noise_impact()     → prints before/after stats per level

--------------------------------------------------------------------------------
Dataset:
    engineered_iot_weather_dataset.csv
    Test split: 2000 rows (20%), stratified, random_state=42
    Test failures: 68 (3.40%) — matches overall 3.39% failure rate

--------------------------------------------------------------------------------
Output Files (saved to data/noisy/):
    test_noisy_5pct.csv
    test_noisy_10pct.csv
    test_noisy_15pct.csv

--------------------------------------------------------------------------------
Usage:
    from noise_injection import generate_all_noise_levels
    noisy_datasets = generate_all_noise_levels()
    # Member 2 then loads these for model evaluation
================================================================================
"""



