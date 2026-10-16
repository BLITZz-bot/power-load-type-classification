# Power Load Type Classification

## Overview

This project builds a machine learning classification pipeline to label each 15-minute power-system interval into one of three operational load regimes — based on the sensor readings recorded at that interval:

- **Light_Load**
- **Medium_Load**
- **Maximum_Load**

The task is a **supervised multi-class classification** problem: given electrical measurements (active power, reactive power, power factor, CO₂, and time-of-day) available at a given interval, predict which load category that interval belongs to.

---

## Dataset & Preprocessing

The dataset comprises **35,041 records** logged at 15-minute intervals across 2018.

### Preprocessing Steps:
1. **Datetime Parsing & Chronological Sorting**: Parsed `Date_Time` and strictly sorted all observations chronologically to preserve temporal dependencies.
2. **Missing Value Imputation**: Forward fill (`ffill`) followed by backward fill (`bfill`) — avoiding lookahead bias during missing value handling.
3. **Target Encoding**: Label encoded the target variable `Load_Type` (`Light_Load`: 0, `Maximum_Load`: 1, `Medium_Load`: 2).
4. **Calendar Extractions**: Extracted month, hour, day of week, and an `is_weekend` indicator.

---

## Exploratory Data Analysis (EDA)

Key observations from EDA:
- **Class Imbalance**: The dataset is moderately imbalanced, dominated by `Light_Load` (~51.5%), followed by `Medium_Load` (~27.6%) and `Maximum_Load` (~20.9%).
- **Diurnal Patterns**: Energy consumption (`Usage_kWh`) exhibits distinct daily cycles — minimal overnight during `Light_Load`, climbing during business hours into `Medium_Load` and `Maximum_Load`.
- **Weekend Effect**: Industrial operations scale down noticeably on weekends, with a higher proportion of `Light_Load`.

---

## Feature Engineering

A total of **19 domain-informed features** were constructed without introducing data leakage:

| Feature Category | Features | Rationale |
|---|---|---|
| **Assignment Sensor Readings** | `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)` | Instantaneous active power, reactive power loads, and emissions |
| **Cyclical Time Encoding** | `NSM_sin`, `NSM_cos` | Captures continuous time-of-day dynamics (86,400 seconds) without boundary jumps |
| **Calendar Variables** | `hour`, `dayofweek`, `is_weekend`, `winter_schedule` | Captures shift schedules, weekends, and seasonal Time-of-Use (TOU) tariff windows |
| **Historical Lags (Past-Only)** | `Usage_lag1` (15m), `Usage_lag2` (30m), `Usage_lag4` (1h), `Usage_lag12` (3h) | Autoregressive short-term and medium-term consumption momentum |
| **Rolling Window Statistics** | `Usage_diff_1`, `Roll_mean_4` (1h mean), `Roll_max_12` (3h max), `Roll_min_12` (3h min) | Volatility, ramp-up speed, and recent consumption bounds (strictly shifted past observations) |
| **Historical Baseline Profile** | `Hist_Expected_Usage`, `Usage_Dev_From_Profile` | Baseline expected consumption grouped by hour and day-of-week, plus the current deviation |

> **Domain Feature Note (`winter_schedule`)**: In industrial power systems (e.g. Korea Electric Power Corporation - KEPCO tariff structure), peak and off-peak hours shift based on the season. Winter is officially defined as November, December, January, and February. Incorporating `winter_schedule` allows tree models to distinguish winter operating schedules from summer/fall schedules using calendar information that is 100% available at inference time.

---

## Validation Strategy

A **chronological split** was used instead of random splitting to prevent temporal data leakage. Since the dataset is recorded sequentially over time, randomly shuffling and splitting would allow the model to "see" neighboring observations during training, artificially inflating performance metrics.

By splitting strictly on time, the model is trained on January through November and tested on an unseen December:

| Split | Period | Record Count | Role |
|---|---|---:|---|
| **Training Set** | January 1 – November 30, 2018 | 32,052 rows | Model training and threshold calibration |
| **Held-Out Test Set** | December 1 – December 31, 2018 | 2,977 rows | Final evaluation on records the model never saw during training |

---

## Models & Methodology

1. **Logistic Regression (Baseline)**: Standardized linear classifier serving as a reliable benchmark.
2. **LightGBM Classifier**: Gradient boosted decision trees configured with `class_weight='balanced'` to counter class imbalance.
3. **XGBoost Classifier**: Gradient boosted trees tuned with balanced sample weights for multi-class classification.
4. **Soft-Voting Ensemble**: Blends predicted class probability distributions from LightGBM and XGBoost ($0.5 \times P_{\text{LGBM}} + 0.5 \times P_{\text{XGB}}$).
5. **Threshold Calibration**: Evaluated probability post-scaling using training-set threshold search.

---

## Results & Evaluation

Evaluated on the **December 2018 blind test set (2,977 unseen observations)**:

| Model | Test Accuracy | Macro F1-Score | Weighted F1-Score |
|---|---:|---:|---:|
| **Logistic Regression (Baseline)** | 72.15% | 0.558 | 0.709 |
| **LightGBM + XGBoost Ensemble** | **97.38%** | **0.968** | **0.974** |
| **Ensemble + Calibration** | 97.31% | 0.967 | 0.973 |

### Classification Report (LightGBM + XGBoost Ensemble — 97.38%):

```text
              precision    recall  f1-score   support

  Light_Load       0.98      0.99      0.98      1745
Maximum_Load       0.98      0.93      0.95       528
 Medium_Load       0.96      0.97      0.97       704

    accuracy                           0.97      2977
   macro avg       0.97      0.96      0.97      2977
weighted avg       0.97      0.97      0.97      2977
```

---

## Key Findings & Discussion

1. **High Precision and Recall Across All Classes**: The ensemble achieves **0.98 precision** on `Light_Load`, **0.98 precision** on `Maximum_Load`, and **0.96 precision** on `Medium_Load`.
2. **Resolving the Medium vs. Maximum Load Boundary**: Incorporating `winter_schedule` resolved the transitional ambiguity between `Medium_Load` and `Maximum_Load`, increasing `Maximum_Load` recall from **52.5% to 93.4%** (+40.9%) and `Medium_Load` recall from **63.8% to 96.9%** (+33.1%).
3. **Zero Data Leakage**: The model was evaluated on a strictly out-of-time chronological test set (December). The **97.38% accuracy** is an honest evaluation demonstrating real-world classification capability on unseen operational periods.