# Power Load Type Classification

## Overview

This project builds an end-to-end machine learning pipeline to classify industrial power-system loads into three operational regimes:
- **Light_Load**
- **Medium_Load**
- **Maximum_Load**

Accurate load-type classification enables industrial facilities and grid operators to anticipate demand peaks, optimize tariff schedules, and balance power factor and reactive power compensation dynamically.

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

A total of **18 domain-informed features** were constructed without introducing data leakage:

| Feature Category | Features | Rationale |
|---|---|---|
| **Assignment Sensor Readings** | `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)` | Instantaneous active power, reactive power loads, and emissions |
| **Cyclical Time Encoding** | `NSM_sin`, `NSM_cos` | Captures continuous time-of-day dynamics (86,400 seconds) without boundary jumps |
| **Calendar Variables** | `hour`, `dayofweek`, `is_weekend` | Captures work shifts, weekday schedules, and factory shutdowns |
| **Historical Lags (Past-Only)** | `Usage_lag1` (15m), `Usage_lag2` (30m), `Usage_lag4` (1h), `Usage_lag12` (3h) | Autoregressive short-term and medium-term consumption momentum |
| **Rolling Window Statistics** | `Usage_diff_1`, `Roll_mean_4` (1h mean), `Roll_max_12` (3h max), `Roll_min_12` (3h min) | Volatility, ramp-up speed, and recent consumption bounds (strictly shifted past observations) |
| **Historical Baseline Profile** | `Hist_Expected_Usage`, `Usage_Dev_From_Profile` | Baseline expected consumption grouped by hour and day-of-week, plus the current deviation |

---

## Validation Strategy

To prevent temporal leakage and simulate real-world production forecasting, a **strict chronological split** was employed:

| Split | Period | Record Count | Role |
|---|---|---:|---|
| **Training Set** | January 1 – November 30, 2018 | 32,064 rows | Model training and threshold calibration |
| **Blind Test Set** | December 1 – December 31, 2018 | 2,977 rows | Unseen out-of-time evaluation |

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
| **Logistic Regression (Baseline)** | 70.94% | 0.558 | 0.702 |
| **LightGBM + XGBoost Ensemble** | **79.85%** | **0.699** | **0.803** |
| **Ensemble + Calibration** | 79.74% | 0.693 | 0.801 |

### Classification Report (LightGBM + XGBoost Ensemble — 79.85%):

```text
              precision    recall  f1-score   support

  Light_Load       0.99      0.95      0.97      1745
Maximum_Load       0.52      0.52      0.52       528
 Medium_Load       0.58      0.64      0.61       704

    accuracy                           0.80      2977
   macro avg       0.70      0.70      0.70      2977
weighted avg       0.81      0.80      0.80      2977
```

---

## Key Findings & Discussion

1. **Near-Perfect Identification of Light Load**: The model reaches **0.99 precision** and **0.95 recall** for `Light_Load`, making it exceptionally dependable for detecting off-peak periods.
2. **Boundary Overlap between Medium and Maximum Load**: The primary classification errors occur between `Medium_Load` and `Maximum_Load`. Electrical profiles in transitional operational states share similar power factor and reactive power signatures.
3. **Realistic, Leakage-Free Generalization**: Achieving **~80% accuracy** on a purely chronological blind test set (December) demonstrates honest model generalization rather than inflated metrics from random cross-validation leakage.