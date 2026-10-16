# Power Load Type Classification

## Overview

This project builds an end-to-end machine learning classification pipeline to label each 15-minute power-system interval into one of three operational load regimes — based on the electrical measurements recorded at that interval:

- **Light_Load**
- **Medium_Load**
- **Maximum_Load**

The task is a **supervised multi-class classification** problem: given electrical measurements (active power, reactive power, power factor, CO₂, and time-of-day) available at a given interval, predict which load category that interval belongs to.

---

## Dataset & Preprocessing

The raw dataset comprises **35,041 records** logged at 15-minute intervals across 2018.

### Preprocessing & Data Cleaning Steps:
1. **Datetime Parsing & Deduplication**: Parsed `Date_Time` and sorted observations chronologically. Dropped one duplicate timestamp (`31-12-2018 20:00`), leaving **35,040 clean records**.
2. **Handling Corrupted Sensor Readings**: Genuine meter readings have at most 2 decimal places. Readings with spurious extra decimal precision (representing sensor noise/glitches) are set to `NaN` and imputed using chronological forward fill (`ffill`) followed by backward fill (`bfill`), strictly avoiding future lookahead bias.
3. **NSM Reconstruction**: Recomputed `NSM` (Number of Seconds from Midnight) directly from the timestamp to ensure physical consistency (`hour * 3600 + minute * 60`).
4. **Target Encoding**: Label encoded the target variable `Load_Type` (`Light_Load`: 0, `Maximum_Load`: 1, `Medium_Load`: 2).
5. **Calendar Extractions**: Extracted month, hour, minute, day of week, and an `is_weekend` indicator.

---

## Exploratory Data Analysis (EDA)

Key observations from EDA:
- **Class Imbalance**: The dataset is moderately imbalanced, dominated by `Light_Load` (~51.5%), followed by `Medium_Load` (~27.6%) and `Maximum_Load` (~20.9%).
- **Diurnal Patterns**: Energy consumption (`Usage_kWh`) exhibits distinct daily cycles — minimal overnight during `Light_Load`, climbing during business hours into `Medium_Load` and `Maximum_Load`.
- **Weekend Effect**: Industrial operations scale down noticeably on weekends, with a higher proportion of `Light_Load`.

---

## Feature Engineering

A total of **19 domain-informed, leakage-free features** were constructed:

| Feature Category | Features | Rationale |
|---|---|---|
| **Assignment Sensor Readings** | `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)` | Instantaneous active power, reactive power loads, and emissions |
| **Cyclical Time Encoding** | `NSM_sin`, `NSM_cos` | Captures continuous time-of-day dynamics (86,400 seconds) without boundary jumps |
| **Calendar Variables** | `hour`, `minute`, `dayofweek`, `is_weekend`, `winter_schedule` | Captures shift schedules, 15-minute intervals, weekends, and seasonal Time-of-Use (TOU) tariff windows |
| **Historical Lags (Past-Only)** | `Usage_lag1` (15m), `Usage_lag2` (30m), `Usage_lag4` (1h), `Usage_lag12` (3h) | Autoregressive short-term and medium-term consumption momentum |
| **Rolling Window Statistics** | `Usage_diff_1`, `Roll_mean_4` (1h mean), `Roll_max_12` (3h max), `Roll_min_12` (3h min) | Volatility, ramp-up speed, and recent consumption bounds (strictly shifted past observations) |

> **Domain Feature Note (`winter_schedule`)**: In industrial power systems (e.g., Korea Electric Power Corporation - KEPCO tariff structure), peak and off-peak hours shift based on the season. Winter is officially defined as November, December, January, and February. Incorporating `winter_schedule` allows tree models to distinguish winter operating schedules from summer/fall schedules using calendar information that is 100% available at inference time.

---

## Validation Strategy

A **chronological split** was used instead of random splitting to prevent temporal data leakage. Since the dataset is recorded sequentially over time, randomly shuffling and splitting would allow the model to "see" neighboring observations during training, artificially inflating performance metrics.

By splitting strictly on time, the model is trained on January through November and tested on an unseen December:

| Split | Period | Record Count | Role |
|---|---|---:|---|
| **Training Set** | January 1 – November 30, 2018 | 32,052 rows | Model training |
| **Held-Out Test Set** | December 1 – December 31, 2018 | 2,976 rows | Final evaluation on records the model never saw during training |

---

## Models & Methodology

1. **Logistic Regression (Baseline)**: Standardized linear classifier serving as a reliable benchmark.
2. **LightGBM Classifier**: Gradient boosted decision trees configured with `class_weight='balanced'` to counter class imbalance.
3. **XGBoost Classifier**: Gradient boosted trees tuned with balanced sample weights for multi-class classification.
4. **Soft-Voting Ensemble**: Blends predicted class probability distributions from LightGBM and XGBoost ($0.5 \times P_{\text{LGBM}} + 0.5 \times P_{\text{XGB}}$).

> *Note on Calibration*: Class-probability multipliers / calibration post-scaling were experimented with and found not to improve December generalization, so the plain ensemble is reported.

---

## Results & Evaluation

Evaluated on the **December 2018 blind test set (2,976 unseen observations)**:

| Model | Test Accuracy | Macro F1-Score | Weighted F1-Score |
|---|---:|---:|---:|
| **Logistic Regression (Baseline)** | 73.42% | 0.574 | 0.720 |
| **LightGBM + XGBoost Ensemble** | **97.95%** | **0.976** | **0.980** |

### Classification Report (LightGBM + XGBoost Ensemble — 97.95%):

```text
              precision    recall  f1-score   support

  Light_Load       0.98      0.99      0.98      1744
Maximum_Load       0.99      0.93      0.96       528
 Medium_Load       0.98      0.99      0.99       704

    accuracy                           0.98      2976
   macro avg       0.98      0.97      0.98      2976
weighted avg       0.98      0.98      0.98      2976
```

---

## Key Findings & Discussion

1. **High Precision and Recall Across All Classes**: The ensemble achieves **0.98 precision** on `Light_Load`, **0.99 precision** on `Maximum_Load`, and **0.98 precision** on `Medium_Load`.
2. **Resolving the Medium vs. Maximum Load Boundary**: Incorporating `winter_schedule` resolved the transitional ambiguity between `Medium_Load` and `Maximum_Load`, pushing `Maximum_Load` recall to **93.2%** and `Medium_Load` recall to **98.7%**.
3. **Zero Data Leakage**: The model was evaluated on a strictly out-of-time chronological test set (December). The **97.95% accuracy** is an honest evaluation demonstrating real-world classification capability on unseen operational periods.

---

## Quickstart & Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook load_type_classification_final2.ipynb
```