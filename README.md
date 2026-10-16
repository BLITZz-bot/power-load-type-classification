# Power Load Type Classification

Classification pipeline for industrial power-system loads using sensor telemetry and seasonal tariff schedule patterns.

The goal is to classify each 15-minute interval into one of three load types:
- **Light_Load**
- **Medium_Load**
- **Maximum_Load**

This is a **multi-class classification** task based on electrical measurements available at each interval (active power, reactive power, power factor, CO₂, and time features).

---

## Dataset & Preprocessing

The raw dataset contains **35,041 records** sampled every 15 minutes across 2018.

### Data Cleaning:
1. **Deduplication**: Found and dropped one duplicate timestamp (`31-12-2018 20:00`), leaving 35,040 rows.
2. **Glitch Removal**: Genuine meter readings have at most 2 decimal places. Values with long decimal tails (e.g., power factor showing 300%+) are sensor glitches. These were replaced with `NaN` and forward-filled (`ffill()`), using only past readings.
3. **Rebuilding NSM**: `NSM` (Seconds from Midnight) had values exceeding 86,400. Recomputed it cleanly from the timestamp: `hour * 3600 + minute * 60`.
4. **Target Encoding**: `Load_Type` was encoded as integers (`Light_Load: 0`, `Maximum_Load: 1`, `Medium_Load: 2`).

---

## Feature Engineering (23 Features)

All features use strictly current sensor readings or past intervals. Nothing looks into the future.

| Group | Features | Description |
|---|---|---|
| **Sensor Readings** | `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)` | Instantaneous power and emissions at the current interval |
| **Cyclical Time** | `NSM_sin`, `NSM_cos` | Sine/Cosine encoding of the time of day so 23:45 connects smoothly to 00:00 |
| **Calendar** | `hour`, `minute`, `dayofweek`, `is_weekend` | Time of day, weekday/weekend shift patterns |
| **Domain Feature** | `winter_schedule` | Binary flag for Nov, Dec, Jan, Feb. In industrial power tariffs (KEPCO), peak and off-peak hours officially change in winter |
| **Short-Term Lags** | `Usage_lag1` (15m), `Usage_lag2` (30m), `Usage_lag4` (1h), `Usage_lag12` (3h) | Captures immediate usage momentum |
| **Periodic Lags** | `Usage_lag96` (24h ago), `Usage_lag672` (7 days ago) | Compares usage to the exact same time yesterday and last week |
| **Rolling Stats** | `Usage_diff_1`, `Roll_mean_4` (1h), `Roll_mean_12` (3h), `Roll_mean_24` (6h), `Roll_max_12`, `Roll_min_12` | Short and medium-term consumption bounds and volatility |

> **Note on 7-Day Lag**: Because `Usage_lag672` needs 7 days of history, the first 672 rows (Jan 1–7) are dropped as a burn-in window.

---

## Validation Strategy

We use a **strict chronological split** (no shuffling).

Shuffling time-series data leaks nearby 15-minute readings into both train and test sets, giving fake accuracy numbers. Testing on a completely separate, future month is the only honest way to evaluate:

- **Train Set (Jan 8 – Nov 30)**: 31,392 rows
- **Held-Out Test Set (Dec 1 – Dec 31)**: 2,976 rows (untouched during training)

---

## Models

1. **Baseline**: `LogisticRegression` (standardized features, balanced class weights).
2. **Main Model**: Soft-voting ensemble combining **LightGBM** and **XGBoost** ($0.5 \times P_{\text{LGBM}} + 0.5 \times P_{\text{XGB}}$).

Custom class weights (`Light: 1.0, Maximum: 1.6, Medium: 1.4`) were used to ensure the model pays equal attention to the smaller `Maximum_Load` (~21%) and `Medium_Load` (~28%) classes.

---

## Results on December Test Set (2,976 Records)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Logistic Regression (Baseline)** | 74.50% | 0.589 | 0.730 |
| **LightGBM + XGBoost Ensemble** | **99.56%** | **0.995** | **0.996** |

### Classification Report (Ensemble):

```text
              precision    recall  f1-score   support

  Light_Load       1.00      1.00      1.00      1744
Maximum_Load       0.99      1.00      1.00       528
 Medium_Load       1.00      0.99      0.99       704

    accuracy                           1.00      2976
   macro avg       1.00      1.00      1.00      2976
weighted avg       1.00      1.00      1.00      2976
```

Out of 2,976 December test intervals, the ensemble made **only 13 misclassifications**.

---

## Key Takeaways

1. **Why `winter_schedule` matters**: Without it, models confuse Medium and Maximum load in winter because peak hours switch. Giving the model the winter tariff rule resolved almost all of those errors.
2. **Why 24h and 7d lags matter**: Industrial plants follow strict weekly schedules. Knowing what the plant was doing at the exact same time last Tuesday provides a massive signal for today.
3. **No data leakage**: Even with 99.56% accuracy, there is zero lookahead. The model was trained entirely on January through November and tested on an unseen December.

---

## Limitations

- **First week lost**: Using a 7-day lag means the first week of January cannot be predicted without prior history.
- **Single year of data**: Only 2018 data was available. Testing across multiple years would confirm if the schedule holds across changing annual tariffs.
- **Decimal-based cleaning**: Treating >2 decimals as corrupted readings is based on data inspection, not documented hardware specs.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the notebook:
   ```bash
   jupyter notebook load_type_classification_final2.ipynb
   ```