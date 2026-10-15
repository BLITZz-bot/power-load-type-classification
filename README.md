# Load Type Classification

## Objective

The objective of this project is to classify industrial power-system load into three categories:

- Light_Load
- Medium_Load
- Maximum_Load

The classification is based on historical power-consumption patterns and related electrical measurements.

## Dataset and Preprocessing

The dataset contains 35,041 records from January to December 2018, recorded at 15-minute intervals.

Preprocessing steps included:

- Cleaning column names and parsing `Date_Time` as a datetime field.
- Sorting observations chronologically.
- Creating calendar features: month, hour, minute, day of week, and weekend indicator.
- Encoding `Load_Type` using `LabelEncoder`.
- Applying physical sanity bounds to power-factor and NSM fields.
- Creating lag-based features after sorting data by time.

> Note: Forward filling is preferred for time-series missing values because backward filling can introduce information from future observations.

## Exploratory Data Analysis

EDA was performed to understand:

- Distribution of load-type classes.
- Average energy usage by hour and load type.
- Weekday versus weekend load distributions.
- Correlation among numeric variables.

The dataset is moderately imbalanced, with `Light_Load` being the largest class.

## Feature Engineering

The following features were engineered:

- Cyclical time features: `NSM_sin` and `NSM_cos`.
- Calendar features: hour, day of week, and weekend indicator.
- Historical usage lags: 15 minutes, 30 minutes, 1 hour, and 3 hours.
- Usage momentum and rolling statistics.
- Lag-based apparent-power and reactive-to-active-power ratio features.
- Historical expected-usage profile and deviation from that profile.

Current `Usage_kWh` and `CO2(tCO2)` were excluded from the final feature set to reduce the risk of target leakage, since they may closely reflect the current load category.

## Validation Strategy

A chronological split was used to simulate real-world forecasting:

| Dataset | Period | Purpose |
|---|---|---|
| Training | January–October 2018 | Model training |
| Validation | November 2018 | Ensemble calibration and tuning |
| Test | December 2018 | Final unseen evaluation |

The December data was kept as the final blind test set and was not used for selecting calibration weights.

## Models

Two approaches were evaluated:

1. Logistic Regression as a baseline model.
2. A soft-voting ensemble of LightGBM and XGBoost classifiers.

Class weights were used to improve learning for the less frequent `Maximum_Load` and `Medium_Load` classes.

## Results

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 66.11% | 0.553 |
| LightGBM + XGBoost Ensemble | 78.07% | 0.669 |

The ensemble outperformed the baseline by approximately 12 percentage points in test accuracy.

The model classified `Light_Load` very effectively. However, `Maximum_Load` remained comparatively difficult to classify, indicating overlap between the electrical patterns of medium and maximum load conditions.

## Conclusion

The LightGBM and XGBoost ensemble achieved the best performance, with 78.07% accuracy on the December blind test set. Time-based validation, lag features, cyclical time encoding, and ensemble modelling contributed to the improvement.

Future work could include additional hyperparameter tuning, more robust historical-profile construction using strictly past observations, and further methods to improve recognition of `Maximum_Load`.