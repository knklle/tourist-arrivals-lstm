# Philippine Tourist Arrivals Forecasting with LSTM & XAI

This project is a Streamlit-based forecasting application for Philippine tourist arrivals using Long Short-Term Memory (LSTM) and Explainable Artificial Intelligence (XAI).

It follows a complete time-series forecasting workflow, including data inspection, cleaning, feature selection, leakage-safe preprocessing, LSTM training, evaluation, SHAP explainability, and next-month forecasting.

## Features

- Dataset summary and monthly sequence checking
- Duplicate, missing-value, and outlier detection
- Spearman correlation feature selection
- Iterative Variance Inflation Factor (VIF) filtering
- Chronological train/test split
- Training-only scaling to prevent data leakage
- 12-month LSTM sequence preparation
- Hyperparameter tuning using validation data
- Evaluation using:
  - MAE
  - RMSE
  - MAPE
  - R²
- Comparison with:
  - Naive baseline
  - Seasonal-naive baseline
- SHAP-based explainability
- Editable forecasting interface using Streamlit

## Selected Predictors

The final model used 8 predictors:

- `quarter`
- `is_holiday_peak`
- `rainy_days`
- `humidity_pct`
- `typhoon_max_wind_kt`
- `storm_signal_days`
- `pm25_ugm3`
- `wave_height_m`

## Model Configuration

The best LSTM configuration obtained during tuning was:

- LSTM units: `64`
- Dropout: `0.3`
- Batch size: `16`
- Lookback window: `12 months`

## Evaluation

The trained LSTM was evaluated on a held-out test period and compared with naive and seasonal-naive forecasting methods.

The results showed that the simple naive baseline performed better than the LSTM on the test period. This is still a valid forecasting result and suggests that recent tourist arrival values were highly informative for short-term prediction in this dataset.

## Explain

SHAP was used to explain the trained LSTM model.

The most influential predictors in the sampled forecasts included:

- `is_holiday_peak`
- `wave_height_m`
- `pm25_ugm3`
- `quarter`

SHAP values were used to show both global feature importance and feature contributions for individual forecasts.

## Project Structure

```text
tourist-arrivals-app/
├── Home.py
├── requirements.txt
├── data/
│   └── tourist_arrivals.csv
├── pages/
│   ├── 1_Dataset.py
│   ├── 2_Clean.py
│   ├── 3_Features.py
│   ├── 4_Prepare.py
│   ├── 5_Train.py
│   ├── 6_Evaluate.py
│   ├── 7_Explain.py
│   └── 8_Forecast.py
