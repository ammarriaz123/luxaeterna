# Frontend & Backend Integration Guide: Lighting Prediction Ensemble

This document outlines how the client application (Frontend/Backend) should interface with the LuxAeterna forecasting models to predict exceptional lighting events. 

## 1. The Core Paradigm
The production model **does not process imagery**. It evaluates **sequential weather data** and **solar positioning** to predict the future lighting quality (3 hours ahead) without the need for live webcams.

## 2. Model Outputs (The Predictions)
The ensemble outputs a probability distribution across **4 distinct classes** (Option D):
*   **Class 0:** `No Event` (Standard lighting, flat daylight, or night/uninteresting)
*   **Class 1:** `Golden Hour Only` (Good low-angle sun, but no complex cloud diffusion)
*   **Class 2:** `Dramatic Diffusion Only` (High cloud-based ALQS, dynamic skies, daytime)
*   **Class 3:** `Golden Hour + Diffusion` (The "Holy Grail" – low angle sun + dynamic cloud cover)

**Decision Logic:** The frontend should alert the user if Class 1, 2, or 3 exceeds a certain probability threshold (e.g., > 40%).

## 3. Required Inference Inputs
To generate a prediction for a target time `T`, the client/server must fetch weather forecasts and historical states for a specific geographic location.

### A. Core Features required per timestep:
1.  `latitude` & `longitude`
2.  `temperature`
3.  `relative_humidity`
4.  `visibility`
5.  `cloud_cover_low`
6.  `cloud_cover_mid`
7.  `cloud_cover_high`
8.  `weather_code` (Standard WMO code)
9.  `solar_elevation` (Needs to be calculated using the location/timestamp via a library like `PyEphem` or `Suncalc`)

### B. Time-Series Construction:
Because we run an ensemble involving an LSTM, you must pass a **history window** to the inference engine. 
*   **Sequence Window:** To predict lighting 3 hours in the future (`T+3`), you must provide weather data for `[T-5, T-4, T-3, T-2, T-1, T]` (A 6-hour sliding window). 
*   **Tabular Lags:** For the XGBoost and MLP models, the final timestep (`T`) requires pre-calculated lag features relative to `T`:
    *   `temp_lag_1`, `temp_lag_2`, `temp_lag_3`
    *   `rh_lag_1`, `rh_lag_2`, `rh_lag_3`
    *   `cloud_low_lag_1`, `cloud_low_lag_2`, `cloud_low_lag_3`
    *   `temp_change_1h` (T.temp - T-1.temp)
    *   `cloud_low_change_1h` (T.cloud_low - T-1.cloud_low)

## 4. Inference Execution Flow
We use a **Power-Weighted Ensemble** because it offers the perfect balance of catching rare events (LSTM) while preventing false alarms (XGBoost).

Your inference server should load the pre-trained weights from `/models/artifacts/`:
1.  **XGBoost** (`xgb_multiclass_model.json`): Predicts using the 21 flattened tabular features.
2.  **MLP** (`mlp_multiclass_model.keras` & `mlp_multiclass_scaler.joblib`): Predicts using the scaled 21 flattened tabular features.
3.  **LSTM** (`lstm_multiclass_model.keras` & `lstm_multiclass_scaler.joblib`): Predicts using the scaled 3D sequence matrix `(6, 10)` base features.

**Final Probability Calculation:**
```python
final_probabilities = (XGB_Prob * 0.50) + (LSTM_Prob * 0.35) + (MLP_Prob * 0.15)
predicted_class = argmax(final_probabilities)
```

## 5. Architecture Notes for Frontend Devs
*   **Scaling:** Do not forget to apply the specific `StandardScaler` (`.joblib` files) to the data *before* passing it into the MLP and LSTM. XGBoost does not require scaling.
*   **NaN Handling:** If weather APIs return nulls (e.g., missing visibility), impute them with `0.0` or regional medians *before* scaling, using `np.nan_to_num()`.