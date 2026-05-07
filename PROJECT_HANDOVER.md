# Project Handover Report: PhotometricAI (LuxAeterna)

**Date:** May 6, 2026  
**Status:** Ensemble Architecture Finalized & Production Ready  

---

## 1. Executive Summary
The LuxAeterna project's goal is to predict visually stunning natural lighting events (like dramatic sunrises, intense twilight diffusion, and golden hours) for photographers. 

Initially, the project relied heavily on computer vision (CNN-LSTM networks processing live webcam imagery). However, live image processing is highly restrictive for scalable mobile/backend consumer applications. **The breakthrough of this project** was refactoring the pipeline to use the expensive image-based ALQS (Aesthetic Lighting Quality Score) models strictly as *offline teachers* to generate labels, and then training lightweight tabular/sequential *student* models to predict those labels using nothing but highly-available public weather API data.

---

## 2. The Data Pipeline
The data ingestion and engineering phases have been robustly automated:
1.  **Ingestion (`data/global_ingestion.py`):** Scrapes geographic webcam imagery alongside synchronous weather API metadata over 72-hour windows.
2.  **ALQS Labelling:** Imagery is scored for aesthetic quality using contrast, brightness, and color variance algorithms.
3.  **Solar Geometry (`feature_engineer.py`):** PyEphem calculates exact global positional data (`solar_elevation`, `azimuth`) based on coordinates and timestamps.
4.  **Target Generation (`xgboost_feature_engineer.py`):** 
    *   Creates a `forecast_horizon` offset (currently highly optimized for +3 hours).
    *   Constructs 4 distinct Target Classes (No Event, Golden Hour, Diffusion, Both) representing differing combinations of ALQS severity and Solar angle (-6 to +6 degrees).
    *   Computes time-series derivatives (Lags and Deltas) to capture weather momentum.

---

## 3. The Predictive Architecture: Power-Weighted Ensemble
Instead of relying on a single algorithm, the final system features a multi-modal ensemble. This mitigates the massive class imbalance of weather data (where 80% of days are uninteresting).

### Model 1: Gradient Boosted Trees (XGBoost)
*   **Role:** Precision anchor. Prevents false positives.
*   **Weight:** 50%
*   **Why it works:** Decision trees are phenomenal at setting strict logical boundaries (e.g., *if solar_elevation > 6 AND cloud_cover == 100 -> No Golden Hour*). It achieved ~70% overall accuracy but struggled slightly to catch the rarest diffusion events (lower recall).

### Model 2: Temporal Sequence Network (LSTM)
*   **Role:** Recall optimizer and momentum tracker.
*   **Weight:** 35%
*   **Why it works:** By feeding a 6-hour sliding window `(6 timesteps x 10 base features)` into a Bidirectional LSTM, the network learned to track trajectories (e.g., dropping temps colliding with incoming low-level clouds). It achieved a staggering **68% recall** on the rarest target class (Golden Hour + Diffusion), saving the XGBoost from missing events.

### Model 3: Deep Tabular Network (MLP)
*   **Role:** Probability Regularizer.
*   **Weight:** 15%
*   **Why it works:** A 3-layer dense network with Swish activations and Batch Normalization. It smooths the stark decision boundaries of the XGBoost model, ensuring edge-case weather patterns output moderate confidences rather than binary 0s or 1s.

### Final Ensemble Capabilities (3-Hour Forecast)
*   **Overall Structural Accuracy:** ~68%
*   **Event Generation Recall:** Catches > 50-60% of all distinct lighting phenomenons 3 hours before they happen.
*   **Execution Runtime:** < 50 milliseconds CPU inference (ideal for serverless functions or mobile client edge deployment).

---

## 4. Deliverables & File Structure
All functional code and artifacts generated are preserved in the repository securely without overwriting previous benchmarks:
*   `FRONTEND_INTEGRATION_GUIDE.md` (Crucial for the web/app dev team)
*   `models/ensemble_classifier.py` (The final orchestrated prediction script)
*   `models/lstm_classifier.py` (The sequence-window LSTM model)
*   `models/mlp_classifier.py` (The swish-activated dense tabular network)
*   `models/xgb_predictor.py` (The baseline tree classifier)
*   `models/artifacts/*` (Contains `.json`, `.keras`, and `.joblib` production weights)
*   `data/xgboost_feature_engineer.py` (Crucial tabular lag-generator logic)

---

## 5. Future Roadmap for the Handover Team
1.  **Continuous Learning Setup:** As the app gains users, prompt them to "verify" if the lighting was good. Use these real human ground-truth metrics to periodically re-train the Ensemble (fine-tuning the keras models on the new classification target vectors).
2.  **API Deployment:** Wrap `models/ensemble_classifier.py` inside a fast FastAPI or AWS Lambda container. The endpoints just need to accept a JSON array of regional hourly weather reports. 
3.  **Hyper-Local Models:** The coordinates (`latitude`, `longitude`) are already included as features. Eventually, if a specific region (like mountains vs. oceans) behaves differently, you can train localized XGBoost forks that branch off the main prediction.