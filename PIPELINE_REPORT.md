# LuxAeterna: Global Webcam Light Quality Prediction
## Data Ingestion, Preprocessing, Feature Engineering, Training & Results Report

**Date:** May 2, 2026

---

# 🔴 EXECUTIVE SUMMARY: FAILURE

| Aspect | Status | Details |
|--------|--------|---------|
| **Dataset Construction** | ✅ Success | 1,000 webcams, 14,233 sequences, 13-hour collection window |
| **Model Training** | ✅ Completed | LSTM trained, converged, checkpoints saved |
| **MAE Target (< 12)** | ✅ PASSED | Test MAE = 7.88 ALQS points |
| **Baseline Comparison** | ❌ FAILED | -145% worse than persistence baseline |
| **Signal Detection** | ❌ NONE | No meaningful temporal patterns learned |
| **Model Quality** | ❌ USELESS | Persistence model (MAE 3.21) vastly superior |

---

# 1. DATA INGESTION

## 1.1 Webcam Discovery

- **Provider:** Windy v3 API
- **Target:** 1,000 unique global webcams
- **Method:** Grid-based geosearch (24 seed locations spanning 30°--60° latitude/longitude bands)
- **Deduplication:** Tracked seen IDs across seeds to avoid redundancy
- **Result:** Successfully cached exactly 1,000 unique webcams in `data/webcams.json`

## 1.2 Collection Campaign

| Parameter | Value |
|-----------|-------|
| Duration per Webcam | ~13 hours |
| Timesteps per Webcam | ~26 observations |
| Intended Interval | 30 minutes |
| Actual Intervals | 15--45 minutes (irregular) |
| Total Observations (raw) | 26,233 rows |
| Ingestion Mode | Async batch (50 webcams/batch, 8 concurrent requests) |
| Output Format | Parquet (~461 files in `data/processed/global_dataset/`) |
| Weather Source | Open-Meteo (archive + forecast blend) |

## 1.3 Data Fields Captured

| Field | Type | Source |
|-------|------|--------|
| `timestamp` | datetime64[ns] | System UTC clock (image capture time) |
| `webcam_id` | string | Windy API metadata |
| `latitude`, `longitude` | float | Webcam geographic coordinates |
| `image_path` | string | Local JPEG file path |
| `alqs` | float [0--100] | OpenCV photometry pipeline |
| `temperature` | float | Open-Meteo (°C) |
| `relative_humidity` | float | Open-Meteo (%) |
| `visibility` | float | Open-Meteo (meters) |
| `cloud_cover_low`, `mid`, `high` | float | Open-Meteo (0--100% per layer) |
| `weather_code` | int | WMO weather code (Open-Meteo) |

---

# 2. DATA PREPROCESSING & CLEANING

## 2.1 Cleaning Pipeline

| Step | Action | Result |
|------|--------|--------|
| 1. **Timestamp Normalization** | Convert all timestamps to UTC datetime64 | ✅ Standardized |
| 2. **Missing Value Handling** | Dropped rows with missing `{alqs, temperature, humidity, cloud_cover_low}`; imputed `visibility` with median (10,000m) | ✅ 26,233 → 26,233 rows (0 dropped) |
| 3. **Coordinate Validation** | Enforced $-90 \leq \text{lat} \leq 90$, $-180 \leq \text{lon} \leq 180$ | ✅ All valid |
| 4. **Deduplication** | Removed duplicate (webcam_id, timestamp) pairs; kept latest | ✅ No duplicates found |
| 5. **Image File Validation** | Checked JPEG files exist on disk | ✅ All present |

**Final Cleaned Dataset:** 26,233 rows, zero rows discarded (excellent data quality)

## 2.2 Derived Features

1. **`total_cloud_cover`**: Sum of cloud_cover_low + cloud_cover_mid + cloud_cover_high
   - Range: [0, 300]
   
2. **`delta_time_minutes`**: Elapsed time (in minutes) since previous observation within same webcam
   - First timestep per webcam: 0 minutes
   - Median per webcam: ~30 minutes
   - Captures irregular sampling intervals explicitly for LSTM

---

# 3. FEATURE ENGINEERING & SEQUENCE CONSTRUCTION

## 3.1 Per-Webcam Grouping & Temporal Sorting

**Critical Design Decision:** Strict per-webcam grouping to avoid temporal leakage.

- Grouped 26,233 rows into 1,000 independent time-series (one per webcam)
- Within each group: sorted by timestamp ascending
- Result: Each webcam's sequence preserves natural temporal order

## 3.2 Sliding Window Approach

### Window Specification
- **Window Length:** 12 timesteps (~6 hours at 30-min intervals)
- **Input (X):** Previous 12 observations
- **Target (y):** ALQS at timestep t+1 (one-step-ahead prediction)
- **Stride:** 1 (maximum overlap for data utilization)
- **Minimum Requirement:** ≥13 timesteps per webcam to create at least 1 sequence

### Feature Vector (per timestep)
```
[temperature, relative_humidity, cloud_cover_low, cloud_cover_mid, 
 cloud_cover_high, weather_code, visibility, total_cloud_cover, 
 delta_time_minutes]
```
**Total:** 9 features × 12 timesteps = shape (12, 9)

## 3.3 Sequence Construction Results

| Metric | Count |
|--------|-------|
| Total Sequences Built | 14,233 |
| Unique Webcams Contributing | 1,000 |
| Sequences per Webcam (avg) | 14.2 |
| Webcams Skipped (< 13 timesteps) | 0 |
| Webcams Skipped (missing features) | 0 |

## 3.4 Feature Normalization

- **Method:** MinMaxScaler (scikit-learn)
- **Fit Data:** All 14,233 sequences
- **Range:** [0, 1]
- **Saved Scaler:** `data/processed/feature_artifacts.joblib`
- **Target (ALQS):** Kept in original [0--100] scale

---

# 4. TRAIN / VALIDATION / TEST SPLIT

## 4.1 Split Strategy: By Webcam ID (Not Random Rows)

**Rationale:** Prevent temporal leakage. All sequences from a given webcam stay in one partition.

- **Train Webcams:** 700 / 1,000 (70.0%)
- **Validation Webcams:** 150 / 1,000 (15.0%)
- **Test Webcams:** 150 / 1,000 (15.0%)

## 4.2 Split Distribution

| Partition | Webcams | Sequences | % of Total | Avg Seq/Webcam |
|-----------|---------|-----------|------------|-----------------|
| Train | 700 | 9,968 | 70.0% | 14.2 |
| Validation | 150 | 2,137 | 15.0% | 14.2 |
| Test | 150 | 2,128 | 15.0% | 14.2 |
| **TOTAL** | **1,000** | **14,233** | **100%** | **14.2** |

---

# 5. MODEL ARCHITECTURE & TRAINING

## 5.1 LSTM Architecture

```
Input Layer:           (batch_size, 12 timesteps, 9 features)
                       ↓
LSTM Layer 1:          128 units
                       Dropout: 0.2
                       return_sequences=True
                       ↓
LSTM Layer 2:          64 units
                       Dropout: 0.2
                       return_sequences=False
                       ↓
Dense Layer:           32 units, ReLU activation
                       ↓
Output Layer:          1 unit, Linear (regression)
                       ↓
Output:                (batch_size, 1) → predicted ALQS
```

## 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Metrics | Mean Absolute Error (MAE) |
| Batch Size | 64 |
| Max Epochs | 120 |
| Early Stopping Patience | 10 epochs |
| ReduceLROnPlateau Patience | 5 epochs |
| LR Decay Factor | 0.5 |

## 5.3 Training Dynamics

### Learning Rate Schedule

| Epoch Range | Learning Rate | Trigger |
|-------------|---------------|---------|
| 1--20 | 0.001 | Initial |
| 21--27 | 0.0005 | ReduceLROnPlateau at epoch 21 |
| 28--32 | 0.00025 | ReduceLROnPlateau at epoch 28 |
| 33 | 0.000125 | ReduceLROnPlateau at epoch 33 |

### Training Progression

| Epoch | Loss | MAE | Val Loss | Val MAE | Notes |
|-------|------|-----|----------|---------|-------|
| 1 | 152.44 | 9.43 | 93.99 | 7.67 | Initialization |
| 2 | 99.80 | 7.80 | 93.99 | 7.70 | Sharp drop to plateau |
| 5 | 99.96 | 7.81 | 94.14 | 7.66 | Plateauing |
| 10 | 99.85 | 7.80 | 94.90 | 7.77 | Plateau continues |
| 15 | 99.91 | 7.81 | 94.06 | 7.70 | Minimal change |
| 20 | 100.09 | 7.81 | 93.94 | 7.69 | Still plateau |
| **23** | **98.70** | **7.77** | **93.40** | **7.65** | **BEST (restored)** |
| 28 | 98.26 | 7.75 | 94.68 | 7.72 | LR reduced, no improvement |
| 33 | 97.79 | 7.75 | 94.08 | 7.68 | **Early stop triggered** |

### Key Observations
- ✅ Rapid convergence: loss drops sharply epoch 1→2
- ⚠️ Validation plateau: no meaningful improvement after epoch 2 (val_loss ≈ 93.4--94.0)
- ⚠️ No overfitting: train loss (≈99) ≈ val loss (≈94) throughout
- ⚠️ LR decay ineffective: reduced LR multiple times without recovery
- 🛑 Early stopping: triggered at epoch 33 (no improvement for 10 epochs); best weights restored from epoch 23

---

# 6. RESULTS & EVALUATION

## 6.1 Test Set Performance

### Quantitative Metrics

| Metric | LSTM Model | Persistence Baseline | Delta |
|--------|-----------|----------------------|-------|
| **MAE** | **7.88** | **3.21** | +4.67 (LSTM worse) |
| **MSE** | 99.98 | 33.80 | +66.18 (LSTM worse) |
| **RMSE** | 9.999 | 5.82 | +4.18 (LSTM worse) |
| **Residual Std** | 9.999 | — | — |

### Improvement Calculation

```
Improvement (%) = (Baseline MAE - LSTM MAE) / Baseline MAE × 100
                = (3.21 - 7.88) / 3.21 × 100
                = -4.67 / 3.21 × 100
                = -145.21%
```

**Interpretation:** Model is **145.21% worse** than the persistence baseline.

## 6.2 Target Assessment

| Target | Threshold | Result | Status |
|--------|-----------|--------|--------|
| **MAE < 12** | 12.0 | 7.88 | ✅ PASS |
| **Improvement > 20%** | 20.0% | -145.21% | ❌ FAIL |

---

# 7. ANALYSIS: WHY DID THE MODEL FAIL?

## 7.1 Strong Autocorrelation Dominates

**The Critical Issue:**

The persistence baseline (predict $\text{ALQS}_{t+1} = \text{ALQS}_t$) achieves MAE = 3.21. This demonstrates that ALQS values are **highly autocorrelated** at 30-minute intervals.

**Implication:** Light quality doesn't change dramatically within 30 minutes. The previous observation is nearly as good as any prediction. The LSTM cannot exploit weather features to beat this strong prior signal.

## 7.2 Insufficient Temporal Window

- **Current Data:** 26 observations per webcam over 13 hours
- **Observation Density:** ~1 per 30 minutes
- **Window Size:** 12 timesteps ≈ 6 hours of history
- **Problem:** 6-hour window is too short to capture meaningful photometric cycles (dawn/dusk/night transitions occur over 12+ hours)

## 7.3 Weak Weather-ALQS Coupling

The 9 weather features show **insufficient predictive power** for ALQS:

- ❌ **Missing Solar Geometry:** Sun altitude/azimuth (time of day, latitude, season) — **strong ALQS driver** not captured
- ❌ **Missing Aerosol Data:** Aerosol Optical Depth (AOD), atmospheric water vapor — **major light attenuation factors** not available
- ❌ **Missing Image Quality Metrics:** Image brightness, contrast, exposure — **direct photometry indicators** unavailable
- ❌ **Missing Scene Context:** Urban/rural/altitude — **persistent scene properties** not included

Current weather proxies alone are **insufficient** to predict light quality.

## 7.4 Limited Training Data Regime

- **Training Sequences:** 9,968
- **Training Timesteps:** 9,968 × 12 = 119,616 total observations
- **Model Complexity:** 3 LSTM layers + dropout + 128/64 units
- **Problem:** For complex LSTM architectures, 119k observations is **sparse** when temporal patterns are weak. Model struggles to learn reliable weights.

## 7.5 Label Noise (Possible)

If ALQS computation (OpenCV-based photometry) contains **systematic bias or measurement noise**:
- Multiple noisy observations in a 12-step window may **amplify noise** rather than average it out
- Persistence (using one recent value) might have lower variance
- LSTM tries to learn patterns that don't exist in the noisy labels

---

# 7.6 Data Quality Assessment

| Aspect | Grade | Evidence |
|--------|-------|----------|
| **Completeness** | A+ | Zero rows dropped after cleaning |
| **Consistency** | A | All coordinates valid, timestamps normalized cleanly |
| **Representativeness** | B | 1,000 global webcams; however, only 13-hour window per location limits observability |
| **Signal Strength** | **F** | **WEAK SIGNAL.** Autocorrelation dominates; exogenous weather features contribute minimally. |

---

# 8. MODEL QUALITY ASSESSMENT

## 8.1 FINAL VERDICT: 🔴 **USELESS**

The LSTM model is **not suitable for production deployment.**

## 8.2 Evidence for Failure

1. **Worse Than Baseline:** 145% worse than trivial persistence model
   - Any practitioner would use the baseline instead
   
2. **No Learned Structure:** The LSTM failed to extract meaningful temporal patterns
   - Network is learning to regress toward the mean of the input window
   - Does not beat a "predict last value" strategy
   
3. **Residuals Too Large:** Std = 9.999 ALQS points (≈10% of 0--100 scale)
   - Predictions scatter widely; predictions are unreliable
   
4. **Impractical for Operations:** Deploying this model would degrade system accuracy
   - Using the prior observation is simpler, faster, and more accurate

## 8.3 Root Cause

**The data collection window (13 hours per webcam) is too short to reveal meaningful temporal dynamics.**

ALQS is primarily driven by:
- **Diurnal cycles** (dawn → day → dusk → night) — requires 24+ hours
- **Weather transitions** (clear → cloudy → clear) — requires 48+ hours per region
- **Seasonal patterns** — requires months of data

With only 13 hours per location, the model sees only a slice of the cycle and cannot learn generalizable patterns.

---

# 9. RECOMMENDATIONS FOR FUTURE WORK

## 9.1 Data Collection: Extend Window

- **Current:** 13 hours per webcam
- **Recommended:** 48--72 hours per webcam (minimum 96--144 observations)
- **Benefit:** Captures full day/night cycle + weather transitions
- **Impact:** Should generate 100--200+ sequences per webcam vs. current 14

## 9.2 Feature Engineering: Add Solar Geometry

- Compute **solar elevation**, **azimuth**, **clear-sky index** using PyEphem
- These are **strong ALQS drivers** missing from current feature set
- Solar features capture **diurnal patterns** without extending data collection

## 9.3 Sequence Design: Longer Context

- Increase window size from 12 to 24--48 timesteps
- Captures 12--24 hours of history (full diurnal cycle)
- Better alignment with photometric timescales

## 9.4 Baseline Comparison: Multi-Method

Before claiming LSTM superiority, compare against:
- Seasonal naive (predict same hour last day)
- Exponential smoothing (ETS)
- ARIMA
- Random forest / XGBoost
- Simple linear model on weather features

## 9.5 Model Architecture: Advanced Methods

- **Attention Mechanisms:** Learn which timesteps/features matter
- **Transformer Models:** Better capture long-range dependencies
- **Hybrid Approach:** CNN for image features + weather features combined

## 9.6 Causality: Proper Lag Structure

- Current: weather_t predicts ALQS_t+1
- Better: weather_t predicts ALQS_{t+1} to {t+6} (6-30 min lag)
- Accounts for weather → photometry propagation delay

---

# 10. CONCLUSION

## 10.1 What Succeeded

✅ **Data Pipeline:**
- Successfully discovered 1,000 global webcams
- Ingested 26,233 observations across 13-hour window
- Built 14,233 irregular time-series sequences

✅ **Preprocessing & Feature Engineering:**
- Cleaned data rigorously (zero rows discarded)
- Properly grouped by webcam ID
- Handled irregular 30-min sampling intervals
- Normalized features appropriately

✅ **Model & Training:**
- LSTM architecture built and trained end-to-end
- Training converged without errors
- Model saved and artifacts persisted

✅ **Evaluation:**
- Rigorous test set evaluation
- Baseline comparison performed
- Achieved stated MAE target (7.88 < 12)

## 10.2 What Failed

❌ **Model Utility:**
- LSTM is **145% worse** than persistence baseline
- Does NOT capture meaningful temporal patterns
- **Unsuitable for production**

❌ **Signal Detection:**
- **Weak signal** in 13-hour per-webcam window
- ALQS autocorrelation dominates; weather features insufficient
- No generalizability across webcams

❌ **Success Criteria:**
- ❌ Improvement target (-145% vs. target +20%)
- ✅ MAE target achieved (7.88 < 12) — **misleading success**

## 10.3 Overall Assessment

**Status: 🔴 FAILURE**

While the model technically achieves the MAE target, it is **not useful for prediction.** The persistence baseline (simply predicting the last observed ALQS value) is dramatically superior.

**Path Forward:**
1. Extend data collection to 48--72 hours per webcam
2. Add solar geometry features (elevation, azimuth, clear-sky index)
3. Increase sequence window to 24--48 timesteps
4. Benchmark against multiple baseline methods
5. Consider hybrid image + weather approaches

Without these changes, the LSTM approach is **not recommended** for deployment.

---

# APPENDIX: Key File Locations

```
✓ Webcam Cache:         data/webcams.json (1,000 entries)
✓ Raw Parquet Data:     data/processed/global_dataset/batch_*.parquet (~461 files)
✓ Cleaned + Grouped:    data/processed/sequence_dataset.npz
✓ Feature Metadata:     data/processed/feature_metadata.json
✓ Feature Scaler:       data/processed/feature_artifacts.joblib
✓ Trained LSTM Model:   models/artifacts/lstm_predictor.keras
✓ Training Metadata:    models/artifacts/lstm_metadata.json
✓ LaTeX Report:         PIPELINE_REPORT.tex
✓ Markdown Report:      PIPELINE_REPORT.md (this file)
```

---

**Report Generated:** May 2, 2026  
**Status:** Complete ✅
