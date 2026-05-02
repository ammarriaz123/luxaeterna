# LuxAeterna Extended Pipeline: Quick Start Guide

**Status:** ✅ All improvements implemented and ready to deploy

---

## TL;DR - 30 Second Summary

The original pipeline failed because it collected only **13 hours per webcam** with insufficient temporal context. This update:

| Improvement | What's New | Impact |
|-------------|-----------|--------|
| **Duration** | 48–72 hours (vs 13h) | 3–5× more training data |
| **Sampling** | 1-hour intervals (vs 30-min) | Larger diurnal window |
| **Features** | Solar geometry (elevation/azimuth/clear-sky) | Captures direct light drivers |
| **Sequences** | 24-hour context (vs 6-hour) | Full day/night cycle visibility |
| **Benchmarking** | 7 baseline models | Rigorous performance validation |

**Expected Result:** LSTM should now capture meaningful temporal patterns or definitively show that baseline methods are superior.

---

## Implementation Summary

### ✅ Code Changes

#### 1. Ingestion Interval (30 min → 1 hour)
```bash
# File: data/global_ingestion.py
# Changed:
#   --interval: 1800s → 3600s
#   --max-cycles: None → 72

# New defaults:
#   Collection: 72 hours
#   Sampling: 1 hour apart
#   Expected observations: 72 per webcam (vs 26 before)
```

#### 2. Solar Geometry Features
```bash
# File: data/feature_engineer.py
# Added:
#   - solar_elevation (sun angle above horizon)
#   - solar_azimuth (sun compass direction)
#   - clear_sky_index (normalized elevation)

# Feature count: 9 → 12 (33% increase)
# All features scale-normalized independently
```

#### 3. Longer Sequences (12 → 24 timesteps)
```bash
# File: data/feature_engineer.py
# Changed:
#   --window-size: 12 → 24

# Window context:
#   12 timesteps @ 30min = 6 hours
#   24 timesteps @ 1hour = 24 hours (FULL DAY)
```

#### 4. Baseline Benchmarking
```bash
# File: models/baseline_comparison.py (NEW)
# Baselines implemented:
#   1. Persistence (y_t = y_{t-1})
#   2. Seasonal Naive (y_t ≈ y_{t-24})
#   3. Linear Regression
#   4. Random Forest (100 trees)
#   5. XGBoost (optional)
#   6. Exponential Smoothing (optional)
#   7. ARIMA(1,1,1) (optional)

# Output: JSON + console summary
# Success metric: LSTM MAE < best baseline MAE
```

---

## How to Run

### Option 1: Full Pipeline (Recommended First Time)

```bash
cd c:\Users\Dell\Desktop\projects\LuxAeterna
.venv\Scripts\activate

# Run everything end-to-end
python run_extended_pipeline.py --phase all --max-hours 72
```

**Timeline:**
- Discovery: 5 min
- **Ingest: 72 hours** ⏱️ (runs in real-time)
- Features: 5–10 min
- Train: 10–20 min
- Baselines: 5–15 min
- Report: < 1 min
- **Total: ~72 hours wall-clock**

### Option 2: Quick Test (12 Hours)

```bash
# For development/testing
python run_extended_pipeline.py --phase all --max-hours 12
# Total: ~12 hours wall-clock
```

### Option 3: Run Phases Individually

```bash
# Phase 1: Discover webcams (5 min)
python data/webcam_discovery.py --sample-size 1000

# Phase 2: Collect data (1 hour to 72 hours - real-time)
python data/global_ingestion.py --interval 3600 --max-cycles 72

# Phase 3: Engineer features (5–10 min)
python data/feature_engineer.py --window-size 24

# Phase 4: Train model (10–20 min)
python models/lstm_predictor.py --data-path data/processed/sequence_dataset.npz

# Phase 5: Benchmark baselines (5–15 min)
python models/baseline_comparison.py --data-path data/processed/sequence_dataset.npz

# Phase 6: View results
cat models/artifacts/baseline_comparison_results.json
```

---

## Expected Results

### Before (Original Pipeline)
```
Test MAE:       7.88 ALQS points
Baseline MAE:   3.21 (persistence)
Comparison:     LSTM is 145% WORSE than baseline
Verdict:        ❌ USELESS - model learns noise, not patterns
```

### After (Extended Pipeline)
```
Best Case Scenario:
  Test MAE:       4–6 ALQS points
  Best Baseline:  3–4 (likely persistence still)
  Comparison:     LSTM within 20% of baseline (or better)
  Verdict:        ✓ USEFUL - temporal patterns captured

Worst Case Scenario:
  Test MAE:       7–9 ALQS points
  Best Baseline:  3–4 (persistence dominates)
  Comparison:     LSTM still worse than baseline
  Verdict:        ✗ STILL FAILS - problem not temporal
  → Recommend: Use persistence baseline operationally
```

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `run_extended_pipeline.py` | Main orchestration script | NEW ✅ |
| `data/global_ingestion.py` | Extended 1-hour, 72-hour collection | MODIFIED ✅ |
| `data/feature_engineer.py` | Solar features + 24-step windows | MODIFIED ✅ |
| `models/baseline_comparison.py` | 7-model benchmark suite | NEW ✅ |
| `IMPROVEMENTS_GUIDE.md` | Detailed technical documentation | NEW ✅ |
| `PIPELINE_REPORT.md` | Original diagnosis report | EXISTING ✅ |

---

## Monitoring During Ingestion

Since ingestion takes 48–72 hours, monitor progress:

```bash
# In a separate terminal, watch data accumulation
watch -n 60 'ls -lah data/processed/global_dataset/*.parquet | tail -5'

# Or check row count
python -c "
import pandas as pd
from pathlib import Path
import glob
parquet_files = glob.glob('data/processed/global_dataset/*.parquet')
total_rows = sum(len(pd.read_parquet(f)) for f in parquet_files)
print(f'Total rows collected: {total_rows}')
print(f'Expected for 72h: ~72,000')
print(f'Progress: {total_rows/72000*100:.1f}%')
"
```

---

## Troubleshooting

### Problem: "No parquet files found"
**Cause:** Ingestion hasn't started or is still in phase 1  
**Solution:** Wait for ingestion phase to complete

### Problem: "Solar geometry computation failed"
**Cause:** Missing PyEphem or invalid coordinates  
**Solution:** Ensure PyEphem installed (`pip install pyephem`)

### Problem: "ARIMA/XGBoost not found"
**Cause:** Optional dependencies not installed  
**Solution:** Install with `pip install statsmodels xgboost` (optional)

### Problem: "Out of memory during baseline training"
**Cause:** RandomForest/XGBoost using too much RAM with large datasets  
**Solution:** Reduce max observations or use smaller train set

---

## Success Criteria

### ✅ Extended Collection Success
- [ ] 48+ hours of data collected per webcam
- [ ] ~48–72 observations per webcam
- [ ] All 1,000 webcams represented
- [ ] No gaps > 2 hours (except overnight if collection stops)

### ✅ Feature Engineering Success
- [ ] 14,233 → 50,000+ sequences generated
- [ ] Solar features computed without errors
- [ ] Window size = 24 timesteps
- [ ] Feature metadata saved with 12 features

### ✅ Model Training Success
- [ ] LSTM trains without OOM errors
- [ ] Early stopping triggers after 10–30 epochs
- [ ] Test MAE: 4–9 ALQS points
- [ ] Model saved to `.keras` format

### ✅ Baseline Success
- [ ] All 7 baselines computed
- [ ] Results JSON contains MAE, MSE, improvement%
- [ ] Console output shows ranked models
- [ ] Best baseline identified

### ✅ Report Generated
- [ ] `EXTENDED_PIPELINE_REPORT.md` created
- [ ] Shows LSTM vs all baselines
- [ ] Includes recommendations

---

## Decision Tree

After running extended pipeline, check results:

```
LSTM MAE < Best Baseline MAE?
│
├─ YES (LSTM wins)
│  ├─ Improvement > 20%?
│  │  ├─ YES → 🎉 SUCCESS! Model is useful
│  │  └─ NO  → ⚠️ Marginal improvement; consider production trade-offs
│  └─ → Deploy LSTM with validation monitoring
│
└─ NO (Baseline wins)
   ├─ LSTM MAE - Baseline MAE < 1?
   │  └─ YES → 🤔 Close call; use baseline (simpler, faster)
   └─ NO  → ❌ LSTM fails; use best baseline operationally
      └─ → Recommend: Persistence baseline (MAE ~3.2)
```

---

## Next Steps After Results

### If LSTM Wins ✓
1. **Deploy:** Move LSTM to production with monitoring
2. **Optimize:** Fine-tune hyperparameters on validation set
3. **Ensemble:** Try LSTM + best baseline weighted average
4. **Monitor:** Track real-time predictions vs actuals

### If Baseline Wins ✗
1. **Admit defeat:** Temporal ML won't help for this problem
2. **Go operational:** Use persistence baseline (simplest + best)
3. **Research:** Investigate root cause
   - Is ALQS truly just autocorrelated?
   - Are weather features proxies, not drivers?
   - Is label noise overwhelming signal?
4. **Redesign:** Consider alternative approaches
   - Longer collection (1–2 weeks)
   - Different features (aerosol, water vapor)
   - Image-based features (CNN encoder)
   - Hybrid models (LSTM + CNN)

---

## Performance Notes

### Ingestion
- ~1000 concurrent HTTP requests over 72 hours
- Network bandwidth: ~100–500 MB for images + weather
- Storage: ~10–20 GB for raw data + processed dataset

### Feature Engineering
- Solar computation: ~0.1–0.5 sec per observation (depends on ephem overhead)
- Total time: 30–60 min for 72,000 observations
- Memory: ~2–4 GB during processing

### Training
- LSTM: 10–30 epochs to convergence
- Time: 10–20 min on CPU; 2–5 min on GPU
- Memory: ~2 GB for 50,000+ sequences

### Baselines
- Persistence: < 1 sec (trivial)
- Linear/RF/XGB: 1–10 min each (depends on dataset size)
- ARIMA/ETS: 5–30 min (statsmodels slower)

---

## Contact & Support

For issues or questions, refer to:
1. **IMPROVEMENTS_GUIDE.md** — Detailed technical docs
2. **PIPELINE_REPORT.md** — Original failure analysis
3. **Code comments** — Inline documentation
4. **Logs** — Check console output for phase-by-phase progress

---

**Last Updated:** May 2, 2026  
**Status:** Ready for Production  
**Recommendation:** Start with `--max-hours 12` for validation, then deploy with `--max-hours 72`
