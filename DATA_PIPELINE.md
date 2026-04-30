# LuxAeterna Data Ingestion & Dataset Formation Pipeline

## Overview

LuxAeterna ingests weather data, labels atmospheric light quality, and transforms raw observations into ML-ready sequences for time-series prediction and genre classification.

## Global Ingestion (Automated)

The refactored ingestion pipeline discovers webcams globally, samples them per run, and builds a scalable dataset without manual webcam IDs.

**Modules**:
- `data/webcam_discovery.py` caches global webcam metadata to `data/webcams.json`
- `data/global_ingestion.py` samples webcams, downloads images, pairs weather, and writes dataset batches

**Dataset Output (per batch)**:
- `timestamp`, `latitude`, `longitude`, `webcam_id`, `image_path`
- weather features: `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`, `relative_humidity`, `temperature`, `visibility`, `weather_code`
- `alqs` (placeholder hook; computed via OpenCV if enabled)

---

## Complete Data Ingestion Flow

### **Stage 1: Weather Collection** (`data/collector.py`)

**Purpose**: Fetch historical weather observations with solar geometry

**Data Source**:
- **Primary**: Open-Meteo Archive API (free, no authentication)
- **Optional**: OpenWeatherMap (requires `OPENWEATHERMAP_API_KEY` for PM2.5)

**Collected Features** (7 continuous):
- `cloud_cover_low` — Low altitude cloud coverage (0–100%)
- `cloud_cover_mid` — Mid altitude cloud coverage (0–100%)
- `cloud_cover_high` — High altitude cloud coverage (0–100%)
- `relative_humidity` — Relative humidity (0–100%)
- `pm25` — Fine particulate matter (µg/m³) [optional, requires API key]
- `visibility` — Horizontal visibility (meters)
- `temperature` — Air temperature (°C)

**Solar Geometry** (computed via PyEphem):
- `solar_elevation` — Sun height above horizon (degrees)
- `solar_azimuth` — Sun compass direction (0–360°)

**Temporal Resolution**: Hourly observations

**Lookback Period**: Configurable (default: 168 hours = 1 week)

**Storage Format**: Partitioned Parquet files by date
```
data/raw/weather/
├── 2026-04-26/
│   └── observations.parquet    # ~24 rows (hourly)
├── 2026-04-27/
│   └── observations.parquet
└── 2026-04-28/
    └── observations.parquet
```

**Example Row**:
```
timestamp               | cloud_cover_low | humidity | visibility | temp | solar_elevation | solar_azimuth
2026-04-26 10:00:00   | 30.0            | 65.0     | 10000      | 18.5 | 42.3            | 150.2
```

---

### **Stage 2: Webcam Frame Collection** (`data/webcam_scraper.py`)

**Purpose**: Retrieve reference images for light quality labeling

**Data Source**: Windy Public Webcam Archive
- Template: `https://images-webcams.windy.com/{WEBCAM_ID}/{timestamp}/full/{WEBCAM_ID}.jpg`
- Configurable via `.env`: `WEBCAM_ARCHIVE_URL_TEMPLATE`

**Timing Strategy**:
1. Compute sunrise & sunset times for each day (PyEphem)
2. Request frames at ±15 minutes around both events
3. Example: If sunrise is 06:30, request frames at 06:15 and 06:45

**Error Handling** (Critical Design Point):
- **404 responses**: Logged as INFO-level warning, gracefully skipped
- **Timeout errors**: Retried up to 3 times with exponential backoff
- **Pipeline behavior**: **Continues without frames** — ALQS labels are synthesized if webcam unavailable

**Storage Format**: Images + metadata parquet
```
data/raw/webcam/
├── 2026-04-26/
│   ├── 06_15_sunrise.jpg
│   ├── 18_42_sunset.jpg
│   └── metadata.parquet
└── 2026-04-27/
    ├── 06_14_sunrise.jpg
    └── metadata.parquet
```

---

### **Stage 3: ALQS Labeling** (`data/labeller.py`)

**Purpose**: Compute Atmospheric Light Quality Score (0–100) from collected frames

**Algorithm**: Weighted scoring from OpenCV frame analysis
```
ALQS = (saturation_score × 0.40) + (contrast_score × 0.30) + (warm_hue_ratio × 0.30)
```

**Component Metrics**:

| Component | Weight | Calculation | Range |
|-----------|--------|-------------|-------|
| **Saturation** | 40% | Mean saturation in HSV color space | 0–100 |
| **Contrast** | 30% | Std dev of Laplacian edge detection | 0–100 (normalized) |
| **Warm-hue Ratio** | 30% | (orange + gold + red pixels) / total pixels | 0–1, scaled to 0–100 |

**Example Calculation**:
- Frame 1 (sunrise): saturation=85, contrast=45, warm_hue=0.92 → ALQS = 85×0.4 + 45×0.3 + 92×0.3 = **72.5**
- Frame 2 (cloudy morning): saturation=40, contrast=25, warm_hue=0.30 → ALQS = 40×0.4 + 25×0.3 + 30×0.3 = **32.5**

**Storage Format**: Parquet with labeled scores
```
data/processed/alqs_labels.parquet
timestamp               | alqs
2026-04-26 06:15:00    | 72.5
2026-04-26 06:45:00    | 68.2
2026-04-26 18:42:00    | 75.1
```

---

## **SYNTHETIC ALQS FALLBACK** ⚠️

### When Synthetic ALQS is Used

If webcam frames are **unavailable** (404s, all network failures, or no URL configured):
- `data/labeller.py` is skipped
- `data/feature_engineer.py` generates synthetic ALQS scores

### Synthetic ALQS Formula

```python
alqs_synthetic = 50 + (solar_elevation × 0.2)
```

**Rationale**: 
- Solar elevation correlates strongly with visual light quality
- Base of 50 = neutral/average lighting
- `×0.2` scaling: 
  - Solar elevation 90° (sun at zenith) → ALQS = 50 + 18 = **68**
  - Solar elevation 0° (sunrise/sunset) → ALQS = 50 + 0 = **50**
  - Solar elevation -30° (30 min after sunset) → ALQS = 50 - 6 = **44**

### Example Synthetic Scenario

**Given**: No webcam images available for 2026-04-26
- 06:30 sunrise, solar_elevation = 2°
  - Synthetic ALQS = 50 + (2 × 0.2) = **50.4** (dim dawn)
- 12:00 noon, solar_elevation = 58°
  - Synthetic ALQS = 50 + (58 × 0.2) = **61.6** (moderate brightness)
- 18:45 sunset, solar_elevation = -5°
  - Synthetic ALQS = 50 + (-5 × 0.2) = **49.0** (twilight)

### Why Synthetic Works

| Condition | Solar Elevation | Synthetic ALQS | Interpretation |
|-----------|-----------------|---|---|
| Clear midday | 60°+ | 62–70 | Bright daylight |
| Morning/evening | 10–20° | 52–54 | Golden hour |
| Civil twilight | -6° to 0° | 48–50 | Dusk/dawn |
| Nautical twilight | -12° to -6° | 47–48 | Deep twilight |
| Night | -18° to -12° | 46–47 | Dim night sky |

This ensures the model trains even without reference images, using a physically-grounded approximation.

---

## Dataset Formation Method

### **Step 1: Data Merging & Resampling** (`data/feature_engineer.py`)

**Input Sources**:
1. Weather observations: `data/raw/weather/*.parquet`
2. ALQS labels: `data/processed/alqs_labels.parquet` (if available)
   - If missing: Synthesize from solar_elevation

**Process**:
```python
# Load all weather observations
weather_df = read all parquet files from data/raw/weather/

# Load ALQS labels (or synthesize)
if alqs_labels_exist:
    alqs_df = read data/processed/alqs_labels.parquet
else:
    alqs_df = synthesize from solar_elevation in weather_df

# Merge on timestamp with 30-min tolerance
merged = pd.merge_asof(weather_df, alqs_df, 
                       on='timestamp', 
                       tolerance=pd.Timedelta('30min'))

# Resample to 15-minute intervals (weather is hourly)
resampled = merged.resample('15T').apply({
    'cloud_cover_low': 'interpolate',
    'humidity': 'interpolate',
    'pm25': 'interpolate',
    'visibility': 'interpolate',
    'temperature': 'interpolate',
    'solar_elevation': 'interpolate',
    'alqs': 'ffill'  # Forward fill ALQS
})

# Drop rows with any NaN in ALQS
clean_data = resampled.dropna(subset=['alqs'])
```

**Output**: Time-series with uniform 15-minute spacing, ALQS filled forward from labeling time

---

### **Step 2: Feature Scaling & Encoding**

**Continuous Feature Scaling**:
```python
# MinMaxScaler fitted on training data
scaler = MinMaxScaler(feature_range=(0, 1))
continuous_features_scaled = scaler.fit_transform([
    cloud_cover_low, cloud_cover_mid, cloud_cover_high,
    relative_humidity, pm25, visibility, temperature
])
# Result: 7 normalized features in [0, 1]
```

**Weather Category Encoding**:
```python
# Derive weather_code from conditions (e.g., WMO code)
weather_categories = ['clear', 'partly_cloudy', 'cloudy', 'rain', 'storm', 'fog', ...]

# One-hot encode
weather_encoded = pd.get_dummies(weather_code)
# Result: N binary columns (one per category)
```

**Time-of-Day Encoding**:
```python
# Convert hour (0–23) to sin/cos pair for circular periodicity
sin_hour = np.sin(2 * np.pi * hour / 24)
cos_hour = np.cos(2 * np.pi * hour / 24)
# Ensures 23:00 and 01:00 are close, not opposite
```

**Example Row After Encoding**:
```
timestamp: 2026-04-26 10:15:00

Continuous (7): [0.3, 0.25, 0.15, 0.65, 0.1, 1.0, 0.55]
                (cloud_low, cloud_mid, cloud_high, humidity, pm25, visibility, temp)

Weather (8):    [1, 0, 0, 0, 0, 0, 0, 0]
                (clear=1, rest=0)

Time (2):       [0.259, 0.966]
                (sin(2π×10/24), cos(2π×10/24))

ALQS Target:    0.62
                (normalized to [0, 1] via MinMaxScaler)
```

---

### **Step 3: Sliding Window Sequences**

**Window Configuration**:
- **Duration**: 6 hours
- **Timestep Spacing**: 15 minutes
- **Timesteps per Window**: 24 (6 hours × 4 intervals/hour)
- **Temporal Axis**: Look-back only (i.e., history → current prediction)

**Window Construction**:
```python
# For each position i where i >= 24 (need 24 prior steps):
for i in range(24, len(data)):
    # Extract 24-step history (6 hours back)
    X[i] = continuous_features_scaled[i-24:i]  # Shape: (24, 7)
    
    # Target is ALQS at current timestamp
    y[i] = alqs_normalized[i]
    
    # Additional features for MLP classifier
    weather_state[i] = weather_encoded[i]
    time_encoding[i] = [sin_hour[i], cos_hour[i]]
```

**Example Sequence**:
```
Input X[1000] = [
  [0.30, 0.25, 0.15, 0.65, 0.10, 1.00, 0.55],  # t-24 (6 hrs ago)
  [0.32, 0.26, 0.16, 0.64, 0.11, 0.99, 0.56],  # t-23
  ...
  [0.35, 0.28, 0.18, 0.63, 0.12, 0.98, 0.58],  # t-1 (1 timestep ago)
  [0.36, 0.29, 0.19, 0.62, 0.13, 0.97, 0.59]   # t (current)
]
Shape: (24, 7)

Target y[1000] = 0.62  # ALQS at current timestamp
```

---

### **Step 4: Stratified Train/Val/Test Split (70/15/15)**

**Why Stratified?**
- Most datasets are skewed: mostly daylight (high ALQS) with few twilight/night samples (low ALQS)
- Random split might put all rare cases in train, leaving val/test incomplete
- Stratified ensures each split sees the full ALQS distribution

**Process**:
```python
# 1. Compute ALQS quintiles (5 bins)
alqs_strata = pd.qcut(y, q=5, labels=['Q0', 'Q1', 'Q2', 'Q3', 'Q4'], 
                      duplicates='drop')

# 2. Train/test split (70/30) stratified by strata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=alqs_strata, random_state=42
)

# 3. Validation/test split of remaining (50/50) stratified
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.50, 
    stratify=alqs_strata[X_test.index], random_state=42
)
```

**Result Distribution**:
```
Original: 2000 sequences
├── Train: 1400 (70%)
│   ├── Q0: 280 (low ALQS, ~20%)
│   ├── Q1: 280
│   ├── Q2: 280 (medium ALQS)
│   ├── Q3: 280
│   └── Q4: 280 (high ALQS, ~20%)
│
├── Validation: 300 (15%)
│   ├── Q0: 60
│   ├── Q1: 60
│   ├── Q2: 60
│   ├── Q3: 60
│   └── Q4: 60
│
└── Test: 300 (15%)
    ├── Q0: 60
    ├── Q1: 60
    ├── Q2: 60
    ├── Q3: 60
    └── Q4: 60
```

Each split sees the same proportion of rare/bright conditions → better generalization.

---

## Output Artifacts

After feature engineering, four artifacts are produced:

### **1. sequence_dataset.npz**

NumPy compressed archive containing LSTM training data.

**Contents**:
```python
{
    'X_train':           np.array(shape: (n_train, 24, 7)),
    'y_train':           np.array(shape: (n_train,)),
    'X_val':             np.array(shape: (n_val, 24, 7)),
    'y_val':             np.array(shape: (n_val,)),
    'X_test':            np.array(shape: (n_test, 24, 7)),
    'y_test':            np.array(shape: (n_test,)),
    'baseline_prev_test': np.array(shape: (n_test,))  # Persistence baseline
}
```

**Usage**: Loaded by `models/lstm_predictor.py` for time-series training

---

### **2. classifier_features.parquet**

Parquet table for MLP genre recommender training.

**Schema**:
```
timestamp         | alqs_norm | weather_state | sin_time | cos_time | genre
2026-04-26 10:15 | 0.62      | clear         | 0.259    | 0.966    | golden_hour
2026-04-26 11:00 | 0.65      | clear         | 0.342    | 0.940    | golden_hour
2026-04-26 20:30 | 0.25      | cloudy        |-0.793    |-0.609    | moody
```

**Genre Rules**:
- `golden_hour`: ALQS > 0.75 AND weather in {clear, partly_cloudy}
- `night_astro`: ALQS < 0.25
- `moody`: weather in {fog, rain, storm}
- `street`: default if no other rule matches
- `landscape`: secondary classification logic

**Usage**: Loaded by `models/mlp_recommender.py` for classification training

---

### **3. feature_artifacts.joblib**

Serialized scikit-learn objects for data transformation during inference.

**Contents**:
```python
{
    'scaler': MinMaxScaler,                    # Fitted on train continuous features
    'weather_encoder': OneHotEncoder,          # Fitted on train weather categories
    'alqs_scaler': MinMaxScaler,               # Fitted on train ALQS values (for normalization)
    'feature_config': {                        # Metadata
        'continuous_features': [...],
        'weather_categories': [...],
        'window_size': 24,
        'timestep_minutes': 15
    }
}
```

**Usage**: Loaded by `api/main.py` to transform inference inputs before model prediction

---

### **4. feature_metadata.json**

Configuration and statistics from feature engineering.

**Example**:
```json
{
  "dataset": {
    "start_date": "2026-04-26",
    "end_date": "2026-04-30",
    "total_sequences": 2000,
    "train_count": 1400,
    "val_count": 300,
    "test_count": 300
  },
  "features": {
    "window_size": 24,
    "timestep_minutes": 15,
    "window_duration_hours": 6,
    "continuous": [
      "cloud_cover_low",
      "cloud_cover_mid",
      "cloud_cover_high",
      "relative_humidity",
      "pm25",
      "visibility",
      "temperature"
    ],
    "weather_categories": ["clear", "partly_cloudy", "cloudy", "rain", "storm", "fog"],
    "encoding": {
      "time": "sin_cos_hour",
      "weather": "onehot"
    }
  },
  "data_sources": {
    "weather_provider": "open-meteo",
    "alqs_source": "webcam_frames_or_synthetic",
    "solar_provider": "pyephem"
  },
  "split_strategy": "stratified_by_alqs_quintile",
  "scaling_params": {
    "alqs_min": 25.0,
    "alqs_max": 95.0,
    "alqs_mean": 55.3,
    "alqs_std": 18.7
  }
}
```

---

## Complete Data Flow Diagram

```
┌──────────────────────┐        ┌─────────────────────┐
│  Open-Meteo API      │        │  Windy Webcam       │
│  (hourly weather)    │        │  Archive (images)   │
└──────────┬───────────┘        └─────────────┬───────┘
           │                                  │
           ▼                                  ▼
    ┌────────────────┐               ┌──────────────────┐
    │  Collector.py  │               │ WebcamScraper.py │
    │                │               │                  │
    │  • Open-Meteo  │               │  • 404 handling  │
    │  • PyEphem     │               │  • ±15 min round │
    │  • 7 features  │               │    sunrise/set   │
    └────────┬───────┘               └────────┬─────────┘
             │                               │
             ▼                               ▼
    data/raw/weather/*.parquet    data/raw/webcam/images
         (hourly obs)             + metadata.parquet
             │                               │
             │        ┌──────────────────────┘
             │        │
             ▼        ▼
        ┌─────────────────────┐
        │    Labeller.py      │
        │                     │
        │  • OpenCV analysis  │
        │  • Saturation (40%) │
        │  • Contrast (30%)   │
        │  • Warm-hue (30%)   │
        │  • ALQS 0–100       │
        └──────────┬──────────┘
                   │
                   ▼
        data/processed/alqs_labels.parquet
             (or SYNTHETIC if unavailable)
                   │
                   └──────────────────┬──────────────────┐
                                      │                  │
                            ┌─────────────────────┐      │
                            │FeatureEngineer.py   │      │
                            │                     │      │
                            │  • Merge & resample │      │
                            │  • Scale continuous │      │
                            │  • Encode weather   │      │
                            │  • 6-hr windows     │      │
                            │  • Stratified split │      │
                            │    (70/15/15)       │      │
                            │  • Serialize        │      │
                            └─────────────────────┘      │
                                      │                  │
                 ┌────────────────────┼──────────────────┤
                 │                    │                  │
                 ▼                    ▼                  ▼
        ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
        │sequence_        │  │classifier_       │  │feature_         │
        │dataset.npz      │  │features.parquet  │  │artifacts.joblib │
        │                 │  │                  │  │                 │
        │ X_train/val/    │  │ alqs_norm        │  │ scalers         │
        │ test: (n, 24,7) │  │ weather_state    │  │ encoders        │
        │ y_train/val/    │  │ time_encoding    │  │ config          │
        │ test: (n,)      │  │ genre            │  │                 │
        │                 │  │                  │  │                 │
        │ LSTM input ────→│  │ MLP input ──────→│  │ API inference ──│
        └─────────────────┘  └──────────────────┘  └─────────────────┘
                 │                    │                  │
                 ▼                    ▼                  ▼
        ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
        │ LSTM Trainer    │  │ MLP Trainer      │  │ FastAPI Serving │
        │                 │  │                  │  │                 │
        │ → lstm_         │  │ → mlp_           │  │ /predict        │
        │   predictor.    │  │   recommender.   │  │ /recommend      │
        │   keras + JSON  │  │   keras + JSON   │  │ /forecast       │
        └─────────────────┘  └──────────────────┘  └─────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **6-hour windows** | Captures diurnal atmospheric patterns without noise (too short) or staleness (too long) |
| **24 × 15-min timesteps** | Exact 6-hour period; 15-min aligns with typical weather variability |
| **Stratified split by ALQS quintile** | Ensures val/test see rare low-light and bright conditions, not just average days |
| **Graceful 404 handling** | Webcams aren't always available; pipeline continues with synthetic labels |
| **Synthetic ALQS fallback** | Physically-grounded approximation (solar_elevation) ensures model trains without images |
| **Forward-fill ALQS** | ALQS is labeled at specific moments (sunrise/sunset); forward-filled between observations maintains temporal integrity |
| **Separate train/val/test scalers** | Scalers fitted only on train, applied to val/test prevents data leakage |

---

## Configuration via `.env`

Key environment variables controlling the pipeline:

```bash
# Global ingestion
SAMPLE_SIZE=50
MAX_WEBCAMS=1000
DATA_OUTPUT_PATH=data/processed/global_dataset
WEBCAMS_API_KEY=

# Optional API keys (legacy local pipeline)
OPENWEATHERMAP_API_KEY=your_key_here

# Legacy/local pipeline (optional)
PHOTO_LAT=40.7128
PHOTO_LON=-74.0060

# Ingestion timing (API background task)
INGEST_INTERVAL_SECONDS=3600

# Training (legacy local pipeline)
TRAINING_LOOPS=0
```

---

## Running the Full Pipeline

### Windows (PowerShell)
```powershell
.\run_pipeline.ps1
```

### Linux/macOS (Bash)
```bash
./run_pipeline.sh
```

Legacy local pipeline (single location/webcam):

```powershell
.\run_pipeline.ps1 -LegacyLocalPipeline -TrainingLoops 1
```

```bash
LEGACY_LOCAL_PIPELINE=1 ./run_pipeline.sh
```

**Pipeline stages (global ingestion)**:
1. Discover global webcams and cache metadata
2. Sample webcams and download images
3. Pair each image with weather data at the same location/time
4. Write a dataset batch to `DATA_OUTPUT_PATH`

**Optional legacy local pipeline**:
1. Collector (weather fetch)
2. WebcamScraper (single-location image download)
3. Labeller (ALQS computation)
4. FeatureEngineer (sequences + split)
5. Trainer (LSTM + MLP if TrainingLoops > 0)

If any stage fails, pipeline halts immediately with a non-zero exit code.

---

## Troubleshooting

### Issue: "No ALQS labels found, using synthetic"
**Cause**: Webcam scraper failed (404s or network error)
**Resolution**: Pipeline continues with synthetic ALQS based on solar_elevation — safe fallback

### Issue: "sequence_dataset.npz not found"
**Cause**: Feature engineer crashed or didn't create output
**Solution**: 
- Check `logs/feature_engineer.log` for errors
- Verify weather parquet files exist in `data/raw/weather/`
- Ensure at least 7 days of data collected

### Issue: "Train set is empty after stratified split"
**Cause**: Dataset too small (< 100 sequences)
**Solution**: Collect more days of weather data (run collector longer)

---

## Summary

**Data Flow**: 
```
Weather (Open-Meteo) + Images (Webcam) + Labels (ALQS or synthetic)
    ↓
Merge & Resample → 15-min intervals
    ↓
Scale & Encode → Normalized continuous + one-hot categories + sin/cos time
    ↓
Sliding Windows → 24-step × 6-hour sequences
    ↓
Stratified Split → 70% train / 15% val / 15% test by ALQS quintile
    ↓
Serialize → sequence_dataset.npz + classifier_features.parquet + artifacts
    ↓
Train LSTM & MLP models on ML-ready tensors
    ↓
Serve predictions via FastAPI
```

**Key resilience**: If webcams fail, synthetic ALQS ensures training proceeds using solar geometry — a physically-grounded approximation of light quality.
