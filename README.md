# LuxAeterna

Enterprise-grade machine learning backend for predicting Atmospheric Light Quality Score (ALQS) and recommending photography genres from weather, time-series, and visual context.

## Overview

LuxAeterna is organized as a modular mono-repo with three core domains:

- data: data engineering, external API ingestion, webcam collection, ALQS label creation, and feature engineering
- models: time-series regression (LSTM) and multi-class recommendation (MLP)
- api: FastAPI serving layer with startup model loading, inference endpoints, health/status checks, and async feedback logging

The system is designed for practical MLOps workflows:

- repeatable pipeline execution
- environment-driven configuration
- artifact versioning metadata
- clear train/serve separation

## Repository Structure

```text
.
|-- api/
|   |-- main.py
|   |-- schemas.py
|   `-- __init__.py
|-- data/
|   |-- collector.py
|   |-- global_ingestion.py
|   |-- webcam_scraper.py
|   |-- webcam_discovery.py
|   |-- labeller.py
|   |-- feature_engineer.py
|   `-- __init__.py
|-- models/
|   |-- lstm_predictor.py
|   |-- mlp_recommender.py
|   |-- artifacts/           # generated model outputs
|   `-- __init__.py
|-- logs/                    # runtime logs and feedback logs
|-- web/                     # Vite + React dashboard (npm commands run *inside* web/)
|-- run_pipeline.ps1         # Windows-native pipeline runner
|-- run_pipeline.sh          # Bash pipeline runner
|-- Dockerfile
|-- requirements.txt
|-- .env
`-- README.md
```

## Core Capabilities

### Data Module

- Asynchronous weather ingestion from Open-Meteo and optional OpenWeatherMap enrichment
- Solar geometry (elevation and azimuth) via PyEphem
- Global webcam discovery and sampling for scalable dataset creation
- Webcam frame retrieval near sunrise/sunset windows
- OpenCV-based ALQS scoring from saturation, contrast, and warm-hue ratio
- 6-hour sliding window dataset generation with stratified train/val/test split

### Model Module

- LSTM regressor for ALQS forecasting from sequence features
- MLP classifier for genre recommendations from ALQS, weather state, and time encodings
- TensorBoard integration, early stopping, LR scheduling, and artifact exports

### API Module

- FastAPI app with strict Pydantic validation
- Startup lifespan model load (single initialization)
- **Lighting-event ensemble** (`POST /predict/event`): XGBoost + multiclass LSTM + MLP (see `FRONTEND_INTEGRATION_GUIDE.md`)
- Legacy ALQS LSTM (`POST /predict`, `GET /forecast`) and genre MLP (`POST /recommend`) when those artifacts exist
- Background weather ingestion task and async feedback logging
- **`GET /`** redirects to **`/docs`**
- **`POST /predict/event/from_location`**: Open-Meteo hourly features → same ensemble as `/predict/event`, plus rules-based (optional OpenAI) **shooting coach**
- **`POST /coach/shooting`**: coach only from client-supplied class probabilities + weather snapshot (no weather fetch)

## Requirements

- Python **3.10, 3.11, or 3.12** (recommended). On Windows, **avoid 3.13+ for `pip install -r requirements.txt`** unless you upgrade NumPy/TensorFlow to versions that publish wheels for your Python, or install Visual Studio C++ Build Tools (otherwise pip may try to compile NumPy from source and fail).
- Windows PowerShell (recommended on Windows) or Bash
- Internet access for weather/webcam data pulls

Optional:

- OpenWeatherMap API key for PM2.5 and forecast enrichment

## Configuration

1. Create or edit `.env` in the repo root (see variables below)

Important variables:

- SAMPLE_SIZE: number of webcams sampled per global ingestion run
- MAX_WEBCAMS: optional cap for discovery cache size
- DATA_OUTPUT_PATH: dataset output directory or file path
- WEBCAMS_API_KEY: optional discovery provider API key (if required)
- OPENWEATHERMAP_API_KEY: optional, enables OWM enrichment (legacy local pipeline)
- PHOTO_LAT and PHOTO_LON: location for legacy local pipeline and API defaults
- INGEST_INTERVAL_SECONDS: API background ingestion interval
- MODEL_ARTIFACT_DIR: folder with `models/artifacts` (ensemble + optional legacy)
- CORS_ORIGINS: comma-separated origins for the web app (e.g. Vite on :5173)
- INGEST_STORAGE: `parquet` or `sqlite` for background weather pulls
- **Shooting coach (optional LLM):** `COACH_LLM` set to `1`, `true`, `yes`, or `on` plus an API key (`OPENAI_API_KEY`, `GROQ_API_KEY`, or `COACH_LLM_API_KEY`). Uses OpenAI-compatible Chat Completions: default base is OpenAI, or Groq when `GROQ_API_KEY` is set or the key looks like `gsk_...`. Optional `COACH_LLM_BASE_URL` overrides the base (e.g. `https://api.groq.com/openai/v1`). `COACH_OPENAI_MODEL` / `COACH_LLM_MODEL` select the model (Groq defaults to `llama-3.3-70b-versatile` when the endpoint is Groq and the model is still an OpenAI default). `COACH_LLM_BACKEND`: `openai_http` (default) or `langgraph`
- **Hourly email alerts (optional):** The web UI can register an address via `POST /notifications/email-subscription`. The API persists subscriptions and sends at most one message per subscriber per clock hour when **`SMTP_HOST`**, **`SMTP_PORT`**, and **`SMTP_FROM`** are set (optional **`SMTP_USERNAME`** / **`SMTP_PASSWORD`**, **`SMTP_USE_TLS`** default `1`). Without SMTP, subscriptions are still saved but no mail is sent (check server logs). Optional **`NOTIFY_INTERVAL_SECONDS`** (e.g. `3600` or `1h`) overrides the default top-of-hour schedule; **`EMAIL_SUBSCRIPTIONS_PATH`** overrides the JSON file location (default `data/processed/email_subscriptions.json`).

Security note:

- Never commit `.env` (it is gitignored); keep secrets local only

## Quickstart (Windows)

Run the complete pipeline with the PowerShell runner:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\run_pipeline.ps1
```

Useful options:

```powershell
# Skip webcam/labeller stages
.\run_pipeline.ps1 -SkipWebcam -SkipLabeller

# Fast local test with no training loops
.\run_pipeline.ps1 -TrainingLoops 0 -LookbackHours 24 -WebcamDays 1

# Enable legacy local pipeline (single location/webcam)
.\run_pipeline.ps1 -LegacyLocalPipeline
```

What the pipeline does:

1. Creates/uses `venv` (or `.venv` if you prefer)
2. Installs dependencies
3. Discovers global webcams and caches metadata
4. Samples webcams and builds a global dataset with image + weather pairing
5. (Optional) Legacy local pipeline for single-location training

## Quickstart (Bash)

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Legacy local pipeline (single location/webcam) can be enabled with:

```bash
LEGACY_LOCAL_PIPELINE=1 ./run_pipeline.sh
```

## Start the API

After training artifacts exist under models/artifacts:

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

- http://127.0.0.1:8000/docs

## Web dashboard (local)

The UI lives under **`web/`** — run npm from that folder, not the repo root:

```powershell
cd web
npm install
npm run dev
```

Open **http://localhost:5173**. The dev server proxies **`/api/*`** to the API on port **8000**, so keep uvicorn running.

Production build: `cd web && npm run build` — set **`VITE_API_URL`** to your deployed API origin (no trailing slash) before building.

## API Endpoints

### POST /predict

Input:

- feature_window: 24x7 numeric matrix

Output:

- predicted_alqs
- ci_lower
- ci_upper
- model_version

### POST /recommend

Requires **`mlp_recommender.keras`** and **`mlp_aux.joblib`**. Returns **503** if not trained.

Input:

- alqs (0-100)
- weather_state
- timestamp (optional)

Output:

- ranked_genres (sorted with probability scores)
- model_version

### POST /predict/event

**4-class lighting ensemble** (power-weighted XGB + LSTM + MLP). Requires multiclass artifacts under `models/artifacts/`. Request body matches `FRONTEND_INTEGRATION_GUIDE.md`: **`sequence`** `6×10` features + **`tabular`** length **21** (lags at final timestep).

Output: class probabilities, predicted label, ensemble weights.

### POST /predict/event/from_location

**Same ensemble as `/predict/event`**, but the server fetches Open-Meteo for **`latitude`**, **`longitude`**, and **`past_hours`** (24–240), builds `sequence` + `tabular`, runs inference, and returns **`weather_snapshot`**, **`prediction`**, and **`coach`** (rules; optional LLM when `COACH_LLM` and a coach API key are set — OpenAI or Groq; see Configuration).

### POST /coach/shooting

**Shooting / camera recommendations** from an existing **`predicted_class_id`**, **`predicted_label`**, **`class_probabilities`**, and optional **`weather_snapshot`** — no model re-run. Same rules + optional LLM enrichment as the location endpoint.

### POST /notifications/email-subscription

Register or update hourly lighting emails for a coordinate window. Body: **`email`**, **`latitude`**, **`longitude`**, **`past_hours`** (24–240), **`enabled`**. Persists to disk (see **`EMAIL_SUBSCRIPTIONS_PATH`**). Background task sends one email per hour per subscriber when SMTP env vars are set; uses the same Open-Meteo → ensemble path as **`/predict/event/from_location`** (rules-only summary in the message).

### GET /forecast

Output:

- 12-hour forecast at 30-minute intervals
- model_version

### POST /feedback

Input:

- post-event quality fields and user rating

Output:

- accepted status; payload is asynchronously appended to logs/feedback.jsonl

### GET /status

Output:

- API status
- legacy LSTM / MLP metadata versions (genre recommender shows **unknown** if `mlp_metadata.json` or recommender artifacts are missing)
- **`capabilities`**: which routes are actually loaded
- **`ensemble_event`**: XGB + multiclass LSTM/MLP metadata and blend weights when ensemble is loaded

### GET /health

Output:

- simple healthy status when models are loaded

## Model Artifacts

Typical outputs in models/artifacts:

**Lighting-event ensemble (used by `POST /predict/event`):**

- `xgb_multiclass_model.json`, `xgb_multiclass_metadata.json`
- `lstm_multiclass_model.keras`, `lstm_multiclass_scaler.joblib`, `lstm_multiclass_metadata.json`
- `mlp_multiclass_model.keras`, `mlp_multiclass_scaler.joblib`, `mlp_multiclass_metadata.json`

**Legacy ALQS + genres (optional):**

- lstm_predictor.keras, lstm_metadata.json
- mlp_recommender.keras, mlp_metadata.json, mlp_aux.joblib

Keras multiclass models may have been saved with a **newer Keras** than `tensorflow-cpu` bundles; `api/ensemble_bundle.py` applies small load-time compat shims so they still load on TF 2.16 / Keras 3.12.

## Data Artifacts

Typical outputs in data/processed:

- sequence_dataset.npz
- classifier_features.parquet
- feature_artifacts.joblib
- feature_metadata.json
- latest_window.npy
- alqs_labels.parquet (if webcam frames exist)

## Troubleshooting

### Many webcam 404 responses

This can be normal for some archive templates/time ranges. The scraper now treats missing frames as non-fatal and continues pipeline execution.

### OpenWeatherMap key warning

If OPENWEATHERMAP_API_KEY is missing, OWM enrichment is skipped. Base weather ingestion still works through Open-Meteo.

### LSTM training fails with missing sequence_dataset.npz

Run feature engineering successfully first:

```powershell
.\venv\Scripts\python.exe -m data.feature_engineer --weather-path data/raw/weather --label-path data/processed/alqs_labels.parquet --output-dir data/processed
```

### API startup error about missing model artifacts

You need **either** the **ensemble** files above **or** **`lstm_predictor.keras`** (or both). Train or copy artifacts into `models/artifacts`, then restart the API.

### npm ENOENT package.json

Run **`npm install`** / **`npm run dev`** from **`web/`**, not the repository root.

## Documentation map (avoid mixing two pipelines)

| Doc | What it describes |
|-----|-------------------|
| **`FRONTEND_INTEGRATION_GUIDE.md`** | **Production inference contract**: 6×10 sequence + 21 tabular features, ensemble weights — matches **`POST /predict/event`**. |
| **`DATA_PIPELINE.md`** | Deep dive on **legacy local** ingestion (collector + labeller + feature engineer): often **24×7** windows and **15‑minute** resampling for **`lstm_predictor`** / **`mlp_recommender`**. Not the same tensor shape as the global multiclass ensemble. |
| **`PROJECT_HANDOVER.md`** | Product narrative and ensemble rationale; filenames align with `models/artifacts/*multiclass*`. |
| **`PIPELINE_REPORT.md`**, **`IMPROVEMENTS_GUIDE.md`**, **`QUICK_START.md`**, **`IMPLEMENTATION_SUMMARY.md`**, **`VALIDATION_CHECKLIST.md`** | Historical / extended-pipeline notes (dates ~May 2026); useful context, may reference machine-specific paths. |

## Next steps (product)

1. **Real inputs for `/predict/event`**: Server or script that pulls Open-Meteo + PyEphem + lag features for a lat/lon (per `FRONTEND_INTEGRATION_GUIDE.md`), then wire the **web** UI to that instead of demo vectors.
2. **Optional `/recommend`**: Run `models/mlp_recommender.py` to produce `mlp_recommender.keras` + `mlp_aux.joblib` if you want genre rankings in-app.
3. **Agent + notifications**: Scheduled job (thresholds, quiet hours) + web push or email; reuse `/status` and `/predict/event`.
4. **Deploy**: API (Docker/Railway) + static `web` build with `VITE_API_URL`; set `CORS_ORIGINS`.
5. **Tests**: Contract tests for `/predict/event` JSON and scaler parity.

## Docker

Build and run:

```bash
docker build -t luxaeterna .
docker run -p 8000:8000 luxaeterna
```

The Dockerfile defines volumes for:

- /app/data/raw
- /app/data/processed
- /app/models/artifacts
- /app/logs

## Development Notes

- Strong typing and modularity are used across modules
- Runtime behavior is controlled through environment variables
- Optional external sources are validated and warned at pipeline startup
- Training and serving are loosely coupled via model artifacts

## Roadmap Ideas

- Automated tests for data and API contracts
- Model drift and data quality monitoring
- CI/CD for training + model registry promotion
- Better webcam provider abstraction with fallback sources
