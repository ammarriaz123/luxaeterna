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
|-- run_pipeline.ps1         # Windows-native pipeline runner
|-- run_pipeline.sh          # Bash pipeline runner
|-- Dockerfile
|-- requirements.txt
|-- .env.example
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
- Forecast and recommendation endpoints
- Background weather ingestion task and async feedback logging

## Requirements

- Python 3.12
- Windows PowerShell (recommended on Windows) or Bash
- Internet access for weather/webcam data pulls

Optional:

- OpenWeatherMap API key for PM2.5 and forecast enrichment

## Configuration

1. Copy .env.example to .env
2. Set environment variables as needed

Important variables:

- SAMPLE_SIZE: number of webcams sampled per global ingestion run
- MAX_WEBCAMS: optional cap for discovery cache size
- DATA_OUTPUT_PATH: dataset output directory or file path
- WEBCAMS_API_KEY: optional discovery provider API key (if required)
- OPENWEATHERMAP_API_KEY: optional, enables OWM enrichment (legacy local pipeline)
- PHOTO_LAT and PHOTO_LON: location for legacy local pipeline and API defaults
- INGEST_INTERVAL_SECONDS: API background ingestion interval

Security note:

- Never commit real secrets in .env or .env.example
- Keep .env local only

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

1. Creates/uses .venv
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
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

- http://127.0.0.1:8000/docs

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

Input:

- alqs (0-100)
- weather_state
- timestamp (optional)

Output:

- ranked_genres (sorted with probability scores)
- model_version

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
- model versions
- data freshness in minutes

### GET /health

Output:

- simple healthy status when models are loaded

## Model Artifacts

Typical outputs in models/artifacts:

- lstm_predictor.keras
- lstm_predictor_savedmodel/
- lstm_metadata.json
- mlp_recommender.keras
- mlp_recommender_savedmodel/
- mlp_metadata.json
- mlp_aux.joblib

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
.\.venv\Scripts\python.exe -m data.feature_engineer --weather-path data/raw/weather --label-path data/processed/alqs_labels.parquet --output-dir data/processed
```

### API startup error about missing model artifacts

Train models first with run_pipeline.ps1 (TrainingLoops >= 1), then restart API.

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
