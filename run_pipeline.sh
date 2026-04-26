#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINING_LOOPS="${TRAINING_LOOPS:-1}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENWEATHERMAP_API_KEY:-}" ]]; then
  echo "[PhotometricAI][WARN] OPENWEATHERMAP_API_KEY is not set. PM2.5 and forecast enrichment will be unavailable."
fi

if [[ -z "${PHOTO_LAT:-}" || -z "${PHOTO_LON:-}" ]]; then
  echo "[PhotometricAI][WARN] PHOTO_LAT/PHOTO_LON not fully set. Default module values will be used."
fi

if [[ -z "${WEBCAM_ARCHIVE_URL_TEMPLATE:-}" ]]; then
  echo "[PhotometricAI][WARN] WEBCAM_ARCHIVE_URL_TEMPLATE not set. Default template may produce many 404 responses."
elif [[ "${WEBCAM_ARCHIVE_URL_TEMPLATE}" != *"{timestamp}"* ]]; then
  echo "[PhotometricAI][WARN] WEBCAM_ARCHIVE_URL_TEMPLATE is missing {timestamp}; webcam scraping may fail."
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/raw data/processed models/artifacts logs

echo "[PhotometricAI] Collecting weather + solar ground truth"
python -m data.collector --lookback-hours 168 --storage parquet

echo "[PhotometricAI] Scraping webcam reference frames"
if [[ -n "${WEBCAM_ARCHIVE_URL_TEMPLATE:-}" ]]; then
  python -m data.webcam_scraper --days 7 --archive-url-template "${WEBCAM_ARCHIVE_URL_TEMPLATE}" || true
else
  python -m data.webcam_scraper --days 7 || true
fi

echo "[PhotometricAI] Computing ALQS labels"
python -m data.labeller --input-dir data/raw/webcam --output-path data/processed/alqs_labels.parquet || true

echo "[PhotometricAI] Building ML features"
python -m data.feature_engineer --weather-path data/raw/weather --label-path data/processed/alqs_labels.parquet --output-dir data/processed

for ((i=1; i<=TRAINING_LOOPS; i++)); do
  echo "[PhotometricAI] Training loop ${i}/${TRAINING_LOOPS}: LSTM"
  python -m models.lstm_predictor --data-path data/processed/sequence_dataset.npz --artifact-dir models/artifacts

  echo "[PhotometricAI] Training loop ${i}/${TRAINING_LOOPS}: MLP recommender"
  python -m models.mlp_recommender --features-path data/processed/classifier_features.parquet --artifact-dir models/artifacts
done

echo "[PhotometricAI] Pipeline complete"
