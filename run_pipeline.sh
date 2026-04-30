#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINING_LOOPS="${TRAINING_LOOPS:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
MAX_WEBCAMS="${MAX_WEBCAMS:-1000}"
DATA_OUTPUT_PATH="${DATA_OUTPUT_PATH:-data/processed/global_dataset}"
WEBCAM_CACHE_PATH="${WEBCAM_CACHE_PATH:-data/webcams.json}"
LEGACY_LOCAL_PIPELINE="${LEGACY_LOCAL_PIPELINE:-0}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENWEATHERMAP_API_KEY:-}" ]]; then
  echo "[LuxAeterna][WARN] OPENWEATHERMAP_API_KEY is not set. PM2.5 and forecast enrichment will be unavailable."
fi

if [[ -z "${SAMPLE_SIZE:-}" ]]; then
  echo "[LuxAeterna][WARN] SAMPLE_SIZE is not set. Default of 50 will be used."
fi

if [[ -z "${DATA_OUTPUT_PATH:-}" ]]; then
  echo "[LuxAeterna][WARN] DATA_OUTPUT_PATH is not set. Default of data/processed/global_dataset will be used."
fi

if [[ "${LEGACY_LOCAL_PIPELINE}" == "1" ]]; then
  if [[ -z "${PHOTO_LAT:-}" || -z "${PHOTO_LON:-}" ]]; then
    echo "[LuxAeterna][WARN] PHOTO_LAT/PHOTO_LON not fully set. Default module values will be used."
  fi

  if [[ -z "${WEBCAM_ARCHIVE_URL_TEMPLATE:-}" ]]; then
    echo "[LuxAeterna][WARN] WEBCAM_ARCHIVE_URL_TEMPLATE not set. Default template may produce many 404 responses."
  elif [[ "${WEBCAM_ARCHIVE_URL_TEMPLATE}" != *"{timestamp}"* ]]; then
    echo "[LuxAeterna][WARN] WEBCAM_ARCHIVE_URL_TEMPLATE is missing {timestamp}; webcam scraping may fail."
  fi
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/raw data/processed models/artifacts logs

echo "[LuxAeterna] Discovering global webcams"
python -m data.webcam_discovery --cache-path "${WEBCAM_CACHE_PATH}" --max-webcams "${MAX_WEBCAMS}"

echo "[LuxAeterna] Ingesting global webcam batch"
python -m data.global_ingestion --cache-path "${WEBCAM_CACHE_PATH}" --sample-size "${SAMPLE_SIZE}" --max-webcams "${MAX_WEBCAMS}" --output-path "${DATA_OUTPUT_PATH}"

if [[ "${LEGACY_LOCAL_PIPELINE}" == "1" ]]; then
  echo "[LuxAeterna] Collecting weather + solar ground truth"
  python -m data.collector --lookback-hours 168 --storage parquet

  echo "[LuxAeterna] Scraping webcam reference frames"
  if [[ -n "${WEBCAM_ARCHIVE_URL_TEMPLATE:-}" ]]; then
    python -m data.webcam_scraper --days 7 --archive-url-template "${WEBCAM_ARCHIVE_URL_TEMPLATE}" || true
  else
    python -m data.webcam_scraper --days 7 || true
  fi

  echo "[LuxAeterna] Computing ALQS labels"
  python -m data.labeller --input-dir data/raw/webcam --output-path data/processed/alqs_labels.parquet || true

  echo "[LuxAeterna] Building ML features"
  python -m data.feature_engineer --weather-path data/raw/weather --label-path data/processed/alqs_labels.parquet --output-dir data/processed
fi

if [[ "${LEGACY_LOCAL_PIPELINE}" == "1" ]]; then
  for ((i=1; i<=TRAINING_LOOPS; i++)); do
    echo "[LuxAeterna] Training loop ${i}/${TRAINING_LOOPS}: LSTM"
    python -m models.lstm_predictor --data-path data/processed/sequence_dataset.npz --artifact-dir models/artifacts

    echo "[LuxAeterna] Training loop ${i}/${TRAINING_LOOPS}: MLP recommender"
    python -m models.mlp_recommender --features-path data/processed/classifier_features.parquet --artifact-dir models/artifacts
  done
fi

echo "[LuxAeterna] Pipeline complete"
