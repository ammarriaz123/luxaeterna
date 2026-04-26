from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from tensorflow import keras

from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    FeatureWindowRequest,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    StatusResponse,
)
from data.collector import collect_recent_weather

LOGGER = logging.getLogger("luxaeterna.api")


@dataclass(slots=True)
class ModelRegistry:
    lstm_model: keras.Model
    mlp_model: keras.Model
    lstm_metadata: dict[str, Any]
    mlp_metadata: dict[str, Any]
    mlp_aux: dict[str, Any]
    ingest_task: asyncio.Task[None] | None = None


registry: ModelRegistry | None = None


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_data_freshness_minutes() -> float | None:
    weather_root = Path("data/raw/weather")
    parquet_files = sorted(weather_root.rglob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not parquet_files:
        return None

    newest_mtime = datetime.fromtimestamp(parquet_files[0].stat().st_mtime, tz=UTC)
    return (datetime.now(UTC) - newest_mtime).total_seconds() / 60.0


async def _periodic_ingestion_loop() -> None:
    interval = int(os.getenv("INGEST_INTERVAL_SECONDS", "1800"))
    latitude = float(os.getenv("PHOTO_LAT", "40.7128"))
    longitude = float(os.getenv("PHOTO_LON", "-74.0060"))
    storage = os.getenv("INGEST_STORAGE", "parquet")

    while True:
        try:
            await collect_recent_weather(
                latitude=latitude,
                longitude=longitude,
                hours_back=48,
                storage=storage,
            )
            LOGGER.info("Background ingestion tick completed")
        except Exception as exc:
            LOGGER.exception("Background ingestion failed: %s", exc)

        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global registry

    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "models/artifacts"))
    lstm_path = artifact_dir / "lstm_predictor.keras"
    mlp_path = artifact_dir / "mlp_recommender.keras"

    if not lstm_path.exists() or not mlp_path.exists():
        raise RuntimeError(
            "Model artifacts not found. Train models before starting API. "
            f"Expected {lstm_path} and {mlp_path}"
        )

    lstm_model = keras.models.load_model(lstm_path)
    mlp_model = keras.models.load_model(mlp_path)

    lstm_metadata = _read_json(artifact_dir / "lstm_metadata.json", fallback={"version": "unknown", "metrics": {}})
    mlp_metadata = _read_json(artifact_dir / "mlp_metadata.json", fallback={"version": "unknown", "metrics": {}})
    mlp_aux = joblib.load(artifact_dir / "mlp_aux.joblib")

    registry = ModelRegistry(
        lstm_model=lstm_model,
        mlp_model=mlp_model,
        lstm_metadata=lstm_metadata,
        mlp_metadata=mlp_metadata,
        mlp_aux=mlp_aux,
    )

    registry.ingest_task = asyncio.create_task(_periodic_ingestion_loop())
    LOGGER.info("API lifespan startup complete")

    try:
        yield
    finally:
        if registry and registry.ingest_task:
            registry.ingest_task.cancel()
            try:
                await registry.ingest_task
            except asyncio.CancelledError:
                pass
        LOGGER.info("API lifespan shutdown complete")


app = FastAPI(title="LuxAeterna", version="1.0.0", lifespan=lifespan)


def _require_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    return registry


async def _persist_feedback(payload: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(log_path, mode="a", encoding="utf-8") as handle:
        await handle.write(json.dumps(payload) + "\n")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: FeatureWindowRequest) -> PredictResponse:
    state = _require_registry()

    x = np.asarray(request.feature_window, dtype=np.float32).reshape(1, 24, 7)
    pred = float(state.lstm_model.predict(x, verbose=0).reshape(-1)[0])

    residual_std = float(state.lstm_metadata.get("metrics", {}).get("residual_std", 5.0))
    delta = 1.96 * residual_std

    return PredictResponse(
        predicted_alqs=pred,
        ci_lower=max(0.0, pred - delta),
        ci_upper=min(100.0, pred + delta),
        model_version=str(state.lstm_metadata.get("version", "unknown")),
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    state = _require_registry()

    alqs_scaler = state.mlp_aux["alqs_scaler"]
    weather_encoder = state.mlp_aux["weather_encoder"]
    label_encoder = state.mlp_aux["label_encoder"]

    ts = request.timestamp or datetime.now(UTC)
    hour = ts.hour + ts.minute / 60.0
    sin_time = np.sin(2 * np.pi * hour / 24.0)
    cos_time = np.cos(2 * np.pi * hour / 24.0)

    alqs_norm = alqs_scaler.transform(np.asarray([[request.alqs / 100.0]], dtype=np.float32))
    weather_vec = weather_encoder.transform(np.asarray([[request.weather_state]], dtype=object))

    x = np.concatenate([alqs_norm, weather_vec, np.asarray([[sin_time, cos_time]], dtype=np.float32)], axis=1)
    probs = state.mlp_model.predict(x, verbose=0).reshape(-1)

    labels = label_encoder.inverse_transform(np.arange(len(probs)))
    ranking = sorted(zip(labels, probs.tolist()), key=lambda item: item[1], reverse=True)

    return RecommendResponse(
        ranked_genres=[{"genre": genre, "score": float(score)} for genre, score in ranking],
        model_version=str(state.mlp_metadata.get("version", "unknown")),
    )


@app.get("/forecast", response_model=ForecastResponse)
async def forecast() -> ForecastResponse:
    state = _require_registry()

    latest_window_path = Path("data/processed/latest_window.npy")
    if latest_window_path.exists():
        window = np.load(latest_window_path).astype(np.float32)
    else:
        window = np.zeros((24, 7), dtype=np.float32)

    now = datetime.now(UTC)
    points: list[ForecastPoint] = []

    for step in range(24):
        pred = float(state.lstm_model.predict(window.reshape(1, 24, 7), verbose=0).reshape(-1)[0])
        ts = now + timedelta(minutes=30 * step)
        points.append(ForecastPoint(timestamp=ts, predicted_alqs=pred))

        next_row = window[-1].copy()
        next_row = np.clip(next_row * 0.99, 0.0, 1.0)
        window = np.vstack([window[1:], next_row])

    return ForecastResponse(
        points=points,
        model_version=str(state.lstm_metadata.get("version", "unknown")),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks) -> FeedbackResponse:
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **request.model_dump(),
    }
    background_tasks.add_task(_persist_feedback, payload, Path("logs/feedback.jsonl"))
    return FeedbackResponse(status="accepted")


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    state = _require_registry()
    return StatusResponse(
        api_status="ok",
        lstm_model_version=str(state.lstm_metadata.get("version", "unknown")),
        mlp_model_version=str(state.mlp_metadata.get("version", "unknown")),
        data_freshness_minutes=_compute_data_freshness_minutes(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    _require_registry()
    return HealthResponse(status="healthy")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()
