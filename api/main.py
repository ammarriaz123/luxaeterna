from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import aiofiles
import httpx
import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from tensorflow import keras

from api.coach import (
    build_rules_coach,
    coach_llm_available,
    iter_anchor_chat_sse,
    iter_coach_shooting_sse,
    maybe_enrich_coach_with_openai,
)
from api.ensemble_bundle import (
    CLASS_LABELS,
    WEIGHT_LSTM,
    WEIGHT_MLP,
    WEIGHT_XGB,
    EnsembleBundle,
    ensemble_loadable,
    load_ensemble_bundle,
    predict_lighting_event_probs,
)
from api.lighting_features import (
    build_ensemble_arrays,
    fetch_forecast_hourly_dataframe,
    nan_safe_arrays,
)
from api.notifications import (
    is_smtp_configured,
    list_enabled_subscriptions,
    mark_subscription_sent,
    send_hourly_email,
    upsert_email_subscription,
)
from api.schemas import (
    CoachChatRequest,
    CoachFromPredictionRequest,
    CoachFromPredictionResponse,
    EmailSubscriptionRequest,
    EmailSubscriptionResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeatureWindowRequest,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    LightingClassScore,
    PredictFromLocationRequest,
    PredictFromLocationResponse,
    PredictLightingEventRequest,
    PredictLightingEventResponse,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    StatusResponse,
)
from data.collector import collect_recent_weather

LOGGER = logging.getLogger("luxaeterna.api")


@dataclass(slots=True)
class ModelRegistry:
    lstm_model: keras.Model | None
    mlp_model: keras.Model | None
    lstm_metadata: dict[str, Any]
    mlp_metadata: dict[str, Any]
    mlp_aux: dict[str, Any] | None
    ensemble: EnsembleBundle | None
    ingest_task: asyncio.Task[None] | None = None
    notify_task: asyncio.Task[None] | None = None


registry: ModelRegistry | None = None
_smtp_missing_logged: bool = False


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


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


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


def _seconds_until_next_hour() -> float:
    now = datetime.now(UTC)
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return max(5.0, (next_hour - now).total_seconds())


def _interval_to_seconds(raw: str) -> float:
    value = raw.strip().lower()
    if value.endswith("h"):
        return float(value[:-1]) * 3600.0
    if value.endswith("m"):
        return float(value[:-1]) * 60.0
    if value.endswith("s"):
        return float(value[:-1])
    return float(value)


def _notify_sleep_seconds() -> float:
    raw = os.getenv("NOTIFY_INTERVAL_SECONDS", "").strip()
    if not raw:
        return _seconds_until_next_hour()
    try:
        return max(10.0, _interval_to_seconds(raw))
    except ValueError:
        LOGGER.warning("Invalid NOTIFY_INTERVAL_SECONDS=%r; falling back to top-of-hour schedule", raw)
        return _seconds_until_next_hour()


def _notification_key(reference_time_utc: str) -> str:
    try:
        ts = datetime.fromisoformat(reference_time_utc.replace("Z", "+00:00"))
        return ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0).isoformat()
    except ValueError:
        return reference_time_utc


async def _send_hourly_notifications_once(state: ModelRegistry) -> None:
    global _smtp_missing_logged
    subs = await asyncio.to_thread(list_enabled_subscriptions)
    if not subs:
        return
    if not await asyncio.to_thread(is_smtp_configured):
        if not _smtp_missing_logged:
            LOGGER.warning(
                "Hourly email subscriptions are registered but SMTP is not configured "
                "(set SMTP_HOST, SMTP_PORT, SMTP_FROM). Skipping sends until configured."
            )
            _smtp_missing_logged = True
        return
    if state.ensemble is None:
        LOGGER.warning("Skipping hourly notifications because ensemble is unavailable")
        return

    for sub in subs:
        try:
            frame = await asyncio.to_thread(
                fetch_forecast_hourly_dataframe,
                sub.latitude,
                sub.longitude,
                past_hours=sub.past_hours,
            )
            sequence, tabular, meta = await asyncio.to_thread(
                build_ensemble_arrays,
                frame,
                sub.latitude,
                sub.longitude,
            )
            seq, tab = nan_safe_arrays(sequence, tabular)
            pred = _predict_lighting_event_arrays(state, seq, tab)
            ref_key = _notification_key(meta["reference_time_utc"])
            if sub.last_reference_time_utc == ref_key:
                continue
            await asyncio.to_thread(
                send_hourly_email,
                to_email=sub.email,
                reference_time_utc=meta["reference_time_utc"],
                latitude=sub.latitude,
                longitude=sub.longitude,
                predicted_label=pred.predicted_label,
                scores=pred.class_probabilities,
            )
            await asyncio.to_thread(mark_subscription_sent, sub.email, ref_key)
        except Exception as exc:
            LOGGER.exception("Hourly notify failed for %s: %s", sub.email, exc)


async def _periodic_notification_loop() -> None:
    while True:
        await asyncio.sleep(_notify_sleep_seconds())
        state = registry
        if state is None:
            continue
        try:
            await _send_hourly_notifications_once(state)
        except Exception as exc:
            LOGGER.exception("Hourly notification loop tick failed: %s", exc)


def _load_models(artifact_dir: Path) -> ModelRegistry:
    ensemble: EnsembleBundle | None = None
    if ensemble_loadable(artifact_dir):
        LOGGER.info("Loading multiclass lighting ensemble")
        ensemble = load_ensemble_bundle(artifact_dir)
    else:
        LOGGER.warning("Ensemble artifacts incomplete; /predict/event unavailable")

    lstm_path = artifact_dir / "lstm_predictor.keras"
    mlp_path = artifact_dir / "mlp_recommender.keras"
    mlp_aux_path = artifact_dir / "mlp_aux.joblib"

    lstm_model: keras.Model | None = None
    mlp_model: keras.Model | None = None
    mlp_aux: dict[str, Any] | None = None

    if lstm_path.exists():
        lstm_model = keras.models.load_model(lstm_path)
    else:
        LOGGER.warning("lstm_predictor.keras missing; legacy /predict and /forecast unavailable")

    if mlp_path.exists() and mlp_aux_path.exists():
        mlp_model = keras.models.load_model(mlp_path)
        mlp_aux = joblib.load(mlp_aux_path)
    else:
        LOGGER.warning("MLP recommender artifacts missing; /recommend unavailable")

    lstm_metadata = _read_json(artifact_dir / "lstm_metadata.json", fallback={"version": "unknown", "metrics": {}})
    mlp_metadata = _read_json(artifact_dir / "mlp_metadata.json", fallback={"version": "unknown", "metrics": {}})

    if ensemble is None and lstm_model is None:
        raise RuntimeError(
            "No usable models found. Provide ensemble (xgb_multiclass_model.json + mlp/lstm multiclass) "
            f"and/or {lstm_path.name} under {artifact_dir}"
        )

    return ModelRegistry(
        lstm_model=lstm_model,
        mlp_model=mlp_model,
        lstm_metadata=lstm_metadata,
        mlp_metadata=mlp_metadata,
        mlp_aux=mlp_aux,
        ensemble=ensemble,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global registry

    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "models/artifacts"))
    registry = _load_models(artifact_dir)

    registry.ingest_task = asyncio.create_task(_periodic_ingestion_loop())
    registry.notify_task = asyncio.create_task(_periodic_notification_loop())
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
        if registry and registry.notify_task:
            registry.notify_task.cancel()
            try:
                await registry.notify_task
            except asyncio.CancelledError:
                pass
        LOGGER.info("API lifespan shutdown complete")


app = FastAPI(title="LuxAeterna", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Browser default; this API has no HTML homepage — send people to interactive docs."""
    return RedirectResponse(url="/docs")


def _require_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    return registry


def _capabilities(state: ModelRegistry) -> dict[str, bool]:
    ensemble_on = state.ensemble is not None
    return {
        "legacy_alqs_predict": state.lstm_model is not None,
        "legacy_forecast": state.lstm_model is not None,
        "legacy_recommend": state.mlp_model is not None and state.mlp_aux is not None,
        "lighting_event_ensemble": ensemble_on,
        "predict_event_from_location": ensemble_on,
        "shooting_coach": True,
        "coach_llm": coach_llm_available(),
    }


async def _persist_feedback(payload: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(log_path, mode="a", encoding="utf-8") as handle:
        await handle.write(json.dumps(payload) + "\n")


def _predict_lighting_event_arrays(
    state: ModelRegistry,
    seq: np.ndarray,
    tab: np.ndarray,
) -> PredictLightingEventResponse:
    if state.ensemble is None:
        raise HTTPException(status_code=503, detail="Lighting ensemble not loaded (missing multiclass artifacts)")
    probs = predict_lighting_event_probs(state.ensemble, seq, tab)
    predicted_id = int(np.argmax(probs))
    scores = [
        LightingClassScore(class_id=i, label=CLASS_LABELS[i], probability=float(probs[i]))
        for i in range(len(CLASS_LABELS))
    ]
    return PredictLightingEventResponse(
        predicted_class_id=predicted_id,
        predicted_label=CLASS_LABELS[predicted_id],
        class_probabilities=scores,
        ensemble_weights={"xgb": WEIGHT_XGB, "lstm": WEIGHT_LSTM, "mlp": WEIGHT_MLP},
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: FeatureWindowRequest) -> PredictResponse:
    state = _require_registry()
    if state.lstm_model is None:
        raise HTTPException(status_code=503, detail="Legacy LSTM regressor not loaded")

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


@app.post("/predict/event", response_model=PredictLightingEventResponse)
async def predict_lighting_event(request: PredictLightingEventRequest) -> PredictLightingEventResponse:
    state = _require_registry()
    seq, tab = nan_safe_arrays(request.sequence, request.tabular)
    return _predict_lighting_event_arrays(state, seq, tab)


@app.post("/predict/event/from_location", response_model=PredictFromLocationResponse)
async def predict_lighting_from_location(request: PredictFromLocationRequest) -> PredictFromLocationResponse:
    state = _require_registry()
    if state.ensemble is None:
        raise HTTPException(status_code=503, detail="Lighting ensemble not loaded (missing multiclass artifacts)")

    try:
        frame = await asyncio.to_thread(
            fetch_forecast_hourly_dataframe,
            request.latitude,
            request.longitude,
            past_hours=request.past_hours,
        )
        sequence, tabular, meta = await asyncio.to_thread(
            build_ensemble_arrays,
            frame,
            request.latitude,
            request.longitude,
        )
        seq, tab = nan_safe_arrays(sequence, tabular)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Open-Meteo request failed: {exc}") from exc

    pred = _predict_lighting_event_arrays(state, seq, tab)
    coach = build_rules_coach(
        predicted_label=pred.predicted_label,
        predicted_class_id=pred.predicted_class_id,
        class_probabilities=pred.class_probabilities,
        weather_snapshot=meta["snapshot"],
    )
    coach = await asyncio.to_thread(
        maybe_enrich_coach_with_openai,
        coach,
        predicted_class_id=pred.predicted_class_id,
        class_probabilities=pred.class_probabilities,
        weather_snapshot=meta["snapshot"],
    )

    return PredictFromLocationResponse(
        latitude=request.latitude,
        longitude=request.longitude,
        reference_time_utc=meta["reference_time_utc"],
        weather_snapshot=meta["snapshot"],
        prediction=pred,
        coach=coach,
    )


@app.post("/coach/shooting", response_model=CoachFromPredictionResponse)
async def coach_shooting(request: CoachFromPredictionRequest) -> CoachFromPredictionResponse:
    """Agentic-style shooting tips from class probabilities + weather (no model re-run)."""
    coach = build_rules_coach(
        predicted_label=request.predicted_label,
        predicted_class_id=request.predicted_class_id,
        class_probabilities=request.class_probabilities,
        weather_snapshot=request.weather_snapshot,
    )
    coach = await asyncio.to_thread(
        maybe_enrich_coach_with_openai,
        coach,
        predicted_class_id=request.predicted_class_id,
        class_probabilities=request.class_probabilities,
        weather_snapshot=request.weather_snapshot,
    )
    return CoachFromPredictionResponse(coach=coach)


@app.post("/coach/shooting/stream")
async def coach_shooting_stream(request: CoachFromPredictionRequest) -> StreamingResponse:
    """SSE: rules coach, structured field refinements, streamed narrative tokens, final merged coach."""

    def event_bytes() -> Iterator[bytes]:
        yield from iter_coach_shooting_sse(
            predicted_class_id=request.predicted_class_id,
            predicted_label=request.predicted_label,
            class_probabilities=request.class_probabilities,
            weather_snapshot=request.weather_snapshot,
        )

    return StreamingResponse(
        event_bytes(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/coach/chat/stream")
async def coach_chat_stream(request: CoachChatRequest) -> StreamingResponse:
    """Anchored Salle de conseil — multi-turn chat grounded in one forecast bundle (SSE)."""
    if not coach_llm_available():
        raise HTTPException(
            status_code=503,
            detail="Coach LLM is not enabled (set COACH_LLM=1 and OPENAI_API_KEY or GROQ_API_KEY).",
        )

    def chat_bytes() -> Iterator[bytes]:
        yield from iter_anchor_chat_sse(request)

    return StreamingResponse(
        chat_bytes(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/notifications/email-subscription", response_model=EmailSubscriptionResponse)
async def set_email_subscription(request: EmailSubscriptionRequest) -> EmailSubscriptionResponse:
    status, item = await asyncio.to_thread(
        upsert_email_subscription,
        email=request.email,
        latitude=request.latitude,
        longitude=request.longitude,
        past_hours=request.past_hours,
        enabled=request.enabled,
    )
    return EmailSubscriptionResponse(status=status, subscription=asdict(item))


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    state = _require_registry()
    if state.mlp_model is None or state.mlp_aux is None:
        raise HTTPException(status_code=503, detail="MLP recommender not loaded")

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
    if state.lstm_model is None:
        raise HTTPException(status_code=503, detail="Legacy LSTM regressor not loaded")

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
    caps = _capabilities(state)
    ensemble_event: dict[str, Any] | None = None
    if state.ensemble is not None:
        ensemble_event = {
            "blend_weights": {"xgb": WEIGHT_XGB, "lstm": WEIGHT_LSTM, "mlp": WEIGHT_MLP},
            "xgboost_classifier": state.ensemble.metadata.get("xgb_multiclass_metadata"),
            "lstm_multiclass": state.ensemble.metadata.get("lstm_multiclass_metadata"),
            "mlp_multiclass": state.ensemble.metadata.get("mlp_multiclass_metadata"),
        }
    return StatusResponse(
        api_status="ok",
        lstm_model_version=str(state.lstm_metadata.get("version", "unknown")),
        mlp_model_version=str(state.mlp_metadata.get("version", "unknown")),
        data_freshness_minutes=_compute_data_freshness_minutes(),
        capabilities=caps,
        ensemble_event_model_loaded=state.ensemble is not None,
        ensemble_event=ensemble_event,
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
