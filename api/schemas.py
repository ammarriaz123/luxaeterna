from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureWindowRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    feature_window: list[list[float]] = Field(..., description="24x7 feature window")

    @field_validator("feature_window")
    @classmethod
    def validate_window(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 24:
            raise ValueError("feature_window must contain exactly 24 timesteps")
        for row in value:
            if len(row) != 7:
                raise ValueError("each timestep must contain exactly 7 features")
        return value


class PredictResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    predicted_alqs: float
    ci_lower: float
    ci_upper: float
    model_version: str


class RecommendRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    alqs: float = Field(..., ge=0.0, le=100.0)
    weather_state: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime | None = None


class GenreScore(BaseModel):
    model_config = ConfigDict(strict=True)

    genre: str
    score: float


class RecommendResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    ranked_genres: list[GenreScore]
    model_version: str


class ForecastPoint(BaseModel):
    model_config = ConfigDict(strict=True)

    timestamp: datetime
    predicted_alqs: float


class ForecastResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    points: list[ForecastPoint]
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    event_id: str | None = Field(default=None, min_length=1, max_length=128)
    predicted_alqs: float = Field(..., ge=0.0, le=100.0)
    observed_alqs: float = Field(..., ge=0.0, le=100.0)
    rating: int = Field(..., ge=1, le=5)
    weather_state: str = Field(..., min_length=1, max_length=64)
    notes: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["accepted"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["healthy"]


class LightingClassScore(BaseModel):
    model_config = ConfigDict(strict=True)

    class_id: int = Field(..., ge=0, le=3)
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictLightingEventRequest(BaseModel):
    """6×10 sequence (LSTM) + 21 tabular features at final timestep (XGB/MLP). Feature order matches api.ensemble_bundle."""

    model_config = ConfigDict(strict=True)

    sequence: list[list[float]] = Field(
        ...,
        description="6 timesteps × 10 features: lat, lon, temp, rh, visibility, cloud low/mid/high, weather_code, solar_elevation",
    )
    tabular: list[float] = Field(..., description="21 lag/tabular features for the last timestep")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: list[list[float]]) -> list[list[float]]:
        from api.ensemble_bundle import LSTM_WINDOW, N_LSTM_FEATURES

        if len(value) != LSTM_WINDOW:
            raise ValueError(f"sequence must have exactly {LSTM_WINDOW} timesteps")
        for row in value:
            if len(row) != N_LSTM_FEATURES:
                raise ValueError(f"each timestep must have exactly {N_LSTM_FEATURES} features")
        return value

    @field_validator("tabular")
    @classmethod
    def validate_tabular(cls, value: list[float]) -> list[float]:
        from api.ensemble_bundle import N_TABULAR_FEATURES

        if len(value) != N_TABULAR_FEATURES:
            raise ValueError(f"tabular must have exactly {N_TABULAR_FEATURES} values")
        return value


class PredictLightingEventResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    predicted_class_id: int = Field(..., ge=0, le=3)
    predicted_label: str
    class_probabilities: list[LightingClassScore]
    ensemble_weights: dict[str, float]
    model_note: str = Field(
        default="power_weighted_xgb_lstm_mlp",
        description="0.5 XGB + 0.35 LSTM + 0.15 MLP",
    )


class StatusResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    api_status: Literal["ok"]
    lstm_model_version: str
    mlp_model_version: str
    data_freshness_minutes: float | None
    capabilities: dict[str, bool] = Field(default_factory=dict)
    ensemble_event_model_loaded: bool = False
    ensemble_event: dict[str, Any] | None = Field(
        default=None,
        description="XGB + multiclass LSTM/MLP ensemble (lighting events); null if not loaded",
    )


class CoachRecommendation(BaseModel):
    model_config = ConfigDict(strict=True)

    predicted_label: str
    shooting_mode: str
    iso_suggestion: str
    aperture_guidance: str
    shutter_guidance: str
    white_balance: str
    gear_notes: str
    checklist: list[str]
    creative_brief: str
    source: Literal["rules", "rules+openai"] = "rules"
    llm_addon: str | None = None


class PredictFromLocationRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    past_hours: int = Field(
        default=72,
        ge=24,
        le=240,
        description="Open-Meteo past_hours window; need ≥6 usable hourly rows before now",
    )


class PredictFromLocationResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    latitude: float
    longitude: float
    reference_time_utc: str
    weather_snapshot: dict[str, Any]
    prediction: PredictLightingEventResponse
    coach: CoachRecommendation


class CoachFromPredictionRequest(BaseModel):
    """Rules (+ optional LLM) coach from an existing prediction + weather snapshot (no Open-Meteo call)."""

    model_config = ConfigDict(strict=True)

    predicted_class_id: int = Field(..., ge=0, le=3)
    predicted_label: str = Field(..., min_length=1, max_length=64)
    class_probabilities: list[LightingClassScore] = Field(..., min_length=1)
    weather_snapshot: dict[str, Any] = Field(default_factory=dict)


class CoachFromPredictionResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    coach: CoachRecommendation
