from __future__ import annotations

from datetime import datetime
from typing import Literal

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


class StatusResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    api_status: Literal["ok"]
    lstm_model_version: str
    mlp_model_version: str
    data_freshness_minutes: float | None


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["healthy"]
