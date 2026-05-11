from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    EmailSubscriptionRequest,
    FeatureWindowRequest,
    FeedbackRequest,
    HealthResponse,
)


def _valid_feature_window() -> list[list[float]]:
    return [[0.0] * 7 for _ in range(24)]


def test_feature_window_request_shape() -> None:
    FeatureWindowRequest(feature_window=_valid_feature_window())
    with pytest.raises(ValidationError):
        FeatureWindowRequest(feature_window=[[0.0] * 7 for _ in range(23)])
    with pytest.raises(ValidationError):
        FeatureWindowRequest(feature_window=[[0.0] * 6 for _ in range(24)])


def test_email_subscription_request() -> None:
    m = EmailSubscriptionRequest(
        email=" User@Example.com ",
        latitude=10.0,
        longitude=-20.0,
        past_hours=48,
        enabled=True,
    )
    assert m.email == "user@example.com"

    with pytest.raises(ValidationError):
        EmailSubscriptionRequest(email="not-an-email", latitude=0.0, longitude=0.0)


def test_feedback_request_bounds() -> None:
    FeedbackRequest(
        predicted_alqs=50.0,
        observed_alqs=51.0,
        rating=3,
        weather_state="clear",
    )
    with pytest.raises(ValidationError):
        FeedbackRequest(
            predicted_alqs=50.0,
            observed_alqs=51.0,
            rating=6,
            weather_state="clear",
        )


def test_health_response_literal() -> None:
    h = HealthResponse(status="healthy")
    assert h.status == "healthy"
