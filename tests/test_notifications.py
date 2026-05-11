from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.notifications import (
    is_smtp_configured,
    list_enabled_subscriptions,
    mark_subscription_sent,
    send_hourly_email,
    upsert_email_subscription,
)
from api.schemas import LightingClassScore


@pytest.fixture
def subscriptions_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "subs.json"
    monkeypatch.setenv("EMAIL_SUBSCRIPTIONS_PATH", str(path))
    return path


def test_upsert_subscribe_and_update(subscriptions_path: Path) -> None:
    status, item = upsert_email_subscription(
        email="  Test@Example.COM ",
        latitude=40.0,
        longitude=-74.0,
        past_hours=48,
        enabled=True,
    )
    assert status == "subscribed"
    assert item.email == "test@example.com"
    assert item.past_hours == 48

    status2, item2 = upsert_email_subscription(
        email="test@example.com",
        latitude=41.0,
        longitude=-73.0,
        past_hours=72,
        enabled=True,
    )
    assert status2 == "updated"
    assert item2.latitude == 41.0
    assert item2.past_hours == 72
    assert item2.created_at_utc == item.created_at_utc


def test_upsert_unsubscribed_status(subscriptions_path: Path) -> None:
    upsert_email_subscription(
        email="a@b.co",
        latitude=0.0,
        longitude=0.0,
        past_hours=24,
        enabled=True,
    )
    status, _ = upsert_email_subscription(
        email="a@b.co",
        latitude=0.0,
        longitude=0.0,
        past_hours=24,
        enabled=False,
    )
    assert status == "unsubscribed"


def test_list_enabled_subscriptions(subscriptions_path: Path) -> None:
    upsert_email_subscription(
        email="on@x.co", latitude=1.0, longitude=2.0, past_hours=24, enabled=True
    )
    upsert_email_subscription(
        email="off@x.co", latitude=1.0, longitude=2.0, past_hours=24, enabled=False
    )
    enabled = list_enabled_subscriptions()
    assert len(enabled) == 1
    assert enabled[0].email == "on@x.co"


def test_mark_subscription_sent(subscriptions_path: Path) -> None:
    upsert_email_subscription(
        email="u@x.co", latitude=10.0, longitude=20.0, past_hours=48, enabled=True
    )
    mark_subscription_sent("U@X.CO", "2026-01-01T12:00:00+00:00")
    raw = json.loads(subscriptions_path.read_text(encoding="utf-8"))
    row = raw["u@x.co"]
    assert row["last_reference_time_utc"] == "2026-01-01T12:00:00+00:00"
    assert row["last_sent_at_utc"]


def test_is_smtp_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_FROM", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert is_smtp_configured() is False

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_FROM", "noreply@example.com")
    assert is_smtp_configured() is True


def test_send_hourly_email_requires_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_FROM"):
        monkeypatch.delenv(key, raising=False)
    scores = [
        LightingClassScore(class_id=0, label="no_event", probability=0.9),
        LightingClassScore(class_id=1, label="golden_hour_only", probability=0.1),
    ]
    with pytest.raises(RuntimeError, match="SMTP"):
        send_hourly_email(
            to_email="x@y.co",
            reference_time_utc="2026-01-01T00:00:00+00:00",
            latitude=0.0,
            longitude=0.0,
            predicted_label="no_event",
            scores=scores,
        )


def test_compose_helpers_body_and_subject() -> None:
    from api.notifications import _compose_body, _compose_subject

    assert "golden hour" in _compose_subject("golden_hour_only").lower()
    body = _compose_body(
        reference_time_utc="2026-05-01T14:00:00Z",
        latitude=40.7128,
        longitude=-74.006,
        predicted_label="no_event",
        scores=[
            LightingClassScore(class_id=0, label="no_event", probability=0.55),
            LightingClassScore(class_id=3, label="golden_hour_and_diffusion", probability=0.45),
        ],
    )
    assert "2026-05-01" in body
    assert "40.71280" in body
    assert "no event" in body.lower() or "No event" in body
