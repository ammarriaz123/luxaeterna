from __future__ import annotations

import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from api.schemas import LightingClassScore

UTC = timezone.utc
LOGGER = logging.getLogger("luxaeterna.api.notifications")


@dataclass(slots=True)
class EmailSubscription:
    email: str
    latitude: float
    longitude: float
    past_hours: int
    enabled: bool
    created_at_utc: str
    updated_at_utc: str
    last_sent_at_utc: str | None = None
    last_reference_time_utc: str | None = None


def _subscriptions_path() -> Path:
    return Path(os.getenv("EMAIL_SUBSCRIPTIONS_PATH", "data/processed/email_subscriptions.json"))


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_store(path: Path) -> dict[str, EmailSubscription]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, EmailSubscription] = {}
    for email, data in raw.items():
        out[email] = EmailSubscription(
            email=str(data["email"]),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            past_hours=int(data.get("past_hours", 72)),
            enabled=bool(data.get("enabled", True)),
            created_at_utc=str(data.get("created_at_utc", _utc_iso())),
            updated_at_utc=str(data.get("updated_at_utc", _utc_iso())),
            last_sent_at_utc=data.get("last_sent_at_utc"),
            last_reference_time_utc=data.get("last_reference_time_utc"),
        )
    return out


def _write_store(path: Path, items: dict[str, EmailSubscription]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {
            "email": item.email,
            "latitude": item.latitude,
            "longitude": item.longitude,
            "past_hours": item.past_hours,
            "enabled": item.enabled,
            "created_at_utc": item.created_at_utc,
            "updated_at_utc": item.updated_at_utc,
            "last_sent_at_utc": item.last_sent_at_utc,
            "last_reference_time_utc": item.last_reference_time_utc,
        }
        for key, item in items.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_email_subscription(
    *,
    email: str,
    latitude: float,
    longitude: float,
    past_hours: int,
    enabled: bool,
) -> tuple[str, EmailSubscription]:
    path = _subscriptions_path()
    store = _read_store(path)
    key = _normalize_email(email)
    now = _utc_iso()
    existing = store.get(key)
    if existing is None:
        status = "subscribed" if enabled else "unsubscribed"
        item = EmailSubscription(
            email=key,
            latitude=latitude,
            longitude=longitude,
            past_hours=past_hours,
            enabled=enabled,
            created_at_utc=now,
            updated_at_utc=now,
        )
    else:
        status = "updated"
        if not enabled:
            status = "unsubscribed"
        item = EmailSubscription(
            email=key,
            latitude=latitude,
            longitude=longitude,
            past_hours=past_hours,
            enabled=enabled,
            created_at_utc=existing.created_at_utc,
            updated_at_utc=now,
            last_sent_at_utc=existing.last_sent_at_utc,
            last_reference_time_utc=existing.last_reference_time_utc,
        )
    store[key] = item
    _write_store(path, store)
    return status, item


def list_enabled_subscriptions() -> list[EmailSubscription]:
    path = _subscriptions_path()
    store = _read_store(path)
    return [item for item in store.values() if item.enabled]


def mark_subscription_sent(email: str, reference_time_utc: str) -> None:
    path = _subscriptions_path()
    store = _read_store(path)
    key = _normalize_email(email)
    existing = store.get(key)
    if existing is None:
        return
    updated = EmailSubscription(
        email=existing.email,
        latitude=existing.latitude,
        longitude=existing.longitude,
        past_hours=existing.past_hours,
        enabled=existing.enabled,
        created_at_utc=existing.created_at_utc,
        updated_at_utc=_utc_iso(),
        last_sent_at_utc=_utc_iso(),
        last_reference_time_utc=reference_time_utc,
    )
    store[key] = updated
    _write_store(path, store)


def _top_two(scores: list[LightingClassScore]) -> list[LightingClassScore]:
    return sorted(scores, key=lambda x: x.probability, reverse=True)[:2]


def _compose_subject(predicted_label: str) -> str:
    label = predicted_label.replace("_", " ")
    return f"LuxAeterna hourly lighting update: {label}"


def _compose_body(
    *,
    reference_time_utc: str,
    latitude: float,
    longitude: float,
    predicted_label: str,
    scores: list[LightingClassScore],
) -> str:
    lines = [
        "Your LuxAeterna hourly lighting update is ready.",
        "",
        f"Reference time (UTC): {reference_time_utc}",
        f"Location: {latitude:.5f}, {longitude:.5f}",
        f"Top class: {predicted_label.replace('_', ' ')}",
        "",
        "Top probabilities:",
    ]
    for item in _top_two(scores):
        lines.append(f"- {item.label.replace('_', ' ')}: {item.probability * 100:.1f}%")
    lines.append("")
    lines.append("Open the app for full verdict and coach details.")
    return "\n".join(lines)


def is_smtp_configured() -> bool:
    required = [
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_FROM",
    ]
    return all(os.getenv(key, "").strip() for key in required)


def send_hourly_email(
    *,
    to_email: str,
    reference_time_utc: str,
    latitude: float,
    longitude: float,
    predicted_label: str,
    scores: list[LightingClassScore],
) -> None:
    if not is_smtp_configured():
        raise RuntimeError("SMTP is not configured")

    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "1").strip().lower() in {"1", "true", "yes", "on"}
    sender = os.getenv("SMTP_FROM", "").strip()

    msg = EmailMessage()
    msg["Subject"] = _compose_subject(predicted_label)
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        _compose_body(
            reference_time_utc=reference_time_utc,
            latitude=latitude,
            longitude=longitude,
            predicted_label=predicted_label,
            scores=scores,
        )
    )

    if use_tls:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if username:
                server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if username:
                server.login(username, password)
            server.send_message(msg)
    LOGGER.info("Sent hourly lighting email to %s", to_email)
