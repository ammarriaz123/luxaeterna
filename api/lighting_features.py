"""Build ensemble inputs (6×10 sequence + 21 tabular) from Open-Meteo hourly data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd

from data.collector import _compute_solar_geometry

LOGGER = logging.getLogger("luxaeterna.api.lighting_features")

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


@dataclass(slots=True)
class HourlyWeatherSnapshot:
    """Most recent hour used as ensemble \"now\" timestep."""

    timestamp_utc: datetime
    temperature_c: float
    relative_humidity_pct: float
    visibility_m: float
    cloud_cover_low_pct: float
    cloud_cover_mid_pct: float
    cloud_cover_high_pct: float
    weather_code: float
    solar_elevation_deg: float


def fetch_forecast_hourly_dataframe(
    latitude: float,
    longitude: float,
    *,
    past_hours: int = 72,
    forecast_hours: int = 6,
    timeout: float = 45.0,
) -> pd.DataFrame:
    """Hourly rows from Open-Meteo forecast API (past + short future for API completeness)."""
    hourly = [
        "temperature_2m",
        "relative_humidity_2m",
        "visibility",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "weather_code",
    ]
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            OPEN_METEO_FORECAST,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": hourly,
                "past_hours": past_hours,
                "forecast_hours": forecast_hours,
                "timezone": "UTC",
            },
        )
        response.raise_for_status()
        payload = response.json()

    block = payload.get("hourly") or {}
    times = block.get("time") or []
    if not times:
        raise ValueError("Open-Meteo returned no hourly rows")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times, utc=True),
            "temperature": block.get("temperature_2m"),
            "relative_humidity": block.get("relative_humidity_2m"),
            "visibility": block.get("visibility"),
            "cloud_cover_low": block.get("cloud_cover_low"),
            "cloud_cover_mid": block.get("cloud_cover_mid"),
            "cloud_cover_high": block.get("cloud_cover_high"),
            "weather_code": block.get("weather_code"),
        }
    )
    frame["visibility"] = frame["visibility"].ffill().bfill()
    frame["visibility"] = frame["visibility"].fillna(10_000.0)
    frame = frame.dropna(subset=["temperature", "relative_humidity", "cloud_cover_low"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _add_solar_elevation(frame: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    out = frame.copy()

    def elev(ts: pd.Timestamp) -> float:
        el, _ = _compute_solar_geometry(latitude, longitude, ts)
        return el

    out["solar_elevation"] = out["timestamp"].map(elev)
    return out


def trim_to_observed_hours(frame: pd.DataFrame) -> pd.DataFrame:
    """Use rows up to current UTC hour so we do not rely on future weather for inference."""
    now = pd.Timestamp.now(tz=timezone.utc).floor("h")
    trimmed = frame[frame["timestamp"] <= now].copy()
    if len(trimmed) < 6:
        raise ValueError(
            f"Only {len(trimmed)} hourly rows at or before the current hour (need ≥ 6). "
            "Increase `past_hours` on the request (e.g. 96) or retry after Open-Meteo updates."
        )
    return trimmed.reset_index(drop=True)


def build_ensemble_arrays(
    frame: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    """
    Returns (sequence 6×10, tabular length 21, debug meta).
    Feature order matches api.ensemble_bundle.
    """
    df = _add_solar_elevation(frame, latitude, longitude)
    df = trim_to_observed_hours(df)

    last_n = df.tail(6).reset_index(drop=True)
    sequence: list[list[float]] = []
    for _, row in last_n.iterrows():
        sequence.append(
            [
                float(latitude),
                float(longitude),
                float(row["temperature"]),
                float(row["relative_humidity"]),
                float(row["visibility"]),
                float(row["cloud_cover_low"]),
                float(row["cloud_cover_mid"]),
                float(row["cloud_cover_high"]),
                float(row["weather_code"]),
                float(row["solar_elevation"]),
            ]
        )

    full = df.reset_index(drop=True)
    i = len(full) - 1
    if i < 3:
        raise ValueError("Insufficient history for lag features")

    row = full.iloc[i]
    base10 = [
        float(latitude),
        float(longitude),
        float(row["temperature"]),
        float(row["relative_humidity"]),
        float(row["visibility"]),
        float(row["cloud_cover_low"]),
        float(row["cloud_cover_mid"]),
        float(row["cloud_cover_high"]),
        float(row["weather_code"]),
        float(row["solar_elevation"]),
    ]

    def lag(series: str, k: int) -> float:
        return float(full.iloc[i - k][series])

    tabular = [
        *base10,
        lag("temperature", 1),
        lag("relative_humidity", 1),
        lag("cloud_cover_low", 1),
        lag("temperature", 2),
        lag("relative_humidity", 2),
        lag("cloud_cover_low", 2),
        lag("temperature", 3),
        lag("relative_humidity", 3),
        lag("cloud_cover_low", 3),
        float(row["temperature"]) - lag("temperature", 1),
        float(row["cloud_cover_low"]) - lag("cloud_cover_low", 1),
    ]

    if len(tabular) != 21:
        raise RuntimeError(f"tabular feature length mismatch: {len(tabular)}")

    snap = HourlyWeatherSnapshot(
        timestamp_utc=row["timestamp"].to_pydatetime(),
        temperature_c=float(row["temperature"]),
        relative_humidity_pct=float(row["relative_humidity"]),
        visibility_m=float(row["visibility"]),
        cloud_cover_low_pct=float(row["cloud_cover_low"]),
        cloud_cover_mid_pct=float(row["cloud_cover_mid"]),
        cloud_cover_high_pct=float(row["cloud_cover_high"]),
        weather_code=float(row["weather_code"]),
        solar_elevation_deg=float(row["solar_elevation"]),
    )
    meta: dict[str, Any] = {
        "reference_time_utc": snap.timestamp_utc.isoformat(),
        "rows_in_window": int(len(last_n)),
        "snapshot": {
            "temperature_c": snap.temperature_c,
            "relative_humidity_pct": snap.relative_humidity_pct,
            "visibility_m": snap.visibility_m,
            "cloud_cover_low_pct": snap.cloud_cover_low_pct,
            "cloud_cover_mid_pct": snap.cloud_cover_mid_pct,
            "cloud_cover_high_pct": snap.cloud_cover_high_pct,
            "weather_code": snap.weather_code,
            "solar_elevation_deg": snap.solar_elevation_deg,
        },
    }
    return sequence, tabular, meta


def nan_safe_arrays(
    sequence: list[list[float]],
    tabular: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    seq = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0)
    tab = np.nan_to_num(np.asarray(tabular, dtype=np.float32), nan=0.0)
    return seq, tab
