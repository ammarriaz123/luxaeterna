from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone

UTC = dt_timezone.utc
from pathlib import Path
from typing import Any

import ephem
import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger("luxaeterna.data.collector")


@dataclass(slots=True)
class CollectorConfig:
    latitude: float
    longitude: float
    start_time: datetime
    end_time: datetime
    timezone: str = "UTC"
    output_root: Path = Path("data/raw/weather")
    sqlite_path: Path = Path("data/raw/weather/weather.db")
    storage: str = "parquet"
    owm_api_key: str | None = None


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _compute_solar_geometry(latitude: float, longitude: float, timestamp: pd.Timestamp) -> tuple[float, float]:
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = timestamp.to_pydatetime()

    sun = ephem.Sun(observer)
    elevation = float(sun.alt) * 180.0 / 3.141592653589793
    azimuth = float(sun.az) * 180.0 / 3.141592653589793
    return elevation, azimuth


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _safe_get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


class WeatherCollector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config

    async def fetch_open_meteo(self, client: httpx.AsyncClient) -> pd.DataFrame:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "start_date": self.config.start_time.date().isoformat(),
            "end_date": self.config.end_time.date().isoformat(),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "visibility",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                "weather_code",
            ],
            "timezone": self.config.timezone,
        }
        payload = await _safe_get(client, url, params)
        hourly = payload.get("hourly", {})

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
                "temperature": hourly.get("temperature_2m", []),
                "relative_humidity": hourly.get("relative_humidity_2m", []),
                "visibility": hourly.get("visibility", []),
                "cloud_cover_low": hourly.get("cloud_cover_low", []),
                "cloud_cover_mid": hourly.get("cloud_cover_mid", []),
                "cloud_cover_high": hourly.get("cloud_cover_high", []),
                "weather_code": hourly.get("weather_code", []),
            }
        )
        return frame.dropna(subset=["timestamp"])

    async def fetch_openweather_forecast(self, client: httpx.AsyncClient) -> pd.DataFrame:
        if not self.config.owm_api_key:
            LOGGER.warning("OpenWeatherMap API key not found. Forecast/PM2.5 enrichment disabled.")
            return pd.DataFrame()

        weather_url = "https://api.openweathermap.org/data/2.5/forecast"
        weather_params = {
            "lat": self.config.latitude,
            "lon": self.config.longitude,
            "appid": self.config.owm_api_key,
            "units": "metric",
        }
        weather_payload = await _safe_get(client, weather_url, weather_params)

        weather_records: list[dict[str, Any]] = []
        for item in weather_payload.get("list", []):
            weather_records.append(
                {
                    "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
                    "owm_temperature": item.get("main", {}).get("temp"),
                    "owm_relative_humidity": item.get("main", {}).get("humidity"),
                    "owm_visibility": item.get("visibility"),
                    "owm_cloud_cover_total": item.get("clouds", {}).get("all"),
                }
            )

        air_url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
        air_params = {
            "lat": self.config.latitude,
            "lon": self.config.longitude,
            "appid": self.config.owm_api_key,
        }
        air_payload = await _safe_get(client, air_url, air_params)

        pm_records: list[dict[str, Any]] = []
        for item in air_payload.get("list", []):
            pm_records.append(
                {
                    "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
                    "pm25": item.get("components", {}).get("pm2_5"),
                }
            )

        weather_df = pd.DataFrame(weather_records)
        pm_df = pd.DataFrame(pm_records)
        if weather_df.empty:
            return pm_df
        return weather_df.merge(pm_df, on="timestamp", how="outer")

    async def collect(self) -> pd.DataFrame:
        async with httpx.AsyncClient(timeout=30) as client:
            meteo_task = asyncio.create_task(self.fetch_open_meteo(client))
            owm_task = asyncio.create_task(self.fetch_openweather_forecast(client))
            meteo_df, owm_df = await asyncio.gather(meteo_task, owm_task)

        merged = meteo_df.merge(owm_df, on="timestamp", how="outer") if not owm_df.empty else meteo_df
        if merged.empty:
            return merged

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        merged = merged[(merged["timestamp"] >= self.config.start_time) & (merged["timestamp"] <= self.config.end_time)]

        if "temperature" not in merged.columns:
            merged["temperature"] = pd.NA
        if "relative_humidity" not in merged.columns:
            merged["relative_humidity"] = pd.NA
        if "visibility" not in merged.columns:
            merged["visibility"] = pd.NA

        if "owm_temperature" in merged.columns:
            merged["temperature"] = merged["temperature"].combine_first(merged["owm_temperature"])
        if "owm_relative_humidity" in merged.columns:
            merged["relative_humidity"] = merged["relative_humidity"].combine_first(merged["owm_relative_humidity"])
        if "owm_visibility" in merged.columns:
            merged["visibility"] = merged["visibility"].combine_first(merged["owm_visibility"])

        if "pm25" not in merged.columns:
            merged["pm25"] = pd.NA

        solar = merged["timestamp"].apply(
            lambda ts: _compute_solar_geometry(self.config.latitude, self.config.longitude, ts)
        )
        merged["solar_elevation"] = solar.apply(lambda x: x[0])
        merged["solar_azimuth"] = solar.apply(lambda x: x[1])

        return merged.reset_index(drop=True)

    def persist(self, frame: pd.DataFrame) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        if frame.empty:
            LOGGER.warning("No weather records collected.")
            return

        if self.config.storage == "sqlite":
            with sqlite3.connect(self.config.sqlite_path) as conn:
                frame.to_sql("weather_observations", conn, index=False, if_exists="append")
            return

        frame = frame.copy()
        frame["partition_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        for partition, partition_df in frame.groupby("partition_date"):
            folder = self.config.output_root / f"date={partition}"
            folder.mkdir(parents=True, exist_ok=True)
            filename = folder / f"weather_{partition}.parquet"
            partition_df.drop(columns=["partition_date"]).to_parquet(filename, index=False)


async def collect_recent_weather(
    latitude: float,
    longitude: float,
    hours_back: int,
    storage: str,
    output_root: Path = Path("data/raw/weather"),
) -> pd.DataFrame:
    end_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=hours_back)
    config = CollectorConfig(
        latitude=latitude,
        longitude=longitude,
        start_time=start_time,
        end_time=end_time,
        storage=storage,
        output_root=output_root,
        owm_api_key=os.getenv("OPENWEATHERMAP_API_KEY"),
    )
    collector = WeatherCollector(config)
    frame = await collector.collect()
    collector.persist(frame)
    return frame


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LuxAeterna weather and solar collector")
    parser.add_argument("--lat", type=float, default=float(os.getenv("PHOTO_LAT", "40.7128")))
    parser.add_argument("--lon", type=float, default=float(os.getenv("PHOTO_LON", "-74.0060")))
    parser.add_argument("--lookback-hours", type=int, default=72)
    parser.add_argument("--storage", choices=["parquet", "sqlite"], default="parquet")
    parser.add_argument("--output-root", default="data/raw/weather")
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()

    end_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=args.lookback_hours)

    config = CollectorConfig(
        latitude=args.lat,
        longitude=args.lon,
        start_time=_to_utc(start_time),
        end_time=_to_utc(end_time),
        storage=args.storage,
        output_root=Path(args.output_root),
        sqlite_path=Path(args.output_root) / "weather.db",
        owm_api_key=os.getenv("OPENWEATHERMAP_API_KEY"),
    )

    collector = WeatherCollector(config)
    frame = asyncio.run(collector.collect())
    collector.persist(frame)
    LOGGER.info("Collected %s weather observations", len(frame))


if __name__ == "__main__":
    main()
