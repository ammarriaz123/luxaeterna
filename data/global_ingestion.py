from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data.labeller import compute_alqs
from data.webcam_discovery import WebcamMeta, ensure_webcam_cache, iter_webcam_cache

LOGGER = logging.getLogger("luxaeterna.data.global_ingestion")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "visibility",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
]


@dataclass(slots=True)
class IngestionConfig:
    sample_size: int
    cache_path: Path
    output_path: Path
    image_output_dir: Path
    max_concurrency: int = 8
    max_time_delta_minutes: int = 90
    skip_alqs: bool = False
    max_webcams: int = 1000
    refresh_cache: bool = False


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _safe_get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _download_image(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
    except httpx.HTTPError as exc:
        LOGGER.warning("Image download failed: %s (%s)", url, exc)
        return None


def _save_image(image_bytes: bytes, output_dir: Path, webcam_id: str, timestamp: datetime) -> Path:
    date_folder = output_dir / timestamp.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    filename = f"{webcam_id}_{timestamp.strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = date_folder / filename
    image_path.write_bytes(image_bytes)
    return image_path


def _compute_alqs_from_path(image_path: Path) -> float | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        LOGGER.warning("Unable to read image for ALQS: %s", image_path)
        return None
    try:
        return float(compute_alqs(image))
    except Exception as exc:
        LOGGER.warning("ALQS computation failed for %s (%s)", image_path, exc)
        return None


def _payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    frame = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for field in WEATHER_FIELDS:
        frame[field] = hourly.get(field, [np.nan] * len(frame))
    return frame


def _select_nearest_weather(
    frame: pd.DataFrame, target_time: datetime, max_delta_minutes: int
) -> dict[str, Any] | None:
    if frame.empty:
        return None

    target = pd.Timestamp(target_time)
    frame = frame.copy()
    frame["delta"] = (frame["timestamp"] - target).abs()
    row = frame.loc[frame["delta"].idxmin()]

    if row["delta"] > pd.Timedelta(minutes=max_delta_minutes):
        return None

    return {
        "weather_timestamp": row["timestamp"].to_pydatetime().replace(tzinfo=timezone.utc),
        "temperature": row.get("temperature_2m"),
        "relative_humidity": row.get("relative_humidity_2m"),
        "visibility": row.get("visibility"),
        "cloud_cover_low": row.get("cloud_cover_low"),
        "cloud_cover_mid": row.get("cloud_cover_mid"),
        "cloud_cover_high": row.get("cloud_cover_high"),
        "weather_code": row.get("weather_code"),
    }


async def _fetch_weather_for_timestamp(
    client: httpx.AsyncClient, latitude: float, longitude: float, timestamp: datetime, max_delta_minutes: int
) -> dict[str, Any] | None:
    date_value = timestamp.date().isoformat()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_value,
        "end_date": date_value,
        "hourly": WEATHER_FIELDS,
        "timezone": "UTC",
    }

    payload = await _safe_get(client, OPEN_METEO_ARCHIVE_URL, params)
    frame = _payload_to_frame(payload)
    weather = _select_nearest_weather(frame, timestamp, max_delta_minutes)
    if weather is not None:
        return weather

    payload = await _safe_get(client, OPEN_METEO_FORECAST_URL, params)
    frame = _payload_to_frame(payload)
    return _select_nearest_weather(frame, timestamp, max_delta_minutes)


def _validate_location(meta: WebcamMeta) -> bool:
    if meta.latitude == 0.0 and meta.longitude == 0.0:
        return False
    if not (-90.0 <= meta.latitude <= 90.0):
        return False
    if not (-180.0 <= meta.longitude <= 180.0):
        return False
    return True


async def _process_webcam(
    meta: WebcamMeta,
    config: IngestionConfig,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    if not _validate_location(meta):
        LOGGER.warning("Skipping webcam with invalid location: %s", meta.webcam_id)
        return None

    if not meta.image_url:
        LOGGER.warning("Skipping webcam without image URL: %s", meta.webcam_id)
        return None

    async with semaphore:
        timestamp = datetime.now(timezone.utc)
        image_bytes = await _download_image(client, meta.image_url)
        if not image_bytes:
            return None

        image_path = _save_image(image_bytes, config.image_output_dir, meta.webcam_id, timestamp)

        alqs = None
        if not config.skip_alqs:
            alqs = await asyncio.to_thread(_compute_alqs_from_path, image_path)

        weather = await _fetch_weather_for_timestamp(
            client=client,
            latitude=meta.latitude,
            longitude=meta.longitude,
            timestamp=timestamp,
            max_delta_minutes=config.max_time_delta_minutes,
        )
        if weather is None:
            LOGGER.warning("Weather unavailable for webcam %s", meta.webcam_id)
            return None

        row: dict[str, Any] = {
            "timestamp": timestamp,
            "latitude": meta.latitude,
            "longitude": meta.longitude,
            "webcam_id": meta.webcam_id,
            "image_path": str(image_path),
            "image_url": meta.image_url,
            "alqs": alqs,
        }
        row.update(weather)
        return row


def _write_dataset(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if not rows:
        LOGGER.warning("No valid rows to write.")
        return None

    frame = pd.DataFrame(rows)
    output_path = output_path.expanduser()

    timestamp_label = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    if output_path.suffix == ".csv":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists()
        frame.to_csv(output_path, mode="a", header=write_header, index=False)
        return output_path

    if output_path.suffix == ".parquet":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            batch_path = output_path.with_name(f"{output_path.stem}_batch_{timestamp_label}.parquet")
            frame.to_parquet(batch_path, index=False)
            return batch_path
        frame.to_parquet(output_path, index=False)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    batch_path = output_path / f"batch_{timestamp_label}.parquet"
    frame.to_parquet(batch_path, index=False)
    return batch_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Global webcam ingestion and dataset builder")
    parser.add_argument("--cache-path", default="data/webcams.json")
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("SAMPLE_SIZE", "50")))
    parser.add_argument("--max-webcams", type=int, default=int(os.getenv("MAX_WEBCAMS", "1000")))
    parser.add_argument(
        "--output-path",
        default=os.getenv("DATA_OUTPUT_PATH", "data/processed/global_dataset"),
    )
    parser.add_argument("--image-output-dir", default="data/raw/global_webcam_images")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-delta-minutes", type=int, default=90)
    parser.add_argument("--skip-alqs", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interval", type=int, default=int(os.getenv("INGEST_INTERVAL_SECONDS", "60")), help="Interval in seconds between ingestion batches")
    parser.add_argument("--max-cycles", type=int, default=None, help="Maximum number of cycles to run")
    return parser


def ingest_batch(batch: list[WebcamMeta], config: IngestionConfig) -> tuple[int, int]:
    """Runs a single ingestion batch, returning (successes, failures)."""
    if not batch:
        return 0, 0

    async def _run_batch() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(config.max_concurrency)
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = [
                asyncio.create_task(_process_webcam(meta, config, client, semaphore))
                for meta in batch
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            for res in raw_results:
                if isinstance(res, Exception):
                    LOGGER.error("Webcam processing crashed: %s", res)
                elif res is not None:
                    valid_results.append(res)
            return valid_results

    try:
        results = asyncio.run(_run_batch())
        successes = len(results)
        failures = len(batch) - successes
        
        output_file = _write_dataset(results, config.output_path)
        if output_file:
            LOGGER.info("Wrote %s rows to %s", successes, output_file)
        else:
            LOGGER.warning("No dataset written; all samples failed validation.")
            
        return successes, failures
    except Exception as exc:
        LOGGER.error("Batch run failed critically: %s", exc, exc_info=True)
        return 0, len(batch)


def main() -> None:
    load_dotenv()
    configure_logging()
    args = _build_arg_parser().parse_args()

    config = IngestionConfig(
        sample_size=max(args.sample_size, 1),
        cache_path=Path(args.cache_path),
        output_path=Path(args.output_path),
        image_output_dir=Path(args.image_output_dir),
        max_concurrency=max(args.max_concurrency, 1),
        max_time_delta_minutes=max(args.max_delta_minutes, 1),
        skip_alqs=bool(args.skip_alqs),
        max_webcams=max(args.max_webcams, 1),
        refresh_cache=bool(args.refresh_cache),
    )

    try:
        LOGGER.info("Checking and expanding webcam pool to maintain global diversity...")
        from data.webcam_discovery import expand_webcam_pool
        pool_size = expand_webcam_pool(
            cache_path=config.cache_path,
            target_count=1000,
            api_key=os.getenv("WEBCAMS_API_KEY"),
        )
        LOGGER.info("Webcam pool ready. Size: %d", pool_size)
    except Exception as exc:
        LOGGER.error("Failed to initialize or expand webcam cache: %s", exc)
        return

    # Valid webcams from cache
    valid_webcams = []
    for meta in iter_webcam_cache(config.cache_path):
        if meta.webcam_id and meta.image_url and _validate_location(meta):
            valid_webcams.append(meta)

    if not valid_webcams:
        LOGGER.error("No valid webcams found in cache. Cannot start ingestion.")
        return

    LOGGER.info("Loaded %d valid webcams from registry.", len(valid_webcams))

    rng = random.Random(args.seed)
    
    interval = args.interval
    max_cycles = args.max_cycles
    
    LOGGER.info("Starting continuous ingestion loop. Interval: %s seconds", interval)

    cycles = 0
    try:
        while True:
            cycle_start = datetime.now(timezone.utc)
            start_time = time.monotonic()
            
            LOGGER.info("Starting ingestion cycle %d with %d webcams", cycles + 1, len(valid_webcams))
            
            # Recalculate weights based on usage_count for this cycle
            # We want to heavily prefer historically underused webcams
            weights = [1.0 / (1.0 + getattr(wc, 'usage_count', 0)) for wc in valid_webcams]
            
            # Since user wants 10 per cycle, ignore old batching entirely
            # config.sample_size should be explicitly 10 as per API constraints
            cycle_sample_size = min(10, len(valid_webcams))
            
            # Perform Weighted Random Sampling without replacement
            # random.choices does replacement natively, so we simulate without replacement:
            import copy
            pool_copy = list(zip(valid_webcams, weights))
            
            selected_batch = []
            for _ in range(cycle_sample_size):
                if not pool_copy:
                    break
                w_list = [w for _, w in pool_copy]
                chosen = random.choices(pool_copy, weights=w_list, k=1)[0]
                selected_batch.append(chosen[0])
                pool_copy.remove(chosen)
            
            cycle_successes = 0
            cycle_failures = 0
            
            LOGGER.info("Processing cycle %d (%d selected webcams via weighted sampling)", cycles + 1, len(selected_batch))
            try:
                successes, failures = ingest_batch(selected_batch, config)
            except Exception as e:
                LOGGER.error("Unexpected error during batch: %s", e, exc_info=True)
                successes, failures = 0, 0
                
            cycle_successes += successes
            cycle_failures += failures
            
            # Update usage counts
            for wc in selected_batch:
                wc.usage_count = getattr(wc, 'usage_count', 0) + 1
            
            # Persist updated usage count back to json 
            with config.cache_path.open("w", encoding="utf-8") as handle:
                import json
                for wc in valid_webcams:
                    handle.write(json.dumps(wc.to_dict()) + "\n")
                    
            execution_time = time.monotonic() - start_time
            
            LOGGER.info(
                "Cycle %d Completed | Timestamp: %s | Execution Time: %.2fs | Total Successes: %d | Total Failures: %d",
                cycles + 1,
                cycle_start.isoformat(),
                execution_time,
                cycle_successes,
                cycle_failures
            )
            
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                LOGGER.info("Reached max cycles (%d). Exiting.", max_cycles)
                break
                
            sleep_time = max(0.0, interval - execution_time)
            if sleep_time > 0:
                LOGGER.info("Sleeping %.2f seconds before next cycle...", sleep_time)
                time.sleep(sleep_time)
            else:
                LOGGER.warning("Cycle execution time (%.2fs) exceeded interval (%ds). Running next cycle immediately.", execution_time, interval)
                
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received. Shutting down gracefully.")


if __name__ == "__main__":
    main()
