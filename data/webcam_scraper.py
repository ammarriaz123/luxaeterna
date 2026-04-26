from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import ephem
import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger("luxaeterna.data.webcam_scraper")


def _is_retryable_httpx_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError)):
        return True
    return False


@dataclass(slots=True)
class WebcamScraperConfig:
    archive_url_template: str
    latitude: float
    longitude: float
    start_date: date
    days: int
    output_dir: Path = Path("data/raw/webcam")


def _sun_events_for_date(target_date: date, latitude: float, longitude: float) -> tuple[datetime, datetime]:
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = datetime.combine(target_date, time.min, tzinfo=UTC)

    sunrise = ephem.localtime(observer.next_rising(ephem.Sun())).astimezone(UTC)
    sunset = ephem.localtime(observer.next_setting(ephem.Sun())).astimezone(UTC)
    return sunrise, sunset


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(4),
    retry=retry_if_exception(_is_retryable_httpx_error),
)
async def _download(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


class WebcamScraper:
    def __init__(self, config: WebcamScraperConfig) -> None:
        self.config = config

    def _candidate_timestamps(self) -> list[tuple[datetime, str]]:
        candidates: list[tuple[datetime, str]] = []
        for day_offset in range(self.config.days):
            day = self.config.start_date + timedelta(days=day_offset)
            sunrise, sunset = _sun_events_for_date(day, self.config.latitude, self.config.longitude)
            for offset_min in (-15, 0, 15):
                candidates.append((sunrise + timedelta(minutes=offset_min), "sunrise"))
                candidates.append((sunset + timedelta(minutes=offset_min), "sunset"))
        return candidates

    async def run(self) -> pd.DataFrame:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=20) as client:
            tasks: list[asyncio.Task[tuple[datetime, str, bytes | None, str | None]]] = []
            for ts, event_type in self._candidate_timestamps():
                tasks.append(asyncio.create_task(self._fetch_one(client, ts, event_type)))

            for result in await asyncio.gather(*tasks):
                ts, event_type, payload, source_url = result
                if payload is None or source_url is None:
                    continue

                date_folder = self.config.output_dir / ts.strftime("%Y-%m-%d")
                date_folder.mkdir(parents=True, exist_ok=True)
                filename = date_folder / f"{event_type}_{ts.strftime('%H%M')}.jpg"
                filename.write_bytes(payload)

                rows.append(
                    {
                        "timestamp": ts,
                        "event_type": event_type,
                        "image_path": str(filename),
                        "source_url": source_url,
                    }
                )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("timestamp").reset_index(drop=True)
            frame.to_parquet(self.config.output_dir / "metadata.parquet", index=False)
        return frame

    async def _fetch_one(
        self, client: httpx.AsyncClient, ts: datetime, event_type: str
    ) -> tuple[datetime, str, bytes | None, str | None]:
        timestamp_text = ts.strftime("%Y%m%d%H%M")
        url = self.config.archive_url_template.format(timestamp=timestamp_text)
        try:
            payload = await _download(client, url)
            return ts, event_type, payload, url
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                LOGGER.info("Frame not found (404): %s", url)
            else:
                LOGGER.warning("Frame unavailable: %s (%s)", url, exc)
            return ts, event_type, None, None
        except httpx.HTTPError as exc:
            LOGGER.warning("Frame unavailable: %s (%s)", url, exc)
            return ts, event_type, None, None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch webcam archives near sunrise and sunset")
    parser.add_argument(
        "--archive-url-template",
        default=(
            "https://images-webcams.windy.com/89/{timestamp}/full/89.jpg"
        ),
        help="URL template that includes {timestamp} placeholder in YYYYMMDDHHMM format",
    )
    parser.add_argument("--lat", type=float, default=40.7128)
    parser.add_argument("--lon", type=float, default=-74.0060)
    parser.add_argument("--start-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output-dir", default="data/raw/webcam")
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()

    config = WebcamScraperConfig(
        archive_url_template=args.archive_url_template,
        latitude=args.lat,
        longitude=args.lon,
        start_date=date.fromisoformat(args.start_date),
        days=args.days,
        output_dir=Path(args.output_dir),
    )
    scraper = WebcamScraper(config)
    frame = asyncio.run(scraper.run())
    LOGGER.info("Downloaded %s webcam frames", len(frame))


if __name__ == "__main__":
    main()
