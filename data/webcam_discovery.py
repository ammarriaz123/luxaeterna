from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger("luxaeterna.data.webcam_discovery")

WINDY_V3_ENDPOINT = "https://api.windy.com/webcams/api/v3/webcams"
WINDY_API_BASES = [
    "https://api.windy.com/webcams/api/v2/list/orderby=popularity",
    "https://api.windy.com/api/webcams/v2/list/orderby=popularity",
    "https://api.windy.com/webcams/api/v2/list",
    "https://api.windy.com/api/webcams/v2/list",
]

DEFAULT_DISCOVERY_RADIUS_KM = 250
DEFAULT_DISCOVERY_SEEDS = [
    # Global sweep coordinates (approx every 60 degrees longitude, 30 degrees latitude)
    (60.0, -120.0), (60.0, -60.0), (60.0, 0.0), (60.0, 60.0), (60.0, 120.0), (60.0, 180.0),
    (30.0, -120.0), (30.0, -60.0), (30.0, 0.0), (30.0, 60.0), (30.0, 120.0), (30.0, 180.0),
    (0.0, -120.0), (0.0, -60.0), (0.0, 0.0), (0.0, 60.0), (0.0, 120.0), (0.0, 180.0),
    (-30.0, -120.0), (-30.0, -60.0), (-30.0, 0.0), (-30.0, 60.0), (-30.0, 120.0), (-30.0, 180.0),
    # Additional high density areas (Europe, North America, Japan)
    (40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298),
    (51.5074, -0.1278), (48.8566, 2.3522), (41.9028, 12.4964), (52.5200, 13.4050),
    (35.6762, 139.6503), (34.6937, 135.5023),
]


@dataclass(slots=True)
class WebcamMeta:
    webcam_id: str
    image_url: str
    latitude: float
    longitude: float
    provider: str
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "webcam_id": self.webcam_id,
            "image_url": self.image_url,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "provider": self.provider,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WebcamMeta":
        return cls(
            webcam_id=str(data.get("webcam_id", "")),
            image_url=str(data.get("image_url", "")),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            provider=str(data.get("provider", "unknown")),
            usage_count=int(data.get("usage_count", 0)),
        )


def _is_valid(meta: WebcamMeta) -> bool:
    if not meta.webcam_id or not meta.image_url:
        return False
    if not (-90.0 <= meta.latitude <= 90.0):
        return False
    if not (-180.0 <= meta.longitude <= 180.0):
        return False
    return True


def _is_retryable_httpx_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    if isinstance(exc, httpx.HTTPError):
        return True
    return False


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(4),
    retry=retry_if_exception(_is_retryable_httpx_error),
)
async def _safe_get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _safe_get_v3(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], api_key: str
) -> dict[str, Any]:
    headers = {"x-windy-api-key": api_key}
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def _extract_image_url(images: dict[str, Any] | str | None) -> str | None:
    if not images:
        return None

    if isinstance(images, str):
        return images if images else None

    preferred_groups = ["current", "daylight", "year", "month"]
    preferred_keys = ["preview", "thumbnail", "toenail", "image", "link"]

    for group_key in preferred_groups:
        group = images.get(group_key)
        if isinstance(group, dict):
            for key in preferred_keys:
                value = group.get(key)
                if isinstance(value, str) and value:
                    return value

    for group in images.values():
        if not isinstance(group, dict):
            continue
        for key in preferred_keys:
            value = group.get(key)
            if isinstance(value, str) and value:
                return value

    return None


def _parse_windy_payload(payload: dict[str, Any]) -> list[WebcamMeta]:
    result = payload.get("result", {})
    webcams = result.get("webcams") or result.get("webcam") or []
    parsed: list[WebcamMeta] = []

    for entry in webcams:
        if not isinstance(entry, dict):
            continue
        webcam_id = entry.get("id") or entry.get("webcamId") or entry.get("webcam_id")
        location = entry.get("location", {}) if isinstance(entry.get("location"), dict) else {}
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        images_source: dict[str, Any] | str | None = None
        if isinstance(entry.get("images"), dict):
            images_source = entry.get("images")
        elif isinstance(entry.get("image"), (dict, str)):
            images_source = entry.get("image")

        image_url = _extract_image_url(images_source)

        if webcam_id is None or latitude is None or longitude is None or not image_url:
            continue

        meta = WebcamMeta(
            webcam_id=str(webcam_id),
            image_url=image_url,
            latitude=float(latitude),
            longitude=float(longitude),
            provider="windy",
        )
        if _is_valid(meta):
            parsed.append(meta)

    return parsed


def _extract_webcam_list(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    for key in ("webcams", "data", "items", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for subkey in ("webcams", "items", "data"):
                subvalue = value.get(subkey)
                if isinstance(subvalue, list):
                    return [item for item in subvalue if isinstance(item, dict)]
    return []


def _extract_lat_lon(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    location = entry.get("location") if isinstance(entry.get("location"), dict) else None
    if location:
        lat = location.get("latitude") or location.get("lat")
        lon = location.get("longitude") or location.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    coordinates = entry.get("coordinates") if isinstance(entry.get("coordinates"), dict) else None
    if coordinates:
        lat = coordinates.get("latitude") or coordinates.get("lat")
        lon = coordinates.get("longitude") or coordinates.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    position = entry.get("position") if isinstance(entry.get("position"), dict) else None
    if position:
        lat = position.get("latitude") or position.get("lat")
        lon = position.get("longitude") or position.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    return None, None


def _extract_image_from_entry(entry: dict[str, Any]) -> str | None:
    for key in ("images", "image", "thumbnail", "preview", "player"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            url = _extract_image_url(value)
            if url:
                return url
    return None


def _parse_windy_v3_payload(payload: dict[str, Any] | list[Any]) -> list[WebcamMeta]:
    webcams = _extract_webcam_list(payload)
    parsed: list[WebcamMeta] = []

    for entry in webcams:
        webcam_id = entry.get("id") or entry.get("webcamId") or entry.get("webcam_id")
        lat, lon = _extract_lat_lon(entry)
        image_url = _extract_image_from_entry(entry)

        if webcam_id is None or lat is None or lon is None or not image_url:
            continue

        meta = WebcamMeta(
            webcam_id=str(webcam_id),
            image_url=image_url,
            latitude=float(lat),
            longitude=float(lon),
            provider="windy_v3",
        )
        if _is_valid(meta):
            parsed.append(meta)

    return parsed


async def _discover_windy_to_cache(
    cache_path: Path,
    max_webcams: int,
    api_key: str | None,
    page_limit: int = 50,
) -> int:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_ids: set[str] = set()
    mode = "a" if cache_path.exists() else "w"
    
    # Pre-populate seen_ids from existing cache to prevent duplicates when appending
    if cache_path.exists():
        try:
            for meta in iter_webcam_cache(cache_path):
                seen_ids.add(meta.webcam_id)
        except Exception:
            pass

    total_written = 0

    async with httpx.AsyncClient(timeout=30) as client:
        with cache_path.open(mode, encoding="utf-8") as handle:
            if api_key:
                v3_failed = False
                for lat, lon in DEFAULT_DISCOVERY_SEEDS:
                    if total_written >= max_webcams or v3_failed:
                        break
                        
                    LOGGER.info("Sweeping webcams near %s, %s", lat, lon)
                    offset = 0
                    
                    while total_written < max_webcams:
                        params = {
                            "nearby": f"{lat},{lon},{DEFAULT_DISCOVERY_RADIUS_KM}",
                            "limit": page_limit,
                            "offset": offset,
                            "include": "location",
                        }

                        try:
                            payload = await _safe_get_v3(client, WINDY_V3_ENDPOINT, params, api_key)
                        except httpx.HTTPStatusError as exc:
                            status_code = exc.response.status_code
                            if status_code in {401, 403}:
                                LOGGER.warning("Windy v3 auth failed (%s); falling back.", status_code)
                                v3_failed = True
                                break
                            if status_code in {400, 404}:
                                break # Usually means end of pagination or invalid parameters
                            raise

                        webcams = payload.get("webcams", [])
                        if not webcams:
                            break

                        new_added = 0
                        for entry in webcams:
                            webcam_id = str(entry.get("webcamId", ""))
                            if not webcam_id or webcam_id in seen_ids:
                                continue
                                
                            image_url = f"https://imgproxy.windy.com/_/preview/plain/current/{webcam_id}/original.jpg"
                            w_lat = entry.get("location", {}).get("latitude")
                            w_lon = entry.get("location", {}).get("longitude")
                            
                            if w_lat is None or w_lon is None:
                                continue
                                
                            meta = WebcamMeta(
                                webcam_id=webcam_id,
                                image_url=image_url,
                                latitude=float(w_lat),
                                longitude=float(w_lon),
                                provider="windy_v3",
                            )
                            
                            if _is_valid(meta):
                                seen_ids.add(meta.webcam_id)
                                d = meta.to_dict()
                                d["usage_count"] = 0
                                handle.write(json.dumps(d) + "\n")
                                new_added += 1
                                total_written += 1
                                
                                if total_written >= max_webcams:
                                    break

                        if new_added == 0 or len(webcams) < page_limit:
                            break

                        offset += page_limit
                        await asyncio.sleep(0.5)

    return total_written


def ensure_webcam_cache(
    cache_path: Path,
    max_webcams: int,
    refresh: bool = False,
    api_key: str | None = None,
    page_limit: int = 50,
) -> int:
    if cache_path.exists() and not refresh:
        if cache_path.stat().st_size > 0 and _cache_has_entries(cache_path):
            LOGGER.info("Webcam cache exists at %s; skipping discovery.", cache_path)
            return 0
        LOGGER.warning("Webcam cache at %s is empty; refreshing discovery.", cache_path)

    LOGGER.info("Discovering webcams (provider=windy, max=%s)", max_webcams)
    total_written = asyncio.run(
        _discover_windy_to_cache(
            cache_path=cache_path,
            max_webcams=max_webcams,
            api_key=api_key,
            page_limit=page_limit,
        )
    )
    LOGGER.info("Cached %s webcams to %s", total_written, cache_path)
    return total_written


def _cache_has_entries(cache_path: Path) -> bool:
    try:
        for meta in iter_webcam_cache(cache_path):
            if _is_valid(meta):
                return True
    except Exception:
        return False
    return False


def iter_webcam_cache(cache_path: Path) -> Iterable[WebcamMeta]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with cache_path.open("r", encoding="utf-8") as handle:
        first_line = None
        for line in handle:
            if line.strip():
                first_line = line
                break

        if first_line is None:
            return

        if first_line.lstrip().startswith("["):
            handle.seek(0)
            payload = json.load(handle)
            for item in payload:
                if not isinstance(item, dict):
                    continue
                yield WebcamMeta.from_dict(item)
            return

        yield WebcamMeta.from_dict(json.loads(first_line))
        for line in handle:
            if not line.strip():
                continue
            yield WebcamMeta.from_dict(json.loads(line))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover global webcams and cache metadata")
    parser.add_argument("--cache-path", default="data/webcams.json")
    parser.add_argument("--max-webcams", type=int, default=1000)
    parser.add_argument("--page-limit", type=int, default=50)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("WEBCAMS_API_KEY"))
    return parser


def main() -> None:
    load_dotenv()
    configure_logging()
    args = _build_arg_parser().parse_args()
    cache_path = Path(args.cache_path)
    ensure_webcam_cache(
        cache_path=cache_path,
        max_webcams=max(args.max_webcams, 1),
        refresh=args.refresh,
        api_key=args.api_key,
        page_limit=max(args.page_limit, 1),
    )


if __name__ == "__main__":
    main()
