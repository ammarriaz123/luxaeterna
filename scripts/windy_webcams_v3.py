from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests

DEFAULT_ENDPOINT = "https://api.windy.com/webcams/api/v3/webcams"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call Windy Webcams API v3 and print results")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (e.g., 40.7128)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (e.g., -74.0060)")
    parser.add_argument("--radius", type=int, default=50, help="Radius in km")
    parser.add_argument("--api-key", default=os.getenv("WEBCAMS_API_KEY", ""))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    return parser


def _error(message: str) -> None:
    print(f"[windy] {message}", file=sys.stderr)


def fetch_webcams(endpoint: str, api_key: str, lat: float, lon: float, radius: int) -> dict[str, Any]:
    params = {
        "lat": lat,
        "lon": lon,
        "radius": radius,
        "include": "images,player,location",
    }
    headers = {"x-windy-api-key": api_key}

    response = requests.get(endpoint, params=params, headers=headers, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        _error(f"HTTP error {response.status_code}: {exc}")
        _error(f"Response body: {response.text}")
        raise

    try:
        return response.json()
    except ValueError as exc:
        _error("Failed to decode JSON response")
        _error(f"Response body: {response.text}")
        raise RuntimeError("Invalid JSON response") from exc


def main() -> None:
    args = _build_parser().parse_args()

    if not args.api_key:
        _error("Missing API key. Provide --api-key or set WEBCAMS_API_KEY.")
        sys.exit(1)

    try:
        payload = fetch_webcams(args.endpoint, args.api_key, args.lat, args.lon, args.radius)
    except Exception:
        sys.exit(1)

    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
