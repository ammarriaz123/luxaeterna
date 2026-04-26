from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("luxaeterna.data.labeller")


@dataclass(slots=True)
class AlqsWeights:
    saturation: float = 0.4
    contrast: float = 0.3
    warm_ratio: float = 0.3


def compute_alqs(image: np.ndarray, weights: AlqsWeights | None = None) -> float:
    if weights is None:
        weights = AlqsWeights()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    saturation_norm = float(np.mean(hsv[:, :, 1]) / 255.0)
    contrast_norm = float(np.std(gray) / 128.0)
    contrast_norm = float(np.clip(contrast_norm, 0.0, 1.0))

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    warm_mask = ((hue <= 25) | (hue >= 160)) & (sat >= 50) & (val >= 40)
    warm_ratio = float(np.mean(warm_mask.astype(np.float32)))

    score = 100.0 * (
        weights.saturation * saturation_norm
        + weights.contrast * contrast_norm
        + weights.warm_ratio * warm_ratio
    )
    return float(np.clip(score, 0.0, 100.0))


def _score_image(path: Path, weights: AlqsWeights) -> dict[str, float | str]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    score = compute_alqs(image, weights)
    timestamp = _timestamp_from_path(path)
    return {
        "image_path": str(path),
        "timestamp": timestamp,
        "alqs": score,
    }


def _timestamp_from_path(path: Path) -> str | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None

    hhmm = parts[-1]
    if len(hhmm) != 4 or not hhmm.isdigit():
        return None

    folder_name = path.parent.name
    try:
        date_val = pd.to_datetime(folder_name, format="%Y-%m-%d")
        ts = date_val.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0)
        return ts.tz_localize("UTC").isoformat()
    except Exception:
        return None


def label_directory(input_dir: Path, output_path: Path, max_workers: int = 8) -> pd.DataFrame:
    image_paths = sorted(input_dir.rglob("*.jpg")) + sorted(input_dir.rglob("*.jpeg"))
    if not image_paths:
        LOGGER.warning("No frames found under %s", input_dir)
        return pd.DataFrame(columns=["image_path", "timestamp", "alqs"])

    weights = AlqsWeights()
    rows: list[dict[str, float | str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_score_image, path, weights) for path in image_paths]
        for future in futures:
            try:
                rows.append(future.result())
            except Exception as exc:
                LOGGER.warning("Skipping frame due to processing error: %s", exc)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)

    return frame


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ALQS labels from webcam frames")
    parser.add_argument("--input-dir", default="data/raw/webcam")
    parser.add_argument("--output-path", default="data/processed/alqs_labels.parquet")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()
    frame = label_directory(Path(args.input_dir), Path(args.output_path), args.workers)
    LOGGER.info("Computed ALQS labels for %s frames", len(frame))


if __name__ == "__main__":
    main()
