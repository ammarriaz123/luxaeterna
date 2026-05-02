from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ephem
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger("luxaeterna.data.feature_engineer")

BASE_FEATURES = [
    "temperature",
    "relative_humidity",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "visibility",
]

DERIVED_FEATURES = [
    "total_cloud_cover",
    "delta_time_minutes",
    "solar_elevation",
    "solar_azimuth",
    "clear_sky_index",
]

FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES

CRITICAL_COLUMNS = [
    "webcam_id",
    "timestamp",
    "alqs",
    "temperature",
    "relative_humidity",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
]


@dataclass(slots=True)
class FeatureArtifacts:
    feature_scaler: MinMaxScaler


def _read_weather_frame(weather_path: Path) -> pd.DataFrame:
    if weather_path.is_dir():
        paths = sorted(weather_path.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {weather_path}")
        frames = [pd.read_parquet(path) for path in paths]
        return pd.concat(frames, ignore_index=True)

    if weather_path.suffix == ".parquet":
        return pd.read_parquet(weather_path)

    raise ValueError(f"Unsupported weather source: {weather_path}")


def _normalize_timestamp_column(frame: pd.DataFrame, col: str) -> pd.Series:
    series = frame.get(col)
    if series is None:
        return pd.Series(pd.NaT, index=frame.index)

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        max_val = float(numeric.dropna().abs().max())
        # Heuristic: values above 1e11 are likely UNIX milliseconds.
        unit = "ms" if max_val > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")


def _clean_frame(frame: pd.DataFrame, max_visibility_fill: float = 10000.0) -> pd.DataFrame:
    frame = frame.copy()

    missing_columns = [col for col in CRITICAL_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    frame["timestamp"] = _normalize_timestamp_column(frame, "timestamp")
    if "weather_timestamp" in frame.columns:
        frame["weather_timestamp"] = _normalize_timestamp_column(frame, "weather_timestamp")

    for col in [
        "alqs",
        "temperature",
        "relative_humidity",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "weather_code",
        "visibility",
        "latitude",
        "longitude",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before_drop = len(frame)
    frame = frame.dropna(subset=CRITICAL_COLUMNS)

    if "latitude" in frame.columns and "longitude" in frame.columns:
        frame = frame[
            frame["latitude"].between(-90.0, 90.0)
            & frame["longitude"].between(-180.0, 180.0)
        ]

    if "visibility" not in frame.columns:
        frame["visibility"] = np.nan

    visibility_median = frame["visibility"].median(skipna=True)
    if pd.isna(visibility_median):
        visibility_median = max_visibility_fill
    frame["visibility"] = frame["visibility"].fillna(float(visibility_median))

    frame["total_cloud_cover"] = (
        frame["cloud_cover_low"] + frame["cloud_cover_mid"] + frame["cloud_cover_high"]
    )

    frame = frame.drop_duplicates(subset=["webcam_id", "timestamp"], keep="last")

    LOGGER.info(
        "Dropped %d rows during cleaning; %d rows remain",
        before_drop - len(frame),
        len(frame),
    )

    return frame


def _compute_solar_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute solar elevation, azimuth, and clear-sky index for each observation."""
    frame = frame.copy()
    
    elevations = []
    azimuths = []
    clear_sky_indices = []
    
    for idx, row in frame.iterrows():
        try:
            if pd.isna(row.get("timestamp")) or pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                elevations.append(np.nan)
                azimuths.append(np.nan)
                clear_sky_indices.append(np.nan)
                continue
            
            # Create observer at webcam location
            observer = ephem.Observer()
            observer.lat = str(float(row["latitude"]))
            observer.lon = str(float(row["longitude"]))
            observer.date = pd.Timestamp(row["timestamp"]).to_pydatetime()
            
            # Compute solar position
            sun = ephem.Sun(observer)
            elevation_rad = float(sun.alt)
            azimuth_rad = float(sun.az)
            
            # Convert from radians to degrees
            elevation_deg = np.degrees(elevation_rad)
            azimuth_deg = np.degrees(azimuth_rad)
            
            elevations.append(elevation_deg)
            azimuths.append(azimuth_deg)
            
            # Clear-sky index: normalized solar elevation
            # Ranges from 0 (sun below horizon) to 1 (sun at zenith, clear sky)
            clear_sky = max(0.0, np.sin(elevation_rad))
            clear_sky_indices.append(clear_sky)
            
        except Exception as e:
            LOGGER.debug(f"Solar geometry computation failed at index {idx}: {e}")
            elevations.append(np.nan)
            azimuths.append(np.nan)
            clear_sky_indices.append(np.nan)
    
    frame["solar_elevation"] = elevations
    frame["solar_azimuth"] = azimuths
    frame["clear_sky_index"] = clear_sky_indices
    
    # Forward fill missing values per webcam to maintain continuity
    for col in ["solar_elevation", "solar_azimuth", "clear_sky_index"]:
        frame[col] = frame.groupby("webcam_id")[col].fillna(method="ffill").groupby("webcam_id").fillna(method="bfill")
    
    return frame


def _build_webcam_splits(webcam_ids: list[str]) -> tuple[set[str], set[str], set[str]]:
    ids = np.array(sorted(set(webcam_ids)))
    if len(ids) < 3:
        raise ValueError("Need at least 3 unique webcams for train/val/test split")

    train_ids, rem_ids = train_test_split(ids, test_size=0.30, random_state=42, shuffle=True)
    val_ids, test_ids = train_test_split(rem_ids, test_size=0.50, random_state=42, shuffle=True)

    return set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())


def _build_sequences(
    frame: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[pd.Timestamp]]:
    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    y_prev: list[float] = []
    seq_webcam_ids: list[str] = []
    seq_target_timestamps: list[pd.Timestamp] = []

    grouped = frame.groupby("webcam_id", sort=False)

    skipped_short = 0
    skipped_missing = 0

    for webcam_id, group in grouped:
        group = group.sort_values("timestamp").copy()

        # Irregular-interval signal: elapsed minutes from previous timestamp.
        group["delta_time_minutes"] = group["timestamp"].diff().dt.total_seconds().div(60.0)
        median_delta = group["delta_time_minutes"].median(skipna=True)
        if pd.isna(median_delta):
            median_delta = 30.0
        group["delta_time_minutes"] = group["delta_time_minutes"].fillna(float(median_delta))

        group = group.dropna(subset=FEATURE_COLUMNS + ["alqs"])

        if len(group) < (window_size + 1):
            skipped_short += 1
            continue

        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = group["alqs"].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].to_numpy()

        for end_idx in range(window_size, len(group)):
            start_idx = end_idx - window_size
            x_window = values[start_idx:end_idx]
            if x_window.shape[0] != window_size:
                skipped_missing += 1
                continue

            x_seq.append(x_window)
            y_seq.append(float(targets[end_idx]))
            y_prev.append(float(targets[end_idx - 1]))
            seq_webcam_ids.append(str(webcam_id))
            seq_target_timestamps.append(pd.Timestamp(timestamps[end_idx]))

    LOGGER.info(
        "Built %d sequences across %d webcams (skipped short=%d, skipped malformed=%d)",
        len(x_seq),
        frame["webcam_id"].nunique(),
        skipped_short,
        skipped_missing,
    )

    if not x_seq:
        raise ValueError("No valid sequences built. Check data quality and window size.")

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(y_prev, dtype=np.float32),
        seq_webcam_ids,
        seq_target_timestamps,
    )


def build_sequence_dataset(
    weather_path: Path,
    output_dir: Path,
    window_size: int = 12,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_weather_frame(weather_path)
    frame = _clean_frame(raw)
    
    # Add solar geometry features
    LOGGER.info("Computing solar geometry features (elevation, azimuth, clear-sky index)...")
    frame = _compute_solar_geometry(frame)

    x, y, baseline_prev, seq_webcam_ids, seq_target_timestamps = _build_sequences(
        frame=frame,
        window_size=window_size,
    )

    feature_scaler = MinMaxScaler()
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_scaled = feature_scaler.fit_transform(x_2d).reshape(original_shape).astype(np.float32)

    train_webcams, val_webcams, test_webcams = _build_webcam_splits(seq_webcam_ids)

    train_idx = np.array([i for i, w in enumerate(seq_webcam_ids) if w in train_webcams], dtype=np.int64)
    val_idx = np.array([i for i, w in enumerate(seq_webcam_ids) if w in val_webcams], dtype=np.int64)
    test_idx = np.array([i for i, w in enumerate(seq_webcam_ids) if w in test_webcams], dtype=np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Split produced an empty partition; cannot continue")

    np.savez_compressed(
        output_dir / "sequence_dataset.npz",
        X_train=x_scaled[train_idx],
        y_train=y[train_idx],
        X_val=x_scaled[val_idx],
        y_val=y[val_idx],
        X_test=x_scaled[test_idx],
        y_test=y[test_idx],
        baseline_prev_test=baseline_prev[test_idx],
    )

    classifier_frame = pd.DataFrame(
        {
            "timestamp": [seq_target_timestamps[i] for i in test_idx],
            "webcam_id": [seq_webcam_ids[i] for i in test_idx],
            "alqs": y[test_idx],
            "baseline_prev": baseline_prev[test_idx],
        }
    )
    classifier_frame.to_parquet(output_dir / "classifier_features.parquet", index=False)

    joblib.dump(
        FeatureArtifacts(feature_scaler=feature_scaler),
        output_dir / "feature_artifacts.joblib",
    )

    metadata = {
        "window_size": window_size,
        "feature_columns": FEATURE_COLUMNS,
        "split_by": "webcam_id",
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "webcam_counts": {
            "all": int(len(set(seq_webcam_ids))),
            "train": int(len(train_webcams)),
            "val": int(len(val_webcams)),
            "test": int(len(test_webcams)),
        },
    }
    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Feature engineering complete. Output written to %s", output_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LuxAeterna sequence features from global webcam data")
    parser.add_argument("--weather-path", default="data/processed/global_dataset")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-size", type=int, default=24, help="Sliding window size in timesteps (default: 24 = 24 hours at 1-hour intervals)")
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()

    build_sequence_dataset(
        weather_path=Path(args.weather_path),
        output_dir=Path(args.output_dir),
        window_size=max(2, args.window_size),
    )


if __name__ == "__main__":
    main()
