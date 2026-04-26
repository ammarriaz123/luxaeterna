from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

LOGGER = logging.getLogger("luxaeterna.data.feature_engineer")

CONTINUOUS_FEATURES = [
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity",
    "pm25",
    "visibility",
    "temperature",
]

WEATHER_STATE_MAP = {
    0: "clear",
    1: "clear",
    2: "partly_cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    61: "rain",
    63: "rain",
    65: "rain",
    71: "snow",
    73: "snow",
    75: "snow",
    95: "storm",
}


@dataclass(slots=True)
class FeatureArtifacts:
    scaler: MinMaxScaler
    weather_encoder: OneHotEncoder
    alqs_scaler: MinMaxScaler


def _read_weather_frame(weather_path: Path) -> pd.DataFrame:
    if weather_path.is_dir():
        paths = sorted(weather_path.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {weather_path}")
        frames = [pd.read_parquet(path) for path in paths]
        return pd.concat(frames, ignore_index=True)

    if weather_path.suffix == ".db":
        import sqlite3

        with sqlite3.connect(weather_path) as conn:
            frame = pd.read_sql_query("SELECT * FROM weather_observations", conn)
        return frame

    if weather_path.suffix == ".parquet":
        return pd.read_parquet(weather_path)

    raise ValueError(f"Unsupported weather source: {weather_path}")


def _assign_weather_state(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["weather_code"] = frame.get("weather_code", pd.Series(np.nan, index=frame.index))
    frame["weather_state"] = frame["weather_code"].map(WEATHER_STATE_MAP).fillna("unknown")
    return frame


def _merge_with_labels(weather_df: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True, errors="coerce")
    weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if not label_path.exists():
        synthetic = weather_df[["timestamp"]].copy()
        synthetic["alqs"] = np.clip(50 + 0.2 * weather_df.get("solar_elevation", 0).fillna(0), 0, 100)
        labels = synthetic
    else:
        labels = pd.read_parquet(label_path)
        labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True, errors="coerce")
        labels = labels.dropna(subset=["timestamp", "alqs"]).sort_values("timestamp")

    merged = pd.merge_asof(
        weather_df,
        labels[["timestamp", "alqs"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    merged["alqs"] = merged["alqs"].interpolate(limit_direction="both")
    merged = merged.dropna(subset=["alqs"]).reset_index(drop=True)
    return merged


def _resample_to_15min(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.set_index("timestamp").sort_index()
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [col for col in frame.columns if col not in numeric_cols]

    resampled_numeric = frame[numeric_cols].resample("15min").interpolate(method="time").ffill().bfill()
    if other_cols:
        resampled_other = frame[other_cols].resample("15min").ffill().bfill()
        frame = pd.concat([resampled_numeric, resampled_other], axis=1).sort_index()
    else:
        frame = resampled_numeric

    return frame.reset_index()


def _safe_split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 20:
        raise ValueError("Not enough samples to create 70/15/15 split; need at least 20 sequence samples")

    indices = np.arange(len(y))
    try:
        strata = pd.qcut(y, q=5, labels=False, duplicates="drop")
        if len(np.unique(strata)) < 2:
            raise ValueError("Insufficient quintile diversity")
    except Exception:
        jitter = y + np.random.default_rng(42).normal(0, 1e-6, size=len(y))
        strata = pd.qcut(jitter, q=5, labels=False, duplicates="drop")

    strata_arr = np.asarray(strata)

    try:
        train_idx, rem_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=42,
            stratify=strata_arr,
        )
    except ValueError:
        train_idx, rem_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=42,
            stratify=None,
        )

    rem_strata = strata_arr[rem_idx]
    try:
        val_idx, test_idx = train_test_split(
            rem_idx,
            test_size=0.50,
            random_state=42,
            stratify=rem_strata,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            rem_idx,
            test_size=0.50,
            random_state=42,
            stratify=None,
        )
    return train_idx, val_idx, test_idx


def build_sequence_dataset(
    weather_path: Path,
    label_path: Path,
    output_dir: Path,
    window_size: int = 24,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_df = _read_weather_frame(weather_path)
    weather_df = _assign_weather_state(weather_df)
    merged = _merge_with_labels(weather_df, label_path)
    merged = _resample_to_15min(merged)

    for feature in CONTINUOUS_FEATURES:
        merged[feature] = pd.to_numeric(
            merged.get(feature, pd.Series(np.nan, index=merged.index)),
            errors="coerce",
        )
        if merged[feature].isna().all():
            merged[feature] = 0.0
        else:
            merged[feature] = merged[feature].interpolate(limit_direction="both").ffill().bfill()

    merged["hour"] = merged["timestamp"].dt.hour + merged["timestamp"].dt.minute / 60.0
    merged["sin_time"] = np.sin(2 * np.pi * merged["hour"] / 24.0)
    merged["cos_time"] = np.cos(2 * np.pi * merged["hour"] / 24.0)

    weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    weather_encoder.fit(merged[["weather_state"]])

    scaler = MinMaxScaler()
    scaled_continuous = scaler.fit_transform(merged[CONTINUOUS_FEATURES])

    alqs_scaler = MinMaxScaler()
    merged["alqs_norm"] = alqs_scaler.fit_transform(merged[["alqs"]])

    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    y_prev: list[float] = []
    seq_times: list[str] = []
    weather_state_for_y: list[str] = []
    sin_time_for_y: list[float] = []
    cos_time_for_y: list[float] = []

    for idx in range(window_size, len(merged)):
        x_seq.append(scaled_continuous[idx - window_size : idx])
        y_seq.append(float(merged.iloc[idx]["alqs"]))
        y_prev.append(float(merged.iloc[idx - 1]["alqs"]))
        seq_times.append(merged.iloc[idx]["timestamp"].isoformat())
        weather_state_for_y.append(str(merged.iloc[idx]["weather_state"]))
        sin_time_for_y.append(float(merged.iloc[idx]["sin_time"]))
        cos_time_for_y.append(float(merged.iloc[idx]["cos_time"]))

    x = np.asarray(x_seq, dtype=np.float32)
    y = np.asarray(y_seq, dtype=np.float32)
    baseline_prev = np.asarray(y_prev, dtype=np.float32)

    train_idx, val_idx, test_idx = _safe_split_indices(y)

    np.savez_compressed(
        output_dir / "sequence_dataset.npz",
        X_train=x[train_idx],
        y_train=y[train_idx],
        X_val=x[val_idx],
        y_val=y[val_idx],
        X_test=x[test_idx],
        y_test=y[test_idx],
        baseline_prev_test=baseline_prev[test_idx],
    )

    classifier_frame = pd.DataFrame(
        {
            "timestamp": seq_times,
            "alqs_norm": alqs_scaler.transform(pd.DataFrame({"alqs": y})).reshape(-1),
            "weather_state": weather_state_for_y,
            "sin_time": sin_time_for_y,
            "cos_time": cos_time_for_y,
        }
    )

    domain_genres = []
    for _, row in classifier_frame.iterrows():
        if row["alqs_norm"] > 0.75 and row["weather_state"] in {"clear", "partly_cloudy"}:
            domain_genres.append("golden_hour")
        elif row["weather_state"] in {"fog", "rain", "storm"}:
            domain_genres.append("moody")
        elif row["alqs_norm"] < 0.25:
            domain_genres.append("night_astro")
        elif row["weather_state"] in {"cloudy", "snow"}:
            domain_genres.append("landscape")
        else:
            domain_genres.append("street")
    classifier_frame["genre"] = domain_genres

    classifier_frame.to_parquet(output_dir / "classifier_features.parquet", index=False)

    latest_window = x[-1] if len(x) else np.zeros((window_size, len(CONTINUOUS_FEATURES)), dtype=np.float32)
    np.save(output_dir / "latest_window.npy", latest_window)

    joblib.dump(
        FeatureArtifacts(
            scaler=scaler,
            weather_encoder=weather_encoder,
            alqs_scaler=alqs_scaler,
        ),
        output_dir / "feature_artifacts.joblib",
    )

    metadata = {
        "window_size": window_size,
        "continuous_features": CONTINUOUS_FEATURES,
        "weather_categories": weather_encoder.categories_[0].tolist(),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LuxAeterna sequence and classifier features")
    parser.add_argument("--weather-path", default="data/raw/weather")
    parser.add_argument("--label-path", default="data/processed/alqs_labels.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-size", type=int, default=24)
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
        label_path=Path(args.label_path),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
    )
    LOGGER.info("Feature engineering complete. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
