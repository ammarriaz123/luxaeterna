import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import cv2

import ephem
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import math
from datetime import datetime

LOGGER = logging.getLogger("luxaeterna.data.feature_engineer")

CONTINUOUS_FEATURES = [
    "temperature",
    "relative_humidity",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "total_cloud_cover",
    "visibility",
    "solar_elevation",
    "solar_azimuth",
]

WEATHER_STATE_MAP = {
    0: "clear", 1: "clear", 2: "partly_cloudy", 3: "cloudy",
    45: "fog", 48: "fog", 51: "drizzle", 53: "drizzle", 55: "drizzle",
    61: "rain", 63: "rain", 65: "rain", 71: "snow", 73: "snow", 75: "snow",
    95: "storm",
}

@dataclass(slots=True)
class FeatureArtifacts:
    scaler: MinMaxScaler
    weather_encoder: OneHotEncoder
    alqs_scaler: MinMaxScaler

def _compute_solar(row):
    obs = ephem.Observer()
    obs.lat = str(row['latitude'])
    obs.lon = str(row['longitude'])
    obs.date = row['timestamp']
    sun = ephem.Sun(obs)
    return pd.Series([math.degrees(sun.alt), math.degrees(sun.az)])

def _is_valid_image(img_path_str):
    try:
        path = Path(img_path_str)
        if not path.exists(): return False
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        if np.var(img) < 10.0: return False
        if np.mean(img) < 5.0: return False
        return True
    except Exception:
        return False

def _read_and_clean_data(data_dir: Path) -> pd.DataFrame:
    LOGGER.info("Loading Parquet files from %s...", data_dir)
    paths = sorted(data_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    
    LOGGER.info("Initial loaded rows: %d", len(df))
    # Mandatory fields
    df = df.dropna(subset=["alqs", "temperature", "relative_humidity", "cloud_cover_low", "latitude", "longitude", "timestamp", "image_path"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df['visibility'] = df['visibility'].fillna(df['visibility'].median() if not df['visibility'].isna().all() else 10000.0)

    # Cloud cover
    df["total_cloud_cover"] = df["cloud_cover_low"] + df["cloud_cover_mid"] + df["cloud_cover_high"]

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["webcam_id", "timestamp"])

    # Solar features
    LOGGER.info("Computing solar features...")
    solar_features = df.apply(_compute_solar, axis=1)
    df[["solar_elevation", "solar_azimuth"]] = solar_features

    # Weather states
    df["weather_code"] = df.get("weather_code", pd.Series(np.nan, index=df.index))
    df["weather_state"] = df["weather_code"].map(WEATHER_STATE_MAP).fillna("unknown")

    # Downsample temporally and validate images
    LOGGER.info("Deduplicating and downsampling...")
    df = df.sort_values(["webcam_id", "timestamp"])
    
    # 10 min downsampling
    df["timestamp_rounded"] = df["timestamp"].dt.floor("10min")
    df = df.drop_duplicates(subset=["webcam_id", "timestamp_rounded"]).copy()

    LOGGER.info("Validating images (this may take a bit)...")
    valid_mask = df["image_path"].apply(_is_valid_image)
    df = df[valid_mask]
    LOGGER.info("Rows after full cleaning: %d", len(df))
    
    return df

def build_sequence_dataset(
    weather_path: Path,
    output_dir: Path,
    window_size: int = 24,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _read_and_clean_data(weather_path)
    
    # Sort and group
    df = df.sort_values(["webcam_id", "timestamp"]).reset_index(drop=True)
    
    # Force 15 min resampling to build consistent windows
    LOGGER.info("Resampling to generic 15min intervals per webcam...")
    webcam_dfs = []
    for wid, group in df.groupby("webcam_id"):
        group = group.set_index("timestamp").sort_index()
        # Remove duplicate times completely if any
        group = group[~group.index.duplicated(keep="first")]
        numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = [col for col in group.columns if col not in numeric_cols]

        res_num = group[numeric_cols].resample("15min").interpolate(method="time").ffill().bfill()
        res_oth = group[other_cols].resample("15min").ffill().bfill()
        res_group = pd.concat([res_num, res_oth], axis=1).reset_index()
        res_group["webcam_id"] = wid
        webcam_dfs.append(res_group)
        
    if not webcam_dfs:
        raise ValueError("No viable webcams survived cleaning.")
        
    merged = pd.concat(webcam_dfs, ignore_index=True)
    merged["hour"] = merged["timestamp"].dt.hour + merged["timestamp"].dt.minute / 60.0
    merged["sin_time"] = np.sin(2 * np.pi * merged["hour"] / 24.0)
    merged["cos_time"] = np.cos(2 * np.pi * merged["hour"] / 24.0)
    
    weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    weather_encoder.fit(merged[["weather_state"]])

    scaler = MinMaxScaler()
    scaled_continuous = scaler.fit_transform(merged[CONTINUOUS_FEATURES])

    alqs_scaler = MinMaxScaler()
    merged["alqs_norm"] = alqs_scaler.fit_transform(merged[["alqs"]])
    
    x_seq, y_seq, y_prev_seq, seq_times = [], [], [], []
    weather_state_for_y, sin_time_for_y, cos_time_for_y = [], [], []

    for wid, group in merged.groupby("webcam_id"):
        g_idx = group.index.tolist()
        if len(g_idx) <= window_size:
            continue
            
        for i in range(window_size, len(g_idx)):
            start_i = g_idx[i - window_size]
            end_i = g_idx[i]
            x_seq.append(scaled_continuous[start_i:end_i])
            y_seq.append(float(merged.loc[end_i, "alqs"]))
            y_prev_seq.append(float(merged.loc[g_idx[i - 1], "alqs"]))
            
            seq_times.append(merged.loc[end_i, "timestamp"].isoformat())
            weather_state_for_y.append(str(merged.loc[end_i, "weather_state"]))
            sin_time_for_y.append(float(merged.loc[end_i, "sin_time"]))
            cos_time_for_y.append(float(merged.loc[end_i, "cos_time"]))

    x = np.asarray(x_seq, dtype=np.float32)
    y = np.asarray(y_seq, dtype=np.float32)
    baseline_prev = np.asarray(y_prev_seq, dtype=np.float32)

    if len(y) < 10:
        raise ValueError(f"Too few sequences generated ({len(y)})")

    indices = np.arange(len(y))
    train_idx, rem_idx = train_test_split(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(rem_idx, test_size=0.50, random_state=42)

    np.savez_compressed(
        output_dir / "sequence_dataset.npz",
        X_train=x[train_idx], y_train=y[train_idx],
        X_val=x[val_idx], y_val=y[val_idx],
        X_test=x[test_idx], y_test=y[test_idx],
        baseline_prev_test=baseline_prev[test_idx],
    )

    classifier_frame = pd.DataFrame({
        "timestamp": seq_times,
        "alqs_norm": alqs_scaler.transform(pd.DataFrame({"alqs": y})).reshape(-1),
        "weather_state": weather_state_for_y,
        "sin_time": sin_time_for_y,
        "cos_time": cos_time_for_y,
    })
    
    # Generic genre mapping to avoid errors
    domain_genres = []
    for _, row in classifier_frame.iterrows():
        if row["alqs_norm"] > 0.75: domain_genres.append("golden_hour")
        elif row["alqs_norm"] < 0.25: domain_genres.append("night_astro")
        else: domain_genres.append("street")
    classifier_frame["genre"] = domain_genres
    classifier_frame.to_parquet(output_dir / "classifier_features.parquet", index=False)

    joblib.dump(FeatureArtifacts(scaler=scaler, weather_encoder=weather_encoder, alqs_scaler=alqs_scaler), output_dir / "feature_artifacts.joblib")
    
    metadata = {
        "window_size": window_size, "continuous_features": CONTINUOUS_FEATURES,
        "weather_categories": weather_encoder.categories_[0].tolist(),
        "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))}
    }
    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_sequence_dataset(Path("data/processed/global_dataset"), Path("data/processed"), 24)

if __name__ == "__main__":
    main()
