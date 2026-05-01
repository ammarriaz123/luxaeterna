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
    try:
        obs = ephem.Observer()
        obs.lat = str(row['latitude'])
        obs.lon = str(row['longitude'])
        obs.date = row['timestamp']
        sun = ephem.Sun(obs)
        return pd.Series([math.degrees(sun.alt), math.degrees(sun.az)])
    except Exception:
        return pd.Series([0.0, 0.0])

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

def build_sequence_dataset(
    weather_path: Path,
    output_dir: Path,
    window_size: int = 24,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading Parquet files from %s...", weather_path)
    paths = sorted(weather_path.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No parquet files found")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    
    df = df.dropna(subset=["alqs", "temperature", "relative_humidity", "cloud_cover_low", "latitude", "longitude", "timestamp", "image_path"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df['visibility'] = df['visibility'].fillna(10000.0)
    df["total_cloud_cover"] = df["cloud_cover_low"] + df["cloud_cover_mid"] + df["cloud_cover_high"]
    df = df.drop_duplicates(subset=["webcam_id", "timestamp"])

    LOGGER.info("Computing solar features...")
    df[["solar_elevation", "solar_azimuth"]] = df.apply(_compute_solar, axis=1)

    df["weather_code"] = df.get("weather_code", pd.Series(np.nan, index=df.index))
    df["weather_state"] = df["weather_code"].map(WEATHER_STATE_MAP).fillna("unknown")

    valid_mask = df["image_path"].apply(_is_valid_image)
    df = df[valid_mask]
    
    if len(df) == 0:
        raise ValueError("No valid data left after cleaning")

    df = df.sort_values(["webcam_id", "timestamp"]).reset_index(drop=True)
    
    # Process into exactly 24-step windows per webcam by padding back from the latest timestamp
    LOGGER.info("Generating sequences padded to %d timesteps...", window_size)
    webcam_dfs = []
    for wid, group in df.groupby("webcam_id"):
        group = group.set_index("timestamp").sort_index()
        group = group[~group.index.duplicated(keep="last")]
        
        last_ts = group.index.max()
        # Force a 24-step index spaced by 15 minutes (or 1 hour) leading up to the last known timestamp
        # The prompt says 6-hour window and 24 timesteps -> exactly 15 min apart.
        time_index = pd.date_range(end=last_ts, periods=window_size, freq="15min")
        
        # Reindex and pad
        reindexed = group.reindex(time_index)
        reindexed = reindexed.ffill().bfill() # forward fill then backfill the rest
        
        # In case some columns stayed NaN if the original group didn't have them cleanly
        reindexed['webcam_id'] = wid
        reindexed = reindexed.fillna(0) # Ultimate fallback
        
        reindexed = reindexed.reset_index(names="timestamp")
        webcam_dfs.append(reindexed)

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
    
    x_seq, y_seq, y_prev_seq = [], [], []

    for wid, group in merged.groupby("webcam_id"):
        # group has exactly 24 rows
        x_seq.append(scaled_continuous[group.index])
        y_seq.append(float(group.iloc[-1]["alqs"]))
        y_prev_seq.append(float(group.iloc[-2]["alqs"]))

    x = np.asarray(x_seq, dtype=np.float32)
    y = np.asarray(y_seq, dtype=np.float32)
    baseline_prev = np.asarray(y_prev_seq, dtype=np.float32)

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

    joblib.dump(FeatureArtifacts(scaler=scaler, weather_encoder=weather_encoder, alqs_scaler=alqs_scaler), output_dir / "feature_artifacts.joblib")
    
    LOGGER.info("Successfully generated %d sequences.", len(x))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_sequence_dataset(Path("data/processed/global_dataset"), Path("data/processed"), 24)
