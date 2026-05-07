import os
import glob
import math
import argparse
from typing import List

import pandas as pd
import numpy as np
import ephem


def find_parquet_files(data_path: str) -> List[str]:
    if os.path.isdir(data_path):
        return glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True)
    if os.path.isfile(data_path) and data_path.endswith(".parquet"):
        return [data_path]
    raise FileNotFoundError(f"No parquet files found at {data_path}")


def calculate_solar_features(row):
    """Calculate the Sun's altitude (elevation) in degrees for a given lat/lon and timestamp."""
    try:
        obs = ephem.Observer()
        obs.lat = str(row['latitude'])
        obs.lon = str(row['longitude'])
        obs.date = ephem.Date(pd.to_datetime(row['timestamp']))
        sun = ephem.Sun(obs)
        alt_deg = math.degrees(sun.alt)
        return alt_deg
    except Exception:
        return np.nan


def prepare_xgboost_dataset(data_path: str, output_path: str, forecast_horizon_hours: int = 1):
    print("Loading parquet files...")
    files = find_parquet_files(data_path)
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)

    print("Sorting and indexing...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(["webcam_id", "timestamp"])

    print("Computing fundamental solar features...")
    df['solar_elevation'] = df.apply(calculate_solar_features, axis=1)

    print("Engineering tabular features (lags, differences, interactions)...")
    engineered_dfs = []
    for wid, g in df.groupby("webcam_id"):
        g = g.copy()
        
        # We calculate the Target (ALQS shifted by the forecast horizon)
        # If the horizon is 1, we want to predict the ALQS 1 step in the future.
        g[f'target_alqs_in_{forecast_horizon_hours}'] = g['alqs'].shift(-forecast_horizon_hours)

        # Binary Target
        qt = g['alqs'].quantile(0.85)
        g[f'target_event_in_{forecast_horizon_hours}'] = ((g['alqs'] > qt) | ((g['solar_elevation'] > -6) & (g['solar_elevation'] < 6))).shift(-forecast_horizon_hours).fillna(False).astype(int)

        # MULTI-CLASS Target (Option D separated out)
        is_solar = (g['solar_elevation'] > -6) & (g['solar_elevation'] < 6)
        is_diff = g['alqs'] > qt
        
        cond_3 = is_solar & is_diff
        cond_1 = is_solar & ~is_diff
        cond_2 = ~is_solar & is_diff
        
        target_class = np.zeros(len(g))
        target_class[cond_1] = 1  # Golden/Blue Hour
        target_class[cond_2] = 2  # Dramatic Diffusion
        target_class[cond_3] = 3  # Both (Golden + Diffusion)
        
        g[f'target_multiclass_in_{forecast_horizon_hours}'] = pd.Series(target_class, index=g.index).shift(-forecast_horizon_hours).fillna(0).astype(int)

        # Lag features: past weather trends
        for lag in [1, 2, 3]:
            g[f'temp_lag_{lag}'] = g['temperature'].shift(lag)
            g[f'rh_lag_{lag}'] = g['relative_humidity'].shift(lag)
            g[f'cloud_low_lag_{lag}'] = g['cloud_cover_low'].shift(lag)

        # Delta features: changes in weather (derivatives)
        g['temp_change_1h'] = g['temperature'] - g['temp_lag_1']
        g['cloud_low_change_1h'] = g['cloud_cover_low'] - g['cloud_low_lag_1']

        engineered_dfs.append(g)

    df_final = pd.concat(engineered_dfs, ignore_index=True)
    
    # Drop rows where we don't have enough history to make lags, or no future target
    df_final = df_final.dropna(subset=[f'target_alqs_in_{forecast_horizon_hours}', 'temp_lag_3'])

    print(f"Saving engineered dataset to {output_path} - {len(df_final)} rows")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, default="data/processed/global_dataset")
    p.add_argument("--output-path", required=True, default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()
    prepare_xgboost_dataset(args.data_path, args.output_path, args.horizon)