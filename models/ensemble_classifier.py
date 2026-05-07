import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report

def create_aligned_dataset(df, xgb_mlp_features, lstm_features, target_col, window_size=6):
    """
    Creates perfectly aligned inputs for LSTM (Sequence) and Tabular (MLP/XGB) models
    so they can be evaluated together on the exact same predictions.
    """
    X_lstm_seq = []
    X_tabular_seq = []
    y_seq = []
    
    for wid, group in df.groupby('webcam_id'):
        group_lstm_x = group[lstm_features].values
        group_tabular_x = group[xgb_mlp_features].values
        group_y = group[target_col].values
        
        if len(group_lstm_x) < window_size:
            continue
            
        for i in range(len(group_lstm_x) - window_size):
            # LSTM sequence (window_size timesteps)
            X_lstm_seq.append(group_lstm_x[i : i+window_size])
            
            # Tabular uses the LAST timestep in the window (same as target offset)
            X_tabular_seq.append(group_tabular_x[i + window_size - 1])
            y_seq.append(group_y[i + window_size - 1])
            
    return np.array(X_lstm_seq), np.array(X_tabular_seq), np.array(y_seq)

def run_ensemble(args):
    print("Loading dataset...")
    df = pd.read_parquet(args.data_path)
    
    target_col = f'target_multiclass_in_{args.forecast_horizon}'
    
    xgb_mlp_features = ['latitude', 'longitude', 'temperature', 'relative_humidity', 'visibility', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'weather_code', 'solar_elevation', 'temp_lag_1', 'rh_lag_1', 'cloud_low_lag_1', 'temp_lag_2', 'rh_lag_2', 'cloud_low_lag_2', 'temp_lag_3', 'rh_lag_3', 'cloud_low_lag_3', 'temp_change_1h', 'cloud_low_change_1h']
    lstm_features = ['latitude', 'longitude', 'temperature', 'relative_humidity', 'visibility', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'weather_code', 'solar_elevation']
    
    # Drop NaNs just like training
    df = df.dropna(subset=xgb_mlp_features + lstm_features + [target_col])
    df = df.sort_values(["webcam_id", "timestamp"])

    # Align sequence windows with tabular states
    X_lstm, X_tab, y = create_aligned_dataset(df, xgb_mlp_features, lstm_features, target_col, window_size=args.window_size)
    
    # We only care about evaluation, so replicate the identical chronological split to get the test set
    split_idx = int(len(X_lstm) * 0.8) 
    
    X_lstm_test = X_lstm[split_idx:]
    X_tab_test = X_tab[split_idx:]
    y_test = y[split_idx:]
    X_tab_test = np.nan_to_num(X_tab_test, nan=0.0) # Correct lags
    
    print(f"Aligned Test set size: {len(y_test)}")
    
    # -- Load Models --
    print("Loading Models from artifacts/")
    
    # XGB
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(args.artifact_dir, "xgb_multiclass_model.json"))
    
    # MLP
    mlp_model = keras.models.load_model(os.path.join(args.artifact_dir, "mlp_multiclass_model.keras"))
    mlp_scaler = joblib.load(os.path.join(args.artifact_dir, "mlp_multiclass_scaler.joblib"))
    
    # LSTM
    lstm_model = keras.models.load_model(os.path.join(args.artifact_dir, "lstm_multiclass_model.keras"))
    lstm_scaler = joblib.load(os.path.join(args.artifact_dir, "lstm_multiclass_scaler.joblib"))
    
    # -- Predict --
    print("Generating Predictions (XGB)...")
    xgb_probs = xgb_model.predict_proba(X_tab_test)
    
    print("Generating Predictions (MLP)...")
    X_tab_test_scaled = mlp_scaler.transform(X_tab_test)
    mlp_probs = mlp_model.predict(X_tab_test_scaled, verbose=0)
    
    print("Generating Predictions (LSTM)...")
    original_shape = X_lstm_test.shape
    X_lstm_test_scaled = lstm_scaler.transform(X_lstm_test.reshape(-1, len(lstm_features))).reshape(original_shape)
    lstm_probs = lstm_model.predict(X_lstm_test_scaled, verbose=0)
    
    # Ensemble exactly aligns probability arrays and averages them
    ensemble_probs = (xgb_probs + mlp_probs + lstm_probs) / 3.0
    ensemble_preds = np.argmax(ensemble_probs, axis=-1)
    
    print("\n" + "="*50)
    print("   ENSEMBLE (SOFT VOTING AVERAGING) METRICS")
    print("="*50)
    acc = float(accuracy_score(y_test, ensemble_preds))
    print(f"Overall Accuracy:  {acc:.4f}\n")
    
    labels_present = np.unique(np.concatenate([y_test, ensemble_preds]))
    all_names = ["0: No Event", "1: Golden Hour Only", "2: Dramatic Diffusion Only", "3: Golden Hour + Diffusion"]
    names_present = [all_names[int(i)] for i in labels_present]
    
    print("--- Classification Report ---")
    print(classification_report(y_test, ensemble_preds, target_names=names_present))
    
    # Approach 2: Weighted combining strengths (XGB precision + LSTM minority recall)
    ensemble_probs_weighted = (xgb_probs * 0.5) + (lstm_probs * 0.35) + (mlp_probs * 0.15)
    ensemble_preds_weighted = np.argmax(ensemble_probs_weighted, axis=-1)
    
    print("\n" + "="*50)
    print("   ENSEMBLE (POWER WEIGHTED VOTING) METRICS")
    print("   (0.5 XGBoost, 0.35 LSTM, 0.15 MLP)")
    print("="*50)
    acc_w = float(accuracy_score(y_test, ensemble_preds_weighted))
    print(f"Overall Accuracy:  {acc_w:.4f}\n")
    print("--- Classification Report ---")
    print(classification_report(y_test, ensemble_preds_weighted, target_names=names_present))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--forecast-horizon", type=int, default=3)
    p.add_argument("--window-size", type=int, default=6)
    args = p.parse_args()
    run_ensemble(args)