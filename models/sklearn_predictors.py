import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_rf(args):
    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)

    # Features: Everything except identifiers, raw image data, and targets
    cols_to_drop = [
        'webcam_id', 'timestamp', 'image_path', 'image_url', 'weather_timestamp', 
        'alqs', 'target_alqs_in_1', 'target_alqs_in_3', 
        'target_event_in_1', 'target_event_in_3',
        'target_multiclass_in_1', 'target_multiclass_in_3'
    ]
    
    target_col = f'target_multiclass_in_{args.forecast_horizon}'
    features = [c for c in df.columns if c not in cols_to_drop]
    
    X = df[features].values
    X = np.nan_to_num(X, nan=0.0) # Handle lag NaNs
    y = df[target_col].values

    # Time-series aware split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("Training Random Forest with Balanced Class Weights...")
    # Restrict max_depth so it doesn't overfit given we are using purely tabular weather logic
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = float(accuracy_score(y_test, preds))
    
    print("\n" + "="*50)
    print("   MULTI-CLASS RANDOM FOREST METRICS")
    print("="*50)
    print(f"Overall Accuracy:  {acc:.4f}\n")
    
    all_names = ["0: No Event", "1: Golden Hour Only", "2: Dramatic Diffusion Only", "3: Golden Hour + Diffusion"]
    labels_present = np.unique(np.concatenate([y_test, preds]))
    names_present = [all_names[i] for i in labels_present]
    
    print("--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=names_present))

    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(importances.head(10))

    # Save artifacts safely
    os.makedirs(args.artifact_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.artifact_dir, "rf_multiclass_model.joblib"))

    meta = {
        "model_type": "random_forest",
        "target": target_col,
        "features": features,
        "metrics": {"accuracy": acc}
    }
    with open(os.path.join(args.artifact_dir, "rf_multiclass_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved Random Forest to models/artifacts/rf_multiclass_model.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--forecast-horizon", type=int, default=3)
    args = p.parse_args()
    train_rf(args)