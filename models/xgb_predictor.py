import argparse
import json
import os
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

def train_xgboost(args):
    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)

    # Features: Everything except identifiers, raw image data, and targets
    cols_to_drop = [
        'webcam_id', 'timestamp', 'image_path', 'image_url', 'weather_timestamp', 
        'alqs', 'target_alqs_in_1', 'target_alqs_in_3', 
        'target_event_in_1', 'target_event_in_3',
        'target_multiclass_in_1', 'target_multiclass_in_3'
    ]
    
    target_col = f'target_{args.target}_in_{args.forecast_horizon}'
    
    features = [c for c in df.columns if c not in cols_to_drop]
    
    print(f"Training features ({len(features)}): {features}")
    
    X = df[features]
    y = df[target_col]

    # Time-series aware split: Train on the past, test on the future.
    # In a real environment you split by date, here we just do a standard split for the POC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    if args.target == "event":
        print("Training XGBoost Classifier...")
        # Scale pos_weight for imbalanced event datasets
        vc = y_train.value_counts()
        scale_pos_weight = vc[0] / vc[1] if 1 in vc else 1
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            early_stopping_rounds=20
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = float(accuracy_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)
        
        results = {
            "accuracy": acc, "roc_auc": auc, 
            "precision": prec, "recall": rec, "f1": f1
        }
        
        print("\n" + "="*40)
        print("   DETAILED CLASSIFICATION METRICS")
        print("="*40)
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f} (When we notify the user, we are right {prec*100:.1f}% of the time)")
        print(f"Recall:    {rec:.4f} (Out of all real events, we caught {rec*100:.1f}%)")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\n--- Confusion Matrix ---")
        print("                 Predicted No Event | Predicted EVENT")
        print(f"Actual No Event |       {cm[0][0]:<11} |     {cm[0][1]:<11}")
        print(f"Actual EVENT    |       {cm[1][0]:<11} |     {cm[1][1]:<11}")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, preds, target_names=["No Event", "Event"]))

    elif args.target == "multiclass":
        print("Training Multi-Class XGBoost with Balanced Class Weights...")
        
        # Calculate weights to heavily penalize missing the rare Dramatic Diffusion events
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=20
        )
        
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        
        print("\n" + "="*50)
        print("   MULTI-CLASS METRICS (OPTION D)")
        print("="*50)
        print(f"Overall Accuracy:  {acc:.4f}\n")
        
        # In case a specific class didn't appear in the test set
        labels_present = np.unique(np.concatenate([y_test, preds]))
        all_names = ["0: No Event", "1: Golden Hour Only", "2: Dramatic Diffusion Only", "3: Golden Hour + Diffusion"]
        names_present = [all_names[i] for i in labels_present]
        
        print("--- Classification Report ---")
        print(classification_report(y_test, preds, target_names=names_present))
        
        results = {"accuracy": acc}

    else:
        print("Training XGBoost Regressor (ALQS)...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        
        results = {"mae": mae, "rmse": rmse}
        print(f"\nFinal Test Results: MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Feature Importance
    importance = model.feature_importances_
    f_imp = pd.DataFrame({'feature': features, 'importance': importance})
    f_imp = f_imp.sort_values('importance', ascending=False).head(10)
    print("\nTop 10 Important Features:")
    print(f_imp)

    # Save
    os.makedirs(args.artifact_dir, exist_ok=True)
    model_path = os.path.join(args.artifact_dir, f"xgb_{args.target}_model.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    meta = {
        "model_type": "xgboost_classifier" if args.target == "event" else "xgboost_regressor",
        "target": target_col,
        "features": features,
        "metrics": results,
    }
    with open(os.path.join(args.artifact_dir, f"xgb_{args.target}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--forecast-horizon", type=int, default=3)
    p.add_argument("--target", choices=["alqs", "event", "multiclass"], default="multiclass", help="Train on raw ALQS, binary Events, or multi-class explicit events")
    args = p.parse_args()
    train_xgboost(args)