import argparse
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

def train_mlp(args):
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
    
    print(f"Training features ({len(features)}): {features}")
    
    X = df[features].values
    y = df[target_col].values

    # Time-series aware split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Neural Networks require feature scaling and NO Missing Values (NaN)
    # The NaN values from the pandas lags are causing the loss to explode to NaN!
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class Weights for imbalance
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), weights)}
    print(f"Computed Class Weights: {class_weight_dict}")

    print("Building MLP Architecture...")
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        
        keras.layers.Dense(256, activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(64, activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        
        # 4 classes for Option D
        keras.layers.Dense(4, activation='softmax')
    ], name="mlp_tabular_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    print("Training MLP...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    probs = model.predict(X_test_scaled)
    preds = np.argmax(probs, axis=-1)
    
    acc = float(accuracy_score(y_test, preds))
    
    print("\n" + "="*50)
    print("   MULTI-CLASS MLP METRICS (OPTION D)")
    print("="*50)
    print(f"Overall Accuracy:  {acc:.4f}\n")
    
    labels_present = np.unique(np.concatenate([y_test, preds]))
    all_names = ["0: No Event", "1: Golden Hour Only", "2: Dramatic Diffusion Only", "3: Golden Hour + Diffusion"]
    names_present = [all_names[i] for i in labels_present]
    
    print("--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=names_present))

    # Save artifacts safely
    os.makedirs(args.artifact_dir, exist_ok=True)
    model.save(os.path.join(args.artifact_dir, "mlp_multiclass_model.keras"))
    
    import joblib
    joblib.dump(scaler, os.path.join(args.artifact_dir, "mlp_multiclass_scaler.joblib"))

    meta = {
        "model_type": "mlp_classifier",
        "target": target_col,
        "features": features,
        "metrics": {"accuracy": acc}
    }
    with open(os.path.join(args.artifact_dir, "mlp_multiclass_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved MLP to models/artifacts/mlp_multiclass_model.keras")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--forecast-horizon", type=int, default=3)
    args = p.parse_args()
    train_mlp(args)