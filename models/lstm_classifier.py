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
import joblib

def create_sequences(df, features, target_col, window_size=6):
    """
    Given a sorted dataframe (grouped by webcam generally), slide a window 
    of `window_size` to create seq elements.
    """
    X_seq = []
    y_seq = []
    
    # We group by webcam so we don't accidentally leak data between different webcams
    for wid, group in df.groupby('webcam_id'):
        group_x = group[features].values
        group_y = group[target_col].values
        
        if len(group_x) < window_size:
            continue
            
        for i in range(len(group_x) - window_size):
            X_seq.append(group_x[i : i+window_size])
            # The target for this sequence is the target value of the *last* element
            # in the sequence (or the one immediately after).
            # The parquet target_multiclass_in_3 already represents +3 hours from the current!
            # So the target from the last timestep in the window is perfect.
            y_seq.append(group_y[i + window_size - 1])
            
    return np.array(X_seq), np.array(y_seq)

def train_lstm(args):
    print(f"Loading tabular engineered data from {args.data_path}")
    df = pd.read_parquet(args.data_path)

    # Base features to run sequentially
    # We do NOT use lag features here because the LSTM models time explicitly!
    features = [
        'latitude', 'longitude', 'temperature', 'relative_humidity', 
        'visibility', 'cloud_cover_low', 'cloud_cover_mid', 
        'cloud_cover_high', 'weather_code', 'solar_elevation'
    ]
    target_col = f'target_multiclass_in_{args.forecast_horizon}'
    
    # Ensure no naive NaN rows breaking sequence extraction
    df = df.dropna(subset=features + [target_col])
    df = df.sort_values(["webcam_id", "timestamp"])

    X, y = create_sequences(df, features, target_col, window_size=args.window_size)
    print(f"Constructed Multi-Class Sequence Dataset: X={X.shape}, y={y.shape}")

    if len(X) == 0:
        raise ValueError("Sequence dataset is empty!")

    # Time-series aware split (split chronologically by webcam implicitly with shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Training set sequences: {X_train.shape}, Test set: {X_test.shape}")

    # Normalize 3D Sequences (fit on 2D version of Train only)
    scaler = StandardScaler()
    
    # Reshape train, fit scaler, reshape back
    X_train_2d = X_train.reshape(-1, len(features))
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    
    # Transform test
    X_test_2d = X_test.reshape(-1, len(features))
    X_test_scaled_2d = scaler.transform(X_test_2d)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    # Class weights! Neural nets will die on minority classes without this.
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), weights)}
    print(f"Class Weights explicitly mapped for LSTM: {class_weight_dict}")

    print("Building LSTM Classifier Architecture...")
    inputs = keras.layers.Input(shape=(args.window_size, len(features)))
    
    # Spatial dropout randomly drops features out across the whole sequence step
    x = keras.layers.SpatialDropout1D(0.15)(inputs)
    
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)
    x = keras.layers.LayerNormalization()(x)
    
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False, dropout=0.2)
    )(x)
    x = keras.layers.LayerNormalization()(x)
    
    x = keras.layers.Dense(64, activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # 4 Output Classes for Option D
    outputs = keras.layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_tabular_multiclass")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    print("Training LSTM Classifier...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print("Evaluating LSTM...")
    probs = model.predict(X_test_scaled, verbose=0)
    preds = np.argmax(probs, axis=-1)
    
    acc = float(accuracy_score(y_test, preds))
    
    print("\n" + "="*50)
    print(f"   LSTM SEQUENCE CLASSIFIER METRICS (Window={args.window_size})")
    print("="*50)
    print(f"Overall Accuracy:  {acc:.4f}\n")
    
    labels_present = np.unique(np.concatenate([y_test, preds]))
    all_names = ["0: No Event", "1: Golden Hour Only", "2: Dramatic Diffusion Only", "3: Golden Hour + Diffusion"]
    names_present = [all_names[int(i)] for i in labels_present]
    
    print("--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=names_present))

    os.makedirs(args.artifact_dir, exist_ok=True)
    model.save(os.path.join(args.artifact_dir, "lstm_multiclass_model.keras"))
    joblib.dump(scaler, os.path.join(args.artifact_dir, "lstm_multiclass_scaler.joblib"))

    meta = {
        "model_type": "lstm_classifier",
        "target": target_col,
        "features": features,
        "window_size": args.window_size,
        "metrics": {"accuracy": acc}
    }
    with open(os.path.join(args.artifact_dir, "lstm_multiclass_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, default="data/processed/global_dataset_xgb.parquet")
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--forecast-horizon", type=int, default=3)
    p.add_argument("--window-size", type=int, default=6)
    args = p.parse_args()
    train_lstm(args)