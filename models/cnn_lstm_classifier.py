import argparse
import json
import os
import glob
import math
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
import ephem
from tensorflow import keras
from tensorflow.keras import layers


def find_parquet_files(data_path: str) -> List[str]:
    if os.path.isdir(data_path):
        return glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True)
    if os.path.isfile(data_path) and data_path.endswith(".parquet"):
        return [data_path]
    raise FileNotFoundError(f"No parquet files found at {data_path}")


def load_dataframe(data_path: str) -> pd.DataFrame:
    files = find_parquet_files(data_path)
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["webcam_id", "timestamp"])
    return df


def label_events(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing heuristic event labels (Option D: Solar Angle + ALQS Diffusion)...")
    
    # 1. Alqs threshold (Dramatic cloud diffusion / extraordinary light intensity)
    # Define an event if ALQS is in the top 10% for that specific region/webcam
    df['alqs_thresh'] = df.groupby('webcam_id')['alqs'].transform(lambda x: x > x.quantile(0.90))
    
    # 2. Solar angle (Golden hour / Blue hour)
    def is_golden_blue(row):
        try:
            obs = ephem.Observer()
            obs.lat = str(row['latitude'])
            obs.lon = str(row['longitude'])
            # Ensure timestamp is parsed properly
            obs.date = ephem.Date(pd.to_datetime(row['timestamp']))
            sun = ephem.Sun(obs)
            alt_deg = math.degrees(sun.alt)
            # Golden/Blue hour is roughly when sun altitude is between -6 (civil twilight) and +6 degrees
            return -6 <= alt_deg <= 6
        except Exception:
            return False
            
    if 'latitude' in df.columns and 'longitude' in df.columns and 'timestamp' in df.columns:
        df['is_solar_event'] = df.apply(is_golden_blue, axis=1)
    else:
        df['is_solar_event'] = False

    # 3. Combine heuristics
    df['is_event'] = (df['alqs_thresh'] | df['is_solar_event']).astype(np.float32)
    pos_rate = df['is_event'].mean()
    print(f"Labeling complete. Event-positive rate: {pos_rate*100:.2f}%")
    return df


@lru_cache(maxsize=8192)
def _load_image(path: str, image_width: int, image_height: int) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("invalid image")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (image_width, image_height))
    return im.astype(np.float32) / 255.0


def build_sequence_records(
    df: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    image_col: str = "image_path",
    target_col: str = "is_event",
    metadata_cols: Optional[List[str]] = None,
):
    metadata_cols = metadata_cols or []
    records = []
    for wid, g in df.groupby("webcam_id"):
        g = g.reset_index(drop=True)
        total = len(g)
        if total < (sequence_length + forecast_horizon):
            continue
        # Build windows that predict a future target event (h steps ahead)
        max_start = total - sequence_length - forecast_horizon + 1
        for i in range(0, max_start, 1):
            window = g.iloc[i: i + sequence_length]
            if window[image_col].isnull().any():
                continue
            img_paths = window[image_col].tolist()
            target_index = i + sequence_length + forecast_horizon - 1
            target = float(g[target_col].iloc[target_index])
            metadata = window[metadata_cols].values.astype(np.float32) if metadata_cols else None
            records.append((img_paths, metadata, target))
    return records


class HybridSequence(keras.utils.Sequence):
    def __init__(self, records, batch_size, seq_len, image_size=(224, 224), metadata_len: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.records = records
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.metadata_len = metadata_len

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.records[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs = np.zeros((len(batch), self.seq_len, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        metas = None
        if self.metadata_len:
            metas = np.zeros((len(batch), self.seq_len, self.metadata_len), dtype=np.float32)
        y = np.zeros((len(batch), 1), dtype=np.float32)

        for i, (img_paths, meta, target) in enumerate(batch):
            for t, p in enumerate(img_paths):
                try:
                    imgs[i, t] = _load_image(p, self.image_size[1], self.image_size[0])
                except Exception:
                    imgs[i, t] = 0.0
            if metas is not None and meta is not None:
                metas[i] = meta
            y[i, 0] = target

        if metas is not None:
            inputs = {"images": imgs, "metadata": metas}
        else:
            inputs = {"images": imgs}
        return inputs, y


def build_cnn_feature_model(image_size, feature_dim: int, freeze_backbone: bool, backbone: str):
    if backbone == "mobilenetv2":
        base_cnn = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size[0], image_size[1], image_size[2]),
        )
        base_cnn.trainable = not freeze_backbone
        x = layers.GlobalAveragePooling2D(name="gap")(base_cnn.output)
        x = layers.Dense(feature_dim, activation="relu", name="feature_dense")(x)
        return keras.Model(base_cnn.input, x, name="cnn_feature_model")

    inputs = keras.Input(shape=(image_size[0], image_size[1], image_size[2]))
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.SeparableConv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.SeparableConv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation="relu")(x)
    return keras.Model(inputs, x, name="lite_cnn_feature_model")


def build_hybrid_classifier_model(seq_len: int, image_size=(224, 224, 3), metadata_len: Optional[int] = None,
                       feature_dim: int = 256, lstm_units: int = 128, freeze_backbone: bool = True,
                       learning_rate: float = 1e-4, backbone: str = "lite"):
    
    img_in = keras.Input(shape=(seq_len, image_size[0], image_size[1], image_size[2]), name="images")
    cnn_feature_model = build_cnn_feature_model(image_size, feature_dim, freeze_backbone, backbone)
    td = layers.TimeDistributed(cnn_feature_model, name="td_cnn")(img_in)

    if metadata_len:
        meta_in = keras.Input(shape=(seq_len, metadata_len), name="metadata")
        meta_proj = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="td_meta_proj")(meta_in)
        seq_input = layers.Concatenate(axis=-1)([td, meta_proj])
    else:
        meta_in = None
        seq_input = td

    lstm = layers.LSTM(lstm_units, return_sequences=False, name="lstm_1")(seq_input)
    h = layers.Dense(128, activation="relu")(lstm)
    h = layers.Dropout(0.3)(h)
    
    # Classification head (sigmoid for probability of event)
    out = layers.Dense(1, activation="sigmoid", name="target")(h)

    inputs = [img_in] + ([meta_in] if meta_in is not None else [])
    model = keras.Model(inputs=inputs, outputs=out, name="cnn_lstm_classifier")

    # Optimize for Precision, Recall, and AUC (binary classification)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[
                      keras.metrics.Precision(name="precision"),
                      keras.metrics.Recall(name="recall"),
                      keras.metrics.AUC(name="roc_auc"),
                      keras.metrics.AUC(curve="PR", name="pr_auc")
                  ])
    return model


def train(args):
    os.makedirs(args.artifact_dir, exist_ok=True)
    df = load_dataframe(args.data_path)
    
    # Apply heuristic event labeling (Option D)
    df = label_events(df)

    possible_meta = [c for c in df.columns if c not in ["webcam_id", "timestamp", "image_path", "alqs", "alqs_thresh", "is_solar_event", "is_event"]]
    metadata_cols = possible_meta if args.use_metadata and possible_meta else []

    records = build_sequence_records(
        df,
        args.sequence_length,
        args.forecast_horizon,
        image_col="image_path",
        target_col="is_event",
        metadata_cols=metadata_cols,
    )
    np.random.shuffle(records)
    n = len(records)
    if n == 0:
        raise RuntimeError("No sequence records found; check sequence length and data path")

    print(
        f"Prepared {n} forecasting sequences with sequence_length={args.sequence_length}, "
        f"forecast_horizon={args.forecast_horizon}"
    )

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_recs = records[:train_end]
    val_recs = records[train_end:val_end]
    test_recs = records[val_end:]

    # Calculate class weights for imbalanced datasets
    train_targets = [r[2] for r in train_recs]
    pos = sum(train_targets)
    neg = len(train_targets) - pos
    if pos > 0 and neg > 0:
        weight_for_0 = (1 / neg) * (len(train_targets) / 2.0)
        weight_for_1 = (1 / pos) * (len(train_targets) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Using class weights: {class_weight}")
    else:
        class_weight = None

    metadata_len = len(metadata_cols) if metadata_cols else None

    train_seq = HybridSequence(train_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)
    val_seq = HybridSequence(val_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)
    test_seq = HybridSequence(test_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)

    model = build_hybrid_classifier_model(args.sequence_length, image_size=(args.img_size, args.img_size, 3),
                              metadata_len=metadata_len, feature_dim=args.feature_dim,
                              lstm_units=args.lstm_units, freeze_backbone=args.freeze_backbone,
                              learning_rate=args.learning_rate, backbone=args.backbone)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", # Optimize early stopping for precision/recall AUC
            mode="max",
            patience=10,
            start_from_epoch=args.min_epochs,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    ]

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    metrics = model.evaluate(test_seq, verbose=1)
    metric_names = model.metrics_names
    metric_map = dict(zip(metric_names, metrics))

    result = {
        "test_precision": float(metric_map.get("precision", np.nan)),
        "test_recall": float(metric_map.get("recall", np.nan)),
        "test_roc_auc": float(metric_map.get("roc_auc", np.nan)),
        "test_pr_auc": float(metric_map.get("pr_auc", np.nan)),
        "test_loss": float(metric_map.get("loss", np.nan))
    }

    model_path = os.path.join(args.artifact_dir, "cnn_lstm_classifier.keras")
    model.save(model_path)
    meta = {
        "model_name": "luxaeterna_cnn_lstm_classifier",
        "version": args.run_tag,
        "sequence_length": args.sequence_length,
        "forecast_horizon": args.forecast_horizon,
        "image_size": args.img_size,
        "feature_dim": args.feature_dim,
        "lstm_units": args.lstm_units,
        "backbone": args.backbone,
        "metrics": result,
        "data_counts": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs), "positive_ratio": pos/(pos+neg) if (pos+neg)>0 else 0},
    }
    with open(os.path.join(args.artifact_dir, "cnn_lstm_classifier_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--forecast-horizon", type=int, default=1)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--lstm-units", type=int, default=128)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--use-metadata", action="store_true")
    p.add_argument("--backbone", choices=["lite", "mobilenetv2"], default="lite")
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()
    if args.run_tag is None:
        import datetime
        args.run_tag = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)