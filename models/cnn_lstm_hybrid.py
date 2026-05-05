import argparse
import json
import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
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
    df = df.sort_values(["webcam_id", "timestamp"])  # assume these columns exist
    return df


def build_sequence_records(
    df: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    image_col: str = "image_path",
    target_col: str = "alqs",
    metadata_cols: Optional[List[str]] = None,
):
    metadata_cols = metadata_cols or []
    records = []
    for wid, g in df.groupby("webcam_id"):
        g = g.reset_index(drop=True)
        total = len(g)
        if total < (sequence_length + forecast_horizon):
            continue
        # Build windows that predict a future target (h steps ahead)
        max_start = total - sequence_length - forecast_horizon + 1
        for i in range(0, max_start, 1):
            window = g.iloc[i: i + sequence_length]
            # require all images present
            if window[image_col].isnull().any():
                continue
            img_paths = window[image_col].tolist()
            target_index = i + sequence_length + forecast_horizon - 1
            target = float(g[target_col].iloc[target_index])
            last_observed = float(window[target_col].iloc[-1])
            metadata = window[metadata_cols].values.astype(np.float32) if metadata_cols else None
            records.append((img_paths, metadata, target, last_observed))
    return records


class HybridSequence(keras.utils.Sequence):
    def __init__(self, records, batch_size, seq_len, image_size=(224, 224), metadata_len: Optional[int] = None):
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

        for i, (img_paths, meta, target, _last_observed) in enumerate(batch):
            for t, p in enumerate(img_paths):
                try:
                    im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if im is None:
                        raise ValueError("invalid image")
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (self.image_size[1], self.image_size[0]))
                    imgs[i, t] = im.astype(np.float32) / 255.0
                except Exception:
                    imgs[i, t] = 0.0
            if metas is not None and meta is not None:
                metas[i] = meta
            y[i, 0] = target

        # Return a dict keyed to the model input names to avoid tf.data conversion issues
        if metas is not None:
            inputs = {"images": imgs, "metadata": metas}
        else:
            inputs = {"images": imgs}
        return inputs, y


def build_hybrid_model(seq_len: int, image_size=(224, 224, 3), metadata_len: Optional[int] = None,
                       feature_dim: int = 256, lstm_units: int = 128, freeze_backbone: bool = True,
                       learning_rate: float = 1e-4):
    # Image input: sequence of images
    img_in = keras.Input(shape=(seq_len, image_size[0], image_size[1], image_size[2]), name="images")

    # Build a small CNN backbone (MobileNetV2-based but returning a feature vector)
    base_cnn = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_shape=(image_size[0], image_size[1], image_size[2]))
    base_cnn.trainable = not freeze_backbone
    x = layers.GlobalAveragePooling2D(name="gap")(base_cnn.output)
    x = layers.Dense(feature_dim, activation="relu", name="feature_dense")(x)
    cnn_feature_model = keras.Model(base_cnn.input, x, name="cnn_feature_model")

    # Wrap with TimeDistributed
    td = layers.TimeDistributed(cnn_feature_model, name="td_cnn")(img_in)

    # Optional metadata input
    if metadata_len:
        meta_in = keras.Input(shape=(seq_len, metadata_len), name="metadata")
        # simple projection per time-step
        meta_proj = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="td_meta_proj")(meta_in)
        # concat along feature axis per time-step
        seq_input = layers.Concatenate(axis=-1)([td, meta_proj])
    else:
        meta_in = None
        seq_input = td

    # LSTM layers
    lstm = layers.LSTM(lstm_units, return_sequences=False, name="lstm_1")(seq_input)

    # Dense head
    h = layers.Dense(128, activation="relu")(lstm)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="linear", name="target")(h)

    inputs = [img_in] + ([meta_in] if meta_in is not None else [])
    model = keras.Model(inputs=inputs, outputs=out, name="cnn_lstm_hybrid")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                           keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def train(args):
    os.makedirs(args.artifact_dir, exist_ok=True)
    df = load_dataframe(args.data_path)

    # infer metadata columns (optional)
    possible_meta = [c for c in df.columns if c not in ["webcam_id", "timestamp", "image_path", "alqs"]]
    metadata_cols = possible_meta if args.use_metadata and possible_meta else []

    records = build_sequence_records(
        df,
        args.sequence_length,
        args.forecast_horizon,
        image_col="image_path",
        target_col="alqs",
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

    # splits
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_recs = records[:train_end]
    val_recs = records[train_end:val_end]
    test_recs = records[val_end:]

    metadata_len = len(metadata_cols) if metadata_cols else None

    train_seq = HybridSequence(train_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)
    val_seq = HybridSequence(val_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)
    test_seq = HybridSequence(test_recs, args.batch_size, args.sequence_length, image_size=(args.img_size, args.img_size), metadata_len=metadata_len)

    model = build_hybrid_model(args.sequence_length, image_size=(args.img_size, args.img_size, 3),
                              metadata_len=metadata_len, feature_dim=args.feature_dim,
                              lstm_units=args.lstm_units, freeze_backbone=args.freeze_backbone,
                              learning_rate=args.learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    ]

    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=1)

    # evaluate
    metrics = model.evaluate(test_seq, verbose=1)
    # model.metrics_names align with metrics
    metric_names = model.metrics_names
    metric_map = dict(zip(metric_names, metrics))

    # Forecasting baseline: predict future target as the last observed value in the window
    y_true = []
    y_pred_baseline = []
    for rec in test_recs:
        y_true.append(rec[2])
        y_pred_baseline.append(rec[3])

    baseline_mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred_baseline))))

    result = {
        "test_mae": float(metric_map.get("mae", metric_map.get("mean_absolute_error", np.nan))),
        "test_rmse": float(metric_map.get("rmse", metric_map.get("root_mean_squared_error", np.nan))),
        "baseline_mae": baseline_mae,
    }
    result["improvement_pct"] = (baseline_mae - result["test_mae"]) / baseline_mae * 100.0 if baseline_mae else None

    # save model and metadata
    model_path = os.path.join(args.artifact_dir, "cnn_lstm_hybrid.keras")
    model.save(model_path)
    meta = {
        "model_name": "luxaeterna_cnn_lstm_hybrid",
        "version": args.run_tag,
        "sequence_length": args.sequence_length,
        "forecast_horizon": args.forecast_horizon,
        "image_size": args.img_size,
        "feature_dim": args.feature_dim,
        "lstm_units": args.lstm_units,
        "metrics": result,
        "data_counts": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
    }
    with open(os.path.join(args.artifact_dir, "cnn_lstm_hybrid_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--artifact-dir", default="models/artifacts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--forecast-horizon", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--lstm-units", type=int, default=128)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--use-metadata", action="store_true")
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()
    if args.run_tag is None:
        import datetime

        args.run_tag = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    if args.forecast_horizon < 1:
        raise ValueError("--forecast-horizon must be >= 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
