from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

LOGGER = logging.getLogger("luxaeterna.models.cnn_weather_model")

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
WEATHER_FEATURE_CANDIDATES = [
    "temperature",
    "relative_humidity",
    "visibility",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "total_cloud_cover",
]


@dataclass(slots=True)
class CnnWeatherTrainingConfig:
    data_path: Path
    artifact_dir: Path
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    freeze_backbone: bool = False
    log_every_n_steps: int = 50


@dataclass(slots=True)
class PreparedDataset:
    train_records: list[tuple[Path, np.ndarray, float]]
    val_records: list[tuple[Path, np.ndarray, float]]
    test_records: list[tuple[Path, np.ndarray, float]]
    weather_feature_names: list[str]
    skipped_invalid_image: int
    skipped_missing_image: int


def _read_frame_from_path(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset source not found: {data_path}")

    if data_path.is_dir():
        paths = sorted(data_path.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {data_path}")

        frames = [df for df in (pd.read_parquet(path) for path in paths) if not df.empty]
        if not frames:
            raise ValueError(f"All parquet files in {data_path} were empty")
        return pd.concat(frames, ignore_index=True)

    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)

    raise ValueError(f"Unsupported data source: {data_path}")


def _resolve_image_path(raw_path: str, workspace_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return workspace_root / path


def _is_valid_image(image_path: Path) -> bool:
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        image_array = image.astype(np.float32) / 255.0
        # Skip almost blank frames which provide no visual signal.
        if float(np.std(image_array)) < 1e-3:
            return False
        return True
    except Exception:
        return False


def _prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"image_path", "alqs"}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for CNN-weather model: {missing}")

    work = frame.copy()
    work["alqs"] = pd.to_numeric(work["alqs"], errors="coerce")
    work = work.dropna(subset=["image_path", "alqs"])

    if "total_cloud_cover" not in work.columns and {
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
    }.issubset(work.columns):
        work["total_cloud_cover"] = (
            pd.to_numeric(work["cloud_cover_low"], errors="coerce")
            + pd.to_numeric(work["cloud_cover_mid"], errors="coerce")
            + pd.to_numeric(work["cloud_cover_high"], errors="coerce")
        )

    weather_features = [col for col in WEATHER_FEATURE_CANDIDATES if col in work.columns]
    if not weather_features:
        raise ValueError("No usable weather feature columns found")

    for col in weather_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        median = work[col].median(skipna=True)
        fill_val = float(median) if not pd.isna(median) else 0.0
        work[col] = work[col].fillna(fill_val).astype(np.float32)

    return work, weather_features


def _split_frame(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "webcam_id" in work.columns and work["webcam_id"].nunique() >= 3:
        webcam_ids = np.array(sorted(work["webcam_id"].dropna().astype(str).unique().tolist()))
        train_ids, rem_ids = train_test_split(webcam_ids, test_size=0.30, random_state=42, shuffle=True)
        val_ids, test_ids = train_test_split(rem_ids, test_size=0.50, random_state=42, shuffle=True)

        train = work[work["webcam_id"].astype(str).isin(set(train_ids.tolist()))]
        val = work[work["webcam_id"].astype(str).isin(set(val_ids.tolist()))]
        test = work[work["webcam_id"].astype(str).isin(set(test_ids.tolist()))]
    else:
        train, rem = train_test_split(work, test_size=0.30, random_state=42, shuffle=True)
        val, test = train_test_split(rem, test_size=0.50, random_state=42, shuffle=True)

    if train.empty or val.empty or test.empty:
        raise ValueError("Data split produced an empty partition")

    return train, val, test


def _standardize_weather(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weather_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_w = train_df.loc[:, weather_features].to_numpy(dtype=np.float32)
    val_w = val_df.loc[:, weather_features].to_numpy(dtype=np.float32)
    test_w = test_df.loc[:, weather_features].to_numpy(dtype=np.float32)

    mean = train_w.mean(axis=0)
    std = train_w.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df.loc[:, weather_features] = ((train_w - mean) / std).astype(np.float32)
    val_df.loc[:, weather_features] = ((val_w - mean) / std).astype(np.float32)
    test_df.loc[:, weather_features] = ((test_w - mean) / std).astype(np.float32)

    return train_df, val_df, test_df, mean.astype(np.float32), std.astype(np.float32)


def _build_records(
    frame: pd.DataFrame,
    weather_features: Sequence[str],
    workspace_root: Path,
) -> tuple[list[tuple[Path, np.ndarray, float]], int, int]:
    records: list[tuple[Path, np.ndarray, float]] = []
    skipped_invalid = 0
    skipped_missing = 0

    for row in frame.itertuples(index=False):
        image_path = _resolve_image_path(str(getattr(row, "image_path")), workspace_root)
        if not image_path.exists():
            skipped_missing += 1
            continue

        weather_vector = np.asarray([float(getattr(row, feat)) for feat in weather_features], dtype=np.float32)
        target = float(getattr(row, "alqs"))
        records.append((image_path, weather_vector, target))

    return records, skipped_invalid, skipped_missing


def prepare_dataset(config: CnnWeatherTrainingConfig) -> tuple[PreparedDataset, np.ndarray, np.ndarray]:
    raw = _read_frame_from_path(config.data_path)
    frame, weather_features = _prepare_frame(raw)
    train_df, val_df, test_df = _split_frame(frame)

    train_df, val_df, test_df, weather_mean, weather_std = _standardize_weather(
        train_df,
        val_df,
        test_df,
        weather_features,
    )

    workspace_root = Path.cwd()
    train_records, train_bad, train_missing = _build_records(train_df, weather_features, workspace_root)
    val_records, val_bad, val_missing = _build_records(val_df, weather_features, workspace_root)
    test_records, test_bad, test_missing = _build_records(test_df, weather_features, workspace_root)

    if not train_records or not val_records or not test_records:
        raise ValueError(
            "Insufficient valid image records after filtering. "
            "Check image paths and data quality."
        )

    prepared = PreparedDataset(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        weather_feature_names=list(weather_features),
        skipped_invalid_image=train_bad + val_bad + test_bad,
        skipped_missing_image=train_missing + val_missing + test_missing,
    )

    LOGGER.info(
        "Prepared dataset: train=%d val=%d test=%d (skipped missing=%d, skipped invalid=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        prepared.skipped_missing_image,
        prepared.skipped_invalid_image,
    )

    return prepared, weather_mean, weather_std


class ImageWeatherSequence(keras.utils.Sequence):
    def __init__(
        self,
        records: list[tuple[Path, np.ndarray, float]],
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.records) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        start = idx * self.batch_size
        stop = min(len(self.indices), start + self.batch_size)
        batch_indices = self.indices[start:stop]

        images: list[np.ndarray] = []
        weather: list[np.ndarray] = []
        targets: list[float] = []

        for record_idx in batch_indices:
            image_path, weather_vector, target = self.records[int(record_idx)]
            try:
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                arr = img.astype(np.float32) / 255.0
                if float(np.std(arr)) < 1e-3:
                    continue
                images.append(arr)
                weather.append(weather_vector)
                targets.append(target)
            except Exception:
                continue

        if not images:
            # Keras expects non-empty batches; if all failed in this batch,
            # retry from the next batch index modulo total batches.
            return self.__getitem__((idx + 1) % self.__len__())

        return (
            np.asarray(images, dtype=np.float32),
            np.asarray(weather, dtype=np.float32),
        ), np.asarray(targets, dtype=np.float32)


class BatchProgressLogger(keras.callbacks.Callback):
    def __init__(self, train_steps: int, val_steps: int, log_every_n_steps: int) -> None:
        super().__init__()
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._epoch_start = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._epoch_start = time.time()
        LOGGER.info(
            "Epoch %d started (train_steps=%d, val_steps=%d)",
            epoch + 1,
            self.train_steps,
            self.val_steps,
        )

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        step = batch + 1
        if step % self.log_every_n_steps != 0 and step != self.train_steps:
            return

        logs = logs or {}
        LOGGER.info(
            "Train step %d/%d - loss=%.4f mae=%.4f rmse=%.4f",
            step,
            self.train_steps,
            float(logs.get("loss", 0.0)),
            float(logs.get("mae", 0.0)),
            float(logs.get("rmse", 0.0)),
        )

    def on_test_batch_end(self, batch: int, logs: dict | None = None) -> None:
        step = batch + 1
        if step % self.log_every_n_steps != 0 and step != self.val_steps:
            return

        logs = logs or {}
        LOGGER.info(
            "Val step %d/%d - val_loss=%.4f val_mae=%.4f val_rmse=%.4f",
            step,
            self.val_steps,
            float(logs.get("loss", 0.0)),
            float(logs.get("mae", 0.0)),
            float(logs.get("rmse", 0.0)),
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        elapsed = time.time() - self._epoch_start
        logs = logs or {}
        LOGGER.info(
            "Epoch %d finished in %.1fs - loss=%.4f val_loss=%.4f",
            epoch + 1,
            elapsed,
            float(logs.get("loss", 0.0)),
            float(logs.get("val_loss", 0.0)),
        )


def build_model(num_weather_features: int, learning_rate: float, freeze_backbone: bool) -> keras.Model:
    image_input = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS), name="image_input")
    weather_input = keras.layers.Input(shape=(num_weather_features,), name="weather_input")

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS),
    )
    backbone.trainable = not freeze_backbone

    x_img = keras.layers.Lambda(
        lambda t: keras.applications.mobilenet_v2.preprocess_input(t * 255.0),
        name="mobilenet_preprocess",
    )(image_input)
    x_img = backbone(x_img, training=not freeze_backbone)
    x_img = keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
    x_img = keras.layers.Dense(128, activation="relu", name="img_dense_128")(x_img)

    x_weather = keras.layers.Dense(64, activation="relu", name="weather_dense_64")(weather_input)
    x_weather = keras.layers.Dense(32, activation="relu", name="weather_dense_32")(x_weather)

    fused = keras.layers.Concatenate(name="fusion_concat")([x_img, x_weather])
    fused = keras.layers.Dense(128, activation="relu", name="fusion_dense_128")(fused)
    fused = keras.layers.Dropout(0.3, name="fusion_dropout")(fused)
    fused = keras.layers.Dense(64, activation="relu", name="fusion_dense_64")(fused)
    output = keras.layers.Dense(1, activation="linear", name="alqs_output")(fused)

    model = keras.Model(inputs=[image_input, weather_input], outputs=output, name="luxaeterna_cnn_weather")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def train(config: CnnWeatherTrainingConfig) -> dict[str, float]:
    prepared, weather_mean, weather_std = prepare_dataset(config)

    train_ds = ImageWeatherSequence(prepared.train_records, batch_size=config.batch_size, shuffle=True)
    val_ds = ImageWeatherSequence(prepared.val_records, batch_size=config.batch_size, shuffle=False)
    test_ds = ImageWeatherSequence(prepared.test_records, batch_size=config.batch_size, shuffle=False)

    train_steps = len(train_ds)
    val_steps = len(val_ds)
    LOGGER.info("Training configuration: epochs=%d, batch_size=%d, train_steps=%d, val_steps=%d", config.epochs, config.batch_size, train_steps, val_steps)

    model = build_model(
        num_weather_features=len(prepared.weather_feature_names),
        learning_rate=config.learning_rate,
        freeze_backbone=config.freeze_backbone,
    )

    timestamp_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        BatchProgressLogger(
            train_steps=train_steps,
            val_steps=val_steps,
            log_every_n_steps=config.log_every_n_steps,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    eval_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    keras_path = config.artifact_dir / "cnn_weather_model.keras"
    model.save(keras_path)

    savedmodel_path = config.artifact_dir / "cnn_weather_model_savedmodel"
    try:
        model.export(str(savedmodel_path))
    except Exception as e:
        LOGGER.warning("Could not export SavedModel natively: %s. Keras standard save was successful.", e)

    metadata = {
        "model_name": model.name,
        "version": timestamp_tag,
        "input_shape": {
            "image": [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS],
            "weather": [len(prepared.weather_feature_names)],
        },
        "weather_features": prepared.weather_feature_names,
        "metrics": {
            "test_loss": float(eval_metrics.get("loss", 0.0)),
            "test_mae": float(eval_metrics.get("mae", 0.0)),
            "test_rmse": float(eval_metrics.get("rmse", 0.0)),
            "test_mse": float(eval_metrics.get("loss", 0.0)),
        },
        "history": {
            "epochs_trained": len(history.history.get("loss", [])),
            "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
        },
        "data": {
            "split_counts": {
                "train": len(prepared.train_records),
                "val": len(prepared.val_records),
                "test": len(prepared.test_records),
            },
            "skipped_missing_image": prepared.skipped_missing_image,
            "skipped_invalid_image": prepared.skipped_invalid_image,
        },
        "training": {
            "freeze_backbone": bool(config.freeze_backbone),
            "batch_size": int(config.batch_size),
            "epochs": int(config.epochs),
            "learning_rate": float(config.learning_rate),
        },
        "weather_scaler": {
            "mean": weather_mean.tolist(),
            "std": weather_std.tolist(),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    metrics_path = config.artifact_dir / "cnn_weather_metadata.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info(
        "CNN+Weather training complete: test_mae=%.4f test_rmse=%.4f",
        float(eval_metrics.get("mae", 0.0)),
        float(eval_metrics.get("rmse", 0.0)),
    )

    return {
        "test_loss": float(eval_metrics.get("loss", 0.0)),
        "test_mae": float(eval_metrics.get("mae", 0.0)),
        "test_rmse": float(eval_metrics.get("rmse", 0.0)),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LuxAeterna CNN + Weather predictor")
    parser.add_argument("--data-path", default="data/processed/global_dataset")
    parser.add_argument("--artifact-dir", default="models/artifacts")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Log train/val progress every N steps")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained CNN backbone")
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()
    config = CnnWeatherTrainingConfig(
        data_path=Path(args.data_path),
        artifact_dir=Path(args.artifact_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_every_n_steps=args.log_every_n_steps,
        freeze_backbone=args.freeze_backbone,
    )
    metrics = train(config)
    LOGGER.info("CNN+Weather training complete: %s", metrics)


if __name__ == "__main__":
    main()
