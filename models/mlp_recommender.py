from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

LOGGER = logging.getLogger("luxaeterna.models.mlp_recommender")

GENRE_CLASSES = ["landscape", "golden_hour", "night_astro", "street", "moody"]
WEATHER_STATES = ["clear", "partly_cloudy", "cloudy", "rain", "fog", "snow", "storm", "unknown"]
GENRE_TO_INDEX = {genre: idx for idx, genre in enumerate(GENRE_CLASSES)}


@dataclass(slots=True)
class MlpTrainingConfig:
    features_path: Path
    artifact_dir: Path
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 0.001


def generate_synthetic_bootstrap_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    alqs = rng.uniform(0.0, 1.0, n_samples)
    weather = rng.choice(WEATHER_STATES, size=n_samples, p=[0.2, 0.18, 0.18, 0.16, 0.12, 0.08, 0.05, 0.03])
    tod = rng.uniform(0.0, 24.0, n_samples)
    sin_time = np.sin(2 * np.pi * tod / 24.0)
    cos_time = np.cos(2 * np.pi * tod / 24.0)

    genre: list[str] = []
    for idx in range(n_samples):
        score = alqs[idx]
        state = weather[idx]
        if score > 0.78 and state in {"clear", "partly_cloudy"}:
            genre.append("golden_hour")
        elif score < 0.2:
            genre.append("night_astro")
        elif state in {"rain", "fog", "storm"}:
            genre.append("moody")
        elif state in {"cloudy", "snow"} and score > 0.45:
            genre.append("landscape")
        else:
            genre.append("street")

    return pd.DataFrame(
        {
            "alqs_norm": alqs,
            "weather_state": weather,
            "sin_time": sin_time,
            "cos_time": cos_time,
            "genre": genre,
        }
    )


def _load_or_bootstrap(path: Path) -> tuple[pd.DataFrame, bool]:
    if path.exists():
        frame = pd.read_parquet(path)
        required_cols = {"alqs_norm", "weather_state", "sin_time", "cos_time", "genre"}
        if required_cols.issubset(frame.columns):
            return frame.dropna(subset=list(required_cols)), False
        LOGGER.warning("Feature file missing columns; synthetic bootstrap data will be used")
    return generate_synthetic_bootstrap_data(), True


def build_model(input_dim: int, learning_rate: float) -> keras.Model:
    inputs = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.LayerNormalization(name="input_norm")(inputs)
    x = keras.layers.Dense(128, activation="swish", name="dense_128")(x)
    x = keras.layers.BatchNormalization(name="bn_128")(x)
    x = keras.layers.Dropout(0.35, name="dropout_128")(x)
    x = keras.layers.Dense(64, activation="swish", name="dense_64")(x)
    x = keras.layers.BatchNormalization(name="bn_64")(x)
    x = keras.layers.Dropout(0.25, name="dropout_64")(x)
    x = keras.layers.Dense(32, activation="swish", name="dense_32")(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="luxaeterna_mlp_recommender")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


def _safe_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use stratified split when possible, otherwise fall back to unstratified split."""
    class_counts = np.bincount(y_labels, minlength=len(GENRE_CLASSES))
    min_non_zero_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    stratify_labels = y_labels if min_non_zero_count >= 2 else None

    if stratify_labels is None:
        LOGGER.warning("Not enough samples per class for stratified split; using random split")

    return train_test_split(
        x,
        y,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )


def train(config: MlpTrainingConfig) -> dict[str, float]:
    keras.utils.set_random_seed(42)
    frame, used_bootstrap = _load_or_bootstrap(config.features_path)

    frame = frame.copy()
    frame["genre"] = frame["genre"].astype(str)
    frame = frame[frame["genre"].isin(GENRE_CLASSES)]
    if frame.empty:
        raise ValueError("No rows with supported genre labels were found")

    alqs_scaler = MinMaxScaler()
    alqs_scaled = alqs_scaler.fit_transform(frame[["alqs_norm"]])

    weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    weather_encoded = weather_encoder.fit_transform(frame[["weather_state"]])

    x = np.concatenate(
        [
            alqs_scaled,
            weather_encoded,
            frame[["sin_time", "cos_time"]].to_numpy(dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    y_labels = frame["genre"].map(GENRE_TO_INDEX).to_numpy(dtype=np.int64)
    y = tf.keras.utils.to_categorical(y_labels, num_classes=5)

    x_train, x_temp, y_train, y_temp, y_labels_train, y_labels_temp = _safe_train_test_split(
        x,
        y,
        y_labels,
        test_size=0.30,
        random_state=42,
    )
    x_val, x_test, y_val, y_test, y_labels_val, y_labels_test = _safe_train_test_split(
        x_temp,
        y_temp,
        y_labels_temp,
        test_size=0.50,
        random_state=42,
    )

    model = build_model(input_dim=x.shape[1], learning_rate=config.learning_rate)

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_labels_train),
        y=y_labels_train,
    )
    class_weights = {int(k): float(v) for k, v in zip(np.unique(y_labels_train), class_weights_values)}

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=14, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    y_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    precision = precision_score(y_labels_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_labels_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_labels_test, y_pred, average="weighted", zero_division=0)

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    keras_path = config.artifact_dir / "mlp_recommender.keras"
    model.save(keras_path)

    savedmodel_path = config.artifact_dir / "mlp_recommender_savedmodel"
    try:
        model.export(savedmodel_path)
    except Exception:
        tf.saved_model.save(model, str(savedmodel_path))

    joblib.dump(
        {
            "weather_encoder": weather_encoder,
            "genre_to_index": GENRE_TO_INDEX,
            "index_to_genre": {idx: genre for idx, genre in enumerate(GENRE_CLASSES)},
            "alqs_scaler": alqs_scaler,
            "genre_classes": GENRE_CLASSES,
        },
        config.artifact_dir / "mlp_aux.joblib",
    )

    metadata = {
        "version": version,
        "input_dim": int(x.shape[1]),
        "used_bootstrap_data": bool(used_bootstrap),
        "weather_categories": weather_encoder.categories_[0].tolist(),
        "label_classes": GENRE_CLASSES,
        "metrics": {
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(weighted_f1),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (config.artifact_dir / "mlp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("MLP precision=%.4f recall=%.4f weighted_f1=%.4f", precision, recall, weighted_f1)
    return {
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(weighted_f1),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LuxAeterna MLP recommender")
    parser.add_argument("--features-path", default="data/processed/classifier_features.parquet")
    parser.add_argument("--artifact-dir", default="models/artifacts")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = _build_arg_parser().parse_args()
    config = MlpTrainingConfig(
        features_path=Path(args.features_path),
        artifact_dir=Path(args.artifact_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    metrics = train(config)
    LOGGER.info("MLP training complete: %s", metrics)


if __name__ == "__main__":
    main()
