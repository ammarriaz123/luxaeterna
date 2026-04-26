from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

LOGGER = logging.getLogger("luxaeterna.models.lstm_predictor")


@dataclass(slots=True)
class LstmTrainingConfig:
    data_path: Path
    artifact_dir: Path
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 0.001


def load_dataset(data_path: Path) -> dict[str, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    with np.load(data_path) as dataset:
        return {key: dataset[key] for key in dataset.files}


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(buffer_size=min(len(x), 4096), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(input_shape: tuple[int, int], learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(128, dropout=0.2, return_sequences=True),
            keras.layers.LSTM(64, dropout=0.2, return_sequences=False),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ],
        name="luxaeterna_lstm_predictor",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate_persistence_baseline(y_true: np.ndarray, y_pred_persistence: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred_persistence)))


def train(config: LstmTrainingConfig) -> dict[str, float]:
    dataset = load_dataset(config.data_path)

    x_train = dataset["X_train"].astype(np.float32)
    y_train = dataset["y_train"].astype(np.float32)
    x_val = dataset["X_val"].astype(np.float32)
    y_val = dataset["y_val"].astype(np.float32)
    x_test = dataset["X_test"].astype(np.float32)
    y_test = dataset["y_test"].astype(np.float32)
    baseline_prev_test = dataset["baseline_prev_test"].astype(np.float32)

    train_ds = make_tf_dataset(x_train, y_train, config.batch_size, training=True)
    val_ds = make_tf_dataset(x_val, y_val, config.batch_size, training=False)
    test_ds = make_tf_dataset(x_test, y_test, config.batch_size, training=False)

    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]), learning_rate=config.learning_rate)

    timestamp_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = config.artifact_dir / "tensorboard" / f"lstm_{timestamp_tag}"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir), histogram_freq=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    eval_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    y_pred = model.predict(x_test, verbose=0).reshape(-1)
    model_mae = float(np.mean(np.abs(y_test - y_pred)))
    baseline_mae = evaluate_persistence_baseline(y_test, baseline_prev_test)
    improvement_pct = (baseline_mae - model_mae) / max(baseline_mae, 1e-8) * 100.0

    residual_std = float(np.std(y_test - y_pred))

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    keras_path = config.artifact_dir / "lstm_predictor.keras"
    model.save(keras_path)

    savedmodel_path = config.artifact_dir / "lstm_predictor_savedmodel"
    try:
        model.export(savedmodel_path)
    except Exception:
        tf.saved_model.save(model, str(savedmodel_path))

    metadata = {
        "model_name": model.name,
        "version": timestamp_tag,
        "input_shape": [int(x_train.shape[1]), int(x_train.shape[2])],
        "metrics": {
            "test_loss": float(eval_metrics.get("loss", 0.0)),
            "test_mae": float(eval_metrics.get("mae", model_mae)),
            "baseline_mae": baseline_mae,
            "mae_improvement_pct": improvement_pct,
            "target_improvement_met": bool(improvement_pct > 20.0),
            "residual_std": residual_std,
        },
        "history": {
            "epochs_trained": len(history.history.get("loss", [])),
            "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    (config.artifact_dir / "lstm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if improvement_pct <= 20.0:
        LOGGER.warning(
            "Model MAE improvement over persistence baseline is %.2f%% (target > 20%% not met)",
            improvement_pct,
        )
    else:
        LOGGER.info("Model MAE improved over baseline by %.2f%%", improvement_pct)

    return {
        "test_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
        "residual_std": residual_std,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LuxAeterna LSTM predictor")
    parser.add_argument("--data-path", default="data/processed/sequence_dataset.npz")
    parser.add_argument("--artifact-dir", default="models/artifacts")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
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
    config = LstmTrainingConfig(
        data_path=Path(args.data_path),
        artifact_dir=Path(args.artifact_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    metrics = train(config)
    LOGGER.info("LSTM training complete: %s", metrics)


if __name__ == "__main__":
    main()
