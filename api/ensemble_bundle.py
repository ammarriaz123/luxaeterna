"""Load and run the power-weighted multiclass lighting ensemble (XGB + LSTM + MLP)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from tensorflow import keras

# Must match training order in models/ensemble_classifier.py and data pipeline.
LSTM_FEATURE_NAMES: tuple[str, ...] = (
    "latitude",
    "longitude",
    "temperature",
    "relative_humidity",
    "visibility",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "solar_elevation",
)

TABULAR_FEATURE_NAMES: tuple[str, ...] = (
    "latitude",
    "longitude",
    "temperature",
    "relative_humidity",
    "visibility",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "solar_elevation",
    "temp_lag_1",
    "rh_lag_1",
    "cloud_low_lag_1",
    "temp_lag_2",
    "rh_lag_2",
    "cloud_low_lag_2",
    "temp_lag_3",
    "rh_lag_3",
    "cloud_low_lag_3",
    "temp_change_1h",
    "cloud_low_change_1h",
)

CLASS_LABELS: tuple[str, ...] = (
    "no_event",
    "golden_hour_only",
    "dramatic_diffusion_only",
    "golden_hour_and_diffusion",
)

# FRONTEND_INTEGRATION_GUIDE.md — power-weighted ensemble
WEIGHT_XGB = 0.5
WEIGHT_LSTM = 0.35
WEIGHT_MLP = 0.15

LSTM_WINDOW = 6
N_LSTM_FEATURES = len(LSTM_FEATURE_NAMES)
N_TABULAR_FEATURES = len(TABULAR_FEATURE_NAMES)


def _wrap_layer_init_strip_kwargs(module_path: str, class_name: str, strip_keys: tuple[str, ...], marker: str) -> None:
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError):
        return
    if getattr(cls, marker, False):
        return
    _orig_init = cls.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        for k in strip_keys:
            kwargs.pop(k, None)
        return _orig_init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore[method-assign]
    setattr(cls, marker, True)


def _apply_keras_saved_model_compat_patches() -> None:
    """Artifacts saved with newer Keras may pass kwargs this stack's Keras 3 rejects (TF 2.16 / keras 3.12)."""
    _wrap_layer_init_strip_kwargs(
        "keras.src.layers.core.dense",
        "Dense",
        ("quantization_config",),
        "_luxaeterna_dense_quant_patch",
    )
    _wrap_layer_init_strip_kwargs(
        "keras.src.layers.normalization.batch_normalization",
        "BatchNormalization",
        ("renorm", "renorm_clipping", "renorm_momentum", "synchronized"),
        "_luxaeterna_bn_patch",
    )


def _load_keras_ensemble_model(path: Path) -> keras.Model:
    _apply_keras_saved_model_compat_patches()
    return keras.models.load_model(path, compile=False)


@dataclass(slots=True)
class EnsembleBundle:
    xgb_model: Any
    mlp_model: keras.Model
    mlp_scaler: Any
    lstm_model: keras.Model
    lstm_scaler: Any
    metadata: dict[str, Any]


def ensemble_artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "xgb": artifact_dir / "xgb_multiclass_model.json",
        "mlp": artifact_dir / "mlp_multiclass_model.keras",
        "mlp_scaler": artifact_dir / "mlp_multiclass_scaler.joblib",
        "lstm": artifact_dir / "lstm_multiclass_model.keras",
        "lstm_scaler": artifact_dir / "lstm_multiclass_scaler.joblib",
    }


def ensemble_loadable(artifact_dir: Path) -> bool:
    return all(p.exists() for p in ensemble_artifact_paths(artifact_dir).values())


def load_ensemble_bundle(artifact_dir: Path) -> EnsembleBundle:
    paths = ensemble_artifact_paths(artifact_dir)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(paths["xgb"]))
    mlp_model = _load_keras_ensemble_model(paths["mlp"])
    mlp_scaler = joblib.load(paths["mlp_scaler"])
    lstm_model = _load_keras_ensemble_model(paths["lstm"])
    lstm_scaler = joblib.load(paths["lstm_scaler"])

    metadata: dict[str, Any] = {}
    for fname in (
        "xgb_multiclass_metadata.json",
        "mlp_multiclass_metadata.json",
        "lstm_multiclass_metadata.json",
    ):
        p = artifact_dir / fname
        if p.exists():
            metadata[fname.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))

    return EnsembleBundle(
        xgb_model=xgb_model,
        mlp_model=mlp_model,
        mlp_scaler=mlp_scaler,
        lstm_model=lstm_model,
        lstm_scaler=lstm_scaler,
        metadata=metadata,
    )


def predict_lighting_event_probs(bundle: EnsembleBundle, sequence: np.ndarray, tabular: np.ndarray) -> np.ndarray:
    """
    sequence: (6, 10) float32 — hourly (or regular) steps, newest last.
    tabular: (21,) — lag features for the final timestep (XGB/MLP input).
    Returns combined probabilities shape (n_classes,).
    """
    if sequence.shape != (LSTM_WINDOW, N_LSTM_FEATURES):
        raise ValueError(f"sequence must be ({LSTM_WINDOW}, {N_LSTM_FEATURES})")
    if tabular.shape != (N_TABULAR_FEATURES,):
        raise ValueError(f"tabular must be ({N_TABULAR_FEATURES},)")

    X_tab = np.nan_to_num(tabular.astype(np.float64).reshape(1, -1), nan=0.0)
    xgb_probs = np.asarray(bundle.xgb_model.predict_proba(X_tab), dtype=np.float64)

    X_tab_scaled = bundle.mlp_scaler.transform(X_tab)
    mlp_probs = np.asarray(bundle.mlp_model.predict(X_tab_scaled, verbose=0), dtype=np.float64)

    seq_flat = sequence.reshape(-1, N_LSTM_FEATURES)
    seq_scaled = bundle.lstm_scaler.transform(seq_flat).reshape(1, LSTM_WINDOW, N_LSTM_FEATURES)
    lstm_probs = np.asarray(bundle.lstm_model.predict(seq_scaled, verbose=0), dtype=np.float64)

    if xgb_probs.ndim == 1:
        xgb_probs = xgb_probs.reshape(1, -1)
    if mlp_probs.ndim == 1:
        mlp_probs = mlp_probs.reshape(1, -1)
    if lstm_probs.ndim == 1:
        lstm_probs = lstm_probs.reshape(1, -1)

    combined = WEIGHT_XGB * xgb_probs + WEIGHT_LSTM * lstm_probs + WEIGHT_MLP * mlp_probs
    return combined.reshape(-1)
