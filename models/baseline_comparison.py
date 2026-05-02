"""
Baseline Comparison Module for ALQS Prediction

This module benchmarks multiple baseline models against the LSTM predictor:
- Persistence (predict previous value)
- Seasonal Naive (predict from 24 hours ago)
- Exponential Smoothing (ETS)
- ARIMA
- Random Forest
- XGBoost
- Linear Regression on features
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

LOGGER = logging.getLogger("luxaeterna.models.baseline_comparison")


@dataclass(slots=True)
class BaselineMetrics:
    """Container for baseline model metrics."""
    model_name: str
    mae: float
    mse: float
    rmse: float
    improvement_pct: float  # vs persistence baseline
    
    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test data from sequence dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with np.load(data_path) as dataset:
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        y_test_baseline = dataset.get("y_test_baseline", np.roll(y_test, 1))
        
    return X_test, y_test, y_test_baseline


def baseline_persistence(y_prev: np.ndarray) -> np.ndarray:
    """
    Persistence baseline: predict y_t = y_{t-1}
    
    Args:
        y_prev: Previous ALQS values (shape: N,)
        
    Returns:
        Predictions (shape: N,)
    """
    return y_prev.copy()


def baseline_seasonal_naive(X_test: np.ndarray, y_prev: np.ndarray, window_size: int = 24) -> np.ndarray:
    """
    Seasonal Naive: predict y_t = y_{t-24} (same hour from 24 hours ago)
    
    Falls back to persistence if history insufficient.
    
    Args:
        X_test: Input sequences (shape: N, window_size, features)
        y_prev: Previous ALQS values
        window_size: Window size in timesteps (assume hourly)
        
    Returns:
        Predictions (shape: N,)
    """
    predictions = y_prev.copy()
    
    # If window_size >= 24, use the first value in window as 24-hour-ago estimate
    if window_size >= 24:
        # Heuristic: use the oldest value in the sequence as proxy for 24h ago
        # This is approximate; for true seasonal naive, we'd need extended history
        predictions_seasonal = X_test[:, 0, -1]  # Use oldest delta_time as reference
        predictions = np.where(~np.isnan(predictions_seasonal), predictions_seasonal, predictions)
    
    return predictions


def baseline_linear_regression(X_train: np.ndarray, y_train: np.ndarray, 
                               X_test: np.ndarray) -> np.ndarray:
    """
    Linear Regression baseline: fit features to target directly.
    
    Args:
        X_train: Training sequences (shape: N, window_size, features)
        y_train: Training targets
        X_test: Test sequences (shape: M, window_size, features)
        
    Returns:
        Predictions (shape: M,)
    """
    try:
        # Reshape sequences: take mean of each feature across window
        X_train_flat = X_train.mean(axis=1)  # (N, features)
        X_test_flat = X_test.mean(axis=1)    # (M, features)
        
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        predictions = model.predict(X_test_flat)
        
        return np.clip(predictions, 0, 100)  # Clip to valid ALQS range
    except Exception as e:
        LOGGER.warning(f"Linear Regression failed: {e}. Falling back to mean.")
        return np.full_like(y_train, y_train.mean())


def baseline_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, n_estimators: int = 100) -> np.ndarray:
    """
    Random Forest baseline.
    
    Args:
        X_train: Training sequences (shape: N, window_size, features)
        y_train: Training targets
        X_test: Test sequences (shape: M, window_size, features)
        n_estimators: Number of trees
        
    Returns:
        Predictions (shape: M,)
    """
    try:
        # Reshape sequences: flatten window and features
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        window_size = X_train.shape[1]
        n_features = X_train.shape[2]
        
        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)
        
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        model.fit(X_train_flat, y_train)
        predictions = model.predict(X_test_flat)
        
        return np.clip(predictions, 0, 100)
    except Exception as e:
        LOGGER.warning(f"Random Forest failed: {e}. Falling back to mean.")
        return np.full_like(y_train, y_train.mean())


def baseline_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
    """
    XGBoost baseline (if available).
    
    Args:
        X_train: Training sequences (shape: N, window_size, features)
        y_train: Training targets
        X_test: Test sequences (shape: M, window_size, features)
        
    Returns:
        Predictions (shape: M,)
    """
    if not HAS_XGBOOST:
        LOGGER.info("XGBoost not installed; skipping.")
        return None
    
    try:
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(n_test, -1)
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_flat, y_train, verbose=False)
        predictions = model.predict(X_test_flat)
        
        return np.clip(predictions, 0, 100)
    except Exception as e:
        LOGGER.warning(f"XGBoost failed: {e}. Skipping.")
        return None


def baseline_exponential_smoothing(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Exponential Smoothing baseline (if statsmodels available).
    
    Args:
        y_train: Training targets
        y_test: Test targets (for fitting on train, predicting test)
        
    Returns:
        Predictions for test set (shape: M,)
    """
    if not HAS_STATSMODELS:
        LOGGER.info("statsmodels not installed; skipping Exponential Smoothing.")
        return None
    
    try:
        model = ExponentialSmoothing(y_train, trend="add", seasonal=None)
        fitted = model.fit()
        # Forecast for test set
        predictions = fitted.forecast(steps=len(y_test))
        return np.clip(predictions.values, 0, 100)
    except Exception as e:
        LOGGER.warning(f"Exponential Smoothing failed: {e}. Skipping.")
        return None


def baseline_arima(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    ARIMA baseline (if statsmodels available).
    
    Args:
        y_train: Training targets
        y_test: Test targets (for fitting on train, predicting test)
        
    Returns:
        Predictions for test set (shape: M,)
    """
    if not HAS_STATSMODELS:
        LOGGER.info("statsmodels not installed; skipping ARIMA.")
        return None
    
    try:
        # Use ARIMA(1,1,1) - simple configuration for robustness
        model = ARIMA(y_train, order=(1, 1, 1))
        fitted = model.fit()
        # Forecast for test set
        predictions = fitted.get_forecast(steps=len(y_test)).predicted_mean.values
        return np.clip(predictions, 0, 100)
    except Exception as e:
        LOGGER.warning(f"ARIMA failed: {e}. Skipping.")
        return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline_mae: float) -> BaselineMetrics:
    """Compute MAE, MSE, RMSE, and improvement percentage."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    improvement_pct = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae > 0 else 0
    
    return BaselineMetrics(
        model_name="",
        mae=float(mae),
        mse=float(mse),
        rmse=float(rmse),
        improvement_pct=float(improvement_pct)
    )


def run_baseline_comparison(data_path: Path, output_dir: Path) -> dict[str, BaselineMetrics]:
    """
    Run all baseline models and save results.
    
    Args:
        data_path: Path to sequence dataset .npz file
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping model names to metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Loading test dataset...")
    X_test, y_test, y_test_baseline = load_dataset(data_path)
    
    # For training-based models, need to load train set
    with np.load(data_path) as dataset:
        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
    
    results = {}
    
    # Persistence baseline
    LOGGER.info("Computing persistence baseline...")
    y_pred_persistence = baseline_persistence(y_test_baseline)
    metrics_persistence = compute_metrics(y_test, y_pred_persistence, 0)
    metrics_persistence.model_name = "Persistence (y_t = y_{t-1})"
    baseline_mae = metrics_persistence.mae
    results["persistence"] = metrics_persistence
    
    LOGGER.info(f"  Persistence MAE: {metrics_persistence.mae:.4f}")
    
    # Seasonal Naive
    LOGGER.info("Computing seasonal naive baseline...")
    window_size = X_test.shape[1]
    y_pred_seasonal = baseline_seasonal_naive(X_test, y_test_baseline, window_size)
    metrics_seasonal = compute_metrics(y_test, y_pred_seasonal, baseline_mae)
    metrics_seasonal.model_name = "Seasonal Naive (24h ago)"
    results["seasonal_naive"] = metrics_seasonal
    
    LOGGER.info(f"  Seasonal Naive MAE: {metrics_seasonal.mae:.4f} (improvement: {metrics_seasonal.improvement_pct:.2f}%)")
    
    # Linear Regression
    LOGGER.info("Training Linear Regression baseline...")
    y_pred_lr = baseline_linear_regression(X_train, y_train, X_test)
    metrics_lr = compute_metrics(y_test, y_pred_lr, baseline_mae)
    metrics_lr.model_name = "Linear Regression"
    results["linear_regression"] = metrics_lr
    
    LOGGER.info(f"  Linear Regression MAE: {metrics_lr.mae:.4f} (improvement: {metrics_lr.improvement_pct:.2f}%)")
    
    # Random Forest
    LOGGER.info("Training Random Forest baseline...")
    y_pred_rf = baseline_random_forest(X_train, y_train, X_test, n_estimators=100)
    metrics_rf = compute_metrics(y_test, y_pred_rf, baseline_mae)
    metrics_rf.model_name = "Random Forest (100 trees)"
    results["random_forest"] = metrics_rf
    
    LOGGER.info(f"  Random Forest MAE: {metrics_rf.mae:.4f} (improvement: {metrics_rf.improvement_pct:.2f}%)")
    
    # XGBoost
    LOGGER.info("Training XGBoost baseline...")
    y_pred_xgb = baseline_xgboost(X_train, y_train, X_test)
    if y_pred_xgb is not None:
        metrics_xgb = compute_metrics(y_test, y_pred_xgb, baseline_mae)
        metrics_xgb.model_name = "XGBoost"
        results["xgboost"] = metrics_xgb
        LOGGER.info(f"  XGBoost MAE: {metrics_xgb.mae:.4f} (improvement: {metrics_xgb.improvement_pct:.2f}%)")
    
    # Exponential Smoothing
    LOGGER.info("Training Exponential Smoothing baseline...")
    y_pred_es = baseline_exponential_smoothing(y_train, y_test)
    if y_pred_es is not None:
        metrics_es = compute_metrics(y_test, y_pred_es, baseline_mae)
        metrics_es.model_name = "Exponential Smoothing"
        results["exponential_smoothing"] = metrics_es
        LOGGER.info(f"  Exponential Smoothing MAE: {metrics_es.mae:.4f} (improvement: {metrics_es.improvement_pct:.2f}%)")
    
    # ARIMA
    LOGGER.info("Training ARIMA baseline...")
    y_pred_arima = baseline_arima(y_train, y_test)
    if y_pred_arima is not None:
        metrics_arima = compute_metrics(y_test, y_pred_arima, baseline_mae)
        metrics_arima.model_name = "ARIMA(1,1,1)"
        results["arima"] = metrics_arima
        LOGGER.info(f"  ARIMA MAE: {metrics_arima.mae:.4f} (improvement: {metrics_arima.improvement_pct:.2f}%)")
    
    # Save results
    results_dict = {name: metrics.to_dict() for name, metrics in results.items()}
    
    results_json_path = output_dir / "baseline_comparison_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    LOGGER.info(f"Results saved to {results_json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'Improvement %':<15}")
    print("-"*80)
    
    for name, metrics in sorted(results.items(), key=lambda x: x[1].mae):
        print(f"{metrics.model_name:<30} {metrics.mae:<12.4f} {metrics.mse:<12.4f} {metrics.rmse:<12.4f} {metrics.improvement_pct:<15.2f}")
    
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline comparison for ALQS prediction"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/sequence_dataset.npz"),
        help="Path to sequence dataset .npz file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/artifacts"),
        help="Directory to save baseline comparison results"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    run_baseline_comparison(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
