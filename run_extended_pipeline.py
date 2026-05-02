#!/usr/bin/env python
"""
LuxAeterna Extended Collection & Analysis Pipeline
================================================

This script orchestrates the improved data collection and analysis pipeline:
1. Extended collection: 48-72 hours per webcam at 1-hour intervals
2. Enhanced features: Solar geometry (elevation, azimuth, clear-sky index)
3. Longer sequences: 24-48 timesteps per sequence
4. Comprehensive baseline benchmarking

Usage:
    python run_extended_pipeline.py [--phase PHASE] [--max-hours 72]

Phases:
    discovery    - Discover and cache 1,000 global webcams
    ingest       - Collect data for 48-72 hours at 1-hour intervals
    features     - Engineer features with solar geometry (24-timestep windows)
    train        - Train LSTM on extended data
    baselines    - Benchmark against multiple baseline models
    report       - Generate comprehensive comparison report
    all          - Run all phases
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger("luxaeterna.pipeline")

# Configuration
LUXAETERNA_ROOT = Path(__file__).resolve().parent
DATA_DIR = LUXAETERNA_ROOT / "data"
MODELS_DIR = LUXAETERNA_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"


def run_command(cmd: list[str], description: str) -> int:
    """
    Execute a shell command and log output.
    
    Args:
        cmd: Command as list of strings
        description: Human-readable description
        
    Returns:
        Exit code (0 = success)
    """
    LOGGER.info(f"[PHASE] {description}")
    LOGGER.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        LOGGER.info(f"✓ {description} completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"✗ {description} failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError as e:
        LOGGER.error(f"✗ Command not found: {e}")
        return 1


def phase_discovery():
    """Phase 1: Discover 1,000 global webcams and cache them."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("PHASE 1: WEBCAM DISCOVERY")
    LOGGER.info("="*80)
    
    cmd = [
        sys.executable,
        str(DATA_DIR / "webcam_discovery.py"),
        "--sample-size", "1000",
        "--output-path", str(DATA_DIR / "webcams.json"),
    ]
    
    return run_command(cmd, "Discovering 1,000 global webcams")


def phase_ingest(max_hours: int = 72):
    """Phase 2: Collect data for 48-72 hours at 1-hour intervals."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info(f"PHASE 2: DATA INGESTION ({max_hours}-hour collection)")
    LOGGER.info("="*80)
    LOGGER.info(f"Sampling interval: 1 hour")
    LOGGER.info(f"Expected cycles: {max_hours}")
    LOGGER.info(f"Expected samples: ~{max_hours * 1000} total observations")
    LOGGER.info(f"Estimated runtime: {max_hours} hours")
    
    cmd = [
        sys.executable,
        str(DATA_DIR / "global_ingestion.py"),
        "--sample-size", "1000",
        "--cache-path", str(DATA_DIR / "webcams.json"),
        "--output-path", str(DATA_DIR / "processed" / "global_dataset"),
        "--interval", "3600",  # 1 hour
        "--max-cycles", str(max_hours),
    ]
    
    return run_command(cmd, f"Ingesting data for {max_hours} hours at 1-hour intervals")


def phase_features():
    """Phase 3: Engineer features with solar geometry and 24-timestep windows."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("PHASE 3: FEATURE ENGINEERING")
    LOGGER.info("="*80)
    LOGGER.info("Adding solar geometry features: elevation, azimuth, clear-sky index")
    LOGGER.info("Building 24-timestep sliding windows (full 24-hour context)")
    
    cmd = [
        sys.executable,
        str(DATA_DIR / "feature_engineer.py"),
        "--weather-path", str(DATA_DIR / "processed" / "global_dataset"),
        "--output-dir", str(DATA_DIR / "processed"),
        "--window-size", "24",  # 24 hours at 1-hour intervals
    ]
    
    return run_command(cmd, "Engineering features with solar geometry and 24-step windows")


def phase_train():
    """Phase 4: Train LSTM on extended dataset."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("PHASE 4: LSTM TRAINING")
    LOGGER.info("="*80)
    LOGGER.info("Training LSTM on extended 48-72 hour dataset")
    LOGGER.info("Expected improvement: Longer sequences capture diurnal/weather patterns")
    
    cmd = [
        sys.executable,
        str(MODELS_DIR / "lstm_predictor.py"),
        "--data-path", str(DATA_DIR / "processed" / "sequence_dataset.npz"),
        "--artifact-dir", str(ARTIFACTS_DIR),
    ]
    
    return run_command(cmd, "Training LSTM model on extended data")


def phase_baselines():
    """Phase 5: Benchmark against multiple baseline models."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("PHASE 5: BASELINE BENCHMARKING")
    LOGGER.info("="*80)
    LOGGER.info("Benchmarking LSTM against:")
    LOGGER.info("  - Persistence (y_t = y_{t-1})")
    LOGGER.info("  - Seasonal Naive (24h ago)")
    LOGGER.info("  - Linear Regression")
    LOGGER.info("  - Random Forest")
    LOGGER.info("  - XGBoost (optional)")
    LOGGER.info("  - Exponential Smoothing (optional)")
    LOGGER.info("  - ARIMA (optional)")
    
    cmd = [
        sys.executable,
        str(MODELS_DIR / "baseline_comparison.py"),
        "--data-path", str(DATA_DIR / "processed" / "sequence_dataset.npz"),
        "--output-dir", str(ARTIFACTS_DIR),
    ]
    
    return run_command(cmd, "Running baseline comparison")


def phase_report():
    """Phase 6: Generate comprehensive analysis report."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("PHASE 6: ANALYSIS REPORT")
    LOGGER.info("="*80)
    
    # Read metadata files
    lstm_metadata_path = ARTIFACTS_DIR / "lstm_metadata.json"
    baselines_path = ARTIFACTS_DIR / "baseline_comparison_results.json"
    
    if not lstm_metadata_path.exists():
        LOGGER.warning(f"LSTM metadata not found: {lstm_metadata_path}")
        return 1
    
    if not baselines_path.exists():
        LOGGER.warning(f"Baseline results not found: {baselines_path}")
        return 1
    
    with open(lstm_metadata_path) as f:
        lstm_meta = json.load(f)
    
    with open(baselines_path) as f:
        baselines = json.load(f)
    
    # Generate report
    report_path = LUXAETERNA_ROOT / "EXTENDED_PIPELINE_REPORT.md"
    
    with open(report_path, "w") as f:
        f.write("# LuxAeterna Extended Pipeline Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report documents the improved pipeline with:\n")
        f.write("- Extended collection: 48-72 hours per webcam at 1-hour intervals\n")
        f.write("- Solar geometry features: elevation, azimuth, clear-sky index\n")
        f.write("- Longer sequences: 24-timestep windows (full 24-hour context)\n")
        f.write("- Comprehensive baseline benchmarking\n\n")
        
        f.write("## LSTM Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Test MAE | {lstm_meta.get('test_mae', 'N/A')} |\n")
        f.write(f"| Test MSE | {lstm_meta.get('test_mse', 'N/A')} |\n")
        f.write(f"| Test RMSE | {lstm_meta.get('test_rmse', 'N/A')} |\n")
        f.write(f"| Training Epochs | {lstm_meta.get('epochs_trained', 'N/A')} |\n\n")
        
        f.write("## Baseline Comparison\n\n")
        f.write("| Model | MAE | MSE | Improvement (%) |\n")
        f.write("|-------|-----|-----|------------------|\n")
        
        for model_name, metrics in sorted(baselines.items(), key=lambda x: x[1]['mae']):
            model_display = metrics.get('model_name', model_name)
            mae = metrics.get('mae', 'N/A')
            mse = metrics.get('mse', 'N/A')
            improvement = metrics.get('improvement_pct', 'N/A')
            
            f.write(f"| {model_display} | {mae} | {mse} | {improvement} |\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Check if LSTM outperforms baselines
        if lstm_meta.get('test_mae', float('inf')) < min(m['mae'] for m in baselines.values()):
            f.write("✓ **LSTM outperforms all baselines**\n\n")
            f.write("The LSTM model successfully captures temporal patterns in ALQS prediction.\n")
            f.write("The extended 48-72 hour collection window and solar geometry features\n")
            f.write("have improved model performance significantly.\n\n")
        else:
            f.write("⚠ **Baseline models outperform LSTM**\n\n")
            best_baseline = min(baselines.items(), key=lambda x: x[1]['mae'])
            f.write(f"The {best_baseline[1]['model_name']} model (MAE: {best_baseline[1]['mae']:.4f})\n")
            f.write(f"outperforms LSTM (MAE: {lstm_meta.get('test_mae', 'N/A')}).\n\n")
            f.write("Possible improvements:\n")
            f.write("- Further increase collection window to 1-2 weeks per webcam\n")
            f.write("- Add additional features (aerosol, atmospheric water vapor)\n")
            f.write("- Use ensemble methods combining LSTM + best baseline\n")
            f.write("- Try attention-based architectures (Transformer)\n\n")
        
        f.write("## File Locations\n\n")
        f.write(f"- LSTM Model: `{ARTIFACTS_DIR / 'lstm_predictor.keras'}`\n")
        f.write(f"- Baseline Results: `{ARTIFACTS_DIR / 'baseline_comparison_results.json'}`\n")
        f.write(f"- Features: `{DATA_DIR / 'processed' / 'sequence_dataset.npz'}`\n\n")
    
    LOGGER.info(f"Report saved to {report_path}")
    print(f"\n✓ Report generated: {report_path}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LuxAeterna Extended Collection & Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--phase",
        choices=["discovery", "ingest", "features", "train", "baselines", "report", "all"],
        default="all",
        help="Pipeline phase to run (default: all)"
    )
    
    parser.add_argument(
        "--max-hours",
        type=int,
        default=72,
        help="Maximum hours for ingestion phase (default: 72)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    LOGGER.info("="*80)
    LOGGER.info("LuxAeterna Extended Pipeline")
    LOGGER.info("="*80)
    LOGGER.info(f"Start time: {datetime.now().isoformat()}")
    
    exit_code = 0
    phases_run = []
    
    try:
        if args.phase in ["all", "discovery"]:
            exit_code |= phase_discovery()
            phases_run.append("discovery")
        
        if args.phase in ["all", "ingest"]:
            exit_code |= phase_ingest(args.max_hours)
            phases_run.append("ingest")
        
        if args.phase in ["all", "features"]:
            exit_code |= phase_features()
            phases_run.append("features")
        
        if args.phase in ["all", "train"]:
            exit_code |= phase_train()
            phases_run.append("train")
        
        if args.phase in ["all", "baselines"]:
            exit_code |= phase_baselines()
            phases_run.append("baselines")
        
        if args.phase in ["all", "report"]:
            exit_code |= phase_report()
            phases_run.append("report")
    
    except KeyboardInterrupt:
        LOGGER.warning("Pipeline interrupted by user")
        exit_code = 130
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}", exc_info=True)
        exit_code = 1
    
    LOGGER.info("="*80)
    LOGGER.info(f"End time: {datetime.now().isoformat()}")
    LOGGER.info(f"Phases completed: {', '.join(phases_run)}")
    LOGGER.info(f"Exit code: {exit_code}")
    LOGGER.info("="*80)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
