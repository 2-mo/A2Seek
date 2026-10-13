"""Locate the checkout configuration and local output directories."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/A2Seek"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/A2Seek-R1-3B"
