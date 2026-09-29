"""Configuration and global constants for TimesFM-3 probing experiments."""

import os
from pathlib import Path

# Paths
LOCAL_DIR = Path(__file__).resolve().parent.parent
REMOTE_WORK_DIR = Path("/root/autodl-tmp/DASFAA")
DATA_DIR = Path("/root/autodl-tmp/data/ltsf")
CACHE_DIR = Path("/root/autodl-tmp/cache")
CACHE_L1_DIR = CACHE_DIR / "l1"  # Aggregated error summaries (MSE, MAE, WQL, Cov)
CACHE_L2_DIR = CACHE_DIR / "l2"  # fp16 quantile forecasts for sampled windows
RESULTS_DIR = Path("/root/autodl-tmp/results")
MODEL_DIR = Path("/root/autodl-tmp/models/timesfm-3.0-pytorch")

# Horizons
DEFAULT_HORIZONS = [96, 192, 336, 720]
MAX_HORIZON = 720
DEFAULT_CONTEXT_LENGTH = 512

# Quantile levels matching official TimesFM-3
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Random seeds
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
PRIMARY_SEED = 42

# LTSF Dataset Metadata
# format: (name, filename, num_variates, freq, border_type)
LTSF_DATASETS = {
    "ETTh1": {"file": "ETTh1.csv", "variates": 7, "freq": "1h", "split": (12, 4, 4)},
    "ETTh2": {"file": "ETTh2.csv", "variates": 7, "freq": "1h", "split": (12, 4, 4)},
    "ETTm1": {"file": "ETTm1.csv", "variates": 7, "freq": "15min", "split": (12, 4, 4)},
    "ETTm2": {"file": "ETTm2.csv", "variates": 7, "freq": "15min", "split": (12, 4, 4)},
    "weather": {"file": "weather.csv", "variates": 21, "freq": "10min", "split": (0.7, 0.1, 0.2)},
    "electricity": {"file": "electricity.csv", "variates": 321, "freq": "1h", "split": (0.7, 0.1, 0.2)},
    "traffic": {"file": "traffic.csv", "variates": 862, "freq": "1h", "split": (0.7, 0.1, 0.2)},
    "solar": {"file": "solar.csv", "variates": 137, "freq": "10min", "split": (0.7, 0.1, 0.2)},
}

# Max variates per forward pass as defined in official TimesFM-3
MAX_VARIATES_PER_FORWARD = 32
