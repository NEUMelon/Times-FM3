"""Two-level caching system for experiment predictions and metric summaries."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from probe.config import CACHE_L1_DIR, CACHE_L2_DIR


def compute_cache_key(
    model: str,
    dataset: str,
    condition: str,
    seed: int,
    context_length: int,
    horizon: int,
    **kwargs,
) -> str:
    """Compute a deterministic SHA256 cache key from experimental parameters."""
    params = {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "context_length": context_length,
        "horizon": horizon,
        **kwargs,
    }
    # Sort keys for deterministic hash
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class ExperimentCache:
    """Handles Level-1 (metric summaries) and Level-2 (fp16 quantile predictions) persistence."""

    def __init__(self, l1_dir: Optional[Path] = None, l2_dir: Optional[Path] = None):
        self.l1_dir = Path(l1_dir or CACHE_L1_DIR)
        self.l2_dir = Path(l2_dir or CACHE_L2_DIR)
        self.l1_dir.mkdir(parents=True, exist_ok=True)
        self.l2_dir.mkdir(parents=True, exist_ok=True)

    def save_l1_metrics(self, key: str, df_metrics: pd.DataFrame):
        """Save Level-1 metric dataframe (stride 1 error summaries) as parquet."""
        path = self.l1_dir / f"{key}.parquet"
        df_metrics.to_parquet(path, index=False)

    def load_l1_metrics(self, key: str) -> Optional[pd.DataFrame]:
        """Load Level-1 metrics if cached."""
        path = self.l1_dir / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_l2_forecasts(
        self,
        key: str,
        point_preds: np.ndarray,
        quant_preds: Optional[np.ndarray],
        window_indices: List[int],
    ):
        """Save Level-2 fp16 quantile forecasts for sampled windows as npz."""
        path = self.l2_dir / f"{key}.npz"
        data = {
            "point": point_preds.astype(np.float16),
            "window_indices": np.array(window_indices, dtype=np.int32),
        }
        if quant_preds is not None:
            data["quantiles"] = quant_preds.astype(np.float16)
        np.savez_compressed(path, **data)

    def load_l2_forecasts(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Load Level-2 forecasts if cached."""
        path = self.l2_dir / f"{key}.npz"
        if path.exists():
            with np.load(path) as data:
                return {k: data[k] for k in data.files}
        return None
