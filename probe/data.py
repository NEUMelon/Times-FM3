"""Dataset loading, standard normalization, and sliding window slicing for LTSF benchmarks."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from probe.config import DATA_DIR, LTSF_DATASETS, PRIMARY_SEED


class LTSFDataset:
    """Manages loading, normalization, and window slicing for an LTSF dataset."""

    def __init__(
        self,
        name: str,
        data_dir: Optional[Path] = None,
        context_length: int = 512,
        horizon: int = 720,
    ):
        if name not in LTSF_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Must be one of {list(LTSF_DATASETS.keys())}")
        
        self.name = name
        self.info = LTSF_DATASETS[name]
        self.data_dir = Path(data_dir or DATA_DIR)
        self.context_length = context_length
        self.horizon = horizon

        self._load_and_preprocess()

    def _load_and_preprocess(self):
        filepath = self.data_dir / self.info["file"]
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

        # Check if file has header or date column
        # Standard LTSF CSVs usually have 'date' as first column, or plain matrix
        sample = pd.read_csv(filepath, nrows=5)
        if "date" in sample.columns:
            df = pd.read_csv(filepath)
            df_data = df.drop(columns=["date"])
        else:
            # Check if first column is timestamp
            try:
                pd.to_datetime(sample.iloc[:, 0])
                df = pd.read_csv(filepath)
                df_data = df.iloc[:, 1:]
            except Exception:
                df_data = pd.read_csv(filepath, header=None)

        # Convert to float32 numpy array [T, N]
        raw_values = df_data.values.astype(np.float32)
        self.raw_values = raw_values
        total_len = len(raw_values)
        num_variates = raw_values.shape[1]

        # Calculate standard LTSF split boundaries
        split_spec = self.info["split"]
        if isinstance(split_spec[0], int) and len(split_spec) == 3 and sum(split_spec) == 20:
            # ETT dataset: 12 months train, 4 months val, 4 months test
            # Hourly: 30 * 24 = 720 points/month
            # 15min: 30 * 24 * 4 = 2880 points/month
            freq = self.info["freq"]
            points_per_month = 2880 if "15min" in freq or "15m" in freq else 720
            border1s = [0, 12 * points_per_month - self.context_length, 12 * points_per_month + 4 * points_per_month - self.context_length]
            border2s = [12 * points_per_month, 12 * points_per_month + 4 * points_per_month, 12 * points_per_month + 8 * points_per_month]
        else:
            # Ratio split (e.g. 7:1:2)
            train_ratio, val_ratio, test_ratio = split_spec
            num_train = int(total_len * train_ratio)
            num_test = int(total_len * test_ratio)
            num_val = total_len - num_train - num_test

            border1s = [0, num_train - self.context_length, total_len - num_test - self.context_length]
            border2s = [num_train, num_train + num_val, total_len]

        self.border1s = border1s
        self.border2s = border2s

        # Train data slice strictly for computing scaler statistics
        train_data = raw_values[border1s[0] : border2s[0]]
        self.mean = np.mean(train_data, axis=0, keepdims=True)
        self.std = np.std(train_data, axis=0, keepdims=True)
        self.std[self.std < 1e-5] = 1.0  # prevent division by zero

        # Standardize entire array using train stats
        self.norm_values = (raw_values - self.mean) / self.std

        # Extract test segment
        # In LTSF, test evaluation starts from border1s[2] to border2s[2]
        self.test_data = self.norm_values[border1s[2] : border2s[2]]
        self.num_variates = num_variates
        self.total_test_len = len(self.test_data)

        # Max valid start index for a window: start + context_length + horizon <= total_test_len
        self.num_windows_stride1 = max(0, self.total_test_len - self.context_length - self.horizon + 1)

    def get_window(self, start_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single window.
        Returns:
            ctx: [num_variates, context_length]
            target: [num_variates, horizon]
        """
        end_ctx = start_idx + self.context_length
        end_target = end_ctx + self.horizon
        if end_target > self.total_test_len:
            raise IndexError(f"Window exceeds test data length ({end_target} > {self.total_test_len})")

        # Slice: [L, N] -> transpose to [N, L]
        ctx = self.test_data[start_idx:end_ctx].T
        target = self.test_data[end_ctx:end_target].T
        return ctx, target

    def get_sampled_window_indices(self, num_samples: int = 300, seed: int = PRIMARY_SEED) -> List[int]:
        """
        Uniformly sample window indices deterministically across test set.
        """
        if self.num_windows_stride1 <= num_samples:
            return list(range(self.num_windows_stride1))

        # Uniformly space windows or sample deterministically
        rng = np.random.RandomState(seed)
        step = max(1, self.num_windows_stride1 // num_samples)
        indices = [i * step for i in range(num_samples)]
        return [idx for idx in indices if idx < self.num_windows_stride1]


def load_dataset(name: str, context_length: int = 512, horizon: int = 720) -> LTSFDataset:
    """Convenience helper to load an LTSFDataset."""
    return LTSFDataset(name=name, context_length=context_length, horizon=horizon)
