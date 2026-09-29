"""Channel selection and grouping strategies: Column-order, Random, Correlation clustering, Top-K, Bottom-K.
Ensures that all subset selection functions strictly return (subset, target_pos) where target_pos is always index 0.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def chunk_by_column_order(num_variates: int, chunk_size: int = 32) -> List[List[int]]:
    """Official column-order chunking: consecutive blocks of chunk_size."""
    groups = []
    for i in range(0, num_variates, chunk_size):
        groups.append(list(range(i, min(i + chunk_size, num_variates))))
    return groups


def chunk_by_random(num_variates: int, chunk_size: int = 32, seed: int = 42) -> List[List[int]]:
    """Randomly partitions channels into groups of size chunk_size."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_variates).tolist()
    groups = []
    for i in range(0, num_variates, chunk_size):
        groups.append(perm[i : min(i + chunk_size, num_variates)])
    return groups


def select_top_k_correlated(
    corr_matrix: np.ndarray,
    target_idx: int,
    k: int = 32,
) -> Tuple[List[int], int]:
    """
    Selects target_idx plus the top (k - 1) channels with highest |correlation| to target.
    Guaranteed: target is strictly at index 0.
    Returns: (subset, target_pos=0)
    """
    N = corr_matrix.shape[0]
    if k >= N:
        other_indices = [i for i in range(N) if i != target_idx]
        return [target_idx] + other_indices, 0

    corrs = np.abs(corr_matrix[target_idx])
    corrs = corrs.copy()
    corrs[target_idx] = -1.0  # Exclude target itself
    top_indices = np.argsort(corrs)[::-1][:(k - 1)].tolist()
    subset = [target_idx] + top_indices
    return subset, 0


def select_bottom_k_correlated(
    corr_matrix: np.ndarray,
    target_idx: int,
    k: int = 32,
) -> Tuple[List[int], int]:
    """
    Selects target_idx plus the (k - 1) channels with lowest |correlation| to target.
    Guaranteed: target is strictly at index 0.
    Returns: (subset, target_pos=0)
    """
    N = corr_matrix.shape[0]
    if k >= N:
        other_indices = [i for i in range(N) if i != target_idx]
        return [target_idx] + other_indices, 0

    corrs = np.abs(corr_matrix[target_idx])
    corrs = corrs.copy()
    corrs[target_idx] = 999.0  # Exclude target
    bot_indices = np.argsort(corrs)[:(k - 1)].tolist()
    subset = [target_idx] + bot_indices
    return subset, 0


def select_random_k(
    num_variates: int,
    target_idx: int,
    k: int = 32,
    seed: int = 42,
) -> Tuple[List[int], int]:
    """
    Selects target_idx plus (k - 1) randomly chosen other channels.
    Guaranteed: target is strictly at index 0.
    Returns: (subset, target_pos=0)
    """
    if k >= num_variates:
        other_indices = [i for i in range(num_variates) if i != target_idx]
        return [target_idx] + other_indices, 0

    rng = np.random.RandomState(seed)
    other_indices = [idx for idx in range(num_variates) if idx != target_idx]
    sampled_others = rng.choice(other_indices, size=k - 1, replace=False).tolist()
    subset = [target_idx] + sampled_others
    return subset, 0


def sample_stratified_targets(
    corr_matrix: np.ndarray,
    num_targets: int = 64,
    chunk_size: int = 32,
    seed: int = 42,
) -> List[int]:
    """
    Stratified target channel sampling across 4 inter-channel correlation quartiles (Tasks 7 & 10).
    Excludes any channels residing in single-channel blocks (e.g. Electricity ch 320 in a 1-channel block).
    """
    N = corr_matrix.shape[0]
    # Filter out single-channel block targets
    valid_channels = []
    for i in range(N):
        block_start = (i // chunk_size) * chunk_size
        block_end = min(block_start + chunk_size, N)
        if (block_end - block_start) > 1:
            valid_channels.append(i)

    if not valid_channels:
        valid_channels = list(range(N))

    # Mean absolute correlation to all other channels in dataset
    mean_corrs = []
    for i in valid_channels:
        other_mask = np.ones(N, dtype=bool)
        other_mask[i] = False
        mean_corrs.append(float(np.mean(np.abs(corr_matrix[i, other_mask]))))
    mean_corrs = np.array(mean_corrs)

    # 4 Quartiles
    q25, q50, q75 = np.percentile(mean_corrs, [25, 50, 75])
    q_groups: List[List[int]] = [[], [], [], []]
    for ch, c in zip(valid_channels, mean_corrs):
        if c <= q25:
            q_groups[0].append(ch)
        elif c <= q50:
            q_groups[1].append(ch)
        elif c <= q75:
            q_groups[2].append(ch)
        else:
            q_groups[3].append(ch)

    rng = np.random.RandomState(seed)
    n_per_q = max(1, num_targets // 4)
    selected = []
    for g in q_groups:
        if len(g) == 0:
            continue
        k_take = min(len(g), n_per_q)
        chosen = rng.choice(g, size=k_take, replace=False).tolist()
        selected.extend(chosen)

    # Fill if needed
    if len(selected) < num_targets and len(valid_channels) >= num_targets:
        rem = [ch for ch in valid_channels if ch not in selected]
        needed = num_targets - len(selected)
        extra = rng.choice(rem, size=needed, replace=False).tolist()
        selected.extend(extra)

    return sorted(selected[:num_targets])
