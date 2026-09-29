"""Evaluation and diagnostic metrics: MSE, MAE, Mean Pinball Loss, WQL, Coverage, B, R, S, GC,
Window-level Stationary Block Bootstrap, Paired Wilcoxon Test, and Holm-Bonferroni Correction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from probe.config import QUANTILES


def calc_mse(pred: np.ndarray, target: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
    """Mean Squared Error."""
    return np.mean((pred - target) ** 2, axis=axis)


def calc_mae(pred: np.ndarray, target: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target), axis=axis)


def calc_mean_pinball_loss(
    quantiles: np.ndarray,
    target: np.ndarray,
    quantile_levels: List[float] = QUANTILES,
) -> float:
    """
    Mean pinball loss across all variates, horizons, and quantile levels (Task 8):
      L = mean_{(N, H, Q)} [ 2 * max(q * (y - y_q), (q - 1) * (y - y_q)) ]
    Equivalent to CRPS quantile approximation. Strictly returns a scalar float.
    quantiles: [..., H, len(quantile_levels)]
    target: [..., H]
    """
    target_exp = np.expand_dims(target, axis=-1)  # [..., H, 1]
    q_levels = np.array(quantile_levels, dtype=np.float64)

    diff = target_exp - quantiles
    loss = 2.0 * np.maximum(q_levels * diff, (q_levels - 1.0) * diff)
    return float(np.mean(loss))


def calc_wql(
    quantiles: np.ndarray,
    target: np.ndarray,
    quantile_levels: List[float] = QUANTILES,
) -> float:
    """
    Weighted Quantile Loss (WQL) across multiple quantile levels on unnormalized scale.
    Only valid for non-negative datasets where sum(|y|) is positive and well-conditioned.
    """
    target_exp = np.expand_dims(target, axis=-1)
    q_levels = np.array(quantile_levels, dtype=np.float64)

    diff = target_exp - quantiles
    loss = 2.0 * np.maximum(q_levels * diff, (q_levels - 1.0) * diff)

    sum_loss = np.sum(loss)
    sum_target = np.sum(np.abs(target_exp))
    if sum_target < 1e-6:
        return float(np.nan)

    return float(sum_loss / (len(quantile_levels) * sum_target))


def calc_coverage(
    quantiles: np.ndarray,
    target: np.ndarray,
    lower_idx: int = 0,
    upper_idx: int = 8,
) -> float:
    """Coverage of prediction interval [q_lower, q_upper] (default 10% to 90%, i.e. 80% coverage)."""
    low = quantiles[..., lower_idx]
    high = quantiles[..., upper_idx]
    covered = (target >= low) & (target <= high)
    return float(np.mean(covered))


def calc_benefit(e_uv: np.ndarray, e_mv: np.ndarray) -> Tuple[float, float]:
    """
    Compute aggregate Benefit B and median relative Benefit per window:
      B = (sum(e_uv) - sum(e_mv)) / sum(e_uv)
    """
    sum_uv = float(np.sum(e_uv))
    sum_mv = float(np.sum(e_mv))
    agg_b = float((sum_uv - sum_mv) / (sum_uv + 1e-8))
    denom = np.maximum(e_uv, 1e-8)
    median_b = float(np.median((e_uv - e_mv) / denom))
    return agg_b, median_b


def calc_gap_closure(
    e_uv_model: float,
    e_mv_model: float,
    e_uv_star: float,
    e_mv_star: float,
    min_gap_ratio: float = 0.05,
) -> float:
    """
    Computes Oracle Gap Closure (GC):
      GC = (e_uv_model - e_mv_model) / (e_uv_star - e_mv_star)
    If the relative oracle gap (e_uv_star - e_mv_star) / e_uv_star < min_gap_ratio (default 5%),
    returns np.nan to avoid reporting GC on degenerate/near-zero denominators.
    """
    denom = float(e_uv_star - e_mv_star)
    rel_gap = denom / max(float(e_uv_star), 1e-7)
    if rel_gap < min_gap_ratio:
        return float(np.nan)
    return float((e_uv_model - e_mv_model) / denom)


def calc_aggregate_reliance(
    e_pi: np.ndarray,
    e_mv: np.ndarray,
    e_uv: np.ndarray,
    min_denom: float = 1e-6,
) -> float:
    """
    Dataset-level Reliance R_pi computed as a ratio of sums:
      R_pi = sum_w(e_pi_w - e_mv_w) / sum_w(e_uv_w - e_mv_w)
    If denominator |sum(e_uv) - sum(e_mv)| < min_denom, returns np.nan (Task 6).
    """
    delta_pi = float(np.sum(e_pi - e_mv))
    delta_uv = float(np.sum(e_uv - e_mv))
    if abs(delta_uv) < min_denom:
        return float(np.nan)
    return float(delta_pi / delta_uv)


def block_bootstrap_reliance_ci(
    e_pi_windows: np.ndarray,
    e_mv_windows: np.ndarray,
    e_uv_windows: np.ndarray,
    block_length: int = 5,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    min_denom: float = 1e-6,
) -> Dict[str, Union[float, bool]]:
    """
    Window-level stationary block bootstrap for ratio-of-sums metrics (Task 6):
      - First sums target errors within each window (e_*_windows are [num_windows] arrays)
      - Resamples continuous blocks of windows (block_length >= 5) to account for temporal overlap
      - Returns point estimates, 95% CIs via nanpercentile, nan_ratio, and B_significant flag.
    """
    e_pi = np.asarray(e_pi_windows, dtype=np.float64).flatten()
    e_mv = np.asarray(e_mv_windows, dtype=np.float64).flatten()
    e_uv = np.asarray(e_uv_windows, dtype=np.float64).flatten()
    n = len(e_pi)

    sum_pi = float(np.sum(e_pi))
    sum_mv = float(np.sum(e_mv))
    sum_uv = float(np.sum(e_uv))
    denom_r = sum_uv - sum_mv

    point_r = float((sum_pi - sum_mv) / denom_r) if abs(denom_r) >= min_denom else float(np.nan)
    point_b = float((sum_uv - sum_mv) / max(sum_uv, 1e-7)) * 100.0
    point_dpi = float((sum_pi - sum_mv) / max(sum_mv, 1e-7)) * 100.0
    point_delta_mse = float(sum_pi - sum_mv) / n if n > 0 else 0.0

    if n <= 1:
        return {
            "R": point_r, "R_lo": point_r, "R_hi": point_r, "nan_ratio": 0.0,
            "B": point_b, "B_lo": point_b, "B_hi": point_b, "B_significant": False,
            "dPI_pct": point_dpi, "dPI_lo": point_dpi, "dPI_hi": point_dpi,
            "delta_mse": point_delta_mse, "delta_mse_lo": point_delta_mse, "delta_mse_hi": point_delta_mse,
        }

    rng = np.random.RandomState(seed)
    # Task 8: block length >= 5 when n is large; for small n (smoke tests), scale block_len < n
    if n < 10:
        block_len = max(1, n // 2)
    else:
        block_len = max(5, block_length)
    block_len = min(block_len, n - 1)
    num_blocks = int(np.ceil(n / block_len))

    boot_r = np.full(n_bootstraps, np.nan, dtype=np.float64)
    boot_b = np.zeros(n_bootstraps, dtype=np.float64)
    boot_dpi = np.zeros(n_bootstraps, dtype=np.float64)
    boot_delta_mse = np.zeros(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        start_indices = rng.randint(0, n - block_len + 1, size=num_blocks)
        indices = np.concatenate([np.arange(idx, idx + block_len) for idx in start_indices])[:n]

        b_pi = np.sum(e_pi[indices])
        b_mv = np.sum(e_mv[indices])
        b_uv = np.sum(e_uv[indices])

        d_uv = b_uv - b_mv
        if abs(d_uv) >= min_denom:
            boot_r[b] = (b_pi - b_mv) / d_uv

        boot_b[b] = (b_uv - b_mv) / max(b_uv, 1e-7) * 100.0
        boot_dpi[b] = (b_pi - b_mv) / max(b_mv, 1e-7) * 100.0
        boot_delta_mse[b] = (b_pi - b_mv) / n

    alpha = (1.0 - ci_level) / 2.0
    nan_ratio = float(np.isnan(boot_r).mean())

    if nan_ratio > 0.9:
        r_lo, r_hi = float(np.nan), float(np.nan)
    else:
        r_lo = float(np.nanpercentile(boot_r, alpha * 100))
        r_hi = float(np.nanpercentile(boot_r, (1.0 - alpha) * 100))

    b_lo = float(np.percentile(boot_b, alpha * 100))
    b_hi = float(np.percentile(boot_b, (1.0 - alpha) * 100))
    b_significant = bool((b_lo > 0 and b_hi > 0) or (b_lo < 0 and b_hi < 0))

    return {
        "R": point_r,
        "R_lo": r_lo,
        "R_hi": r_hi,
        "nan_ratio": nan_ratio,
        "B": point_b,
        "B_lo": b_lo,
        "B_hi": b_hi,
        "B_significant": b_significant,
        "dPI_pct": point_dpi,
        "dPI_lo": float(np.percentile(boot_dpi, alpha * 100)),
        "dPI_hi": float(np.percentile(boot_dpi, (1.0 - alpha) * 100)),
        "delta_mse": point_delta_mse,
        "delta_mse_lo": float(np.percentile(boot_delta_mse, alpha * 100)),
        "delta_mse_hi": float(np.percentile(boot_delta_mse, (1.0 - alpha) * 100)),
    }


def paired_wilcoxon_test(err_a: np.ndarray, err_b: np.ndarray) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test on window-aggregated error vectors.
    Returns (statistic, pvalue).
    """
    err_a = np.asarray(err_a, dtype=np.float64).flatten()
    err_b = np.asarray(err_b, dtype=np.float64).flatten()
    diff = err_a - err_b
    if np.all(diff == 0.0) or len(diff) < 5:
        return 0.0, 1.0
    res = stats.wilcoxon(err_a, err_b)
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni step-down procedure for family-wise error rate control.
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    m = len(p_vals)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    adj_p = np.zeros(m, dtype=np.float64)

    running_max = 0.0
    for i in range(m):
        raw = sorted_p[i]
        adj = raw * (m - i)
        running_max = max(running_max, adj)
        adj_p[i] = min(1.0, running_max)

    # Invert back to original order
    out = np.zeros(m, dtype=np.float64)
    out[sorted_indices] = adj_p
    return [float(x) for x in out]
