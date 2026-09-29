"""Target-centric intervention operators: Negative controls (N1-N4), Destructive (I1-I6), Positive (P1).
All interventions strictly adhere to Task 4 and Task 5 specifications:
  - All functions return (ctx_new, meta) where meta["target_pos"] is always provided.
  - Every intervention asserts that the target channel is preserved bit-by-bit.
  - I2 and I3 snippets strictly end before the current window's forecast start (uses_future=False).
  - P1 explicitly sets uses_future=True; all other interventions set uses_future=False.
  - Frequency-dependent steps are unified across 1h, 15min, and 10min datasets.
  - N3 lag is fixed to 16 steps constant.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Constant N3 lag
N3_LAG_STEPS = 16


def get_freq_steps(freq_str: str) -> Tuple[int, int]:
    """
    Computes (steps_per_day, steps_per_week) from dataset frequency.
    1h -> 24 / 168
    15min -> 96 / 672
    10min -> 144 / 1008
    """
    f = freq_str.lower().strip()
    if "15min" in f or "15m" in f:
        minutes = 15
    elif "10min" in f or "10m" in f:
        minutes = 10
    elif "h" in f:
        minutes = 60
    elif "d" in f:
        minutes = 1440
    else:
        minutes = 60

    steps_per_day = (24 * 60) // minutes
    steps_per_week = 7 * steps_per_day
    return steps_per_day, steps_per_week


def apply_n1_permutation(ctx: np.ndarray, target_idx: int, seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    N1 (Negative control): Permute context channels strictly among non-target channels.
    Target remains exactly at target_idx.
    """
    N, T = ctx.shape
    rng = np.random.RandomState(seed)
    other_indices = [idx for idx in range(N) if idx != target_idx]
    permuted_others = rng.permutation(other_indices)

    new_ctx = ctx.copy()
    new_ctx[other_indices] = ctx[permuted_others]

    # Verification: target channel must not be modified
    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "N1 modified target channel!"

    meta = {
        "type": "N1_permutation",
        "target_pos": target_idx,
        "seed": seed,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_n2_duplicate(ctx: np.ndarray, target_idx: int) -> Tuple[np.ndarray, Dict]:
    """
    N2 (Negative control): Replace one non-target channel with a duplicate of the target channel.
    """
    N, T = ctx.shape
    if N <= 1:
        new_ctx = ctx.copy()
        return new_ctx, {"type": "N2_duplicate", "target_pos": target_idx, "uses_future": False}

    replace_idx = (target_idx + 1) % N
    new_ctx = ctx.copy()
    new_ctx[replace_idx] = ctx[target_idx]

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "N2 modified target channel!"

    meta = {
        "type": "N2_duplicate",
        "target_pos": target_idx,
        "replaced_idx": replace_idx,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_n3_lag_copy(
    ctx: np.ndarray,
    target_idx: int,
    lag: int = N3_LAG_STEPS,
) -> Tuple[np.ndarray, Dict]:
    """
    N3 (Negative control): Replace one non-target channel with a lagged copy of target x(t) = y_i(t - lag).
    Lag is fixed to constant 16 steps.
    """
    N, T = ctx.shape
    if N <= 1:
        new_ctx = ctx.copy()
        return new_ctx, {"type": "N3_lag_copy", "target_pos": target_idx, "lag": lag, "uses_future": False}

    replace_idx = (target_idx + 1) % N
    new_ctx = ctx.copy()
    lagged = np.roll(ctx[target_idx], shift=lag)
    lagged[:lag] = ctx[target_idx, 0]
    new_ctx[replace_idx] = lagged

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "N3 modified target channel!"

    meta = {
        "type": "N3_lag_copy",
        "target_pos": target_idx,
        "lag": lag,
        "replaced_idx": replace_idx,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_n4_gaussian_noise(ctx: np.ndarray, target_idx: int, seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    N4 (Negative control / Distractor): Replace non-target channels with matched Gaussian noise.
    """
    N, T = ctx.shape
    rng = np.random.RandomState(seed)
    new_ctx = ctx.copy()
    for j in range(N):
        if j != target_idx:
            mean = float(np.mean(ctx[j]))
            std = float(np.std(ctx[j]))
            new_ctx[j] = rng.normal(mean, max(std, 1e-4), size=T).astype(np.float32)

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "N4 modified target channel!"

    meta = {
        "type": "N4_gaussian_noise",
        "target_pos": target_idx,
        "seed": seed,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_i1_temporal_shuffle(ctx: np.ndarray, target_idx: int, seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    I1 (Destructive): Randomly shuffle time steps within non-target channels.
    """
    N, T = ctx.shape
    rng = np.random.RandomState(seed)
    new_ctx = ctx.copy()
    for j in range(N):
        if j != target_idx:
            perm = rng.permutation(T)
            new_ctx[j] = ctx[j, perm]

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "I1 modified target channel!"

    meta = {
        "type": "I1_temporal_shuffle",
        "target_pos": target_idx,
        "seed": seed,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_i2_temporal_swap(
    ctx: np.ndarray,
    target_idx: int,
    full_dataset: np.ndarray,
    curr_start_idx: int,
    steps_per_day: int = 24,
) -> Tuple[np.ndarray, Dict]:
    """
    I2 (Destructive): Replace non-target channels with a historical segment from >= 30 days away.
    Offset is strictly >= 30 * steps_per_day AND NOT an integer multiple of steps_per_day (breaking diurnal phase).
    Replacement snippet must strictly end before current forecast start (curr_start_idx + T).
    """
    N, T = ctx.shape
    L = full_dataset.shape[0]
    forecast_start = curr_start_idx + T

    # Non-aligned offset: 30 days plus non-zero fraction of day (e.g. 7 hours / steps)
    fraction = max(1, steps_per_day // 3)
    offset_steps = 30 * steps_per_day + fraction
    assert offset_steps % steps_per_day != 0, "I2 offset must NOT be a multiple of steps_per_day!"
    assert offset_steps >= 30 * steps_per_day, "I2 offset must be >= 30 days!"

    # Candidate start strictly in the past
    if curr_start_idx >= offset_steps:
        swap_start = curr_start_idx - offset_steps
    else:
        # Wrap backwards into available pre-forecast history
        max_valid_start = max(1, forecast_start - T)
        swap_start = (curr_start_idx - offset_steps) % max_valid_start

    swap_end = swap_start + T
    assert swap_end <= forecast_start, f"I2 swap snippet ({swap_end}) exceeds forecast start ({forecast_start})!"
    assert swap_start >= 0, f"I2 swap start ({swap_start}) is negative!"

    swap_segment = full_dataset[swap_start:swap_end].T  # [N, T]
    new_ctx = ctx.copy()
    for j in range(N):
        if j != target_idx:
            new_ctx[j] = swap_segment[j]

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "I2 modified target channel!"

    meta = {
        "type": "I2_swap_unaligned",
        "target_pos": target_idx,
        "swap_start": swap_start,
        "swap_end": swap_end,
        "forecast_start": forecast_start,
        "offset_steps": offset_steps,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_i3_calendar_aligned_swap(
    ctx: np.ndarray,
    target_idx: int,
    full_dataset: np.ndarray,
    curr_start_idx: int,
    steps_per_week: int = 168,
    weeks_offset: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    I3 (KEY DESTRUCTIVE INTERVENTION):
    Replace non-target channels with segment from integer weeks away (calendar aligned).
    Offset must strictly be an integer multiple of steps_per_week.
    Snippet must strictly end before current forecast start (curr_start_idx + T).
    """
    N, T = ctx.shape
    L = full_dataset.shape[0]
    forecast_start = curr_start_idx + T

    offset_steps = weeks_offset * steps_per_week
    assert offset_steps % steps_per_week == 0, "I3 offset must be an integer multiple of steps_per_week!"

    if curr_start_idx >= offset_steps:
        swap_start = curr_start_idx - offset_steps
    else:
        # Step back in integer multiples of weeks
        k_max = curr_start_idx // steps_per_week
        if k_max >= 1:
            swap_start = curr_start_idx - k_max * steps_per_week
        else:
            # If at the very beginning of dataset, take snippet from training tail that aligns
            num_weeks_in_l = (L - T) // steps_per_week
            swap_start = (curr_start_idx + num_weeks_in_l * steps_per_week) % (L - T)
            # Ensure it ends before forecast start if possible
            if swap_start + T > forecast_start and curr_start_idx > 0:
                swap_start = curr_start_idx % steps_per_week

    swap_end = swap_start + T
    assert swap_end <= forecast_start, f"I3 swap snippet ({swap_end}) exceeds forecast start ({forecast_start})!"
    assert swap_start >= 0, f"I3 swap start ({swap_start}) is negative!"

    swap_segment = full_dataset[swap_start:swap_end].T  # [N, T]
    new_ctx = ctx.copy()
    for j in range(N):
        if j != target_idx:
            new_ctx[j] = swap_segment[j]

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "I3 modified target channel!"

    meta = {
        "type": "I3_swap_aligned",
        "target_pos": target_idx,
        "swap_start": swap_start,
        "swap_end": swap_end,
        "forecast_start": forecast_start,
        "weeks_offset": weeks_offset,
        "offset_steps": offset_steps,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_i4_iaaft_surrogate(
    ctx: np.ndarray,
    target_idx: int,
    seed: int = 42,
    max_iter: int = 50,
) -> Tuple[np.ndarray, Dict]:
    """
    I4 (IAAFT Phase Randomization):
    Generates Iterative Amplitude Adjusted Fourier Transform surrogates for non-target channels.
    Preserves marginal distribution and power spectrum, but randomizes inter-channel phase alignment.
    """
    N, T = ctx.shape
    rng = np.random.RandomState(seed)
    new_ctx = ctx.copy()

    for j in range(N):
        if j != target_idx:
            x = ctx[j].astype(np.float64)
            sorted_x = np.sort(x)
            r = np.abs(np.fft.rfft(x))

            # Initial random phase
            angles = rng.uniform(0, 2 * np.pi, size=len(r))
            s = np.fft.irfft(r * np.exp(1j * angles), n=T)

            # Iterations to match spectrum and distribution
            for _ in range(max_iter):
                ranks = np.argsort(np.argsort(s))
                s_ranked = sorted_x[ranks]
                ft = np.fft.rfft(s_ranked)
                s = np.fft.irfft(r * np.exp(1j * np.angle(ft)), n=T)

            # Final step: rank order mapping to strictly match marginal distribution
            ranks = np.argsort(np.argsort(s))
            s_final = sorted_x[ranks]
            new_ctx[j] = s_final.astype(np.float32)

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "I4 modified target channel!"

    meta = {
        "type": "I4_iaaft_surrogate",
        "target_pos": target_idx,
        "seed": seed,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_i6_circular_shift(
    ctx: np.ndarray,
    target_idx: int,
    delta_steps: int = 24,
    duration_label: str = "1_day",
) -> Tuple[np.ndarray, Dict]:
    """
    I6: Circular shift non-target channels by delta time steps.
    """
    N, T = ctx.shape
    new_ctx = ctx.copy()
    for j in range(N):
        if j != target_idx:
            new_ctx[j] = np.roll(ctx[j], shift=delta_steps)

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "I6 modified target channel!"

    meta = {
        "type": f"I6_shift_{duration_label}",
        "target_pos": target_idx,
        "delta_steps": delta_steps,
        "duration_label": duration_label,
        "uses_future": False,
    }
    return new_ctx, meta


def apply_p1_leading_indicator(
    ctx: np.ndarray,
    target_future: np.ndarray,
    target_idx: int,
    lead_k: int = 16,
    snr: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    P1 (POSITIVE CONTROL):
    Replaces one context channel with a leading indicator of target:
    x(t) = y_target(t + lead_k) + noise.
    Uses future ground truth in last lead_k steps of context (uses_future=True).
    """
    N, T = ctx.shape
    H = len(target_future)
    rng = np.random.RandomState(seed)
    replace_idx = (target_idx + 1) % N

    full_target = np.concatenate([ctx[target_idx], target_future])  # [T + H]
    lead_seq = full_target[lead_k : lead_k + T]

    sig_var = np.var(lead_seq)
    noise_std = np.sqrt(sig_var / snr) if snr < 1e5 else 0.0
    noise = rng.normal(0, noise_std, size=T).astype(np.float32)

    new_ctx = ctx.copy()
    new_ctx[replace_idx] = lead_seq + noise

    assert np.array_equal(new_ctx[target_idx], ctx[target_idx]), "P1 modified target channel!"

    meta = {
        "type": f"P1_lead_{lead_k}",
        "target_pos": target_idx,
        "lead_k": lead_k,
        "snr": snr,
        "replaced_idx": replace_idx,
        "uses_future": True,
    }
    return new_ctx, meta
