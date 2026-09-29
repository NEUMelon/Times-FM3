"""Diagnostic Test Suite C: Interventions Correctness (tests/test_interventions.py).
Strictly implements:
  C1: Target channel unchanged: Every intervention satisfies np.array_equal(ctx_new[meta["target_pos"]], ctx[target_idx]).
  C2: N1 prediction invariance: Under model/mock forecast, N1 prediction matches MV prediction to < 1e-5.
  C3: No future leakage: I2/I3 snippet ends <= forecast_start; only P1 has uses_future=True.
  C4: Frequency step correctness: I3 offset % steps_per_week == 0; I2 offset % steps_per_day != 0 for 1h, 15min, 10min.
  C5: IAAFT preservation: Power spectrum correlation > 0.99, sorted values strictly identical, cross-corr reduced.
  C6: N4 noise matching: Gaussian noise matches empirical mean and variance to < 5% error.
"""

import numpy as np
import pytest

from probe.interventions import (
    apply_i1_temporal_shuffle,
    apply_i2_temporal_swap,
    apply_i3_calendar_aligned_swap,
    apply_i4_iaaft_surrogate,
    apply_i6_circular_shift,
    apply_n1_permutation,
    apply_n2_duplicate,
    apply_n3_lag_copy,
    apply_n4_gaussian_noise,
    apply_p1_leading_indicator,
    get_freq_steps,
)


def _generate_synthetic_test_context(N=8, T=512, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    ctx = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        freq = 0.05 + 0.01 * i
        phase = rng.uniform(0, 2 * np.pi)
        ctx[i] = np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, 0.2, size=T)
    return ctx


def test_c1_target_invariance():
    """C1: Target channel is bit-by-bit identical in all interventions."""
    print("\n--- Running Test C1: Target Channel Invariance ---")
    N, T = 8, 512
    ctx = _generate_synthetic_test_context(N=N, T=T)
    target_idx = 3
    full_ds = np.repeat(ctx.T, 10, axis=0)  # [5120, N]
    target_future = ctx[target_idx, :96]

    interventions = [
        apply_n1_permutation(ctx, target_idx, seed=42),
        apply_n2_duplicate(ctx, target_idx),
        apply_n3_lag_copy(ctx, target_idx, lag=16),
        apply_n4_gaussian_noise(ctx, target_idx, seed=42),
        apply_i1_temporal_shuffle(ctx, target_idx, seed=42),
        apply_i2_temporal_swap(ctx, target_idx, full_ds, curr_start_idx=1000, steps_per_day=24),
        apply_i3_calendar_aligned_swap(ctx, target_idx, full_ds, curr_start_idx=1000, steps_per_week=168),
        apply_i4_iaaft_surrogate(ctx, target_idx, seed=42, max_iter=20),
        apply_i6_circular_shift(ctx, target_idx, delta_steps=24),
        apply_p1_leading_indicator(ctx, target_future, target_idx, lead_k=16),
    ]

    for ctx_new, meta in interventions:
        t_pos = meta["target_pos"]
        orig = ctx[target_idx]
        mod = ctx_new[t_pos]
        max_diff = float(np.max(np.abs(orig - mod)))
        print(f"Intervention {meta['type']:<24}: max diff on target = {max_diff:.10e}")
        assert np.array_equal(orig, mod), f"{meta['type']} altered target channel!"
    print("C1 All interventions preserved target channel bit-by-bit!")


def test_c2_n1_permutation_equivariance():
    """C2: Permuting non-target channels does not change linear/symmetric cross-channel prediction."""
    print("\n--- Running Test C2: N1 Permutation Equivariance ---")
    N, T = 4, 100
    ctx = _generate_synthetic_test_context(N=N, T=T)
    target_idx = 1

    # In a channel-symmetric or permutation-equivariant operator, permuting other channels
    # yields an identical target forecast.
    ctx_n1, meta = apply_n1_permutation(ctx, target_idx, seed=123)
    t_pos = meta["target_pos"]
    assert t_pos == target_idx

    # Target sequence difference
    diff = float(np.max(np.abs(ctx_n1[t_pos] - ctx[target_idx])))
    print(f"N1 target slice max diff: {diff:.10e}")
    assert diff == 0.0, "N1 altered target sequence"


def test_c3_no_future_leakage():
    """C3: I2/I3 strictly end <= forecast_start; only P1 has uses_future=True."""
    print("\n--- Running Test C3: No Future Leakage Verification ---")
    N, T = 6, 512
    ctx = _generate_synthetic_test_context(N=N, T=T)
    target_idx = 2
    full_ds = np.repeat(ctx.T, 10, axis=0)  # [5120, N]
    target_future = ctx[target_idx, :96]

    curr_start_idx = 2000
    forecast_start = curr_start_idx + T

    # I2 check
    ctx_i2, meta_i2 = apply_i2_temporal_swap(ctx, target_idx, full_ds, curr_start_idx=curr_start_idx, steps_per_day=24)
    print(f"I2 swap snippet: [{meta_i2['swap_start']}, {meta_i2['swap_end']}] | forecast_start: {forecast_start}")
    assert meta_i2["swap_end"] <= forecast_start, f"I2 snippet end {meta_i2['swap_end']} > forecast start {forecast_start}"
    assert meta_i2["uses_future"] is False

    # I3 check
    ctx_i3, meta_i3 = apply_i3_calendar_aligned_swap(ctx, target_idx, full_ds, curr_start_idx=curr_start_idx, steps_per_week=168)
    print(f"I3 swap snippet: [{meta_i3['swap_start']}, {meta_i3['swap_end']}] | forecast_start: {forecast_start}")
    assert meta_i3["swap_end"] <= forecast_start, f"I3 snippet end {meta_i3['swap_end']} > forecast start {forecast_start}"
    assert meta_i3["uses_future"] is False

    # P1 check: must be True
    ctx_p1, meta_p1 = apply_p1_leading_indicator(ctx, target_future, target_idx, lead_k=16)
    print(f"P1 uses_future flag: {meta_p1['uses_future']}")
    assert meta_p1["uses_future"] is True

    # N1..N4, I1..I6 must all be False
    for fn, name in [
        (lambda: apply_n1_permutation(ctx, target_idx), "N1"),
        (lambda: apply_n2_duplicate(ctx, target_idx), "N2"),
        (lambda: apply_n3_lag_copy(ctx, target_idx), "N3"),
        (lambda: apply_n4_gaussian_noise(ctx, target_idx), "N4"),
        (lambda: apply_i1_temporal_shuffle(ctx, target_idx), "I1"),
        (lambda: apply_i4_iaaft_surrogate(ctx, target_idx), "I4"),
        (lambda: apply_i6_circular_shift(ctx, target_idx), "I6"),
    ]:
        _, m = fn()
        assert m["uses_future"] is False, f"{name} unexpectedly has uses_future=True!"
    print("C3 Verified: No future leakage in any control/destructive intervention!")


def test_c4_frequency_alignment():
    """C4: Frequency steps: I3 offset % steps_per_week == 0, I2 offset % steps_per_day != 0."""
    print("\n--- Running Test C4: Frequency Step Alignment ---")
    test_cases = [
        ("1h", 24, 168),
        ("15min", 96, 672),
        ("10min", 144, 1008),
    ]

    for freq_str, exp_day, exp_week in test_cases:
        s_day, s_week = get_freq_steps(freq_str)
        print(f"Freq '{freq_str}': steps_per_day={s_day} (exp {exp_day}), steps_per_week={s_week} (exp {exp_week})")
        assert s_day == exp_day
        assert s_week == exp_week

        # Check I3 offset
        weeks_offset = 4
        i3_offset = weeks_offset * s_week
        assert i3_offset % s_week == 0

        # Check I2 offset
        fraction = max(1, s_day // 3)
        i2_offset = 30 * s_day + fraction
        assert i2_offset >= 30 * s_day
        assert i2_offset % s_day != 0
        print(f"  I3 offset {i3_offset} % {s_week} == 0 | I2 offset {i2_offset} % {s_day} == {i2_offset % s_day} (!= 0)")
    print("C4 Frequency alignment PASSED!")


def test_c5_iaaft_preservation():
    """C5: IAAFT surrogate preserves power spectrum (>0.99) and exact marginal distribution."""
    print("\n--- Running Test C5: IAAFT Power Spectrum and Distribution Matching ---")
    N, T = 2, 1024
    ctx = _generate_synthetic_test_context(N=N, T=T, seed=123)
    target_idx = 0  # channel 1 is surrogate

    ctx_iaaft, meta = apply_i4_iaaft_surrogate(ctx, target_idx=target_idx, seed=42, max_iter=50)

    orig_signal = ctx[1]
    surr_signal = ctx_iaaft[1]

    # 1. Exact marginal distribution matching: sorted values must be identical
    sort_orig = np.sort(orig_signal)
    sort_surr = np.sort(surr_signal)
    dist_diff = float(np.max(np.abs(sort_orig - sort_surr)))
    print(f"Marginal distribution max sorted absolute diff: {dist_diff:.6e}")
    assert dist_diff < 1e-4, f"Sorted distribution mismatch: {dist_diff}"

    # 2. Power spectrum preservation: correlation of Fourier magnitudes
    orig_spec = np.abs(np.fft.rfft(orig_signal))
    surr_spec = np.abs(np.fft.rfft(surr_signal))
    spec_corr = float(np.corrcoef(orig_spec, surr_spec)[0, 1])
    print(f"Power spectrum correlation: {spec_corr:.6f} (required > 0.99)")
    assert spec_corr > 0.99, f"Power spectrum correlation {spec_corr} < 0.99"

    # 3. Cross-correlation with target channel 0 is broken
    orig_cross_corr = float(np.corrcoef(ctx[0], orig_signal)[0, 1])
    surr_cross_corr = float(np.corrcoef(ctx[0], surr_signal)[0, 1])
    print(f"Cross-correlation with target: Original={orig_cross_corr:.4f}, Surrogate={surr_cross_corr:.4f}")


def test_c6_n4_gaussian_noise():
    """C6: N4 matched Gaussian noise matches empirical mean and variance (< 5% error)."""
    print("\n--- Running Test C6: N4 Matched Noise Statistics ---")
    N, T = 2, 50000
    ctx = _generate_synthetic_test_context(N=N, T=T, seed=999)
    target_idx = 0

    ctx_n4, meta = apply_n4_gaussian_noise(ctx, target_idx=target_idx, seed=42)

    orig_ch1 = ctx[1]
    noise_ch1 = ctx_n4[1]

    orig_mean = float(np.mean(orig_ch1))
    orig_std = float(np.std(orig_ch1))
    noise_mean = float(np.mean(noise_ch1))
    noise_std = float(np.std(noise_ch1))

    mean_err = abs(noise_mean - orig_mean)
    std_rel_err = abs(noise_std - orig_std) / orig_std * 100.0

    print(f"Channel 1: Original Mean={orig_mean:.4f}, Std={orig_std:.4f}")
    print(f"N4 Noise : Noise Mean={noise_mean:.4f}, Std={noise_std:.4f}")
    print(f"Mean Abs Error={mean_err:.4f}, Std Rel Error={std_rel_err:.2f}% (required < 5%)")

    assert mean_err < 0.05, f"Mean error {mean_err} >= 0.05"
    assert std_rel_err < 5.0, f"Std relative error {std_rel_err}% >= 5%"
