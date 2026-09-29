"""Diagnostic Test Suite D: Metrics Correctness & Numerical Ground Truth (tests/test_metrics.py).
Strictly implements:
  D1: Known answer for Reliance R (e.g. sum(UV)=10, sum(MV)=8, sum(PI)=9 -> R=0.5; zero-denom -> NaN).
  D2: Known answer for Mean Pinball Loss (y=0, q_pred=0 -> loss=0; hand-calculated pinball losses).
  D3: Bootstrap Coverage: 95% CI covers true value in [0.92, 0.98] over 200 synthetic simulations.
  D4: Consistency: E1 benefit and E2 benefit strictly use the exact same calculation function.
"""

import numpy as np
import pytest

from probe.config import QUANTILES
from probe.metrics import (
    block_bootstrap_reliance_ci,
    calc_aggregate_reliance,
    calc_benefit,
    calc_mean_pinball_loss,
    calc_mse,
)


def test_d1_reliance_known_answers():
    """D1: Known answer for Reliance R and NaN for zero denominator."""
    print("\n--- Running Test D1: Reliance Known Answers ---")

    # Case 1: sum(UV) = 10, sum(MV) = 8, sum(PI) = 9 -> R = (9 - 8) / (10 - 8) = 0.5
    e_uv = np.array([5.0, 5.0])
    e_mv = np.array([4.0, 4.0])
    e_pi = np.array([4.5, 4.5])

    r_val = calc_aggregate_reliance(e_pi=e_pi, e_mv=e_mv, e_uv=e_uv)
    print(f"Case 1 (UV=10, MV=8, PI=9): Calculated R = {r_val:.4f}, Expected = 0.5000")
    assert abs(r_val - 0.5) < 1e-6, f"Expected R=0.5, got {r_val}"

    # Case 2: Zero denominator: sum(UV) == sum(MV) -> must return NaN
    e_uv_zero = np.array([4.0, 4.0])
    e_mv_zero = np.array([4.0, 4.0])
    e_pi_zero = np.array([5.0, 5.0])

    r_nan = calc_aggregate_reliance(e_pi=e_pi_zero, e_mv=e_mv_zero, e_uv=e_uv_zero)
    print(f"Case 2 (Zero Denominator): Calculated R = {r_nan}, Expected = NaN")
    assert np.isnan(r_nan), f"Expected NaN for zero denominator, got {r_nan}"


def test_d2_pinball_loss_known_answers():
    """D2: Known answers for mean pinball loss."""
    print("\n--- Running Test D2: Mean Pinball Loss Known Answers ---")

    # 1. Zero error test: y=0, all quantiles predict 0 -> loss = 0.0
    H = 4
    n_q = len(QUANTILES)  # 9
    target_zero = np.zeros(H, dtype=np.float64)
    quants_zero = np.zeros((H, n_q), dtype=np.float64)

    loss_zero = calc_mean_pinball_loss(quants_zero, target_zero, quantile_levels=QUANTILES)
    print(f"Subtest 1 (All zeros): Loss = {loss_zero:.6f}, Expected = 0.000000")
    assert abs(loss_zero - 0.0) < 1e-7, f"Expected 0.0, got {loss_zero}"

    # 2. Hand-calculated single step test:
    # Target y = 1.0. Quantile q = 0.1 predicts 0.0.
    # diff = y - q_pred = 1.0 > 0.
    # Pinball loss = 2 * max(q * diff, (q - 1) * diff) = 2 * 0.1 * 1.0 = 0.2.
    target_single = np.array([1.0], dtype=np.float64)
    quants_single = np.array([[0.0]], dtype=np.float64)
    loss_single = calc_mean_pinball_loss(quants_single, target_single, quantile_levels=[0.1])
    print(f"Subtest 2 (y=1.0, q_pred=0.0, q=0.1): Loss = {loss_single:.6f}, Expected = 0.200000")
    assert abs(loss_single - 0.2) < 1e-6, f"Expected 0.2, got {loss_single}"

    # Target y = -1.0. Quantile q = 0.9 predicts 0.0.
    # diff = y - q_pred = -1.0 < 0.
    # Pinball loss = 2 * (q - 1) * (-1.0) = 2 * (-0.1) * (-1.0) = 0.2.
    target_neg = np.array([-1.0], dtype=np.float64)
    loss_neg = calc_mean_pinball_loss(quants_single, target_neg, quantile_levels=[0.9])
    print(f"Subtest 3 (y=-1.0, q_pred=0.0, q=0.9): Loss = {loss_neg:.6f}, Expected = 0.200000")
    assert abs(loss_neg - 0.2) < 1e-6, f"Expected 0.2, got {loss_neg}"


def test_d3_bootstrap_ci_coverage():
    """D3: Bootstrap 95% confidence interval coverage over 200 synthetic simulations."""
    print("\n--- Running Test D3: Bootstrap CI Coverage Verification ---")
    rng = np.random.RandomState(42)
    n_sims = 200
    n_windows = 50

    # True population distribution:
    # e_uv ~ Gamma(shape=4, scale=1.0) -> mean = 4.0
    # e_mv = 0.8 * e_uv + Noise(0, 0.2^2) -> E[e_mv] = 3.2
    # True Benefit B = (4.0 - 3.2) / 4.0 * 100% = 20.0%
    true_benefit = 20.0
    covered_count = 0

    for s in range(n_sims):
        sim_uv = rng.gamma(shape=4.0, scale=1.0, size=n_windows)
        sim_mv = 0.8 * sim_uv + rng.normal(0, 0.2, size=n_windows)

        ci_res = block_bootstrap_reliance_ci(
            e_pi_windows=sim_mv,
            e_mv_windows=sim_mv,
            e_uv_windows=sim_uv,
            block_length=5,
            n_bootstraps=500,
            ci_level=0.95,
            seed=42 + s,
        )

        b_lo = ci_res["B_lo"]
        b_hi = ci_res["B_hi"]
        if b_lo <= true_benefit <= b_hi:
            covered_count += 1

    coverage_rate = covered_count / n_sims
    print(f"Bootstrap 95% CI Coverage across {n_sims} simulations: {coverage_rate:.3f} ({covered_count}/{n_sims})")
    print(f"Acceptance interval: [0.920, 0.980]")
    assert 0.92 <= coverage_rate <= 0.98, f"Coverage {coverage_rate:.3f} out of expected [0.92, 0.98] bounds!"


def test_d4_benefit_consistency():
    """D4: E1 Benefit B and E2 Benefit B use identical calculation."""
    print("\n--- Running Test D4: Benefit Calculation Consistency ---")
    e_uv = np.array([2.5, 3.0, 3.5, 4.0], dtype=np.float64)
    e_mv = np.array([2.0, 2.5, 3.0, 3.2], dtype=np.float64)

    # Method 1: calc_benefit
    b_val, _ = calc_benefit(e_uv, e_mv)
    b_pct_1 = b_val * 100.0

    # Method 2: block_bootstrap_reliance_ci
    ci_res = block_bootstrap_reliance_ci(
        e_pi_windows=e_mv,
        e_mv_windows=e_mv,
        e_uv_windows=e_uv,
        block_length=2,
        n_bootstraps=100,
        seed=42,
    )
    b_pct_2 = ci_res["B"]

    print(f"calc_benefit B: {b_pct_1:.6f}% | bootstrap B: {b_pct_2:.6f}%")
    assert abs(b_pct_1 - b_pct_2) < 1e-5, f"Benefit inconsistency: {b_pct_1} vs {b_pct_2}"
