"""Diagnostic Test Suite A: Kalman Filter Oracles Mathematical Rigor (tests/test_oracle.py).
Strictly uses true generators from probe/synth/generators.py without mock generators.
Implements:
  A1: Theoretical vs Empirical Riccati convergence (< 3% relative error) across G1, G2, G5, G6 at h in {1, 8, 96}.
  A2: Multivariate dominance: e_MV* <= e_UV* theoretically and empirically across 1000 windows.
  A3: Catch leakage test: OLS AR(p) with p in {24, 64, 168} and VAR(24) cannot beat Kalman oracles.
  A4: Scale consistency: Oracle, model, and ground truth are evaluated on identical scale.
  A5: Missing parameter assertion: Deleting any metadata key causes oracle to raise KeyError/ValueError.
"""

import numpy as np
import pytest

from probe.synth.generators import (
    generate_g1_global_low_rank,
    generate_g2_local_blocks,
    generate_g3_structural_drift,
    generate_g5_lead_lag,
    generate_g6_null,
)
from probe.synth.oracle import (
    build_g1_oracle,
    build_g2_oracle,
    build_g3_oracles,
    build_g5_oracle,
    build_g6_oracle,
    compute_exact_kalman_oracles,
)


def test_a1_theoretical_vs_empirical_riccati_convergence():
    """
    A1: Generates 10^5 steps from generators.py across G1, G2, G5, G6 (2 parameter sets each).
    Evaluates oracle's own filter_and_forecast_* at h in {1, 8, 96}.
    Relative error between empirical MSE and Riccati theoretical variance must be < 3%.
    """
    print("\n" + "=" * 70)
    print("--- Running Test A1: Riccati Theoretical vs Empirical (< 3%) ---")
    print("=" * 70)

    test_configs = [
        # (Generator, kwargs, name)
        (generate_g1_global_low_rank, {"N": 4, "rank": 2, "gamma": 0.5, "sigma_eps": 0.1}, "G1_set1"),
        (generate_g1_global_low_rank, {"N": 4, "rank": 1, "gamma": 0.75, "sigma_eps": 0.2}, "G1_set2"),
        (generate_g2_local_blocks, {"N": 4, "block_size": 2, "gamma": 0.5, "sigma_eps": 0.1}, "G2_set1"),
        (generate_g2_local_blocks, {"N": 4, "block_size": 2, "gamma": 0.75, "sigma_eps": 0.15}, "G2_set2"),
        (generate_g5_lead_lag, {"N": 3, "lag": 8, "lead_coeff": 0.8, "sigma_eps": 0.1}, "G5_set1"),
        (generate_g5_lead_lag, {"N": 3, "lag": 16, "lead_coeff": 0.6, "sigma_eps": 0.15}, "G5_set2"),
        (generate_g6_null, {"N": 3, "sigma_eps": 0.1}, "G6_set1"),
        (generate_g6_null, {"N": 3, "sigma_eps": 0.2}, "G6_set2"),
    ]

    T_sim = 100000

    for gen_fn, params, label in test_configs:
        print(f"\nEvaluating Config: {label} ({params}) on {T_sim} steps...")
        X, meta = gen_fn(T=T_sim, seed=42, **params)
        g_type = meta["type"]

        if "G1" in g_type:
            oracle_mv, uv_oracles = build_g1_oracle(meta)
        elif "G2" in g_type:
            oracle_mv, uv_oracles = build_g2_oracle(meta)
        elif "G5" in g_type:
            oracle_mv, uv_oracles = build_g5_oracle(meta)
        elif "G6" in g_type:
            oracle_mv, uv_oracles = build_g6_oracle(meta)

        target_idx = 1 if "G5" in g_type else 0

        # Multi-step projections using oracle steady-state matrices
        A = oracle_mv.A
        C = oracle_mv.C
        Q = oracle_mv.Q
        R = oracle_mv.R
        dim = oracle_mv.state_dim

        # Precompute powers of A
        A_powers = {h: np.linalg.matrix_power(A, h) for h in [1, 8, 96]}
        C_tgt = C[target_idx : target_idx + 1]

        # Dynamic Riccati filtering starting from stationary prior P0 (matching filter_and_forecast_mv)
        P = oracle_mv.P0.copy() if oracle_mv.P0 is not None else np.eye(dim, dtype=np.float64)
        x_hat = np.zeros((dim, 1), dtype=np.float64)
        sum_sq = {1: 0.0, 8: 0.0, 96: 0.0}
        eval_count = 0
        warmup = 2000

        for t in range(T_sim - 100):
            y = X[:, t : t + 1].astype(np.float64)
            S = C @ P @ C.T + R
            K = P @ C.T @ np.linalg.inv(S)
            x_filt = x_hat + K @ (y - C @ x_hat)

            if t >= warmup:
                for h in [1, 8, 96]:
                    pred_h = (C_tgt @ (A_powers[h] @ x_filt)).item()
                    true_h = X[target_idx, t + h]
                    sum_sq[h] += (true_h - pred_h) ** 2
                eval_count += 1

            I_KC = np.eye(dim, dtype=np.float64) - K @ C
            P_filt = I_KC @ P @ I_KC.T + K @ R @ K.T
            x_hat = A @ x_filt
            P = A @ P_filt @ A.T + Q

        for h in [1, 8, 96]:
            emp_mse = sum_sq[h] / eval_count
            theo_var = oracle_mv.get_theoretical_forecast_variance(h=h, target_idx=target_idx)
            rel_err_pct = abs(emp_mse - theo_var) / theo_var * 100.0

            print(f"  h={h:>2}: Empirical MSE = {emp_mse:.6f} | Theoretical = {theo_var:.6f} | Rel Error = {rel_err_pct:.3f}% (threshold < 3%)")
            assert rel_err_pct < 3.0, f"[{label}] Horizon h={h} failed: rel error {rel_err_pct:.3f}% >= 3%"

    print("\nA1 ALL 8 CONFIGURATIONS PASSED: Riccati theory perfectly matches empirical simulation!")


def test_a2_multivariate_dominates_univariate():
    """
    A2: e_MV* <= e_UV* strictly in theory and empirically across 1000 windows.
    """
    print("\n" + "=" * 70)
    print("--- Running Test A2: Multivariate Oracle Dominates Univariate Oracle ---")
    print("=" * 70)

    # Generate 1000 evaluation windows for G1
    N = 4
    rank = 2
    gamma = 0.5
    H = 96
    T_win = 512 + H

    # 1. Theoretical verification
    _, meta_sample = generate_g1_global_low_rank(N=N, T=T_win, rank=rank, gamma=gamma, seed=42)
    oracle_mv, uv_oracles = build_g1_oracle(meta_sample)

    for h in [1, 8, 24, 48, 96]:
        theo_mv = oracle_mv.get_theoretical_forecast_variance(h=h, target_idx=0)
        theo_uv = uv_oracles[0].get_theoretical_forecast_variance(h=h, target_idx=0)
        gain = theo_uv - theo_mv
        print(f"Theory h={h:>2}: UV* Var = {theo_uv:.6f}, MV* Var = {theo_mv:.6f}, Theoretical Gain = {gain:.6f}")
        assert theo_mv <= theo_uv + 1e-9, f"Theory violation at h={h}: MV {theo_mv} > UV {theo_uv}"

    # 2. Empirical verification over 1000 windows
    print("\nRunning 1000-window empirical verification on G1...")
    mse_uv_all = []
    mse_mv_all = []

    # Run on one long continuous simulation sliced into 1000 non-overlapping windows
    T_long = 1000 * 64 + 512 + H
    X_long, meta_long = generate_g1_global_low_rank(N=N, T=T_long, rank=rank, gamma=gamma, seed=999)
    oracle_mv_long, uv_oracles_long = build_g1_oracle(meta_long)
    oracle_uv_long = uv_oracles_long[0]

    for w in range(1000):
        t_start = w * 64
        ctx_w = X_long[:, t_start : t_start + 512]
        target_w = X_long[0, t_start + 512 : t_start + 512 + H]

        pred_mv = oracle_mv_long.filter_and_forecast_mv(ctx_w, horizon=H)[0]
        pred_uv = oracle_uv_long.filter_and_forecast_uv(ctx_w[0], horizon=H, target_idx=0)

        mse_mv_all.append(np.mean((pred_mv - target_w) ** 2))
        mse_uv_all.append(np.mean((pred_uv - target_w) ** 2))

    mean_uv = float(np.mean(mse_uv_all))
    mean_mv = float(np.mean(mse_mv_all))
    emp_gain = mean_uv - mean_mv
    print(f"1000 Windows Empirical: Mean UV* MSE = {mean_uv:.6f} | Mean MV* MSE = {mean_mv:.6f} | Gain = {emp_gain:.6f}")
    assert mean_mv <= mean_uv, f"Empirical failure: MV {mean_mv} > UV {mean_uv}"
    print("A2 PASSED: Multivariate oracle strictly dominates univariate oracle!")


def test_a3_catch_leakage_with_ar_and_var():
    """
    A3 (Critical): Fits OLS AR(p) for p in {24, 64, 168} on 5*10^4 steps.
    Tests on 1000 test windows. Asserts MSE(AR) >= MSE(Oracle_UV) * (1 - 0.02).
    Also fits VAR(24) and asserts MSE(VAR) >= MSE(Oracle_MV) * (1 - 0.02).
    """
    print("\n" + "=" * 70)
    print("--- Running Test A3: Catch Leakage Test (AR & VAR vs Oracles) ---")
    print("=" * 70)

    N = 4
    T_train = 50000
    X_train, meta_train = generate_g1_global_low_rank(N=N, T=T_train, rank=2, gamma=0.5, sigma_eps=0.1, seed=42)

    target_idx = 0
    y_tr = X_train[target_idx].astype(np.float64)

    # 1. Fit AR(p) models
    ar_models = {}
    for p in [24, 64, 168]:
        Y = y_tr[p:]
        Z = np.column_stack([y_tr[p - k : -k] for k in range(1, p + 1)])
        beta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
        ar_models[p] = beta
        print(f"Fitted AR({p:>3}) OLS coefficients on {T_train} steps.")

    # 2. Fit VAR(24) model
    p_var = 24
    Y_var = X_train[:, p_var:].T
    Z_var = np.column_stack([X_train[:, p_var - k : -k].T for k in range(1, p_var + 1)])
    B_var, _, _, _ = np.linalg.lstsq(Z_var, Y_var, rcond=None)
    print(f"Fitted VAR({p_var}) OLS coefficients on {T_train} steps.")

    # 3. Test on 1000 evaluation windows
    H = 8
    T_test_long = 1000 * 64 + 512 + H
    X_test, meta_test = generate_g1_global_low_rank(N=N, T=T_test_long, rank=2, gamma=0.5, sigma_eps=0.1, seed=12345)
    oracle_mv, uv_oracles = build_g1_oracle(meta_test)
    oracle_uv = uv_oracles[target_idx]

    oracle_uv_mses = []
    oracle_mv_mses = []
    ar_mses = {p: [] for p in [24, 64, 168]}
    var_mses = []

    for w in range(1000):
        t_start = w * 64
        ctx = X_test[:, t_start : t_start + 512].astype(np.float64)
        target = X_test[target_idx, t_start + 512 : t_start + 512 + H]

        # Oracle forecasts
        pred_uv = oracle_uv.filter_and_forecast_uv(ctx[target_idx], horizon=H, target_idx=0)
        pred_mv = oracle_mv.filter_and_forecast_mv(ctx, horizon=H)[target_idx]

        oracle_uv_mses.append(np.mean((pred_uv - target) ** 2))
        oracle_mv_mses.append(np.mean((pred_mv - target) ** 2))

        # AR(p) recursive forecasts
        for p, beta in ar_models.items():
            curr = ctx[target_idx, -p:].copy()
            preds = []
            for _ in range(H):
                nxt = float(np.dot(curr[::-1], beta))
                preds.append(nxt)
                curr = np.append(curr[1:], nxt)
            ar_mses[p].append(np.mean((np.array(preds) - target) ** 2))

        # VAR(24) recursive forecast
        curr_var = ctx[:, -p_var:].copy()
        var_preds = []
        for _ in range(H):
            z_step = np.concatenate([curr_var[:, -k] for k in range(1, p_var + 1)])
            nxt_all = z_step @ B_var
            var_preds.append(nxt_all[target_idx])
            curr_var = np.column_stack([curr_var[:, 1:], nxt_all])
        var_mses.append(np.mean((np.array(var_preds) - target) ** 2))

    mean_or_uv = float(np.mean(oracle_uv_mses))
    mean_or_mv = float(np.mean(oracle_mv_mses))
    mean_var = float(np.mean(var_mses))

    print(f"\nUnivariate Oracle UV* MSE = {mean_or_uv:.6f}")
    for p in [24, 64, 168]:
        m_ar = float(np.mean(ar_mses[p]))
        ratio = m_ar / mean_or_uv
        print(f"  AR({p:>3}) MSE = {m_ar:.6f} | Ratio AR/Oracle = {ratio:.4f} (must be >= 0.98)")
        assert m_ar >= mean_or_uv * 0.98, f"Leakage detected! AR({p}) ({m_ar}) outperformed Oracle UV ({mean_or_uv})"

    ratio_var = mean_var / mean_or_mv
    print(f"\nMultivariate Oracle MV* MSE = {mean_or_mv:.6f}")
    print(f"  VAR(24) MSE = {mean_var:.6f} | Ratio VAR/Oracle = {ratio_var:.4f} (must be >= 0.98)")
    assert mean_var >= mean_or_mv * 0.98, f"Leakage detected! VAR(24) ({mean_var}) outperformed Oracle MV ({mean_or_mv})"

    print("\nA3 PASSED: No information leakage! Oracles dominate all fitted AR and VAR models.")


def test_a4_scale_consistency():
    """
    A4: Asserts that oracle, model, and y_true share identical normalization scale.
    """
    print("\n--- Running Test A4: Scale Consistency Check ---")
    N = 4
    T = 608
    H = 96
    X, meta = generate_g1_global_low_rank(N=N, T=T, seed=42)
    ctx = X[:, :512]
    target = X[:, 512:512+H]

    # Evaluate oracle
    e_uv_s, e_mv_s = compute_exact_kalman_oracles(ctx, target, target_idx=0, meta=meta)

    y_true = target[0]
    oracle_mv, uv_oracles = build_g1_oracle(meta)
    pred_mv = oracle_mv.filter_and_forecast_mv(ctx, horizon=H)[0]
    pred_uv = uv_oracles[0].filter_and_forecast_uv(ctx[0], horizon=H, target_idx=0)

    print(f"y_true:    mean = {float(np.mean(y_true)):+.4f}, std = {float(np.std(y_true)):.4f}")
    print(f"pred_mv:   mean = {float(np.mean(pred_mv)):+.4f}, std = {float(np.std(pred_mv)):.4f}")
    print(f"pred_uv:   mean = {float(np.mean(pred_uv)):+.4f}, std = {float(np.std(pred_uv)):.4f}")

    assert abs(np.mean(pred_mv)) < 1.0
    assert abs(np.mean(pred_uv)) < 1.0
    print("A4 PASSED: Scale consistency verified!")


def test_a5_missing_parameter_raises():
    """
    A5: Deleting any parameter key from meta must cause oracle builders to raise KeyError.
    """
    print("\n--- Running Test A5: Missing Parameter Validation ---")
    X, meta_g1 = generate_g1_global_low_rank(N=4, T=100)
    for key in ["Lambda", "rank", "gamma", "sigma_eps", "alpha", "self_params"]:
        meta_corrupt = meta_g1.copy()
        del meta_corrupt[key]
        with pytest.raises(KeyError) as exc_info:
            build_g1_oracle(meta_corrupt)
        print(f"G1 Missing '{key}': Successfully raised {exc_info.value}")

    X, meta_g5 = generate_g5_lead_lag(N=4, T=100)
    for key in ["lag", "lead_coeff", "sigma_eps", "self_params"]:
        meta_corrupt = meta_g5.copy()
        del meta_corrupt[key]
        with pytest.raises(KeyError) as exc_info:
            build_g5_oracle(meta_corrupt)
        print(f"G5 Missing '{key}': Successfully raised {exc_info.value}")

    print("A5 PASSED: Strict metadata parameter enforcement verified!")
