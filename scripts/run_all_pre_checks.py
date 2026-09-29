"""Comprehensive Pre-Experiment Verification and Red-Flag Diagnostic Suite.
Strictly implements and reports exact numerical values for:
1. Supplementary Checks:
   - C2: On ETTh1, Weather, Electricity, Solar (5 windows each), run_e2 flow reading meta["target_pos"].
         |MSE_N1 - MSE_MV| must be < 1e-6.
   - B6: Other channels as targets vs past_only_covariates input. Max absolute difference on target < 1e-5.
2. Red-Flag Checklist:
   - E1: E2 flow on ALL 8 datasets, N1 |delta_MSE| <= 1e-6.
   - E2: E5 flow on Electricity and Solar, K=64 vs K=32 MSE difference (must differ).
   - E3: E3 flow across synthetic configs, report (MSE_Model_MV - MSE_Oracle_MV) with SE. Halt if Model < Oracle - 2*SE.
   - E4: In all synthetic configs, Oracle_MV <= Oracle_UV.
   - E7: In E2 flow across ALL 8 datasets, P1 (k=32) reduces forecast error compared to MV baseline (Benefit > 0).
"""

import datetime
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# Path setup
sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import MODEL_DIR, PRIMARY_SEED, LTSF_DATASETS
from probe.data import load_dataset
from probe.grouping import sample_stratified_targets, select_top_k_correlated
from probe.interventions import (
    apply_n1_permutation,
    apply_p1_leading_indicator,
    get_freq_steps,
)
from probe.metrics import calc_mse, calc_benefit
from probe.models.timesfm3 import TimesFM3Wrapper
from probe.synth.generators import (
    generate_g1_global_low_rank,
    generate_g2_local_blocks,
    generate_g3_structural_drift,
    generate_g5_lead_lag,
    generate_g6_null,
)
from probe.synth.oracle import compute_exact_kalman_oracles


def run_all_checks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("PRE-EXPERIMENT VERIFICATION & RED-FLAG DIAGNOSTIC AUDIT")
    print(f"Device: {device} | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 85)

    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)

    # -------------------------------------------------------------------------
    # 1. B6: Target vs Past-Only Covariates Equivalence
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHECK B6: Other channels as targets vs past_only_covariates equivalence (< 1e-5)")
    print("=" * 85)
    rng_b6 = np.random.RandomState(42)
    b6_diffs = []
    for w_i in range(3):
        ctx_dummy = rng_b6.normal(0, 1.0, size=(4, 512)).astype(np.float32)
        # Method 1: All 4 channels as target variates in a single multivariate window
        outs_1 = list(wrapper.evaluator.predict_batch(
            contexts=[ctx_dummy],
            horizon=96,
            univariate=False,
            past_only_covariates=None,
            use_symmetric_averaging=False,
            make_positive=False,
        ))
        pred_1 = outs_1[0].forecast[0]

        # Method 2: Channel 0 as target, channels 1..3 as past_only_covariates
        outs_2 = list(wrapper.evaluator.predict_batch(
            contexts=[ctx_dummy[0:1]],
            horizon=96,
            univariate=False,
            past_only_covariates=[ctx_dummy[1:]],
            use_symmetric_averaging=False,
            make_positive=False,
        ))
        pred_2 = outs_2[0].forecast[0]

        diff = float(np.max(np.abs(pred_1 - pred_2)))
        b6_diffs.append(diff)
        print(f"  Window {w_i+1}: Max absolute difference = {diff:.10e}")

    max_b6_diff = max(b6_diffs)
    b6_pass = max_b6_diff < 1e-5
    print(f"--> B6 Summary: Maximum absolute diff = {max_b6_diff:.10e} (Threshold < 1e-5) => {'PASS' if b6_pass else 'FAIL'}")
    assert b6_pass, f"B6 failed with max diff {max_b6_diff}"

    # -------------------------------------------------------------------------
    # 2. C2 & E1: N1 Permutation Equivariance in run_e2 Flow across ALL 8 Datasets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHECK C2 & E1: N1 Permutation Equivariance (|delta_MSE| <= 1e-6) in run_e2 Flow")
    print("=" * 85)
    c2_datasets = ["ETTh1", "weather", "electricity", "solar"]
    all_datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "solar"]
    
    e1_results = {}
    
    for d_name in all_datasets:
        ds = load_dataset(d_name, context_length=512, horizon=96)
        window_indices = ds.get_sampled_window_indices(num_samples=5, seed=PRIMARY_SEED)
        N = ds.num_variates
        
        # Sample up to 4 targets per dataset to keep test nimble while representative
        if N <= 32:
            sampled_targets = list(range(min(4, N)))
        else:
            train_data = ds.norm_values[: ds.border2s[0]]
            corr_matrix = np.corrcoef(train_data, rowvar=False)
            np.nan_to_num(corr_matrix, copy=False, nan=0.0)
            np.fill_diagonal(corr_matrix, 0.0)
            sampled_targets = sample_stratified_targets(corr_matrix, num_targets=4, chunk_size=32, seed=PRIMARY_SEED)

        diffs_n1 = []
        for w_idx in window_indices:
            ctx_full, target_full = ds.get_window(w_idx)
            y_true_full = target_full[:, :96]
            
            for t_idx in sampled_targets:
                chunk_id = t_idx // 32
                start_ch = chunk_id * 32
                end_ch = min(start_ch + 32, N)
                group_indices = list(range(start_ch, end_ch))
                ctx_group = ctx_full[group_indices].copy()
                t_pos = t_idx - start_ch
                y_true = y_true_full[t_idx]

                # MV base prediction
                pred_mv_all, _ = wrapper.forecast(
                    ctx_group,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                )
                pred_mv_tgt = pred_mv_all[t_pos]
                mse_mv = float(calc_mse(pred_mv_tgt, y_true))

                # N1 intervention
                ctx_n1, meta = apply_n1_permutation(ctx_group, t_pos, seed=PRIMARY_SEED + w_idx)
                pred_n1_all, _ = wrapper.forecast(
                    ctx_n1,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                )
                t_pos_n1 = meta["target_pos"]
                pred_n1_tgt = pred_n1_all[t_pos_n1]
                mse_n1 = float(calc_mse(pred_n1_tgt, y_true))

                diff_mse = abs(mse_n1 - mse_mv)
                diffs_n1.append(diff_mse)

        max_diff_d = max(diffs_n1)
        mean_diff_d = float(np.mean(diffs_n1))
        e1_results[d_name] = {"max_diff": max_diff_d, "mean_diff": mean_diff_d}
        is_c2 = " (C2 Target)" if d_name in c2_datasets else ""
        print(f"  Dataset {d_name:<12}{is_c2}: Max |delta_MSE| = {max_diff_d:.10e}, Mean |delta_MSE| = {mean_diff_d:.10e}")
        assert max_diff_d <= 1e-6, f"Dataset {d_name} failed N1 equivariance: max diff {max_diff_d} > 1e-6"

    print("--> C2 & E1 Summary: All datasets satisfied |delta_MSE| <= 1e-6 strictly!")

    # -------------------------------------------------------------------------
    # 3. E2: E5 Channel Budget K=64 vs K=32 on Electricity and Solar
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHECK E2: E5 flow K=64 vs K=32 MSE difference on Electricity and Solar")
    print("=" * 85)
    e2_datasets = ["electricity", "solar"]
    e2_records = []

    for d_name in e2_datasets:
        ds = load_dataset(d_name, context_length=512, horizon=96)
        window_indices = ds.get_sampled_window_indices(num_samples=5, seed=PRIMARY_SEED)
        N = ds.num_variates

        train_data = ds.norm_values[: ds.border2s[0]]
        corr_matrix = np.corrcoef(train_data, rowvar=False)
        np.nan_to_num(corr_matrix, copy=False, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0)
        sampled_targets = sample_stratified_targets(corr_matrix, num_targets=4, chunk_size=32, seed=PRIMARY_SEED)

        mses_k32 = []
        mses_k64 = []

        for w_idx in window_indices:
            ctx_full, target_full = ds.get_window(w_idx)
            for t_idx in sampled_targets:
                y_true = target_full[t_idx, :96]

                # K=32 top_k
                subset_32, t_pos_32 = select_top_k_correlated(corr_matrix, t_idx, k=32)
                ctx_32 = ctx_full[subset_32]
                pred_32, _ = wrapper.forecast(
                    ctx_32,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                    bypass_evaluator_chunking=False,
                )
                mse_32 = float(calc_mse(pred_32[t_pos_32, :96], y_true))
                mses_k32.append(mse_32)

                # K=64 top_k
                subset_64, t_pos_64 = select_top_k_correlated(corr_matrix, t_idx, k=64)
                ctx_64 = ctx_full[subset_64]
                pred_64, _ = wrapper.forecast(
                    ctx_64,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                    bypass_evaluator_chunking=True,
                )
                mse_64 = float(calc_mse(pred_64[t_pos_64, :96], y_true))
                mses_k64.append(mse_64)

        mean_mse_32 = float(np.mean(mses_k32))
        mean_mse_64 = float(np.mean(mses_k64))
        diff_k = abs(mean_mse_64 - mean_mse_32)
        e2_records.append({
            "dataset": d_name,
            "MSE_K32": mean_mse_32,
            "MSE_K64": mean_mse_64,
            "diff_abs": diff_k,
        })
        print(f"  Dataset {d_name:<12}: MSE(K=32) = {mean_mse_32:.6f}, MSE(K=64) = {mean_mse_64:.6f}, |Diff| = {diff_k:.6f}")
        assert diff_k > 1e-5, f"Dataset {d_name} K=64 and K=32 produced identical MSE: {diff_k}"

    print("--> E2 Summary: K=64 and K=32 yield distinct MSEs as expected!")

    # -------------------------------------------------------------------------
    # 4. E3 & E4: Synthetic Controlled Probing vs Exact Kalman Oracles
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHECK E3 & E4: Model vs Exact Kalman Oracle & (Oracle_MV <= Oracle_UV)")
    print("=" * 85)
    synthetic_configs = [
        ("G1_Rank1_g0.5", "G1", {"N": 8, "rank": 1, "gamma": 0.5}),
        ("G1_Rank2_g0.75", "G1", {"N": 8, "rank": 2, "gamma": 0.75}),
        ("G2_Block4_g0.5", "G2", {"N": 16, "block_size": 4, "gamma": 0.5}),
        ("G3_Drift_d0.5", "G3", {"N": 8, "delta": 0.5, "gamma": 0.5, "tau_loc": "pred_start"}),
        ("G5_LeadLag_L16", "G5", {"N": 8, "lag": 16, "lead_coeff": 0.8}),
        ("G6_Null_N8", "G6", {"N": 8}),
    ]

    print(f"{'Config':<20} | {'UV* Oracle':<10} | {'MV* Oracle':<10} | {'Model MV':<10} | {'Model - MV*':<12} | {'SE':<10} | {'t-stat':<8} | {'E4 (MV*<=UV*)':<13}")
    print("-" * 105)

    for cfg_label, g_type, params in synthetic_configs:
        N = params.get("N", 8)
        T_total = 512 + 96 + 50
        
        diff_window_list = []
        oracle_gap_list = []
        mse_uv_star_list = []
        mse_mv_star_list = []
        mse_mv_model_list = []

        for w in range(5):
            w_seed = PRIMARY_SEED + w * 100
            if g_type == "G1":
                X, meta = generate_g1_global_low_rank(N=N, T=T_total, rank=params["rank"], gamma=params["gamma"], seed=w_seed)
            elif g_type == "G2":
                X, meta = generate_g2_local_blocks(N=N, T=T_total, block_size=params["block_size"], gamma=params["gamma"], seed=w_seed)
            elif g_type == "G3":
                tau = 512 if params["tau_loc"] == "pred_start" else 256
                X, meta = generate_g3_structural_drift(N=N, T=T_total, tau=tau, delta=params["delta"], gamma=params["gamma"], seed=w_seed)
            elif g_type == "G5":
                X, meta = generate_g5_lead_lag(N=N, T=T_total, lag=params["lag"], lead_coeff=params["lead_coeff"], seed=w_seed)
            elif g_type == "G6":
                X, meta = generate_g6_null(N=N, T=T_total, seed=w_seed)

            ctx = X[:, :512]
            target = X[:, 512 : 512 + 96]
            t_idx = 0 if g_type != "G5" else 1

            pred_mv, _ = wrapper.forecast(ctx, horizon=96, univariate=False, use_symmetric_averaging=True, return_quantiles=False)
            e_mv_m = float(calc_mse(pred_mv[t_idx], target[t_idx]))

            e_uv_s, e_mv_s = compute_exact_kalman_oracles(ctx, target, target_idx=t_idx, meta=meta)

            mse_uv_star_list.append(e_uv_s)
            mse_mv_star_list.append(e_mv_s)
            mse_mv_model_list.append(e_mv_m)

            diff_window_list.append(e_mv_m - e_mv_s)
            oracle_gap_list.append(e_mv_s - e_uv_s)

        mean_uv_s = float(np.mean(mse_uv_star_list))
        mean_mv_s = float(np.mean(mse_mv_star_list))
        mean_mv_m = float(np.mean(mse_mv_model_list))
        
        mean_diff = float(np.mean(diff_window_list))
        se_diff = float(np.std(diff_window_list, ddof=1) / np.sqrt(len(diff_window_list)))
        t_stat = mean_diff / se_diff if se_diff > 1e-12 else 0.0

        # E4 Check: Oracle MV* <= Oracle UV* (margin 1e-9)
        e4_pass = mean_mv_s <= mean_uv_s + 1e-9
        assert e4_pass, f"E4 Red Flag: Oracle MV* ({mean_mv_s:.6f}) > Oracle UV* ({mean_uv_s:.6f}) for {cfg_label}!"

        # E3 Check: Model should not significantly beat Oracle by > 2 SE (i.e. mean_diff < -2 * se_diff)
        e3_red_flag = (mean_diff < -2.0 * se_diff) and (abs(mean_diff) > 1e-4)
        assert not e3_red_flag, f"E3 Red Flag: Model significantly outperformed Oracle by {t_stat:.2f} SE! HALTING!"

        print(f"{cfg_label:<20} | {mean_uv_s:<10.5f} | {mean_mv_s:<10.5f} | {mean_mv_m:<10.5f} | {mean_diff:<+12.5f} | {se_diff:<10.5f} | {t_stat:<+8.2f} | {'PASS' if e4_pass else 'FAIL'}")

    print("--> E3 & E4 Summary: All oracles verified; MV* <= UV* everywhere; model does not beat oracle.")

    # -------------------------------------------------------------------------
    # 5. E7: Verification that P1 (k=32) reduces forecast error across ALL datasets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHECK E7: P1 Leading Indicator (k=32) Error Reduction across ALL 8 Datasets")
    print("=" * 85)
    print(f"{'Dataset':<12} | {'MSE (MV)':<12} | {'MSE (P1 k=32)':<15} | {'Delta MSE':<12} | {'Benefit B (%)':<15} | {'Status':<10}")
    print("-" * 85)

    for d_name in all_datasets:
        ds = load_dataset(d_name, context_length=512, horizon=96)
        window_indices = ds.get_sampled_window_indices(num_samples=5, seed=PRIMARY_SEED)
        N = ds.num_variates

        if N <= 32:
            sampled_targets = list(range(min(4, N)))
        else:
            train_data = ds.norm_values[: ds.border2s[0]]
            corr_matrix = np.corrcoef(train_data, rowvar=False)
            np.nan_to_num(corr_matrix, copy=False, nan=0.0)
            np.fill_diagonal(corr_matrix, 0.0)
            sampled_targets = sample_stratified_targets(corr_matrix, num_targets=4, chunk_size=32, seed=PRIMARY_SEED)

        mv_mses = []
        p1_mses = []

        for w_idx in window_indices:
            ctx_full, target_full = ds.get_window(w_idx)
            y_true_full = target_full[:, :96]

            for t_idx in sampled_targets:
                chunk_id = t_idx // 32
                start_ch = chunk_id * 32
                end_ch = min(start_ch + 32, N)
                group_indices = list(range(start_ch, end_ch))
                ctx_group = ctx_full[group_indices].copy()
                t_pos = t_idx - start_ch
                y_true = y_true_full[t_idx]

                # MV baseline
                pred_mv_all, _ = wrapper.forecast(
                    ctx_group,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                )
                mse_mv = float(calc_mse(pred_mv_all[t_pos], y_true))
                mv_mses.append(mse_mv)

                # P1 intervention (k=32 lead)
                ctx_p1, meta_p1 = apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=32, snr=10.0, seed=PRIMARY_SEED)
                pred_p1_all, _ = wrapper.forecast(
                    ctx_p1,
                    horizon=96,
                    univariate=False,
                    use_symmetric_averaging=True,
                    return_quantiles=False,
                )
                t_pos_p1 = meta_p1["target_pos"]
                mse_p1 = float(calc_mse(pred_p1_all[t_pos_p1], y_true))
                p1_mses.append(mse_p1)

        mean_mv = float(np.mean(mv_mses))
        mean_p1 = float(np.mean(p1_mses))
        delta_mse = mean_mv - mean_p1
        benefit_pct = (delta_mse / mean_mv) * 100.0 if mean_mv > 1e-12 else 0.0

        pass_e7 = benefit_pct > 0.0
        print(f"{d_name:<12} | {mean_mv:<12.6f} | {mean_p1:<15.6f} | {delta_mse:<+12.6f} | {benefit_pct:<+15.2f}% | {'PASS' if pass_e7 else 'FAIL'}")
        assert pass_e7, f"Dataset {d_name} failed E7: P1 did not reduce error (Benefit = {benefit_pct:+.2f}%)!"

    print("--> E7 Summary: P1 (k=32) consistently reduced forecast error across all 8 datasets!")
    print("\n" + "=" * 85)
    print("ALL AUDIT CHECKS AND RED FLAGS SUCCESSFULLY VERIFIED!")
    print("=" * 85)


if __name__ == "__main__":
    run_all_checks()
