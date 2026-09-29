"""E5 Experiment Runner: Channel Budget K, Selection Strategies, and Efficiency Pareto Front.
Updated according to Task 10:
  1. (subset, target_pos) unpacking from grouping functions.
  2. Stratified target sampling matching E2 (64 targets, single-channel blocks excluded).
  3. 5-run GPU warmup before timing.
  4. Latency reported as median and 90th percentile (ms), noting symmetric averaging toggle.
  5. Window-level block bootstrap confidence intervals for MSE.
  6. Git commit, timestamp, and parameter logging.
"""

import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

# Ensure probe can be imported
sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import MODEL_DIR, PRIMARY_SEED, RESULTS_DIR
from probe.data import load_dataset
from probe.grouping import (
    sample_stratified_targets,
    select_bottom_k_correlated,
    select_random_k,
    select_top_k_correlated,
)
from probe.metrics import calc_mae, calc_mse
from probe.models.timesfm3 import TimesFM3Wrapper


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def run_e5(
    datasets: Optional[List[str]] = None,
    horizon: int = 96,
    context_length: int = 512,
    num_samples: int = 50,
    k_list: Optional[List[int]] = None,
    seed: int = PRIMARY_SEED,
    device: str = "cuda",
    use_symmetric_averaging: bool = True,
):
    commit_hash = get_git_commit_hash()
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("STARTING E5: CHANNEL BUDGET K & SELECTION STRATEGIES EXPERIMENT (REVISED)")
    print(f"Git Commit: {commit_hash} | Timestamp: {run_timestamp}")
    print(f"Horizon: {horizon}, Context Length: {context_length}, Windows: {num_samples}, Seed: {seed}")
    print(f"Symmetric Averaging: {use_symmetric_averaging}")
    print("=" * 80)

    target_datasets = datasets or ["solar", "electricity"]
    test_ks = k_list or [1, 2, 4, 8, 16, 32, 64, 128]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)
    summary_records = []

    for d_name in target_datasets:
        print(f"\n>>> Running E5 on Dataset: {d_name}")
        ds = load_dataset(d_name, context_length=context_length, horizon=horizon)
        window_indices = ds.get_sampled_window_indices(num_samples=num_samples, seed=seed)
        N = ds.num_variates

        # Compute correlation matrix on training split
        train_data = ds.norm_values[: ds.border2s[0]]
        corr_matrix = np.corrcoef(train_data, rowvar=False)
        np.nan_to_num(corr_matrix, copy=False, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0)

        # Stratified sampling of up to 64 targets, excluding 1-channel blocks (Task 10)
        if N <= 32:
            sampled_targets = list(range(N))
        else:
            sampled_targets = sample_stratified_targets(
                corr_matrix,
                num_targets=min(64, N),
                chunk_size=32,
                seed=seed,
            )
        print(f"Evaluating {len(sampled_targets)} stratified target channels.")

        for K in test_ks:
            if K > N:
                continue
            print(f"\n--- Testing Channel Budget K = {K} ---")
            strategies = ["top_k", "bottom_k", "random"] if K > 1 else ["k1_univariate"]

            for strat in strategies:
                # 5 warmup runs before timing (Task 10)
                dummy_ctx = np.zeros((K, context_length), dtype=np.float32)
                for _ in range(5):
                    if K == 1:
                        _ = wrapper.forecast(dummy_ctx, horizon=horizon, univariate=True, use_symmetric_averaging=use_symmetric_averaging, return_quantiles=False)
                    else:
                        _ = wrapper.forecast(dummy_ctx, horizon=horizon, univariate=False, use_symmetric_averaging=use_symmetric_averaging, return_quantiles=False, bypass_evaluator_chunking=(K > 32))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                window_mse_list = []
                all_mse_list = []
                all_mae_list = []
                latency_list = []

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

                for w_idx in window_indices:
                    ctx_full, target_full = ds.get_window(w_idx)
                    win_mses = []

                    for t_idx in sampled_targets:
                        y_true = target_full[t_idx, :horizon]

                        if strat == "k1_univariate":
                            ctx_k = ctx_full[t_idx : t_idx + 1]
                            t_pos = 0
                        elif strat == "top_k":
                            subset, t_pos = select_top_k_correlated(corr_matrix, t_idx, k=K)
                            ctx_k = ctx_full[subset]
                        elif strat == "bottom_k":
                            subset, t_pos = select_bottom_k_correlated(corr_matrix, t_idx, k=K)
                            ctx_k = ctx_full[subset]
                        elif strat == "random":
                            subset, t_pos = select_random_k(N, t_idx, k=K, seed=seed + w_idx)
                            ctx_k = ctx_full[subset]

                        # Timed forward pass
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_start = time.time()

                        if K == 1:
                            pred_k, _ = wrapper.forecast(
                                ctx_k,
                                horizon=horizon,
                                univariate=True,
                                use_symmetric_averaging=use_symmetric_averaging,
                                return_quantiles=False,
                            )
                        else:
                            pred_k, _ = wrapper.forecast(
                                ctx_k,
                                horizon=horizon,
                                univariate=False,
                                use_symmetric_averaging=use_symmetric_averaging,
                                return_quantiles=False,
                                bypass_evaluator_chunking=(K > 32),
                            )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        latency = time.time() - t_start

                        pred_target = pred_k[t_pos, :horizon]
                        m = float(calc_mse(pred_target, y_true))
                        a = float(calc_mae(pred_target, y_true))

                        win_mses.append(m)
                        all_mse_list.append(m)
                        all_mae_list.append(a)
                        latency_list.append(latency)

                    window_mse_list.append(float(np.mean(win_mses)))

                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
                mean_mse = float(np.mean(all_mse_list))
                mean_mae = float(np.mean(all_mae_list))

                # Median and 90th percentile latency (Task 10)
                lat_ms = np.array(latency_list) * 1000.0
                med_latency_ms = float(np.median(lat_ms))
                p90_latency_ms = float(np.percentile(lat_ms, 90))

                # Window-level bootstrap CI for MSE (Task 10)
                n_win = len(window_mse_list)
                rng_boot = np.random.RandomState(seed)
                boot_means = []
                for _ in range(2000):
                    resamp = rng_boot.choice(window_mse_list, size=n_win, replace=True)
                    boot_means.append(float(np.mean(resamp)))
                mse_lo = float(np.percentile(boot_means, 2.5))
                mse_hi = float(np.percentile(boot_means, 97.5))

                print(
                    f"Strategy: {strat:<15} | K: {K:>3} | MSE: {mean_mse:.4f} [{mse_lo:.4f}, {mse_hi:.4f}] | "
                    f"Latency med: {med_latency_ms:.1f}ms, p90: {p90_latency_ms:.1f}ms | Peak VRAM: {peak_mem_mb:.1f}MB",
                    flush=True,
                )

                summary_records.append({
                    "dataset": d_name,
                    "K": K,
                    "strategy": strat,
                    "MSE": mean_mse,
                    "MSE_lo": mse_lo,
                    "MSE_hi": mse_hi,
                    "MAE": mean_mae,
                    "latency_median_ms": med_latency_ms,
                    "latency_p90_ms": p90_latency_ms,
                    "peak_vram_mb": peak_mem_mb,
                    "symmetric_averaging": use_symmetric_averaging,
                    "git_commit": commit_hash,
                    "timestamp": run_timestamp,
                    "horizon": horizon,
                    "context_length": context_length,
                    "seed": seed,
                })

            df_e5 = pd.DataFrame(summary_records)
            out_csv = RESULTS_DIR / "e5_grouping_summary.csv"
            df_e5.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("E5 EXPERIMENT COMPLETE!")
    df_e5 = pd.DataFrame(summary_records)
    print(df_e5[["dataset", "K", "strategy", "MSE", "MSE_lo", "MSE_hi", "latency_median_ms", "latency_p90_ms"]].to_string(index=False))
    print("=" * 80)
    return df_e5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args()

    run_e5(datasets=args.datasets, horizon=args.horizon, num_samples=args.samples, seed=args.seed)
