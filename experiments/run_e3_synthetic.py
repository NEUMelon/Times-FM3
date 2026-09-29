"""E3 Experiment Runner: Controlled Synthetic Structural Probing & Kalman Oracle Gap Closure.
Strictly evaluates G1-G6 synthetic datasets against exact Kalman filter oracles.
Logs git commit hash, timestamp, and key parameters.
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

# Ensure probe can be imported
sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import MODEL_DIR, PRIMARY_SEED, RESULTS_DIR
from probe.metrics import calc_benefit, calc_gap_closure, calc_mse
from probe.models.timesfm3 import TimesFM3Wrapper
from probe.synth.generators import (
    generate_g1_global_low_rank,
    generate_g2_local_blocks,
    generate_g3_structural_drift,
    generate_g5_lead_lag,
    generate_g6_null,
)
from probe.synth.oracle import compute_exact_kalman_oracles


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def run_e3(
    context_length: int = 512,
    horizon: int = 96,
    num_windows: int = 50,
    seed: int = PRIMARY_SEED,
    device: str = "cuda",
    use_symmetric_averaging: bool = True,
):
    commit_hash = get_git_commit_hash()
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("STARTING E3: SYNTHETIC DATA & ORACLE GAP CLOSURE EXPERIMENT (REVISED)")
    print(f"Git Commit: {commit_hash} | Timestamp: {run_timestamp}")
    print(f"Context: {context_length}, Horizon: {horizon}, Windows per config: {num_windows}, Seed: {seed}")
    print(f"Symmetric Averaging: {use_symmetric_averaging}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)

    records = []
    configs = []

    # Configs for G1: Low-Rank Factor
    for rank in [1, 2, 4]:
        for gamma in [0.0, 0.25, 0.5, 0.75]:
            configs.append(("G1", {"N": 8, "rank": rank, "gamma": gamma}))
            if gamma == 0.5:
                configs.append(("G1", {"N": 32, "rank": rank, "gamma": gamma}))

    # Configs for G2: Local Blocks
    for b_size in [2, 4, 8]:
        for gamma in [0.25, 0.5, 0.75]:
            configs.append(("G2", {"N": 16, "block_size": b_size, "gamma": gamma}))
    # Verified G2 configuration with Oracle Gap >= 5% for valid Gap Closure
    configs.append(("G2", {"N": 16, "block_size": 8, "gamma": 0.85, "alpha": 0.98, "sigma_eps": 0.2}))

    # Configs for G3: Structural Drift
    for delta in [0.0, 0.25, 0.5, 1.0]:
        configs.append(("G3", {"N": 8, "delta": delta, "gamma": 0.5, "tau_loc": "pred_start"}))
        configs.append(("G3", {"N": 8, "delta": delta, "gamma": 0.5, "tau_loc": "mid"}))

    # Configs for G5: Lead-Lag
    for lag in [1, 8, 16, 32, 64]:
        configs.append(("G5", {"N": 8, "lag": lag, "lead_coeff": 0.8}))

    # Configs for G6: Null
    configs.append(("G6", {"N": 8}))
    configs.append(("G6", {"N": 32}))

    print(f"Total synthetic configurations to evaluate: {len(configs)}")
    t0 = time.time()

    for cfg_idx, (g_type, params) in enumerate(configs):
        N = params.get("N", 8)
        T_total = context_length + horizon + 50
        mse_uv_model_list = []
        mse_mv_model_list = []
        mse_uv_star_list = []
        mse_mv_star_list = []

        for w in range(num_windows):
            w_seed = seed + cfg_idx * 1000 + w
            if g_type == "G1":
                X, meta = generate_g1_global_low_rank(N=N, T=T_total, rank=params["rank"], gamma=params["gamma"], seed=w_seed)
            elif g_type == "G2":
                X, meta = generate_g2_local_blocks(
                    N=N,
                    T=T_total,
                    block_size=params["block_size"],
                    gamma=params["gamma"],
                    sigma_eps=params.get("sigma_eps", 0.1),
                    alpha=params.get("alpha", 0.95),
                    seed=w_seed,
                )
            elif g_type == "G3":
                tau = context_length if params["tau_loc"] == "pred_start" else context_length // 2
                X, meta = generate_g3_structural_drift(N=N, T=T_total, tau=tau, delta=params["delta"], gamma=params["gamma"], seed=w_seed)
            elif g_type == "G5":
                X, meta = generate_g5_lead_lag(N=N, T=T_total, lag=params["lag"], lead_coeff=params["lead_coeff"], seed=w_seed)
            elif g_type == "G6":
                X, meta = generate_g6_null(N=N, T=T_total, seed=w_seed)

            # Slice window
            ctx = X[:, :context_length]
            target = X[:, context_length : context_length + horizon]

            # Model forecast
            pred_uv, _ = wrapper.forecast(ctx, horizon=horizon, univariate=True, use_symmetric_averaging=use_symmetric_averaging, return_quantiles=False)
            pred_mv, _ = wrapper.forecast(ctx, horizon=horizon, univariate=False, use_symmetric_averaging=use_symmetric_averaging, return_quantiles=False)

            # Evaluate target channel
            t_idx = 0 if g_type != "G5" else 1  # For G5, evaluate follower channel 1
            e_uv_m = float(calc_mse(pred_uv[t_idx], target[t_idx]))
            e_mv_m = float(calc_mse(pred_mv[t_idx], target[t_idx]))

            # Exact Kalman filter oracles
            e_uv_s, e_mv_s = compute_exact_kalman_oracles(ctx, target, target_idx=t_idx, meta=meta)

            mse_uv_model_list.append(e_uv_m)
            mse_mv_model_list.append(e_mv_m)
            mse_uv_star_list.append(e_uv_s)
            mse_mv_star_list.append(e_mv_s)

        mean_uv_m = float(np.mean(mse_uv_model_list))
        mean_mv_m = float(np.mean(mse_mv_model_list))
        mean_uv_s = float(np.mean(mse_uv_star_list))
        mean_mv_s = float(np.mean(mse_mv_star_list))

        benefit_b, _ = calc_benefit(np.array(mse_uv_model_list), np.array(mse_mv_model_list))
        gc = calc_gap_closure(mean_uv_m, mean_mv_m, mean_uv_s, mean_mv_s)

        rec = {
            "generator": g_type,
            "params": str(params),
            "N": N,
            "gamma": params.get("gamma", 0.0),
            "rank": params.get("rank", np.nan),
            "delta": params.get("delta", np.nan),
            "lag": params.get("lag", np.nan),
            "MSE_UV_Model": mean_uv_m,
            "MSE_MV_Model": mean_mv_m,
            "MSE_UV_Oracle": mean_uv_s,
            "MSE_MV_Oracle": mean_mv_s,
            "Benefit_B_pct": benefit_b * 100.0,
            "Gap_Closure_GC": gc,
            "git_commit": commit_hash,
            "timestamp": run_timestamp,
            "seed": seed,
            "use_symmetric_averaging": use_symmetric_averaging,
        }
        records.append(rec)

        if (cfg_idx + 1) % 5 == 0 or (cfg_idx + 1) == len(configs):
            elapsed = time.time() - t0
            print(f"[{cfg_idx+1}/{len(configs)}] {g_type} {params} | B: {benefit_b*100:+.2f}% | GC: {gc:.3f} | Elapsed: {elapsed:.1f}s", flush=True)

        # Checkpoint save
        df_synth = pd.DataFrame(records)
        out_csv = RESULTS_DIR / "e3_synthetic_summary.csv"
        df_synth.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("E3 SYNTHETIC EXPERIMENT COMPLETE!")
    df_synth = pd.DataFrame(records)
    print(df_synth[["generator", "params", "MSE_UV_Model", "MSE_MV_Model", "MSE_UV_Oracle", "MSE_MV_Oracle", "Benefit_B_pct", "Gap_Closure_GC"]].head(15))
    print("=" * 80)
    return df_synth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args()

    run_e3(num_windows=args.windows, horizon=args.horizon, seed=args.seed)
