"""E1 Experiment Runner: LTSF 8 Datasets Univariate vs. Multivariate Evaluation.
Completely updated according to Task 8 and Task 9:
  1. Mean Pinball Loss (Task 8): on normalized data, reports mean pinball loss (scalar float).
  2. Stationary Block Bootstrap for Benefit B (Task 9): block length >= 5, 2000 resamples.
  3. Holm-Bonferroni correction across 8 datasets x 4 horizons.
  4. Raw CSV deduplication on (dataset, horizon, window_idx) when resuming.
  5. Records git commit hash, timestamp, seed, and key parameters.
"""

import argparse
import datetime
import math
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

from probe.config import (
    DEFAULT_HORIZONS,
    LTSF_DATASETS,
    MODEL_DIR,
    PRIMARY_SEED,
    RESULTS_DIR,
)
from probe.data import load_dataset
from probe.metrics import (
    block_bootstrap_reliance_ci,
    calc_benefit,
    calc_coverage,
    calc_mae,
    calc_mean_pinball_loss,
    calc_mse,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
)
from probe.models.timesfm3 import TimesFM3Wrapper


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def run_e1(
    datasets: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    context_length: int = 512,
    num_samples: int = 300,
    stride1: bool = False,
    seed: int = PRIMARY_SEED,
    device: str = "cuda",
    use_symmetric_averaging: bool = True,
):
    commit_hash = get_git_commit_hash()
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("STARTING E1: LTSF 8 DATASETS EXPERIMENT (UV vs MV)")
    print(f"Git Commit: {commit_hash} | Timestamp: {run_timestamp}")
    print(f"Mode: {'Full Stride 1' if stride1 else f'Sampled ({num_samples} windows)'}")
    print(f"Horizons: {horizons or DEFAULT_HORIZONS}")
    print(f"Context Length: {context_length} | Seed: {seed}")
    print(f"Symmetric Averaging: {use_symmetric_averaging}")
    print("=" * 80)

    target_datasets = datasets or list(LTSF_DATASETS.keys())
    target_horizons = horizons or DEFAULT_HORIZONS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)

    summary_csv = RESULTS_DIR / f"e1_ltsf_{'stride1' if stride1 else 'sampled'}.csv"
    raw_e1_csv = RESULTS_DIR / f"e1_ltsf_raw_{'stride1' if stride1 else 'sampled'}.csv"

    # Load existing raw records if present for deduplication (Task 9)
    if raw_e1_csv.exists():
        try:
            df_raw_prev = pd.read_csv(raw_e1_csv)
            raw_records = df_raw_prev.to_dict("records")
            existing_pairs = set(zip(df_raw_prev["dataset"], df_raw_prev["horizon"], df_raw_prev["window_idx"]))
            print(f"Loaded {len(raw_records)} existing raw records from {raw_e1_csv}")
        except Exception as e:
            print(f"Warning reading existing raw csv: {e}")
            raw_records = []
            existing_pairs = set()
    else:
        raw_records = []
        existing_pairs = set()

    for d_name in target_datasets:
        print(f"\n>>> Processing Dataset: {d_name} (variates: {LTSF_DATASETS[d_name]['variates']})")
        ds = load_dataset(d_name, context_length=context_length, horizon=max(target_horizons))

        if stride1:
            window_indices = list(range(ds.num_windows_stride1))
        else:
            window_indices = ds.get_sampled_window_indices(num_samples=num_samples, seed=seed)

        print(f"Total windows to evaluate: {len(window_indices)}")

        batch_windows_size = 32
        num_batches = math.ceil(len(window_indices) / batch_windows_size)

        for b_idx in range(num_batches):
            b_slice = window_indices[b_idx * batch_windows_size : (b_idx + 1) * batch_windows_size]
            needed_horizons = [H for H in target_horizons if any((d_name, H, w) not in existing_pairs for w in b_slice)]
            if not needed_horizons:
                continue

            t0 = time.time()
            ctx_batch = []
            target_full_batch = []
            for w_idx in b_slice:
                c, t_full = ds.get_window(w_idx)
                ctx_batch.append(c)
                target_full_batch.append(t_full[:, :720])

            # 1. Batched Univariate Forecast (Horizon 720 sliced to 96/192/336/720 per B4 prefix consistency)
            preds_uv_720, quants_uv_720 = wrapper.forecast(
                ctx_batch,
                horizon=720,
                univariate=True,
                use_symmetric_averaging=use_symmetric_averaging,
                return_quantiles=True,
            )

            # 2. Batched Multivariate Forecast (Horizon 720 sliced to 96/192/336/720 per B4 prefix consistency)
            preds_mv_720, quants_mv_720 = wrapper.forecast(
                ctx_batch,
                horizon=720,
                univariate=False,
                use_symmetric_averaging=use_symmetric_averaging,
                return_quantiles=True,
            )

            for H in target_horizons:
                for w_idx_val, p_uv_full, q_uv_full, p_mv_full, q_mv_full, t_full_val in zip(
                    b_slice, preds_uv_720, quants_uv_720, preds_mv_720, quants_mv_720, target_full_batch
                ):
                    if (d_name, H, w_idx_val) in existing_pairs:
                        continue
                    p_uv = p_uv_full[:, :H]
                    p_mv = p_mv_full[:, :H]
                    q_uv = q_uv_full[:, :H, :]
                    q_mv = q_mv_full[:, :H, :]
                    t_val = t_full_val[:, :H]

                    e_uv = float(calc_mse(p_uv, t_val))
                    e_mv = float(calc_mse(p_mv, t_val))
                    mae_uv = float(calc_mae(p_uv, t_val))
                    mae_mv = float(calc_mae(p_mv, t_val))
                    # Scalar Mean Pinball Loss (Task 8)
                    pinball_uv = float(calc_mean_pinball_loss(q_uv, t_val))
                    pinball_mv = float(calc_mean_pinball_loss(q_mv, t_val))
                    cov_uv = float(calc_coverage(q_uv, t_val))
                    cov_mv = float(calc_coverage(q_mv, t_val))

                    rec = {
                        "dataset": d_name,
                        "horizon": H,
                        "window_idx": w_idx_val,
                        "MSE_UV": e_uv,
                        "MSE_MV": e_mv,
                        "MAE_UV": mae_uv,
                        "MAE_MV": mae_mv,
                        "Pinball_UV": pinball_uv,
                        "Pinball_MV": pinball_mv,
                        "Cov_UV": cov_uv,
                        "Cov_MV": cov_mv,
                        "Delta_MSE": e_mv - e_uv,
                        "git_commit": commit_hash,
                        "timestamp": run_timestamp,
                        "context_length": context_length,
                        "seed": seed,
                        "use_symmetric_averaging": use_symmetric_averaging,
                    }
                    raw_records.append(rec)
                    existing_pairs.add((d_name, H, w_idx_val))

            elapsed = time.time() - t0
            print(f"    Batch [{b_idx+1}/{num_batches}] done (sliced to {target_horizons}) | Elapsed: {elapsed:.1f}s", flush=True)

            # Checkpoint save raw deduplicated data
            df_raw = pd.DataFrame(raw_records).drop_duplicates(subset=["dataset", "horizon", "window_idx"])
            df_raw.to_csv(raw_e1_csv, index=False)

    # =========================================================================
    # Task 9: Summary with Block Bootstrap CIs and Holm-Bonferroni Correction
    # =========================================================================
    print("\nComputing Summary with Window Block Bootstrap CIs and Holm Correction...")
    df_raw = pd.DataFrame(raw_records).drop_duplicates(subset=["dataset", "horizon", "window_idx"])

    summary_rows = []
    group_keys = []
    raw_p_values = []
    ci_dict = {}

    for (d_name, H), group in df_raw.groupby(["dataset", "horizon"]):
        e_uv = group["MSE_UV"].to_numpy()
        e_mv = group["MSE_MV"].to_numpy()

        ci_res = block_bootstrap_reliance_ci(
            e_pi_windows=e_mv,
            e_mv_windows=e_mv,
            e_uv_windows=e_uv,
            block_length=5,
            n_bootstraps=2000,
            seed=seed,
        )

        _, p_val = paired_wilcoxon_test(e_uv, e_mv)
        benefit_b, median_b = calc_benefit(e_uv, e_mv)

        key = (d_name, H)
        group_keys.append(key)
        raw_p_values.append(p_val)
        ci_dict[key] = {
            "benefit_b": benefit_b * 100.0,
            "b_lo": ci_res["B_lo"],
            "b_hi": ci_res["B_hi"],
            "b_sig": ci_res["B_significant"],
            "median_b": median_b * 100.0,
            "mse_uv": float(group["MSE_UV"].mean()),
            "mse_mv": float(group["MSE_MV"].mean()),
            "mae_uv": float(group["MAE_UV"].mean()),
            "mae_mv": float(group["MAE_MV"].mean()),
            "pinball_uv": float(group["Pinball_UV"].mean()),
            "pinball_mv": float(group["Pinball_MV"].mean()),
            "cov_uv": float(group["Cov_UV"].mean()),
            "cov_mv": float(group["Cov_MV"].mean()),
            "n_win": len(group),
        }

    # Holm-Bonferroni correction across all (dataset x horizon) pairs (Task 9)
    adj_p_values = holm_bonferroni_correction(raw_p_values)

    for (d_name, H), raw_p, adj_p in zip(group_keys, raw_p_values, adj_p_values):
        stats = ci_dict[(d_name, H)]
        summary_rows.append({
            "dataset": d_name,
            "horizon": H,
            "num_windows": stats["n_win"],
            "MSE_UV": stats["mse_uv"],
            "MSE_MV": stats["mse_mv"],
            "MAE_UV": stats["mae_uv"],
            "MAE_MV": stats["mae_mv"],
            "Pinball_UV": stats["pinball_uv"],
            "Pinball_MV": stats["pinball_mv"],
            "Cov_UV": stats["cov_uv"],
            "Cov_MV": stats["cov_mv"],
            "Benefit_B_pct": stats["benefit_b"],
            "B_lo": stats["b_lo"],
            "B_hi": stats["b_hi"],
            "B_significant": stats["b_sig"],
            "Median_B_pct": stats["median_b"],
            "Wilcoxon_p_raw": raw_p,
            "Wilcoxon_p_adj_holm": adj_p,
            "git_commit": commit_hash,
            "timestamp": run_timestamp,
            "context_length": context_length,
            "seed": seed,
            "use_symmetric_averaging": use_symmetric_averaging,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved E1 summary to {summary_csv}")
    print("=" * 80)
    print(df_summary[["dataset", "horizon", "MSE_UV", "MSE_MV", "Benefit_B_pct", "B_significant", "Wilcoxon_p_adj_holm"]].to_string(index=False))
    print("=" * 80)
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, help="List of datasets to run")
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="List of horizons")
    parser.add_argument("--samples", type=int, default=300, help="Number of sampled windows")
    parser.add_argument("--stride1", action="store_true", help="Run full stride 1")
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED, help="Random seed")
    args = parser.parse_args()

    run_e1(
        datasets=args.datasets,
        horizons=args.horizons,
        num_samples=args.samples,
        stride1=args.stride1,
        seed=args.seed,
    )
