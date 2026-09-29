"""E2 Experiment Runner: Target-Centric Interventions and Diagnostic Probing.
Completely rewritten adhering strictly to Tasks 3, 4, 5, 6, 7:
  1. Sensitivity S scale: S = mean_h |y_pi - y_mv| on normalized scale (training std = 1), no division by ds.std.
  2. Unified frequency steps: steps_per_day = 24*60 // freq_min, steps_per_week = 7 * steps_per_day.
  3. Pre-forecast snippet guarantee: I2 & I3 snippets strictly end before forecast start.
  4. N3 lag fixed to constant 16 steps.
  5. I6 shifts: 1 step, 1 hour, 1 day, 1 week with step count and duration label.
  6. Prediction read via meta["target_pos"], target channel invariance asserted.
  7. Block and window caching: MV and UV base forecasts computed once and reused across targets.
  8. Batching interventions per block.
  9. Electricity channel 320 (single-channel block) excluded from target sampling.
  10. Window-level aggregation, stationary block bootstrap (block >= 5), Wilcoxon tests with Holm correction.
  11. NaN handling for near-zero denominators, B_significant indicator column.
  12. Git commit hash, timestamp, and all critical configuration recorded in every CSV.
"""

import argparse
from collections import defaultdict
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Path configuration
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
from probe.grouping import sample_stratified_targets
from probe.interventions import (
    N3_LAG_STEPS,
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
from probe.metrics import (
    block_bootstrap_reliance_ci,
    calc_aggregate_reliance,
    calc_mae,
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


def get_fixed_channel_group(
    target_idx: int,
    total_variates: int,
    block_size: int = 32,
) -> Tuple[List[int], int]:
    """
    Returns the fixed 32-channel block containing target_idx and target's relative index in that block.
    """
    chunk_id = target_idx // block_size
    start_idx = chunk_id * block_size
    end_idx = min(start_idx + block_size, total_variates)
    group_indices = list(range(start_idx, end_idx))
    t_pos = target_idx - start_idx
    return group_indices, t_pos


def run_e2(
    datasets: Optional[List[str]] = None,
    horizon: int = 96,
    context_length: int = 512,
    num_samples: int = 150,
    seed: int = PRIMARY_SEED,
    device: str = "cuda",
    use_symmetric_averaging: bool = True,
):
    commit_hash = get_git_commit_hash()
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("STARTING E2: TARGET-CENTRIC INTERVENTIONS EXPERIMENT (REVISED & OPTIMIZED)")
    print(f"Git Commit: {commit_hash} | Timestamp: {run_timestamp}")
    print(f"Horizon: {horizon}, Context Length: {context_length}, Windows: {num_samples}, Seed: {seed}")
    print(f"Symmetric Averaging: {use_symmetric_averaging}")
    print("=" * 80)

    target_datasets = datasets or ["ETTh1", "weather", "electricity", "solar"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)
    raw_records = []

    for d_name in target_datasets:
        print(f"\n>>> Running Interventions on Dataset: {d_name}")
        ds = load_dataset(d_name, context_length=context_length, horizon=horizon)
        window_indices = ds.get_sampled_window_indices(num_samples=num_samples, seed=seed)
        N = ds.num_variates

        freq_str = ds.info["freq"]
        steps_per_day, steps_per_week = get_freq_steps(freq_str)
        minutes = (24 * 60) // steps_per_day
        steps_1h = max(1, 60 // minutes)

        # Stratified sampling of targets excluding 1-channel blocks (Task 7)
        corr_matrix = np.corrcoef(ds.norm_values[: ds.border2s[0]], rowvar=False)
        np.nan_to_num(corr_matrix, copy=False, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0)

        if N <= 32:
            sampled_targets = list(range(N))
        else:
            sampled_targets = sample_stratified_targets(
                corr_matrix,
                num_targets=min(64, N),
                chunk_size=32,
                seed=seed,
            )

        print(f"Dataset {d_name} ({freq_str}): N={N} channels, steps/day={steps_per_day}, steps/week={steps_per_week}")
        print(f"Evaluating {len(sampled_targets)} stratified targets (single-channel blocks excluded).")

        # Group targets by official 32-channel block for caching (Task 7)
        targets_by_block: Dict[int, List[int]] = defaultdict(list)
        for t_idx in sampled_targets:
            chunk_id = t_idx // 32
            targets_by_block[chunk_id].append(t_idx)

        t0 = time.time()
        for w_count, w_idx in enumerate(window_indices):
            abs_w_idx = ds.border1s[2] + w_idx
            ctx_full, target_full = ds.get_window(w_idx)
            target = target_full[:, :horizon]  # [N, H]

            # 1. Batch univariate predictions across all targets in window (Task 7)
            uv_slices = [ctx_full[t_idx : t_idx + 1] for t_idx in sampled_targets]
            uv_preds, _ = wrapper.forecast(
                uv_slices,
                horizon=horizon,
                univariate=True,
                use_symmetric_averaging=use_symmetric_averaging,
                return_quantiles=False,
            )
            uv_pred_map = {t_idx: uv_preds[idx][0] for idx, t_idx in enumerate(sampled_targets)}

            # 2. Iterate through each block to reuse MV baseline predictions
            for chunk_id, block_targets in targets_by_block.items():
                start_ch = chunk_id * 32
                end_ch = min(start_ch + 32, N)
                group_indices = list(range(start_ch, end_ch))
                ctx_group = ctx_full[group_indices].copy()

                # Multi-variate base prediction on this block (computed once per block)
                mv_group_preds, _ = wrapper.forecast(
                    ctx_group,
                    horizon=horizon,
                    univariate=False,
                    use_symmetric_averaging=use_symmetric_averaging,
                    return_quantiles=False,
                )

                # Now evaluate interventions for each target in this block
                for t_idx in block_targets:
                    t_pos = t_idx - start_ch
                    y_true = target[t_idx]
                    pred_mv_tgt = mv_group_preds[t_pos]
                    e_mv = float(calc_mse(pred_mv_tgt, y_true))
                    e_uv = float(calc_mse(uv_pred_map[t_idx], y_true))

                    # Prepare intervention dictionary with strict parameters
                    interventions = {
                        "N1_perm": lambda: apply_n1_permutation(ctx_group, t_pos, seed=seed + w_idx),
                        "N2_dup": lambda: apply_n2_duplicate(ctx_group, t_pos),
                        "N3_lag_16": lambda: apply_n3_lag_copy(ctx_group, t_pos, lag=N3_LAG_STEPS),
                        "N4_noise": lambda: apply_n4_gaussian_noise(ctx_group, t_pos, seed=seed + w_idx),
                        "I1_shuffle": lambda: apply_i1_temporal_shuffle(ctx_group, t_pos, seed=seed + w_idx),
                        "I2_swap_unaligned": lambda: apply_i2_temporal_swap(
                            ctx_group, t_pos, ds.norm_values[:, group_indices], abs_w_idx, steps_per_day=steps_per_day
                        ),
                        "I3_swap_aligned": lambda: apply_i3_calendar_aligned_swap(
                            ctx_group, t_pos, ds.norm_values[:, group_indices], abs_w_idx, steps_per_week=steps_per_week, weeks_offset=4
                        ),
                        "I4_iaaft": lambda: apply_i4_iaaft_surrogate(ctx_group, t_pos, seed=seed + w_idx, max_iter=20),
                        "I6_shift_1step": lambda: apply_i6_circular_shift(ctx_group, t_pos, delta_steps=1, duration_label="1_step"),
                        "I6_shift_1h": lambda: apply_i6_circular_shift(ctx_group, t_pos, delta_steps=steps_1h, duration_label="1_hour"),
                        "I6_shift_1d": lambda: apply_i6_circular_shift(ctx_group, t_pos, delta_steps=steps_per_day, duration_label="1_day"),
                        "I6_shift_1w": lambda: apply_i6_circular_shift(ctx_group, t_pos, delta_steps=steps_per_week, duration_label="1_week"),
                        "P1_lead_16": lambda: apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=16, snr=10.0, seed=seed),
                        "P1_lead_32": lambda: apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=32, snr=10.0, seed=seed),
                        "P1_lead_1h": lambda: apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=steps_1h, snr=10.0, seed=seed),
                        "P1_lead_6h": lambda: apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=min(95, 6 * steps_1h), snr=10.0, seed=seed),
                    }

                    # Execute each intervention
                    for int_name, int_fn in interventions.items():
                        ctx_pi, meta = int_fn()
                        pred_pi_all, _ = wrapper.forecast(
                            ctx_pi,
                            horizon=horizon,
                            univariate=False,
                            use_symmetric_averaging=use_symmetric_averaging,
                            return_quantiles=False,
                        )
                        # Read target forecast strictly at meta["target_pos"] (Task 5)
                        t_target_pos = meta["target_pos"]
                        pred_pi_tgt = pred_pi_all[t_target_pos]

                        e_pi = float(calc_mse(pred_pi_tgt, y_true))

                        # Sensitivity S on normalized scale: mean_h |y_pi - y_mv| (Task 3)
                        sens = float(np.mean(np.abs(pred_pi_tgt - pred_mv_tgt)))

                        raw_records.append({
                            "dataset": d_name,
                            "window_idx": w_idx,
                            "target_idx": t_idx,
                            "intervention": int_name,
                            "MSE_UV": e_uv,
                            "MSE_MV": e_mv,
                            "MSE_PI": e_pi,
                            "delta_PI": e_pi - e_mv,
                            "delta_UV": e_uv - e_mv,
                            "Sensitivity_S": sens,
                            "uses_future": meta.get("uses_future", False),
                            "git_commit": commit_hash,
                            "timestamp": run_timestamp,
                            "horizon": horizon,
                            "context_length": context_length,
                            "num_windows": num_samples,
                            "num_targets": len(sampled_targets),
                            "seed": seed,
                            "use_symmetric_averaging": use_symmetric_averaging,
                        })

            if (w_count + 1) % 25 == 0 or (w_count + 1) == len(window_indices):
                elapsed = time.time() - t0
                print(f"  [{w_count+1}/{len(window_indices)}] Evaluated interventions | Elapsed: {elapsed:.1f}s", flush=True)

        # Incremental checkpointing
        df_raw = pd.DataFrame(raw_records)
        raw_csv = RESULTS_DIR / "e2_interventions_raw_windows.csv"
        df_raw.to_csv(raw_csv, index=False)
        print(f"  --> Checkpointed raw window records to {raw_csv}", flush=True)

    # =========================================================================
    # Task 6: Window-level aggregation, Block Bootstrap, and Holm Correction
    # =========================================================================
    print("\nComputing Window-Level Block Bootstrap CIs and Holm-Corrected Wilcoxon Tests...")
    df_raw = pd.DataFrame(raw_records)
    summary_rows = []

    for d_name, d_group in df_raw.groupby("dataset"):
        # Collect p-values for Holm correction across all interventions for this dataset
        int_names = []
        raw_p_values = []
        ci_results = {}
        sens_means = {}
        n_windows_map = {}
        n_targets_map = {}

        for int_name, group in d_group.groupby("intervention"):
            # Aggregate across all targets within each window first (Task 6)
            window_agg = group.groupby("window_idx")[["MSE_PI", "MSE_MV", "MSE_UV"]].sum().reset_index()
            e_pi_win = window_agg["MSE_PI"].to_numpy()
            e_mv_win = window_agg["MSE_MV"].to_numpy()
            e_uv_win = window_agg["MSE_UV"].to_numpy()

            # Stationary/block bootstrap over windows (block length >= 5)
            ci_res = block_bootstrap_reliance_ci(
                e_pi_windows=e_pi_win,
                e_mv_windows=e_mv_win,
                e_uv_windows=e_uv_win,
                block_length=5,
                n_bootstraps=2000,
                seed=seed,
            )

            # Window-level paired Wilcoxon test
            _, p_val = paired_wilcoxon_test(e_pi_win, e_mv_win)

            int_names.append(int_name)
            raw_p_values.append(p_val)
            ci_results[int_name] = ci_res
            sens_means[int_name] = float(group["Sensitivity_S"].mean())
            n_windows_map[int_name] = group["window_idx"].nunique()
            n_targets_map[int_name] = group["target_idx"].nunique()

        # Apply Holm-Bonferroni correction across interventions (Task 6)
        adj_p_values = holm_bonferroni_correction(raw_p_values)

        for int_name, raw_p, adj_p in zip(int_names, raw_p_values, adj_p_values):
            ci_res = ci_results[int_name]
            b_sig = ci_res["B_significant"]

            # If B is not significant, leave R column NaN/blank (Task 6)
            r_val = ci_res["R"] if b_sig else np.nan
            r_lo = ci_res["R_lo"] if b_sig else np.nan
            r_hi = ci_res["R_hi"] if b_sig else np.nan

            summary_rows.append({
                "dataset": d_name,
                "intervention": int_name,
                "n_win": n_windows_map[int_name],
                "n_tgt": n_targets_map[int_name],
                "B": ci_res["B"],
                "B_lo": ci_res["B_lo"],
                "B_hi": ci_res["B_hi"],
                "B_significant": b_sig,
                "dPI_pct": ci_res["dPI_pct"],
                "dPI_lo": ci_res["dPI_lo"],
                "dPI_hi": ci_res["dPI_hi"],
                "delta_MSE": ci_res["delta_mse"],
                "delta_MSE_lo": ci_res["delta_mse_lo"],
                "delta_MSE_hi": ci_res["delta_mse_hi"],
                "R": r_val,
                "R_lo": r_lo,
                "R_hi": r_hi,
                "nan_ratio": ci_res["nan_ratio"],
                "Sensitivity_S": sens_means[int_name],
                "p_raw": raw_p,
                "p_adj_holm": adj_p,
                "git_commit": commit_hash,
                "timestamp": run_timestamp,
                "horizon": horizon,
                "context_length": context_length,
                "seed": seed,
                "use_symmetric_averaging": use_symmetric_averaging,
            })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_csv = RESULTS_DIR / "e2_interventions_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"\nSuccessfully saved revised E2 summary to {out_summary_csv}")
    print("=" * 80)
    print(df_summary[["dataset", "intervention", "B", "B_significant", "dPI_pct", "R", "Sensitivity_S", "p_adj_holm"]].to_string(index=False))
    print("=" * 80)
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args()

    run_e2(datasets=args.datasets, horizon=args.horizon, num_samples=args.samples, seed=args.seed)
