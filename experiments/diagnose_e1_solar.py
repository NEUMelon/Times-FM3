"""Solar-Energy Diagnostic Script: Window-by-Window Error vs Diurnal Regime & Normalization Scale.
Directly investigates Claude's hypothesis:
  "Solar 均值收益为 -7.7%, 但中位数收益为 +7.6% (H=96), 说明退化集中在少数窗口。
   怀疑夜间大量为 0 的通道经过逐通道归一化后数值被放大，再通过变量注意力扩散到目标通道。"
Outputs:
  - Raw window-level diagnostic CSV: results/solar_diurnal_diagnostic.csv
  - Diurnal breakdown statistics: results/solar_diurnal_breakdown.csv
  - Spearman rank correlation between patch-zero fraction & Delta_MSE
  - Fisher exact test on degradation rate (Night vs Daytime)
  - Mitigation test via context-tail zero suppression
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Ensure probe can be imported
sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import MODEL_DIR, PRIMARY_SEED, RESULTS_DIR
from probe.data import load_dataset
from probe.metrics import calc_mae, calc_mse, paired_wilcoxon_test
from probe.models.timesfm3 import TimesFM3Wrapper


def detect_solar_frequency(ds) -> int:
    """
    Empirically detect and assert Solar sampling frequency.
    Checks autocorrelation peaks at lag 24 (hourly) vs lag 144 (10-minute).
    """
    raw_sample = ds.raw_values[:min(5000, len(ds.raw_values)), 0]
    raw_sample = raw_sample - np.mean(raw_sample)
    
    # Autocorrelation for lags up to 200
    n = len(raw_sample)
    var = np.var(raw_sample)
    if var < 1e-6:
        # Fallback if first channel is constant
        raw_sample = ds.raw_values[:min(5000, len(ds.raw_values)), 1]
        raw_sample = raw_sample - np.mean(raw_sample)
        var = np.var(raw_sample)
        
    lags = [24, 144]
    acfs = {}
    for lag in lags:
        if n > lag:
            cov = np.mean(raw_sample[:-lag] * raw_sample[lag:])
            acfs[lag] = cov / var
        else:
            acfs[lag] = -1.0
            
    print(f"[Solar Frequency Check] ACF(24) = {acfs.get(24, 0):.4f}, ACF(144) = {acfs.get(144, 0):.4f}")
    if acfs.get(144, -1) > 0.5:
        print("[Solar Frequency Check] Detected 10-minute frequency: 144 steps/day.")
        return 144
    elif acfs.get(24, -1) > 0.5:
        print("[Solar Frequency Check] Detected hourly frequency: 24 steps/day.")
        return 24
    else:
        # Fallback based on dataset total length (52560 = 365 * 144)
        if len(ds.raw_values) > 30000:
            print("[Solar Frequency Check] Length > 30000, using 144 steps/day.")
            return 144
        else:
            print("[Solar Frequency Check] Length <= 30000, using 24 steps/day.")
            return 24


def classify_solar_hour(hour_of_day: float) -> str:
    """
    Classifies hour of day into diurnal regimes according to spec:
    - Daytime: 6:00 - 18:00 (光照充足)
    - Night: 20:00 - 4:00 (全零或接近零)
    - Twilight: 4:00 - 6:00, 18:00 - 20:00 (晨昏快速变化)
    """
    if (hour_of_day >= 20.0) or (hour_of_day < 4.0):
        return "Night (20:00-4:00)"
    elif (4.0 <= hour_of_day < 6.0) or (18.0 <= hour_of_day < 20.0):
        return "Twilight (4:00-6:00, 18:00-20:00)"
    else:
        return "Daytime (6:00-18:00)"


def run_solar_diagnostic(
    horizon: int = 96,
    context_length: int = 512,
    num_samples: int = 200,
    seed: int = PRIMARY_SEED,
    device: str = "cuda",
    max_targets: int = 16,
):
    print("=" * 80)
    print("STARTING SOLAR-ENERGY DIURNAL & NORMALIZATION DIAGNOSTIC")
    print(f"Horizon: {horizon}, Context Length: {context_length}, Windows: {num_samples}, Seed: {seed}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("solar", context_length=context_length, horizon=horizon)
    window_indices = ds.get_sampled_window_indices(num_samples=num_samples, seed=seed)
    N = ds.num_variates  # 137 channels

    # Detect sampling frequency
    steps_per_day = detect_solar_frequency(ds)
    steps_per_hour = steps_per_day / 24.0

    # Sample target channels (e.g. 16 evenly spaced channels across the 137 solar plants)
    target_channels = list(range(0, N, max(1, N // max_targets)))[:max_targets]

    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device=device)
    records = []
    patch_len = 32  # TimesFM patch size

    print(f"Evaluating {len(window_indices)} windows across {len(target_channels)} target channels...")
    t0 = time.time()

    for w_count, w_idx in enumerate(window_indices):
        ctx_norm, target_norm = ds.get_window(w_idx)  # Normalized scale [N, T], [N, H]
        target = target_norm[:, :horizon]

        # Raw slice for zero-channel detection
        global_ctx_start = ds.border1s[2] + w_idx
        raw_ctx = ds.raw_values[global_ctx_start : global_ctx_start + context_length].T  # [N, T]

        # Calculate hour of day corresponding to the start of the prediction horizon
        # Global time index = border2_start + w_idx + context_length
        global_t = global_ctx_start + context_length
        step_in_day = global_t % steps_per_day
        hour_of_day = (step_in_day / steps_per_day) * 24.0
        regime = classify_solar_hour(hour_of_day)

        # Fraction of channels where the ending patch (last 32 steps) in raw data is all zero (< 1e-4)
        raw_tail_patch = raw_ctx[:, -patch_len:]
        is_zero_channel = np.all(raw_tail_patch < 1e-4, axis=1)  # [N] boolean
        zero_channel_fraction = float(np.mean(is_zero_channel))

        # Channel std in normalized context
        ctx_std_per_channel = np.std(ctx_norm, axis=1)  # [N]
        min_channel_std = float(np.min(ctx_std_per_channel))

        # Model forecasts
        pred_uv, _ = wrapper.forecast(
            ctx_norm,
            horizon=horizon,
            univariate=True,
            return_quantiles=False,
        )
        pred_mv, _ = wrapper.forecast(
            ctx_norm,
            horizon=horizon,
            univariate=False,
            return_quantiles=False,
        )

        # Mitigation test: clamp/zero-out tail normalized values for zero channels
        ctx_mitigated = ctx_norm.copy()
        if zero_channel_fraction > 0.0:
            # Set normalized tail of inactive channels to 0 (mean level) to prevent noise explosion
            ctx_mitigated[is_zero_channel, -patch_len:] = 0.0
            pred_mv_mitigated, _ = wrapper.forecast(
                ctx_mitigated,
                horizon=horizon,
                univariate=False,
                return_quantiles=False,
            )
        else:
            pred_mv_mitigated = pred_mv

        for t_idx in target_channels:
            y_true = target[t_idx]
            mse_uv = float(calc_mse(pred_uv[t_idx], y_true))
            mse_mv = float(calc_mse(pred_mv[t_idx], y_true))
            mse_mitigated = float(calc_mse(pred_mv_mitigated[t_idx], y_true))

            delta_mse = mse_mv - mse_uv  # > 0 means MV is WORSE than UV
            rel_benefit = (mse_uv - mse_mv) / max(mse_uv, 1e-6) * 100.0  # < 0 means MV degraded
            is_degraded = (rel_benefit < 0.0)

            rel_benefit_mitigated = (mse_uv - mse_mitigated) / max(mse_uv, 1e-6) * 100.0

            records.append({
                "window_idx": w_idx,
                "global_t": global_t,
                "hour_of_day": hour_of_day,
                "diurnal_regime": regime,
                "target_idx": t_idx,
                "zero_channel_fraction": zero_channel_fraction,
                "min_channel_std": min_channel_std,
                "MSE_UV": mse_uv,
                "MSE_MV": mse_mv,
                "MSE_Mitigated": mse_mitigated,
                "Delta_MSE": delta_mse,
                "Relative_Benefit_pct": rel_benefit,
                "Is_Degraded": is_degraded,
                "Mitigated_Benefit_pct": rel_benefit_mitigated,
            })

        if (w_count + 1) % 25 == 0 or (w_count + 1) == len(window_indices):
            elapsed = time.time() - t0
            print(f"  [{w_count+1}/{len(window_indices)}] Evaluated Solar windows | Elapsed: {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "solar_diurnal_diagnostic.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw diagnostic records to {out_csv}")

    # Summary analysis by diurnal regime
    print("\n" + "=" * 80)
    print("SOLAR REGIME-STRATIFIED PERFORMANCE BREAKDOWN")
    print("=" * 80)
    summary = []
    
    night_degraded_counts = [0, 0]  # [degraded, not_degraded]
    daytime_degraded_counts = [0, 0]

    for regime, group in df.groupby("diurnal_regime"):
        uv = group["MSE_UV"].to_numpy()
        mv = group["MSE_MV"].to_numpy()
        mit = group["MSE_Mitigated"].to_numpy()
        delta = group["Delta_MSE"].to_numpy()
        b_pct = group["Relative_Benefit_pct"].to_numpy()
        is_deg = group["Is_Degraded"].to_numpy()

        n_samples = len(group)
        n_deg = int(np.sum(is_deg))
        deg_rate = (n_deg / n_samples) * 100.0

        mean_uv = float(np.mean(uv))
        mean_mv = float(np.mean(mv))
        agg_b = (np.sum(uv) - np.sum(mv)) / np.sum(uv) * 100.0
        median_b = float(np.median(b_pct))
        agg_b_mit = (np.sum(uv) - np.sum(mit)) / np.sum(uv) * 100.0
        
        w_stat, p_val = paired_wilcoxon_test(uv, mv)

        if "Night" in regime:
            night_degraded_counts = [n_deg, n_samples - n_deg]
        elif "Daytime" in regime:
            daytime_degraded_counts = [n_deg, n_samples - n_deg]

        summary.append({
            "Regime": regime,
            "N_Windows": group["window_idx"].nunique(),
            "N_Samples": n_samples,
            "Mean_Zero_Channel_Frac": float(np.mean(group["zero_channel_fraction"])),
            "Mean_MSE_UV": mean_uv,
            "Mean_MSE_MV": mean_mv,
            "Aggregate_B_%": agg_b,
            "Median_B_%": median_b,
            "Degradation_Rate_%": deg_rate,
            "Mitigated_B_%": agg_b_mit,
            "Wilcoxon_p": p_val,
        })

    df_summary = pd.DataFrame(summary)
    breakdown_csv = RESULTS_DIR / "solar_diurnal_breakdown.csv"
    df_summary.to_csv(breakdown_csv, index=False)
    print(df_summary.to_string(index=False))
    print(f"\nSaved diurnal breakdown to {breakdown_csv}")

    # Fisher's Exact Test: Night vs Daytime degradation rate
    if sum(night_degraded_counts) > 0 and sum(daytime_degraded_counts) > 0:
        table_fisher = [night_degraded_counts, daytime_degraded_counts]
        odds_ratio, p_fisher = stats.fisher_exact(table_fisher)
        print("\n" + "-" * 80)
        print(f"Fisher's Exact Test (Night vs Daytime Degradation Rate):")
        print(f"  Night: {night_degraded_counts[0]}/{sum(night_degraded_counts)} degraded ({night_degraded_counts[0]/sum(night_degraded_counts)*100:.1f}%)")
        print(f"  Daytime: {daytime_degraded_counts[0]}/{sum(daytime_degraded_counts)} degraded ({daytime_degraded_counts[0]/sum(daytime_degraded_counts)*100:.1f}%)")
        print(f"  Odds Ratio = {odds_ratio:.4f}, p-value = {p_fisher:.4e}")
        if p_fisher < 0.05:
            print("  ==> STATISTICALLY SIGNIFICANT: Night degradation rate is significantly higher than Daytime!")
        print("-" * 80)

    # Spearman rank correlation: Zero channel fraction vs Delta_MSE
    spearman_rho, spearman_p = stats.spearmanr(df["zero_channel_fraction"], df["Delta_MSE"])
    print(f"\nSpearman Rank Correlation (Zero-Channel Fraction vs Delta_MSE):")
    print(f"  rho = {spearman_rho:+.4f}, p = {spearman_p:.4e}")
    if spearman_p < 0.05 and spearman_rho > 0:
        print("  ==> STATISTICALLY SIGNIFICANT: Higher fraction of zero channels strongly correlates with worse MV degradation!")

    return df, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=96)
    args = parser.parse_args()

    run_solar_diagnostic(num_samples=args.samples, horizon=args.horizon)
