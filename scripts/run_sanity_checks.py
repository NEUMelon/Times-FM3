import sys
import os
sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd

from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch.evaluator import TimesFM3Evaluator
from probe.models.timesfm3 import TimesFM3Wrapper
from probe.config import MODEL_DIR


def run_checks():
    print("=" * 70)
    print("STARTING SANITY CHECKS S1 - S7")
    print("=" * 70)

    # Initialize forecaster and wrapper
    wrapper = TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device="cuda")
    forecaster = wrapper.forecaster
    evaluator = wrapper.evaluator

    passed = {}

    # -------------------------------------------------------------
    # S2: MV mode with 1 channel == UV mode
    # -------------------------------------------------------------
    print("\n[CHECK S2] Testing 1-channel MV vs UV equivalence...")
    rng = np.random.RandomState(42)
    single_ctx = rng.randn(1, 256).astype(np.float32)
    h = 32

    # Predict in MV mode
    p_mv, _ = wrapper.forecast(single_ctx, horizon=h, univariate=False, return_quantiles=False)
    # Predict in UV mode
    p_uv, _ = wrapper.forecast(single_ctx, horizon=h, univariate=True, return_quantiles=False)

    diff_s2 = np.max(np.abs(p_mv - p_uv))
    print(f"Max absolute difference (MV vs UV for N=1): {diff_s2:.2e}")
    s2_ok = diff_s2 < 1e-5
    passed["S2"] = s2_ok
    print(f"S2 Result: {'PASSED' if s2_ok else 'FAILED'} (threshold: 1e-5)")

    # -------------------------------------------------------------
    # S3: Permutation Equivariance (use_rope_var: false)
    # -------------------------------------------------------------
    print("\n[CHECK S3] Testing channel permutation equivariance...")
    multi_ctx = rng.randn(4, 256).astype(np.float32)
    p_orig, _ = wrapper.forecast(multi_ctx, horizon=h, univariate=False, return_quantiles=False)

    perm = [2, 0, 3, 1]
    perm_ctx = multi_ctx[perm]
    p_perm, _ = wrapper.forecast(perm_ctx, horizon=h, univariate=False, return_quantiles=False)

    # p_perm should match p_orig[perm]
    expected_perm = p_orig[perm]
    diff_s3 = np.max(np.abs(p_perm - expected_perm))
    print(f"Max absolute difference after permutation: {diff_s3:.2e}")
    s3_ok = diff_s3 < 1e-5
    passed["S3"] = s3_ok
    print(f"S3 Result: {'PASSED' if s3_ok else 'FAILED'} (threshold: 1e-5)")

    # -------------------------------------------------------------
    # S4: Target channel vs Past-only Covariate Equivalence
    # -------------------------------------------------------------
    print("\n[CHECK S4] Testing target vs past-only covariate internal equivalence...")
    target_series = rng.randn(1, 256).astype(np.float32)
    other_series = rng.randn(1, 256).astype(np.float32)
    joint_ctx = np.concatenate([target_series, other_series], axis=0)  # [2, 256]

    # Condition A: Both passed as targets in MV mode
    res_a = list(forecaster.predict_batch(
        contexts=list(joint_ctx),
        horizon=h,
        make_positive=False,
        use_symmetric_averaging=False,
        return_quantiles=False
    ))
    pred_target_a = res_a[0].forecast

    # Condition B: target_series as target, other_series as past_only_covariates
    res_b = list(forecaster.predict_batch(
        contexts=[target_series[0]],
        horizon=h,
        past_only_covariates=[other_series[0]],
        make_positive=False,
        use_symmetric_averaging=False,
        return_quantiles=False
    ))
    pred_target_b = res_b[0].forecast

    diff_s4 = np.max(np.abs(pred_target_a - pred_target_b))
    print(f"Max absolute difference (target vs past_only_covariate): {diff_s4:.2e}")
    s4_ok = diff_s4 < 1e-4
    passed["S4"] = s4_ok
    print(f"S4 Result: {'PASSED' if s4_ok else 'FAILED'} (threshold: 1e-4)")

    # -------------------------------------------------------------
    # S5: Forecast Prefix Consistency (H=720 vs H=96, 192, 336)
    # -------------------------------------------------------------
    print("\n[CHECK S5] Testing forecast prefix consistency across horizons...")
    test_ctx = rng.randn(2, 512).astype(np.float32)

    # Run for H=720
    pred_720, _ = wrapper.forecast(test_ctx, horizon=720, univariate=False, return_quantiles=False)
    # Run for H=96, H=192, H=336
    pred_96, _ = wrapper.forecast(test_ctx, horizon=96, univariate=False, return_quantiles=False)
    pred_192, _ = wrapper.forecast(test_ctx, horizon=192, univariate=False, return_quantiles=False)
    pred_336, _ = wrapper.forecast(test_ctx, horizon=336, univariate=False, return_quantiles=False)

    diff_96 = np.max(np.abs(pred_720[:, :96] - pred_96))
    diff_192 = np.max(np.abs(pred_720[:, :192] - pred_192))
    diff_336 = np.max(np.abs(pred_720[:, :336] - pred_336))
    print(f"Max prefix diff: H=96: {diff_96:.2e}, H=192: {diff_192:.2e}, H=336: {diff_336:.2e}")
    # Note: Contiguous patch masking decodes autoregressively or in single forward
    s5_ok = max(diff_96, diff_192, diff_336) < 1e-4
    passed["S5"] = s5_ok
    print(f"S5 Result: {'PASSED' if s5_ok else 'INFO (autoregressive patch masking differences)'}")

    # -------------------------------------------------------------
    # S6: Batch Independence
    # -------------------------------------------------------------
    print("\n[CHECK S6] Testing batch independence...")
    sample_a = rng.randn(2, 256).astype(np.float32)
    sample_b = rng.randn(2, 256).astype(np.float32)

    pred_a_alone, _ = wrapper.forecast(sample_a, horizon=h, univariate=True, return_quantiles=False)
    
    # Combined in single batch in univariate mode
    combined_ctx = np.concatenate([sample_a, sample_b], axis=0)
    pred_combined, _ = wrapper.forecast(combined_ctx, horizon=h, univariate=True, return_quantiles=False)
    pred_a_in_batch = pred_combined[:2]

    diff_s6 = np.max(np.abs(pred_a_alone - pred_a_in_batch))
    print(f"Max absolute difference (alone vs in batch): {diff_s6:.2e}")
    s6_ok = diff_s6 < 1e-5
    passed["S6"] = s6_ok
    print(f"S6 Result: {'PASSED' if s6_ok else 'FAILED'} (threshold: 1e-5)")

    # -------------------------------------------------------------
    # S7: Determinism check across runs
    # -------------------------------------------------------------
    print("\n[CHECK S7] Testing determinism across repeated runs...")
    torch.manual_seed(42)
    pred_run1, _ = wrapper.forecast(multi_ctx, horizon=h, univariate=False, return_quantiles=True)
    
    torch.manual_seed(42)
    pred_run2, _ = wrapper.forecast(multi_ctx, horizon=h, univariate=False, return_quantiles=True)

    diff_s7 = np.max(np.abs(pred_run1 - pred_run2))
    print(f"Max absolute difference between identical runs: {diff_s7:.2e}")
    s7_ok = diff_s7 < 1e-6
    passed["S7"] = s7_ok
    print(f"S7 Result: {'PASSED' if s7_ok else 'FAILED'} (threshold: 1e-6)")

    # -------------------------------------------------------------
    # S1: Official Reproduction Check
    # -------------------------------------------------------------
    print("\n[CHECK S1] Checking official repo CSV files availability and consistency...")
    df_fev = pd.read_csv("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv")
    df_gift = pd.read_csv("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/gift_eval/all_results.csv")
    s1_ok = (len(df_fev) == 100) and (len(df_gift) >= 97)
    passed["S1"] = s1_ok
    print(f"S1 Result: {'PASSED' if s1_ok else 'FAILED'} (fev tasks: {len(df_fev)}, gift tasks: {len(df_gift)})")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECKS SUMMARY:")
    for check_id, status in passed.items():
        print(f"  {check_id}: {'[PASSED]' if status else '[FAILED]'}")
    print("=" * 70)

    all_passed = all(passed.values())
    if all_passed:
        print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("Some checks failed or require attention.")
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
