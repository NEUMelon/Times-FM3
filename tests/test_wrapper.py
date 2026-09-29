"""Diagnostic Test Suite B: TimesFM-3 Wrapper & Invariance Properties (tests/test_wrapper.py).
Requires GPU environment with TimesFM-3 checkpoint.
Strictly implements:
  B1: Univariate == 1-channel Multivariate: max absolute diff < 1e-5.
  B2: Permutation equivariance: Shuffling channel input order yields permuted predictions (< 1e-5).
  B3: Batch size invariance: Single window vs in batch of 32 windows (< 1e-4).
  B4: Horizon prefix consistency: H=720 first 96 steps vs H=96 (< 1e-4).
  B5: Chunking bypass consistency: K <= 32 evaluator == bypass (< 1e-5); K=64 evaluator != bypass.
  B6: Target vs past-covariate equivalence (< 1e-5).
  B7: Determinism: Identical input run twice yields bit-by-bit identical forecasts.
"""

import numpy as np
import pytest
import torch

from probe.config import MODEL_DIR
from probe.models.timesfm3 import TimesFM3Wrapper


@pytest.fixture(scope="module")
def wrapper():
    if not torch.cuda.is_available() or not MODEL_DIR.exists():
        pytest.skip("Test Suite B requires CUDA GPU and TimesFM-3 checkpoint.")
    return TimesFM3Wrapper(checkpoint_path=MODEL_DIR, device="cuda")


def _get_dummy_windows(num_windows=5, num_variates=4, context_length=512, seed=42):
    rng = np.random.RandomState(seed)
    windows = []
    for _ in range(num_windows):
        w = rng.normal(0, 1.0, size=(num_variates, context_length)).astype(np.float32)
        windows.append(w)
    return windows


def test_b1_univariate_vs_single_multivariate(wrapper):
    """B1: Univariate mode == 1-channel multivariate mode (< 1e-5)."""
    print("\n--- Running Test B1: Univariate vs 1-channel Multivariate ---")
    windows = _get_dummy_windows(num_windows=5, num_variates=1)
    diffs = []
    for idx, w in enumerate(windows):
        pred_uv, _ = wrapper.forecast(w, horizon=96, univariate=True, use_symmetric_averaging=False, return_quantiles=False)
        pred_mv, _ = wrapper.forecast(w, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)
        diff = float(np.max(np.abs(pred_uv - pred_mv)))
        diffs.append(diff)
        print(f"Window {idx}: max abs diff = {diff:.10e}")
        assert diff < 1e-5, f"Window {idx} diff {diff} >= 1e-5"
    print("B1 PASSED: Univariate strictly matches 1-channel multivariate mode!")


def test_b2_permutation_equivariance(wrapper):
    """B2: Permuting channel order permutes output predictions (< 1e-5)."""
    print("\n--- Running Test B2: Permutation Equivariance ---")
    windows = _get_dummy_windows(num_windows=5, num_variates=8)
    rng = np.random.RandomState(123)

    for idx, w in enumerate(windows):
        perm = rng.permutation(8)
        w_perm = w[perm]

        pred_orig, _ = wrapper.forecast(w, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)
        pred_perm, _ = wrapper.forecast(w_perm, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)

        # Expected: pred_perm == pred_orig[perm]
        diff = float(np.max(np.abs(pred_perm - pred_orig[perm])))
        print(f"Window {idx}: Permutation equivariance max diff = {diff:.10e}")
        assert diff < 1e-5, f"Window {idx} permutation diff {diff} >= 1e-5"
    print("B2 PASSED: Model is strictly permutation-equivariant across channels!")


def test_b3_batch_invariance(wrapper):
    """B3: Single window vs in batch of 32 windows (< 1e-4)."""
    print("\n--- Running Test B3: Batch Size Invariance ---")
    batch_32 = _get_dummy_windows(num_windows=32, num_variates=4)
    target_idx = 7
    target_window = batch_32[target_idx]

    pred_single, _ = wrapper.forecast(target_window, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)
    preds_batch, _ = wrapper.forecast(batch_32, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)

    pred_in_batch = preds_batch[target_idx]
    diff = float(np.max(np.abs(pred_single - pred_in_batch)))
    print(f"Single vs Batch of 32 max abs diff: {diff:.10e}")
    assert diff < 1e-4, f"Batch invariance violated: diff {diff} >= 1e-4"
    print("B3 PASSED: Batch size does not alter individual predictions!")


def test_b4_horizon_prefix_consistency(wrapper):
    """B4: H=720 prediction prefix (96 steps) matches H=96 prediction (< 1e-4)."""
    print("\n--- Running Test B4: Horizon Prefix Consistency ---")
    windows = _get_dummy_windows(num_windows=3, num_variates=4)

    for idx, w in enumerate(windows):
        pred_96, _ = wrapper.forecast(w, horizon=96, univariate=False, use_symmetric_averaging=False, return_quantiles=False)
        pred_720, _ = wrapper.forecast(w, horizon=720, univariate=False, use_symmetric_averaging=False, return_quantiles=False)

        prefix_720 = pred_720[:, :96]
        diff = float(np.max(np.abs(pred_96 - prefix_720)))
        print(f"Window {idx}: H=720 prefix vs H=96 max abs diff = {diff:.10e}")
        # Note: In autoregressive/patch architectures, past patch tokens are fixed
        assert diff < 1e-4, f"Window {idx} horizon prefix diff {diff} >= 1e-4"
    print("B4 PASSED: Long horizon forecasts can be safely sliced for shorter horizons!")


def test_b5_chunking_path_consistency(wrapper):
    """B5: For K <= 32 evaluator == bypass (< 1e-5); for K=64 evaluator != bypass."""
    print("\n--- Running Test B5: Chunking Bypass Consistency & Activation ---")
    # 1. K = 16 <= 32: both paths must be identical
    w_16 = _get_dummy_windows(num_windows=1, num_variates=16)[0]
    pred_eval_16, _ = wrapper.forecast(w_16, horizon=96, bypass_evaluator_chunking=False, return_quantiles=False)
    pred_byp_16, _ = wrapper.forecast(w_16, horizon=96, bypass_evaluator_chunking=True, return_quantiles=False)
    diff_16 = float(np.max(np.abs(pred_eval_16 - pred_byp_16)))
    print(f"K=16: Evaluator vs Bypass max abs diff = {diff_16:.10e} (required < 1e-5)")
    assert diff_16 < 1e-5, f"K=16 paths differ: {diff_16}"

    # 2. K = 64 > 32: evaluator chunks into 2x32, bypass attends across all 64
    w_64 = _get_dummy_windows(num_windows=1, num_variates=64)[0]
    pred_eval_64, _ = wrapper.forecast(w_64, horizon=96, bypass_evaluator_chunking=False, return_quantiles=False)
    pred_byp_64, _ = wrapper.forecast(w_64, horizon=96, bypass_evaluator_chunking=True, return_quantiles=False)
    diff_64 = float(np.max(np.abs(pred_eval_64 - pred_byp_64)))
    print(f"K=64: Evaluator vs Bypass max abs diff = {diff_64:.10e} (must be > 1e-4 to prove bypass works)")
    assert diff_64 > 1e-4, f"K=64 paths produced identical output! Bypass failed to activate."
    print("B5 PASSED: Bypass path correctly preserves unchunked full attention for K > 32!")


def test_b6_target_vs_past_covariates_equivalence(wrapper):
    """B6: Other channels as targets vs past_only_covariates yields identical prediction (< 1e-5)."""
    print("\n--- Running Test B6: Target vs Past-only Covariates Equivalence ---")
    windows = _get_dummy_windows(num_windows=3, num_variates=4)
    for idx, w in enumerate(windows):
        # Target is channel 0, channels 1..3 are other variates
        # Method 1: all 4 as targets in multivariate window
        outs_m1 = list(wrapper.evaluator.predict_batch(
            contexts=[w],
            horizon=96,
            univariate=False,
            past_only_covariates=None,
            use_symmetric_averaging=False,
            make_positive=False,
        ))
        pred_m1 = outs_m1[0].forecast[0]

        # Method 2: channel 0 as target, channels 1..3 as past_only_covariates
        outs_m2 = list(wrapper.evaluator.predict_batch(
            contexts=[w[0:1]],
            horizon=96,
            univariate=False,
            past_only_covariates=[w[1:]],
            use_symmetric_averaging=False,
            make_positive=False,
        ))
        pred_m2 = outs_m2[0].forecast[0]

        diff = float(np.max(np.abs(pred_m1 - pred_m2)))
        print(f"Window {idx}: Target vs Past-only Covariates max abs diff = {diff:.10e}")
        assert diff < 1e-5, f"Window {idx} B6 violation: diff {diff} >= 1e-5"
    print("B6 PASSED: Covariates and target channels are mathematically equivalent in attention!")


def test_b7_determinism(wrapper):
    """B7: Identical input run twice yields identical bit-by-bit predictions."""
    print("\n--- Running Test B7: Determinism Verification ---")
    w = _get_dummy_windows(num_windows=1, num_variates=8)[0]
    pred_1, _ = wrapper.forecast(w, horizon=96, univariate=False, return_quantiles=False)
    pred_2, _ = wrapper.forecast(w, horizon=96, univariate=False, return_quantiles=False)
    diff = float(np.max(np.abs(pred_1 - pred_2)))
    print(f"Run 1 vs Run 2 max abs diff: {diff:.10e}")
    assert diff == 0.0, "Model is non-deterministic!"
    print("B7 PASSED: Determinism verified bit-by-bit!")
