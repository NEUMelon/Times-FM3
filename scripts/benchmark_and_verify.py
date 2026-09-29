"""
Comprehensive Benchmark, Equivalence, and S5 Verification Script.
Executes:
1. Task 2: Comparative benchmark on 64 Electricity windows for (i), (ii), (iii) + torch.profiler
2. Task 4: S5 Horizon prefix-slicing equivalence test
3. Task 5: 20-window Numerical equivalence check (Max Abs Diff < 1e-4, Relative MSE < 1e-5)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Ensure project root is in sys.path
sys.path.insert(0, "/root/autodl-tmp/DASFAA")

from timesfm3.torch import timesfm3_forecaster as tf
from timesfm3.torch.evaluator import TimesFM3Evaluator
from probe.data import load_dataset
from probe.metrics import calc_mse, calc_mae

print("PyTorch CUDA available:", torch.cuda.is_available())
device = "cuda:0"

# Parameters
context_length = 512
horizon = 720
num_test_windows = 64

# 1. Load Electricity dataset
print("Loading electricity dataset...")
dset = load_dataset("electricity", context_length=context_length, horizon=horizon)
indices = dset.get_sampled_window_indices(num_samples=num_test_windows)
windows = []
targets = []
for idx in indices:
    ctx, tgt = dset.get_window(idx)
    windows.append(ctx)
    targets.append(tgt)
print(f"Prepared {len(windows)} windows, shape of each: {windows[0].shape}, horizon: {horizon}")

checkpoint_path = "/root/autodl-tmp/models/timesfm-3.0-pytorch"
print("Instantiating Evaluator...")
evaluator = TimesFM3Evaluator(checkpoint_path=checkpoint_path, device=device)
original_try_gc = tf.try_gc

# Marker file for timestamp synchronization with external nvidia-smi
marker_file = "/root/autodl-tmp/results/benchmark_markers.txt"
with open(marker_file, "w") as mf:
    mf.write("EXPERIMENT_TIMESTAMPS\n")

def record_marker(name, start_time, end_time):
    with open(marker_file, "a") as mf:
        mf.write(f"{name},{start_time:.3f},{end_time:.3f}\n")


print("\n" + "=" * 60)
print("EXPERIMENT 1: (i) Baseline - Window-by-window (1 window per call, per_core_batch_size=4, original try_gc)")
print("=" * 60)
tf.try_gc = original_try_gc
evaluator.config.per_core_batch_size = 4
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
preds_1 = []
for w in windows:
    out = list(evaluator.predict_batch(
        contexts=[w],
        horizon=horizon,
        univariate=False,
        return_quantiles=False,
        use_symmetric_averaging=True,
        make_positive=False,
    ))[0]
    preds_1.append(out.forecast)
t1 = time.time()
record_marker("exp1_baseline", t0, t1)
t_elapsed_1 = t1 - t0
peak_mem_1 = torch.cuda.max_memory_allocated() / (1024 ** 2)
time_per_window_1 = t_elapsed_1 / len(windows)
print(f"(i) Baseline Results:")
print(f"  Total time: {t_elapsed_1:.2f}s | Per window: {time_per_window_1:.3f}s")
print(f"  Peak VRAM (Allocated): {peak_mem_1:.1f} MiB")


print("\n" + "=" * 60)
print("EXPERIMENT 2: (ii) Multi-window Batching (64 windows in 1 call, per_core_batch_size=32, original try_gc)")
print("=" * 60)
tf.try_gc = original_try_gc
evaluator.config.per_core_batch_size = 32
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
outs_2 = list(evaluator.predict_batch(
    contexts=windows,
    horizon=horizon,
    univariate=False,
    return_quantiles=False,
    use_symmetric_averaging=True,
    make_positive=False,
))
preds_2 = [o.forecast for o in outs_2]
t1 = time.time()
record_marker("exp2_batched", t0, t1)
t_elapsed_2 = t1 - t0
peak_mem_2 = torch.cuda.max_memory_allocated() / (1024 ** 2)
time_per_window_2 = t_elapsed_2 / len(windows)
print(f"(ii) Batched Results:")
print(f"  Total time: {t_elapsed_2:.2f}s | Per window: {time_per_window_2:.3f}s")
print(f"  Peak VRAM (Allocated): {peak_mem_2:.1f} MiB")
print(f"  Speedup vs baseline: {t_elapsed_1 / t_elapsed_2:.2f}x")


print("\n" + "=" * 60)
print("EXPERIMENT 3: (iii) Multi-window Batching + monkeypatch try_gc -> pass")
print("=" * 60)
tf.try_gc = lambda *args, **kwargs: None
evaluator.config.per_core_batch_size = 32
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
outs_3 = list(evaluator.predict_batch(
    contexts=windows,
    horizon=horizon,
    univariate=False,
    return_quantiles=False,
    use_symmetric_averaging=True,
    make_positive=False,
))
preds_3 = [o.forecast for o in outs_3]
t1 = time.time()
record_marker("exp3_batched_nogc", t0, t1)
t_elapsed_3 = t1 - t0
peak_mem_3 = torch.cuda.max_memory_allocated() / (1024 ** 2)
time_per_window_3 = t_elapsed_3 / len(windows)
print(f"(iii) Batched + No-GC Results:")
print(f"  Total time: {t_elapsed_3:.2f}s | Per window: {time_per_window_3:.3f}s")
print(f"  Peak VRAM (Allocated): {peak_mem_3:.1f} MiB")
print(f"  Speedup vs baseline: {t_elapsed_1 / t_elapsed_3:.2f}x")


print("\n" + "=" * 60)
print("TORCH PROFILER: CPU vs CUDA Time for Single Forward Chunk (32 channels)")
print("=" * 60)
test_chunk = [windows[0][:32, :]]
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        _ = list(evaluator.predict_batch(
            contexts=test_chunk,
            horizon=horizon,
            univariate=False,
            return_quantiles=False,
            use_symmetric_averaging=False,
            make_positive=False,
        ))

print("--- Top Ops by CPU Time Total ---")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("--- Top Ops by CUDA Time Total ---")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("\n" + "=" * 60)
print("TASK 5: EQUIVALENCE CHECK (20 windows, MV & UV, Baseline vs Optimized)")
print("=" * 60)
eval_windows = windows[:20]

# Baseline predictions on 20 windows
tf.try_gc = original_try_gc
evaluator.config.per_core_batch_size = 4
base_mv = [list(evaluator.predict_batch([w], horizon=horizon, univariate=False, return_quantiles=False, use_symmetric_averaging=True, make_positive=False))[0].forecast for w in eval_windows]
base_uv = [list(evaluator.predict_batch([w], horizon=horizon, univariate=True, return_quantiles=False, use_symmetric_averaging=True, make_positive=False))[0].forecast for w in eval_windows]

# Optimized predictions on 20 windows
tf.try_gc = lambda *args, **kwargs: None
evaluator.config.per_core_batch_size = 32
opt_mv = [o.forecast for o in evaluator.predict_batch(eval_windows, horizon=horizon, univariate=False, return_quantiles=False, use_symmetric_averaging=True, make_positive=False)]

evaluator.config.per_core_batch_size = 512
opt_uv = [o.forecast for o in evaluator.predict_batch(eval_windows, horizon=horizon, univariate=True, return_quantiles=False, use_symmetric_averaging=True, make_positive=False)]

# Compare MV
diff_mv = np.max([np.max(np.abs(b - o)) for b, o in zip(base_mv, opt_mv)])
mse_base_mv = np.mean([calc_mse(b, t) for b, t in zip(base_mv, targets[:20])])
mse_opt_mv = np.mean([calc_mse(o, t) for o, t in zip(opt_mv, targets[:20])])
rel_mse_diff_mv = abs(mse_base_mv - mse_opt_mv) / (mse_base_mv + 1e-9)

# Compare UV
diff_uv = np.max([np.max(np.abs(b - o)) for b, o in zip(base_uv, opt_uv)])
mse_base_uv = np.mean([calc_mse(b, t) for b, t in zip(base_uv, targets[:20])])
mse_opt_uv = np.mean([calc_mse(o, t) for o, t in zip(opt_uv, targets[:20])])
rel_mse_diff_uv = abs(mse_base_uv - mse_opt_uv) / (mse_base_uv + 1e-9)

print(f"MV Equivalence:")
print(f"  Max Absolute Difference: {diff_mv:.2e} (Threshold: 1e-4) -> {'PASS' if diff_mv < 1e-4 else 'FAIL'}")
print(f"  Relative MSE Difference: {rel_mse_diff_mv:.2e} (Threshold: 1e-5) -> {'PASS' if rel_mse_diff_mv < 1e-5 else 'FAIL'}")

print(f"UV Equivalence:")
print(f"  Max Absolute Difference: {diff_uv:.2e} (Threshold: 1e-4) -> {'PASS' if diff_uv < 1e-4 else 'FAIL'}")
print(f"  Relative MSE Difference: {rel_mse_diff_uv:.2e} (Threshold: 1e-5) -> {'PASS' if rel_mse_diff_uv < 1e-5 else 'FAIL'}")


print("\n" + "=" * 60)
print("S5 CHECK: Horizon Prefix-Slicing Equivalence (H=720 sliced vs H=96/192/336 direct)")
print("=" * 60)
s5_windows = windows[:10]
s5_targets = targets[:10]

# Predict H=720
evaluator.config.per_core_batch_size = 32
pred_720 = [o.forecast for o in evaluator.predict_batch(s5_windows, horizon=720, univariate=False, return_quantiles=False, use_symmetric_averaging=True, make_positive=False)]

# Test 96, 192, 336
for h_test in [96, 192, 336]:
    pred_direct = [o.forecast for o in evaluator.predict_batch(s5_windows, horizon=h_test, univariate=False, return_quantiles=False, use_symmetric_averaging=True, make_positive=False)]
    pred_sliced = [p[:, :h_test] for p in pred_720]
    max_abs_diff = np.max([np.max(np.abs(d - s)) for d, s in zip(pred_direct, pred_sliced)])
    print(f"Horizon H={h_test}: Max Abs Diff (Direct vs Sliced from 720) = {max_abs_diff:.6f}")

print("\nAll benchmark & verification tests complete!")
