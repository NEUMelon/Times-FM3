from probe.data import load_dataset
import numpy as np

print("Testing LTSF dataset loading...")
ds = load_dataset("ETTh1", context_length=512, horizon=720)
print(f"Dataset {ds.name}:")
print(f" - Variates: {ds.num_variates}")
print(f" - Total test length: {ds.total_test_len}")
print(f" - Stride 1 windows: {ds.num_windows_stride1}")

ctx, target = ds.get_window(0)
print(f" - Window 0 ctx shape: {ctx.shape}, target shape: {target.shape}")
print(f" - Ctx mean: {np.mean(ctx):.3f}, std: {np.std(ctx):.3f}")

sampled = ds.get_sampled_window_indices(num_samples=10)
print(f" - Sampled 10 window indices: {sampled}")
print("Data loader test passed successfully!")
