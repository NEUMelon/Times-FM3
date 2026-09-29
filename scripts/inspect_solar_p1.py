import numpy as np
from probe.config import PRIMARY_SEED
from probe.data import load_dataset
from probe.grouping import sample_stratified_targets
from probe.interventions import apply_p1_leading_indicator
from probe.metrics import calc_mse
from probe.models.timesfm3 import TimesFM3Wrapper

wrapper = TimesFM3Wrapper(device="cuda")

ds = load_dataset("solar", context_length=512, horizon=96)
window_indices = ds.get_sampled_window_indices(num_samples=5, seed=PRIMARY_SEED)
N = ds.num_variates

train_data = ds.norm_values[: ds.border2s[0]]
corr_matrix = np.corrcoef(train_data, rowvar=False)
np.nan_to_num(corr_matrix, copy=False, nan=0.0)
np.fill_diagonal(corr_matrix, 0.0)
sampled_targets = sample_stratified_targets(corr_matrix, num_targets=4, chunk_size=32, seed=PRIMARY_SEED)

print("Sampled targets for Solar:", sampled_targets)

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

        # P1 with lead_k = 32
        ctx_p1_32, meta_p1_32 = apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=32, snr=10.0, seed=PRIMARY_SEED)
        pred_p1_32, _ = wrapper.forecast(
            ctx_p1_32,
            horizon=96,
            univariate=False,
            use_symmetric_averaging=True,
            return_quantiles=False,
        )
        mse_p1_32 = float(calc_mse(pred_p1_32[meta_p1_32["target_pos"]], y_true))

        # P1 with lead_k = 16
        ctx_p1_16, meta_p1_16 = apply_p1_leading_indicator(ctx_group, y_true, t_pos, lead_k=16, snr=10.0, seed=PRIMARY_SEED)
        pred_p1_16, _ = wrapper.forecast(
            ctx_p1_16,
            horizon=96,
            univariate=False,
            use_symmetric_averaging=True,
            return_quantiles=False,
        )
        mse_p1_16 = float(calc_mse(pred_p1_16[meta_p1_16["target_pos"]], y_true))

        print(f"Win {w_idx:5d}, Tgt {t_idx:3d} (t_pos {t_pos:2d}): MSE_MV={mse_mv:.5f}, P1_32={mse_p1_32:.5f} (B={(mse_mv-mse_p1_32)/mse_mv*100:+.1f}%), P1_16={mse_p1_16:.5f} (B={(mse_mv-mse_p1_16)/mse_mv*100:+.1f}%)")

