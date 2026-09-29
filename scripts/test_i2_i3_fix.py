from probe.data import load_dataset
from probe.interventions import apply_i2_temporal_swap, apply_i3_calendar_aligned_swap, get_freq_steps

ds = load_dataset('ETTh1', context_length=512, horizon=96)
ctx, tgt = ds.get_window(0)
group_indices = list(range(7))
steps_per_day, steps_per_week = get_freq_steps(ds.info['freq'])
abs_w_idx = ds.border1s[2] + 0

new_ctx, meta = apply_i3_calendar_aligned_swap(
    ctx, 0, ds.norm_values[:, group_indices], abs_w_idx, steps_per_week=steps_per_week, weeks_offset=4
)
print("I3 SUCCESS! swap_start:", meta["swap_start"], "swap_end:", meta["swap_end"], "forecast_start:", meta["forecast_start"])

new_ctx2, meta2 = apply_i2_temporal_swap(
    ctx, 0, ds.norm_values[:, group_indices], abs_w_idx, steps_per_day=steps_per_day
)
print("I2 SUCCESS! swap_start:", meta2["swap_start"], "swap_end:", meta2["swap_end"], "forecast_start:", meta2["forecast_start"])
