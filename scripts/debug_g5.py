import numpy as np
from probe.config import PRIMARY_SEED
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import compute_exact_kalman_oracles
from probe.models.timesfm3 import TimesFM3Wrapper
from probe.metrics import calc_mse

wrapper = TimesFM3Wrapper(device="cuda")
N = 8
T_total = 512 + 96 + 50
lag = 16
lead_coeff = 0.8

print("=== Inspecting G5 LeadLag L16 ===")
for w in range(5):
    w_seed = PRIMARY_SEED + w * 100
    X, meta = generate_g5_lead_lag(N=N, T=T_total, lag=lag, lead_coeff=lead_coeff, seed=w_seed)
    ctx = X[:, :512]
    target = X[:, 512 : 512 + 96]
    t_idx = 1 # follower channel

    pred_mv, _ = wrapper.forecast(ctx, horizon=96, univariate=False, use_symmetric_averaging=True, return_quantiles=False)
    e_mv_m = float(calc_mse(pred_mv[t_idx], target[t_idx]))

    e_uv_s, e_mv_s = compute_exact_kalman_oracles(ctx, target, target_idx=t_idx, meta=meta)
    print(f"Window {w+1}: UV*={e_uv_s:.5f}, MV*={e_mv_s:.5f}, Model={e_mv_m:.5f}, Diff(Model - MV*)={e_mv_m - e_mv_s:+.5f}")

