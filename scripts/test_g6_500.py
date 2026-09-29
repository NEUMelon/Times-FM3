import numpy as np
from probe.synth.generators import generate_g6_null
from probe.synth.oracle import compute_exact_kalman_oracles
from probe.models.timesfm3 import TimesFM3Wrapper
from probe.metrics import calc_mse

wrapper = TimesFM3Wrapper(device="cuda")
N = 8
T_total = 512 + 96 + 50
horizon = 96
num_windows = 500
seed = 42

diffs = []
mses_model = []
mses_oracle = []

print(f"=== Testing G6 across {num_windows} windows ===")
for w in range(num_windows):
    w_seed = seed + w * 17
    X, meta = generate_g6_null(N=N, T=T_total, sigma_eps=0.1, seed=w_seed)
    ctx = X[:, :512]
    target = X[:, 512:512+horizon]
    t_idx = 0

    pred_uv, _ = wrapper.forecast(ctx[t_idx:t_idx+1], horizon=horizon, univariate=True, use_symmetric_averaging=True, return_quantiles=False)
    e_model = float(calc_mse(pred_uv[0], target[t_idx]))

    e_uv_s, e_mv_s = compute_exact_kalman_oracles(ctx, target, target_idx=t_idx, meta=meta)

    diff = e_model - e_uv_s
    diffs.append(diff)
    mses_model.append(e_model)
    mses_oracle.append(e_uv_s)

    if (w + 1) % 50 == 0 or (w + 1) == num_windows:
        mean_d = np.mean(diffs)
        se_d = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        t_stat = mean_d / se_d if se_d > 1e-12 else 0.0
        print(f"[{w+1}/{num_windows}] Mean Model={np.mean(mses_model):.5f}, Mean Oracle={np.mean(mses_oracle):.5f}, Diff={mean_d:+.5f}, SE={se_d:.5f}, t-stat={t_stat:+.2f}")

mean_d = np.mean(diffs)
se_d = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
t_stat = mean_d / se_d if se_d > 1e-12 else 0.0

print("\n=== Final G6 Results (500 windows) ===")
print(f"Model MSE:  {np.mean(mses_model):.6f}")
print(f"Oracle MSE: {np.mean(mses_oracle):.6f}")
print(f"Diff (Model - Oracle): {mean_d:+.6f}")
print(f"SE: {se_d:.6f}")
print(f"t-statistic: {t_stat:+.4f}")
print(f"Red flag triggered (t < -2): {t_stat < -2.0}")
