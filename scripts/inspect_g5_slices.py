import numpy as np
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import build_g5_oracle
from probe.models.timesfm3 import TimesFM3Wrapper

wrapper = TimesFM3Wrapper(device="cuda")

X, meta = generate_g5_lead_lag(N=8, T=512 + 96 + 50, lag=16, lead_coeff=0.8, seed=42)
oracle_mv, uv_oracles = build_g5_oracle(meta)

ctx = X[:, :512]
target = X[:, 512:512+96]

f_mv = oracle_mv.filter_and_forecast_mv(ctx, horizon=96)
pred_model, _ = wrapper.forecast(ctx, horizon=96, univariate=False, use_symmetric_averaging=True, return_quantiles=False)

print("Target mean:", np.mean(target[1]), "std:", np.std(target[1]))
print("Kalman mean:", np.mean(f_mv[1]), "std:", np.std(f_mv[1]))
print("Model mean:", np.mean(pred_model[1]), "std:", np.std(pred_model[1]))

# What is the MSE across all 5 windows for each horizon slice:
# h=1..16, h=17..32, h=33..64, h=65..96
for w in range(5):
    X_w, meta_w = generate_g5_lead_lag(N=8, T=512 + 96 + 50, lag=16, lead_coeff=0.8, seed=42 + w*100)
    oracle_mv_w, _ = build_g5_oracle(meta_w)
    ctx_w = X_w[:, :512]
    target_w = X_w[:, 512:512+96]
    f_mv_w = oracle_mv_w.filter_and_forecast_mv(ctx_w, horizon=96)
    pred_m_w, _ = wrapper.forecast(ctx_w, horizon=96, univariate=False, use_symmetric_averaging=True, return_quantiles=False)
    
    mse_k_16 = np.mean((f_mv_w[1, :16] - target_w[1, :16])**2)
    mse_m_16 = np.mean((pred_m_w[1, :16] - target_w[1, :16])**2)
    mse_k_96 = np.mean((f_mv_w[1] - target_w[1])**2)
    mse_m_96 = np.mean((pred_m_w[1] - target_w[1])**2)
    print(f"Window {w+1}: 1-16 (Kalman={mse_k_16:.4f}, Model={mse_m_16:.4f}) | Full 96 (Kalman={mse_k_96:.4f}, Model={mse_m_96:.4f})")
