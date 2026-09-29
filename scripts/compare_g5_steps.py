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

print("Target 1 variance:", np.var(target[1]))
print("Kalman MV* MSE (all 96 steps):", np.mean((f_mv[1] - target[1])**2))
print("Kalman MV* MSE (first 16 steps):", np.mean((f_mv[1, :16] - target[1, :16])**2))
print("Kalman MV* MSE (steps 16..96):", np.mean((f_mv[1, 16:] - target[1, 16:])**2))

print("\nModel MSE (all 96 steps):", np.mean((pred_model[1] - target[1])**2))
print("Model MSE (first 16 steps):", np.mean((pred_model[1, :16] - target[1, :16])**2))
print("Model MSE (steps 16..96):", np.mean((pred_model[1, 16:] - target[1, 16:])**2))

# Let's inspect the correlation between f_mv[1] and target[1], vs pred_model[1] and target[1]
print("\nKalman correlation with target:", np.corrcoef(f_mv[1], target[1])[0, 1])
print("Model correlation with target:", np.corrcoef(pred_model[1], target[1])[0, 1])
