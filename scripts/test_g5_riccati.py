import numpy as np
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import build_g5_oracle

X, meta = generate_g5_lead_lag(N=8, T=600, lag=16, lead_coeff=0.8, seed=42)
oracle_mv, uv_oracles = build_g5_oracle(meta)

print("A_mv shape:", oracle_mv.A.shape)
print("P0 in oracle_mv:", oracle_mv.P0 is not None)

ctx = X[:, :512]
target = X[:, 512:512+96]

f_mv = oracle_mv.filter_and_forecast_mv(ctx, horizon=96)
mse_mv = np.mean((f_mv[1] - target[1])**2)
print("Kalman MV* MSE on target 1 (horizon 96):", mse_mv)
print("Kalman MV* MSE on target 1 (first 16 steps):", np.mean((f_mv[1, :16] - target[1, :16])**2))
print("Kalman MV* MSE on target 1 (steps 16..96):", np.mean((f_mv[1, 16:] - target[1, 16:])**2))
