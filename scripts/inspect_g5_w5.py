import numpy as np
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import build_g5_oracle

X_w, meta_w = generate_g5_lead_lag(N=8, T=512 + 96 + 50, lag=16, lead_coeff=0.8, seed=42 + 4*100)
oracle_mv_w, _ = build_g5_oracle(meta_w)
ctx_w = X_w[:, :512]
target_w = X_w[:, 512:512+96]
f_mv_w = oracle_mv_w.filter_and_forecast_mv(ctx_w, horizon=96)

print("Target 1 shape:", target_w[1].shape)
print("Target 1 min/max/var:", np.min(target_w[1]), np.max(target_w[1]), np.var(target_w[1]))
print("Kalman min/max/var:", np.min(f_mv_w[1]), np.max(f_mv_w[1]), np.var(f_mv_w[1]))

print("\nKalman forecasts on steps 1..32:")
print(f_mv_w[1, :32])
print("\nTarget values on steps 1..32:")
print(target_w[1, :32])

# Check channel 0 target as well
print("\nChannel 0 target min/max/var:", np.min(target_w[0]), np.max(target_w[0]), np.var(target_w[0]))
print("Kalman ch0 forecasts on steps 1..32:")
print(f_mv_w[0, :32])
print("Target ch0 values on steps 1..32:")
print(target_w[0, :32])
print("Kalman ch0 MSE:", np.mean((f_mv_w[0] - target_w[0])**2))
