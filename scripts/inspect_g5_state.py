import numpy as np
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import build_g5_oracle

X_w, meta_w = generate_g5_lead_lag(N=8, T=512 + 96 + 50, lag=16, lead_coeff=0.8, seed=42 + 4*100)
oracle_mv_w, _ = build_g5_oracle(meta_w)
ctx_w = X_w[:, :512]
target_w = X_w[:, 512:512+96]

# Let's inspect x_hat at t=512
N, T = ctx_w.shape
x_hat = np.zeros((oracle_mv_w.state_dim, 1), dtype=np.float64)
P = oracle_mv_w.solve_riccati_steady_state()[0]

for t in range(T):
    y = ctx_w[:, t : t + 1].astype(np.float64)
    S = oracle_mv_w.C @ P @ oracle_mv_w.C.T + oracle_mv_w.R
    K = P @ oracle_mv_w.C.T @ np.linalg.inv(S)
    x_filt = x_hat + K @ (y - oracle_mv_w.C @ x_hat)
    I_KC = np.eye(oracle_mv_w.state_dim, dtype=np.float64) - K @ oracle_mv_w.C
    P_filt = I_KC @ P @ I_KC.T + K @ oracle_mv_w.R @ K.T
    if t < T - 1:
        x_hat = oracle_mv_w.A @ x_filt
        P = oracle_mv_w.A @ P_filt @ oracle_mv_w.A.T + oracle_mv_w.Q
    else:
        x_hat = x_filt

print("States 0..4 in x_hat (Channel 0):")
print("State 0 (AR):", x_hat[0, 0])
print("State 1 (AR):", x_hat[1, 0])
print("State 2 (Seas):", x_hat[2, 0])
print("State 3 (Seas):", x_hat[3, 0])
print("State 4 (Delay 1):", x_hat[4, 0])
print("P diagonal for states 0..4:")
print(np.diag(P)[:5])
