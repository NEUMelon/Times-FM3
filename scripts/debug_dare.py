import numpy as np
from scipy import linalg
from probe.synth.generators import generate_g1_global_low_rank
from probe.synth.oracle import build_g1_oracle

def riccati_fp(A, Q, C, R, P0, steps=512):
    P = P0.copy()
    dim = A.shape[0]
    for _ in range(steps):
        S = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(S)
        I_KC = np.eye(dim) - K @ C
        P_filt = I_KC @ P @ I_KC.T + K @ R @ K.T
        P = A @ P_filt @ A.T + Q
    return P, P_filt

T_win = 512 + 96
_, meta = generate_g1_global_low_rank(N=4, T=T_win, rank=2, gamma=0.5, seed=42)
oracle_mv, uv_oracles = build_g1_oracle(meta)
oracle_uv = uv_oracles[0]

for steps in [512, 1000]:
    _, P_filt_mv = riccati_fp(oracle_mv.A, oracle_mv.Q, oracle_mv.C, oracle_mv.R, oracle_mv.P0, steps=steps)
    _, P_filt_uv = riccati_fp(oracle_uv.A, oracle_uv.Q, oracle_uv.C, oracle_uv.R, oracle_uv.P0, steps=steps)
    
    print(f"\n--- Comparing with {steps} steps filtering ---")
    for h in [1, 8, 24, 48, 96]:
        A_h_mv = np.linalg.matrix_power(oracle_mv.A, h)
        A_h_uv = np.linalg.matrix_power(oracle_uv.A, h)
        
        Q_sum_mv = sum(np.linalg.matrix_power(oracle_mv.A, j) @ oracle_mv.Q @ np.linalg.matrix_power(oracle_mv.A, j).T for j in range(h))
        Q_sum_uv = sum(np.linalg.matrix_power(oracle_uv.A, j) @ oracle_uv.Q @ np.linalg.matrix_power(oracle_uv.A, j).T for j in range(h))
        
        var_mv = float((oracle_mv.C[0:1] @ (A_h_mv @ P_filt_mv @ A_h_mv.T + Q_sum_mv) @ oracle_mv.C[0:1].T + oracle_mv.R[0, 0]).item())
        var_uv = float((oracle_uv.C @ (A_h_uv @ P_filt_uv @ A_h_uv.T + Q_sum_uv) @ oracle_uv.C.T + oracle_uv.R[0, 0]).item())
        diff = var_uv - var_mv
        print(f"h={h:>2}: MV = {var_mv:.8f}, UV = {var_uv:.8f}, Gain (UV-MV) = {diff:+.8f}")
