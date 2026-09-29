import numpy as np
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import build_g5_oracle

X, meta = generate_g5_lead_lag(N=8, T=512 + 96 + 50, lag=16, lead_coeff=0.8, seed=42)
oracle_mv, uv_oracles = build_g5_oracle(meta)

P_ss, converged = oracle_mv.solve_riccati_steady_state()
print("Riccati converged:", converged)
print("P_ss shape:", P_ss.shape)
print("P_ss min/max diagonal:", np.min(np.diag(P_ss)), np.max(np.diag(P_ss)))
print("P_ss has NaNs:", np.isnan(P_ss).any())
