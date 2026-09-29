import numpy as np
from probe.synth.generators import generate_g2_local_blocks
from probe.synth.oracle import compute_exact_kalman_oracles

for horizon in [16, 24, 48, 96]:
    uv_list, mv_list = [], []
    for w in range(20):
        X, meta = generate_g2_local_blocks(N=16, T=512+horizon+50, block_size=8, gamma=0.85, alpha=0.98, sigma_eps=0.2, seed=42+w*10)
        ctx = X[:, :512]
        target = X[:, 512:512+horizon]
        u_s, m_s = compute_exact_kalman_oracles(ctx, target, target_idx=0, meta=meta)
        uv_list.append(u_s)
        mv_list.append(m_s)
    gap = (np.mean(uv_list) - np.mean(mv_list)) / np.mean(uv_list) * 100
    print(f"Horizon={horizon:2d} => UV*={np.mean(uv_list):.4f}, MV*={np.mean(mv_list):.4f}, Gap = {gap:+.2f}%")
