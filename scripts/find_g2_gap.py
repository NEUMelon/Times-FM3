import numpy as np
from probe.synth.generators import generate_g2_local_blocks
from probe.synth.oracle import compute_exact_kalman_oracles

print("=== Testing G2 parameter variations for Oracle Gap > 5% ===")
# Try different alpha, sigma_eps, gamma, and horizon H
for alpha in [0.98, 0.99]:
    for gamma in [0.7, 0.8, 0.9]:
        for sigma_eps in [0.2, 0.3, 0.5]:
            for b_size in [4, 8]:
                uv_list, mv_list = [], []
                for w in range(10):
                    X, meta = generate_g2_local_blocks(N=16, T=512+96+50, block_size=b_size, gamma=gamma, alpha=alpha, sigma_eps=sigma_eps, seed=42+w*10)
                    ctx = X[:, :512]
                    target = X[:, 512:512+96]
                    u_s, m_s = compute_exact_kalman_oracles(ctx, target, target_idx=0, meta=meta)
                    uv_list.append(u_s)
                    mv_list.append(m_s)
                gap = (np.mean(uv_list) - np.mean(mv_list)) / np.mean(uv_list) * 100
                if gap > 2.0:
                    print(f"BlockSize={b_size}, alpha={alpha}, gamma={gamma}, sigma_eps={sigma_eps} => UV*={np.mean(uv_list):.4f}, MV*={np.mean(mv_list):.4f}, Gap={gap:+.2f}%")
