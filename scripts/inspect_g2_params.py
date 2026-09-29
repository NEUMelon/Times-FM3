import numpy as np
from probe.synth.generators import generate_g2_local_blocks
from probe.synth.oracle import compute_exact_kalman_oracles

print("=== Inspecting G2 Oracle Gap for Parameter Combinations ===")
configs = [
    # (block_size, gamma, alpha, sigma_eps)
    (2, 0.5, 0.95, 0.1),
    (2, 0.75, 0.95, 0.1),
    (2, 0.9, 0.95, 0.05),
    (4, 0.5, 0.95, 0.1),
    (4, 0.75, 0.95, 0.1),
    (4, 0.9, 0.95, 0.05),
    (8, 0.5, 0.95, 0.1),
    (8, 0.75, 0.95, 0.1),
    (8, 0.9, 0.95, 0.05),
]

for b_size, gamma, alpha, sigma_eps in configs:
    uv_stars = []
    mv_stars = []
    for w in range(10):
        X, meta = generate_g2_local_blocks(N=16, T=512+96+50, block_size=b_size, gamma=gamma, alpha=alpha, sigma_eps=sigma_eps, seed=42 + w*10)
        ctx = X[:, :512]
        target = X[:, 512:512+96]
        u_s, m_s = compute_exact_kalman_oracles(ctx, target, target_idx=0, meta=meta)
        uv_stars.append(u_s)
        mv_stars.append(m_s)
    mean_uv = np.mean(uv_stars)
    mean_mv = np.mean(mv_stars)
    gap_pct = (mean_uv - mean_mv) / mean_uv * 100.0
    print(f"BlockSize={b_size}, gamma={gamma:.2f}, alpha={alpha:.2f}, sigma_eps={sigma_eps:.2f} => UV*={mean_uv:.4f}, MV*={mean_mv:.4f}, Gap = {gap_pct:+.2f}%")
