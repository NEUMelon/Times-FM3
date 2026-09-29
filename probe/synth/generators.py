"""Synthetic time series generators G1 to G7 with controlled cross-channel structures.
All generators strictly return complete parameter metadata in `meta` so that
exact Kalman filter oracles can be constructed without heuristics or defaults.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg
from scipy.integrate import odeint


def generate_self_component(
    T: int,
    seed: int = 42,
    period: int = 24,
    phi1: float = 0.5,
    phi2: float = 0.2,
    sigma_u: float = 1.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate independent self-component u_i(t) = (v_i(t) + s_i(t)) / s_{u,i}:
      v_i(t) ~ AR(2) with stationary companion transition and innovation variance sigma_u^2
      s_i(t) = amplitude * sin(2 * pi * t / period + phase), phase ~ Uniform(0, 2*pi)
      s_{u,i} = sqrt(Var(v_i) + amplitude^2 / 2) (exact theoretical standard deviation)
    Returns:
      u_norm: [T] array with exact theoretical mean 0 and variance 1
      meta: dictionary with all structural parameters
    """
    rng = np.random.RandomState(seed)

    # 1. AR(2) stationary covariance via discrete Lyapunov equation
    A_ar = np.array([[phi1, phi2], [1.0, 0.0]], dtype=np.float64)
    Q_ar = np.array([[sigma_u**2, 0.0], [0.0, 0.0]], dtype=np.float64)
    Sigma_ar = scipy.linalg.solve_discrete_lyapunov(A_ar, Q_ar)
    theo_var_ar = float(Sigma_ar[0, 0])

    # 2. Stationary initialization of AR(2)
    # Burn-in 500 steps to ensure stationary regime
    burn_in = 500
    T_total = T + burn_in
    noise = rng.normal(0, sigma_u, size=T_total).astype(np.float64)
    v_full = np.zeros(T_total, dtype=np.float64)
    init_state = rng.multivariate_normal([0.0, 0.0], Sigma_ar)
    v_full[0] = init_state[0]
    v_full[1] = init_state[1]

    for t in range(2, T_total):
        v_full[t] = phi1 * v_full[t - 1] + phi2 * v_full[t - 2] + noise[t]
    v = v_full[burn_in:]

    # 3. Seasonal harmonic: forward rotation sin(2*pi*t/P + phase)
    phase = float(rng.uniform(0, 2 * np.pi))
    t_steps = np.arange(T, dtype=np.float64)
    seasonal = amplitude * np.sin(2 * np.pi * t_steps / period + phase)
    theo_var_seas = (amplitude**2) / 2.0

    # 4. Total unscaled self component and theoretical scaling factor
    s_u = float(np.sqrt(theo_var_ar + theo_var_seas))
    u_norm = ((v + seasonal) / s_u).astype(np.float32)

    meta = {
        "phi1": phi1,
        "phi2": phi2,
        "sigma_u": sigma_u,
        "period": period,
        "amplitude": amplitude,
        "phase": phase,
        "theo_var_ar": theo_var_ar,
        "theo_var_seas": theo_var_seas,
        "s_u": s_u,
        "Sigma_ar": Sigma_ar,
    }
    return u_norm, meta


def generate_g1_global_low_rank(
    N: int,
    T: int,
    rank: int = 2,
    gamma: float = 0.5,
    sigma_eps: float = 0.1,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G1: Global low-rank factor structure.
      f_k(t) = alpha * f_k(t-1) + eta_k(t), eta_k ~ N(0, 1 - alpha^2)  [Var(f_k) = 1]
      c(t) = Lambda @ f(t)
      x_i(t) = sqrt(gamma) * c_i(t) + sqrt(1 - gamma) * u_i(t) + eps_i(t)
    """
    rng = np.random.RandomState(seed)

    # 1. Factors f_k(t) with stationary AR(1) initialization
    f = np.zeros((rank, T), dtype=np.float64)
    factor_noise_std = np.sqrt(1.0 - alpha**2)
    burn_in = 200
    for k in range(rank):
        f_long = np.zeros(T + burn_in, dtype=np.float64)
        f_long[0] = rng.normal(0, 1.0)
        eta = rng.normal(0, factor_noise_std, size=T + burn_in)
        for t in range(1, T + burn_in):
            f_long[t] = alpha * f_long[t - 1] + eta[t]
        f[k] = f_long[burn_in:]

    # 2. Loading matrix Lambda [N, rank] with exact row normalization
    Lambda = rng.normal(0, 1.0, size=(N, rank)).astype(np.float64)
    Lambda /= (np.linalg.norm(Lambda, axis=1, keepdims=True) + 1e-12)

    # 3. Cross-channel component c(t) = Lambda @ f [N, T]
    c = Lambda @ f

    # 4. Self components u_i(t) and combination
    X = np.zeros((N, T), dtype=np.float32)
    self_params = []
    for i in range(N):
        u_i, u_meta = generate_self_component(T, seed=seed + 1000 + i)
        eps = rng.normal(0, sigma_eps, size=T).astype(np.float64)
        X[i] = (np.sqrt(gamma) * c[i] + np.sqrt(1.0 - gamma) * u_i + eps).astype(np.float32)
        self_params.append(u_meta)

    meta = {
        "type": "G1_global_low_rank",
        "N": N,
        "T": T,
        "rank": rank,
        "gamma": gamma,
        "sigma_eps": sigma_eps,
        "alpha": alpha,
        "s_c": 1.0,
        "Lambda": Lambda,
        "self_params": self_params,
    }
    return X, meta


def generate_g2_local_blocks(
    N: int,
    T: int,
    block_size: int = 4,
    gamma: float = 0.5,
    sigma_eps: float = 0.1,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G2: Local block structure with independent factor per block.
    """
    rng = np.random.RandomState(seed)
    num_blocks = int(np.ceil(N / block_size))
    factor_noise_std = np.sqrt(1.0 - alpha**2)
    burn_in = 200

    # Generate one stationary AR(1) factor per block
    block_factors = np.zeros((num_blocks, T), dtype=np.float64)
    for b in range(num_blocks):
        f_long = np.zeros(T + burn_in, dtype=np.float64)
        f_long[0] = rng.normal(0, 1.0)
        eta = rng.normal(0, factor_noise_std, size=T + burn_in)
        for t in range(1, T + burn_in):
            f_long[t] = alpha * f_long[t - 1] + eta[t]
        block_factors[b] = f_long[burn_in:]

    X = np.zeros((N, T), dtype=np.float32)
    self_params = []
    block_assignments = []
    for i in range(N):
        b = i // block_size
        block_assignments.append(b)
        c_i = block_factors[b]
        u_i, u_meta = generate_self_component(T, seed=seed + 2000 + i)
        eps = rng.normal(0, sigma_eps, size=T).astype(np.float64)
        X[i] = (np.sqrt(gamma) * c_i + np.sqrt(1.0 - gamma) * u_i + eps).astype(np.float32)
        self_params.append(u_meta)

    meta = {
        "type": "G2_local_blocks",
        "N": N,
        "T": T,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "block_assignments": block_assignments,
        "gamma": gamma,
        "sigma_eps": sigma_eps,
        "alpha": alpha,
        "s_c": 1.0,
        "self_params": self_params,
    }
    return X, meta


def generate_g3_structural_drift(
    N: int,
    T: int,
    tau: int,
    delta: float = 0.5,
    rank: int = 2,
    gamma: float = 0.5,
    sigma_eps: float = 0.1,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G3: Structural drift at change point tau:
      Lambda2 = sqrt(1 - delta) * Lambda1 + sqrt(delta) * Lambda'
    """
    rng = np.random.RandomState(seed)

    # Factors
    f = np.zeros((rank, T), dtype=np.float64)
    factor_noise_std = np.sqrt(1.0 - alpha**2)
    burn_in = 200
    for k in range(rank):
        f_long = np.zeros(T + burn_in, dtype=np.float64)
        f_long[0] = rng.normal(0, 1.0)
        eta = rng.normal(0, factor_noise_std, size=T + burn_in)
        for t in range(1, T + burn_in):
            f_long[t] = alpha * f_long[t - 1] + eta[t]
        f[k] = f_long[burn_in:]

    # Initial loading Lambda1
    Lambda1 = rng.normal(0, 1.0, size=(N, rank)).astype(np.float64)
    Lambda1 /= (np.linalg.norm(Lambda1, axis=1, keepdims=True) + 1e-12)

    # Shifted loading Lambda2
    Lambda_prime = rng.normal(0, 1.0, size=(N, rank)).astype(np.float64)
    Lambda_prime /= (np.linalg.norm(Lambda_prime, axis=1, keepdims=True) + 1e-12)
    Lambda2_raw = np.sqrt(1.0 - delta) * Lambda1 + np.sqrt(delta) * Lambda_prime
    Lambda2 = Lambda2_raw / (np.linalg.norm(Lambda2_raw, axis=1, keepdims=True) + 1e-12)

    # Form c(t) before and after tau
    c = np.zeros((N, T), dtype=np.float64)
    c[:, :tau] = Lambda1 @ f[:, :tau]
    c[:, tau:] = Lambda2 @ f[:, tau:]

    X = np.zeros((N, T), dtype=np.float32)
    self_params = []
    for i in range(N):
        u_i, u_meta = generate_self_component(T, seed=seed + 3000 + i)
        eps = rng.normal(0, sigma_eps, size=T).astype(np.float64)
        X[i] = (np.sqrt(gamma) * c[i] + np.sqrt(1.0 - gamma) * u_i + eps).astype(np.float32)
        self_params.append(u_meta)

    meta = {
        "type": "G3_structural_drift",
        "N": N,
        "T": T,
        "tau": tau,
        "delta": delta,
        "rank": rank,
        "gamma": gamma,
        "sigma_eps": sigma_eps,
        "alpha": alpha,
        "s_c": 1.0,
        "Lambda1": Lambda1,
        "Lambda2": Lambda2,
        "self_params": self_params,
    }
    return X, meta


def generate_g5_lead_lag(
    N: int,
    T: int,
    lag: int = 16,
    lead_coeff: float = 0.8,
    sigma_eps: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G5: Lead-lag structure where channel 0 leads channel 1:
      x_0(t) = u_0(t) + eps_0(t)
      x_1(t) = a * u_0(t - lag) + sqrt(1 - a^2) * u_1(t) + eps_1(t)
      x_i(t) = u_i(t) + eps_i(t) for i >= 2
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((N, T), dtype=np.float32)
    self_params = []

    # Leader channel 0
    u_0, u_0_meta = generate_self_component(T, seed=seed)
    eps_0 = rng.normal(0, sigma_eps, size=T).astype(np.float64)
    X[0] = (u_0 + eps_0).astype(np.float32)
    self_params.append(u_0_meta)

    # Follower channel 1
    u_1, u_1_meta = generate_self_component(T, seed=seed + 1)
    eps_1 = rng.normal(0, sigma_eps, size=T).astype(np.float64)
    lagged_leader = np.roll(u_0, shift=lag)
    lagged_leader[:lag] = 0.0  # Zero boundary conditions prior to t=0
    rem_coeff = np.sqrt(max(0.0, 1.0 - lead_coeff**2))
    X[1] = (lead_coeff * lagged_leader + rem_coeff * u_1 + eps_1).astype(np.float32)
    self_params.append(u_1_meta)

    # Other channels independent
    for i in range(2, N):
        u_i, u_meta = generate_self_component(T, seed=seed + i)
        eps_i = rng.normal(0, sigma_eps, size=T).astype(np.float64)
        X[i] = (u_i + eps_i).astype(np.float32)
        self_params.append(u_meta)

    meta = {
        "type": "G5_lead_lag",
        "N": N,
        "T": T,
        "lag": lag,
        "lead_coeff": lead_coeff,
        "sigma_eps": sigma_eps,
        "self_params": self_params,
    }
    return X, meta


def generate_g6_null(
    N: int,
    T: int,
    sigma_eps: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G6: Null structure with gamma = 0 (completely independent channels).
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((N, T), dtype=np.float32)
    self_params = []
    for i in range(N):
        u_i, u_meta = generate_self_component(T, seed=seed + i)
        eps = rng.normal(0, sigma_eps, size=T).astype(np.float64)
        X[i] = (u_i + eps).astype(np.float32)
        self_params.append(u_meta)

    meta = {
        "type": "G6_null",
        "N": N,
        "T": T,
        "gamma": 0.0,
        "sigma_eps": sigma_eps,
        "self_params": self_params,
    }
    return X, meta


def generate_g7_lorenz96(
    N: int,
    T: int,
    F: float = 8.0,
    dt: float = 0.05,
    sigma_eps: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    G7: Chaotic coupled Lorenz-96 ODE system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    rng = np.random.RandomState(seed)

    def lorenz_deriv(x, t):
        dxdt = np.zeros(N)
        for i in range(N):
            dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return dxdt

    x0 = F * np.ones(N) + rng.normal(0, 0.01, size=N)
    t = np.linspace(0, T * dt, T)
    sol = odeint(lorenz_deriv, x0, t)

    noise = rng.normal(0, sigma_eps, size=sol.shape)
    X = (sol + noise).T.astype(np.float32)
    return X, {"type": "G7_lorenz96", "N": N, "T": T, "F": F, "sigma_eps": sigma_eps}
