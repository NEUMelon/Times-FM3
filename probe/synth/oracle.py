# -*- coding: utf-8 -*-
"""
Exact, mathematically grounded Kalman filter oracles for linear Gaussian state-space systems.
Zero ad-hoc heuristics: strictly implements Bayes-optimal minimum mean squared error (MMSE)
estimators for both omniscient multivariate and single-channel univariate observers.
All oracles strictly construct state-space matrices from metadata returned by generators.py.
If any required metadata is missing, an explicit KeyError/ValueError is raised immediately.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg


class ExactLinearGaussianKalmanOracle:
    """
    Exact Kalman Filter oracle for linear Gaussian state-space models:
        z_t = A z_{t-1} + w_t,   w_t ~ N(0, Q)
        x_t = C z_t + v_t,       v_t ~ N(0, R)
    where:
        z_t in R^d: latent state vector
        x_t in R^N: observed time series across N channels
        A in R^{d x d}: state transition matrix
        Q in R^{d x d}: process noise covariance matrix
        C in R^{N x d}: observation loading matrix
        R in R^{N x N}: observation noise covariance matrix
    """

    def __init__(
        self,
        A: np.ndarray,
        Q: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        P0: Optional[np.ndarray] = None,
        name: str = "ExactKalmanOracle",
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.name = name

        self.state_dim = self.A.shape[0]
        self.obs_dim = self.C.shape[0]

        assert self.A.shape == (self.state_dim, self.state_dim), f"A shape {self.A.shape} mismatch"
        assert self.Q.shape == (self.state_dim, self.state_dim), f"Q shape {self.Q.shape} mismatch"
        assert self.C.shape[1] == self.state_dim, f"C cols {self.C.shape[1]} != state_dim {self.state_dim}"
        assert self.R.shape == (self.obs_dim, self.obs_dim), f"R shape {self.R.shape} mismatch"

        self.P0 = np.asarray(P0, dtype=np.float64) if P0 is not None else None

    def solve_riccati_steady_state(
        self,
        C_obs: Optional[np.ndarray] = None,
        R_obs: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-11,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves Discrete Algebraic Riccati Equation (DARE) for steady-state
        one-step prediction covariance P_pred and filtered covariance P_filt:
            P_pred = A P_filt A^T + Q
            P_filt = (I - K C) P_pred
        Uses solve_discrete_are, falling back to Riccati fixed-point iteration from P0
        when deterministic seasonal modes lie directly on the unit circle.
        """
        C_mat = self.C if C_obs is None else np.asarray(C_obs, dtype=np.float64)
        R_mat = self.R if R_obs is None else np.asarray(R_obs, dtype=np.float64)

        if self.P0 is not None:
            P_pred = self.P0.copy()
            for _ in range(max_iter):
                S = C_mat @ P_pred @ C_mat.T + R_mat
                K = P_pred @ C_mat.T @ np.linalg.inv(S)
                I_KC = np.eye(self.state_dim, dtype=np.float64) - K @ C_mat
                P_filt = I_KC @ P_pred @ I_KC.T + K @ R_mat @ K.T
                P_next = self.A @ P_filt @ self.A.T + self.Q
                if np.max(np.abs(P_next - P_pred)) < tol:
                    P_pred = P_next
                    break
                P_pred = P_next
        else:
            try:
                P_pred = linalg.solve_discrete_are(self.A.T, C_mat.T, self.Q, R_mat)
            except (linalg.LinAlgError, ValueError):
                # Fixed-point Riccati iteration starting from identity
                P_pred = np.eye(self.state_dim, dtype=np.float64)
                for _ in range(max_iter):
                    S = C_mat @ P_pred @ C_mat.T + R_mat
                    K = P_pred @ C_mat.T @ np.linalg.inv(S)
                    I_KC = np.eye(self.state_dim, dtype=np.float64) - K @ C_mat
                    P_filt = I_KC @ P_pred @ I_KC.T + K @ R_mat @ K.T
                    P_next = self.A @ P_filt @ self.A.T + self.Q
                    if np.max(np.abs(P_next - P_pred)) < tol:
                        P_pred = P_next
                        break
                    P_pred = P_next

        S = C_mat @ P_pred @ C_mat.T + R_mat
        K = P_pred @ C_mat.T @ np.linalg.inv(S)
        I_KC = np.eye(self.state_dim, dtype=np.float64) - K @ C_mat
        # Joseph stabilized form
        P_filt = I_KC @ P_pred @ I_KC.T + K @ R_mat @ K.T
        return P_pred, P_filt

    def get_theoretical_forecast_variance(
        self,
        h: int,
        target_idx: int = 0,
        C_target: Optional[np.ndarray] = None,
        R_target_val: Optional[float] = None,
    ) -> float:
        r"""
        Calculates theoretical forecast error variance for target channel at horizon h (1-indexed):
            Var(x_{i, t+h} - \hat{x}_{i, t+h|t}) = C_i P_h C_i^T + R_ii
        where P_h = A^h P_filt (A^T)^h + sum_{j=0}^{h-1} A^j Q (A^T)^j
        """
        assert h >= 1, f"Horizon h must be >= 1, got {h}"
        C_i = self.C[target_idx : target_idx + 1, :] if C_target is None else np.asarray(C_target, dtype=np.float64)
        R_ii = self.R[target_idx, target_idx] if R_target_val is None else float(R_target_val)

        _, P_filt = self.solve_riccati_steady_state()

        A_h = np.linalg.matrix_power(self.A, h)
        Q_sum = np.zeros_like(self.Q)
        for j in range(h):
            A_j = np.linalg.matrix_power(self.A, j)
            Q_sum += A_j @ self.Q @ A_j.T

        P_h = A_h @ P_filt @ A_h.T + Q_sum
        var_h = float((C_i @ P_h @ C_i.T + R_ii).item())
        return var_h

    def filter_and_forecast_mv(
        self,
        observations: np.ndarray,
        horizon: int,
        C_seq: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Multivariate Kalman Filter Oracle:
        Conditioning on ALL observed channels x_{1:N}(1:T).
        observations: [N, T]
        Returns: [N, horizon] point forecasts
        """
        N, T = observations.shape
        assert N == self.obs_dim, f"Observed channels {N} != model obs_dim {self.obs_dim}"

        x_hat = np.zeros((self.state_dim, 1), dtype=np.float64)
        if self.P0 is not None:
            P = self.P0.copy()
        else:
            P_pred, _ = self.solve_riccati_steady_state()
            P = P_pred.copy()

        for t in range(T):
            C_t = self.C if C_seq is None else C_seq[t]
            y = observations[:, t : t + 1].astype(np.float64)

            # Measurement update
            S = C_t @ P @ C_t.T + self.R
            K = P @ C_t.T @ np.linalg.inv(S)
            x_filt = x_hat + K @ (y - C_t @ x_hat)

            # Joseph stabilized covariance update
            I_KC = np.eye(self.state_dim, dtype=np.float64) - K @ C_t
            P_filt = I_KC @ P @ I_KC.T + K @ self.R @ K.T

            # Time update to next step
            if t < T - 1:
                x_hat = self.A @ x_filt
                P = self.A @ P_filt @ self.A.T + self.Q
            else:
                x_hat = x_filt

        # Multi-step projections from T: \hat{x}_{T+h|T} = C_last A^h \hat{z}_{T|T}
        C_last = self.C if C_seq is None else C_seq[-1]
        forecasts = np.zeros((N, horizon), dtype=np.float64)
        curr_x = x_hat
        for h in range(horizon):
            curr_x = self.A @ curr_x
            forecasts[:, h] = (C_last @ curr_x).flatten()

        return forecasts.astype(np.float32)

    def filter_and_forecast_uv(
        self,
        target_observations: np.ndarray,
        horizon: int,
        target_idx: int = 0,
    ) -> np.ndarray:
        """
        Univariate Kalman Filter Oracle:
        target_observations: [T] or [1, T]
        Returns: [horizon] point forecasts
        """
        y_seq = np.asarray(target_observations, dtype=np.float64).flatten()
        T = len(y_seq)
        C_i = self.C[target_idx : target_idx + 1, :]
        R_i = np.array([[self.R[target_idx, target_idx]]], dtype=np.float64)

        x_hat = np.zeros((self.state_dim, 1), dtype=np.float64)
        if self.P0 is not None:
            P = self.P0.copy()
        else:
            P_pred, _ = self.solve_riccati_steady_state(C_obs=C_i, R_obs=R_i)
            P = P_pred.copy()

        for t in range(T):
            y = np.array([[y_seq[t]]], dtype=np.float64)
            S = C_i @ P @ C_i.T + R_i
            K = P @ C_i.T @ np.linalg.inv(S)
            x_filt = x_hat + K @ (y - C_i @ x_hat)

            I_KC = np.eye(self.state_dim, dtype=np.float64) - K @ C_i
            P_filt = I_KC @ P @ I_KC.T + K @ R_i @ K.T

            if t < T - 1:
                x_hat = self.A @ x_filt
                P = self.A @ P_filt @ self.A.T + self.Q
            else:
                x_hat = x_filt

        forecasts = np.zeros(horizon, dtype=np.float64)
        curr_x = x_hat
        for h in range(horizon):
            curr_x = self.A @ curr_x
            forecasts[h] = (C_i @ curr_x).item()

        return forecasts.astype(np.float32)


# ==============================================================================
# Helper builders for individual channel dynamics
# ==============================================================================

def _extract_self_component_matrices(u_param: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extracts companion AR(2) and rotation seasonal matrices for one channel."""
    phi1 = float(u_param["phi1"])
    phi2 = float(u_param["phi2"])
    sigma_u = float(u_param["sigma_u"])
    period = float(u_param["period"])
    amplitude = float(u_param["amplitude"])
    s_u = float(u_param["s_u"])

    # AR(2) companion
    A_ar = np.array([[phi1, phi2], [1.0, 0.0]], dtype=np.float64)
    Q_ar = np.array([[sigma_u**2, 0.0], [0.0, 0.0]], dtype=np.float64)
    Sigma_ar = u_param.get("Sigma_ar", None)
    if Sigma_ar is None:
        Sigma_ar = linalg.solve_discrete_lyapunov(A_ar, Q_ar)

    # Seasonal forward rotation
    omega = 2.0 * np.pi / period
    A_seas = np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]], dtype=np.float64)
    Q_seas = np.zeros((2, 2), dtype=np.float64)
    P0_seas = ((amplitude**2) / 2.0) * np.eye(2, dtype=np.float64)

    A_self = linalg.block_diag(A_ar, A_seas)
    Q_self = linalg.block_diag(Q_ar, Q_seas)
    P0_self = linalg.block_diag(Sigma_ar, P0_seas)

    return A_self, Q_self, P0_self, s_u


# ==============================================================================
# Generator Oracle Builders
# ==============================================================================

def build_g1_oracle(meta: Dict) -> Tuple[ExactLinearGaussianKalmanOracle, List[ExactLinearGaussianKalmanOracle]]:
    """Builds exact Multivariate and Univariate oracles for G1 Global Low-Rank model."""
    for key in ["N", "rank", "gamma", "sigma_eps", "alpha", "Lambda", "self_params"]:
        if key not in meta:
            raise KeyError(f"G1 metadata missing required key: '{key}'")

    N = int(meta["N"])
    r = int(meta["rank"])
    gamma = float(meta["gamma"])
    sigma_eps = float(meta["sigma_eps"])
    alpha = float(meta["alpha"])
    s_c = float(meta.get("s_c", 1.0))
    Lambda = np.asarray(meta["Lambda"], dtype=np.float64)
    self_params = meta["self_params"]
    assert len(self_params) == N, f"self_params length {len(self_params)} != N {N}"

    # Latent factors state: A_f, Q_f
    A_f = alpha * np.eye(r, dtype=np.float64)
    Q_f = (1.0 - alpha**2) * np.eye(r, dtype=np.float64)
    P0_f = np.eye(r, dtype=np.float64)

    # Full MV state space: factors (r) + N * 4 self states
    A_blocks = [A_f]
    Q_blocks = [Q_f]
    P0_blocks = [P0_f]
    s_u_list = []

    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_blocks.append(A_s)
        Q_blocks.append(Q_s)
        P0_blocks.append(P0_s)
        s_u_list.append(s_u)

    A_mv = linalg.block_diag(*A_blocks)
    Q_mv = linalg.block_diag(*Q_blocks)
    P0_mv = linalg.block_diag(*P0_blocks)
    dim_mv = r + 4 * N

    C_mv = np.zeros((N, dim_mv), dtype=np.float64)
    for i in range(N):
        # Factor loading
        C_mv[i, :r] = (np.sqrt(gamma) / s_c) * Lambda[i]
        # Self AR(2) loading
        off_ar = r + 4 * i
        C_mv[i, off_ar] = np.sqrt(1.0 - gamma) / s_u_list[i]
        # Self Seasonal loading
        off_seas = off_ar + 2
        C_mv[i, off_seas] = np.sqrt(1.0 - gamma) / s_u_list[i]

    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)
    oracle_mv = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C_mv, R_mv, P0_mv, name="G1_MV_Oracle")

    # Univariate oracles: one minimal observable oracle per channel
    uv_oracles = []
    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_uv = linalg.block_diag(A_f, A_s)
        Q_uv = linalg.block_diag(Q_f, Q_s)
        P0_uv = linalg.block_diag(P0_f, P0_s)
        dim_uv = r + 4

        C_uv = np.zeros((1, dim_uv), dtype=np.float64)
        C_uv[0, :r] = (np.sqrt(gamma) / s_c) * Lambda[i]
        C_uv[0, r] = np.sqrt(1.0 - gamma) / s_u
        C_uv[0, r + 2] = np.sqrt(1.0 - gamma) / s_u
        R_uv = np.array([[sigma_eps**2]], dtype=np.float64)

        oracle_uv = ExactLinearGaussianKalmanOracle(A_uv, Q_uv, C_uv, R_uv, P0_uv, name=f"G1_UV_Oracle_tgt{i}")
        uv_oracles.append(oracle_uv)

    return oracle_mv, uv_oracles


def build_g2_oracle(meta: Dict) -> Tuple[ExactLinearGaussianKalmanOracle, List[ExactLinearGaussianKalmanOracle]]:
    """Builds exact Multivariate and Univariate oracles for G2 Local Blocks model."""
    for key in ["N", "block_size", "num_blocks", "block_assignments", "gamma", "sigma_eps", "alpha", "self_params"]:
        if key not in meta:
            raise KeyError(f"G2 metadata missing required key: '{key}'")

    N = int(meta["N"])
    B = int(meta["num_blocks"])
    gamma = float(meta["gamma"])
    sigma_eps = float(meta["sigma_eps"])
    alpha = float(meta["alpha"])
    s_c = float(meta.get("s_c", 1.0))
    block_assignments = meta["block_assignments"]
    self_params = meta["self_params"]

    # B block factors
    A_f = alpha * np.eye(B, dtype=np.float64)
    Q_f = (1.0 - alpha**2) * np.eye(B, dtype=np.float64)
    P0_f = np.eye(B, dtype=np.float64)

    A_blocks = [A_f]
    Q_blocks = [Q_f]
    P0_blocks = [P0_f]
    s_u_list = []

    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_blocks.append(A_s)
        Q_blocks.append(Q_s)
        P0_blocks.append(P0_s)
        s_u_list.append(s_u)

    A_mv = linalg.block_diag(*A_blocks)
    Q_mv = linalg.block_diag(*Q_blocks)
    P0_mv = linalg.block_diag(*P0_blocks)
    dim_mv = B + 4 * N

    C_mv = np.zeros((N, dim_mv), dtype=np.float64)
    for i in range(N):
        b = block_assignments[i]
        C_mv[i, b] = np.sqrt(gamma) / s_c
        off_ar = B + 4 * i
        C_mv[i, off_ar] = np.sqrt(1.0 - gamma) / s_u_list[i]
        off_seas = off_ar + 2
        C_mv[i, off_seas] = np.sqrt(1.0 - gamma) / s_u_list[i]

    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)
    oracle_mv = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C_mv, R_mv, P0_mv, name="G2_MV_Oracle")

    # Univariate: single block factor + self states
    A_f1 = np.array([[alpha]], dtype=np.float64)
    Q_f1 = np.array([[1.0 - alpha**2]], dtype=np.float64)
    P0_f1 = np.array([[1.0]], dtype=np.float64)

    uv_oracles = []
    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_uv = linalg.block_diag(A_f1, A_s)
        Q_uv = linalg.block_diag(Q_f1, Q_s)
        P0_uv = linalg.block_diag(P0_f1, P0_s)
        dim_uv = 1 + 4

        C_uv = np.zeros((1, dim_uv), dtype=np.float64)
        C_uv[0, 0] = np.sqrt(gamma) / s_c
        C_uv[0, 1] = np.sqrt(1.0 - gamma) / s_u
        C_uv[0, 3] = np.sqrt(1.0 - gamma) / s_u
        R_uv = np.array([[sigma_eps**2]], dtype=np.float64)

        oracle_uv = ExactLinearGaussianKalmanOracle(A_uv, Q_uv, C_uv, R_uv, P0_uv, name=f"G2_UV_Oracle_tgt{i}")
        uv_oracles.append(oracle_uv)

    return oracle_mv, uv_oracles


def build_g3_oracles(
    meta: Dict,
) -> Tuple[ExactLinearGaussianKalmanOracle, ExactLinearGaussianKalmanOracle, List[ExactLinearGaussianKalmanOracle]]:
    """
    Builds exact oracles for G3 Structural Drift:
      - oracle_naive: assumes Lambda1 throughout the entire observation period
      - oracle_aware: transitions from Lambda1 to Lambda2 at changepoint tau
      - uv_oracles: univariate oracles for each target channel
    """
    for key in ["N", "T", "tau", "delta", "rank", "gamma", "sigma_eps", "alpha", "Lambda1", "Lambda2", "self_params"]:
        if key not in meta:
            raise KeyError(f"G3 metadata missing required key: '{key}'")

    N = int(meta["N"])
    T = int(meta["T"])
    tau = int(meta["tau"])
    r = int(meta["rank"])
    gamma = float(meta["gamma"])
    sigma_eps = float(meta["sigma_eps"])
    alpha = float(meta["alpha"])
    s_c = float(meta.get("s_c", 1.0))
    Lambda1 = np.asarray(meta["Lambda1"], dtype=np.float64)
    Lambda2 = np.asarray(meta["Lambda2"], dtype=np.float64)
    self_params = meta["self_params"]

    A_f = alpha * np.eye(r, dtype=np.float64)
    Q_f = (1.0 - alpha**2) * np.eye(r, dtype=np.float64)
    P0_f = np.eye(r, dtype=np.float64)

    A_blocks = [A_f]
    Q_blocks = [Q_f]
    P0_blocks = [P0_f]
    s_u_list = []

    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_blocks.append(A_s)
        Q_blocks.append(Q_s)
        P0_blocks.append(P0_s)
        s_u_list.append(s_u)

    A_mv = linalg.block_diag(*A_blocks)
    Q_mv = linalg.block_diag(*Q_blocks)
    P0_mv = linalg.block_diag(*P0_blocks)
    dim_mv = r + 4 * N

    def make_C(Lambda_mat):
        C_mat = np.zeros((N, dim_mv), dtype=np.float64)
        for i in range(N):
            C_mat[i, :r] = (np.sqrt(gamma) / s_c) * Lambda_mat[i]
            off_ar = r + 4 * i
            C_mat[i, off_ar] = np.sqrt(1.0 - gamma) / s_u_list[i]
            off_seas = off_ar + 2
            C_mat[i, off_seas] = np.sqrt(1.0 - gamma) / s_u_list[i]
        return C_mat

    C1 = make_C(Lambda1)
    C2 = make_C(Lambda2)
    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)

    # 1. Regime-naive oracle (always uses C1)
    oracle_naive = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C1, R_mv, P0_mv, name="G3_RegimeNaive_Oracle")

    # 2. Regime-aware oracle (time-varying C sequence)
    C_seq = [C1 if t < tau else C2 for t in range(T)]
    oracle_aware = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C2, R_mv, P0_mv, name="G3_RegimeAware_Oracle")
    oracle_aware.C_seq = C_seq

    # 3. Univariate oracles
    uv_oracles = []
    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_uv = linalg.block_diag(A_f, A_s)
        Q_uv = linalg.block_diag(Q_f, Q_s)
        P0_uv = linalg.block_diag(P0_f, P0_s)
        dim_uv = r + 4

        C_uv = np.zeros((1, dim_uv), dtype=np.float64)
        C_uv[0, :r] = (np.sqrt(gamma) / s_c) * Lambda1[i]
        C_uv[0, r] = np.sqrt(1.0 - gamma) / s_u
        C_uv[0, r + 2] = np.sqrt(1.0 - gamma) / s_u
        R_uv = np.array([[sigma_eps**2]], dtype=np.float64)

        oracle_uv = ExactLinearGaussianKalmanOracle(A_uv, Q_uv, C_uv, R_uv, P0_uv, name=f"G3_UV_Oracle_tgt{i}")
        uv_oracles.append(oracle_uv)

    return oracle_naive, oracle_aware, uv_oracles


def build_g5_oracle(meta: Dict) -> Tuple[ExactLinearGaussianKalmanOracle, List[ExactLinearGaussianKalmanOracle]]:
    """
    Builds exact Multivariate and Univariate oracles for G5 Lead-Lag model.
    Maintains a shift-register tapped delay line for channel 0 to form an exact linear state space.
    """
    for key in ["N", "lag", "lead_coeff", "sigma_eps", "self_params"]:
        if key not in meta:
            raise KeyError(f"G5 metadata missing required key: '{key}'")

    N = int(meta["N"])
    lag = int(meta["lag"])
    lead_coeff = float(meta["lead_coeff"])
    sigma_eps = float(meta["sigma_eps"])
    self_params = meta["self_params"]

    # Channel 0 state: AR(2) (2) + Seas (2) + Delay line (lag)
    A_s0, Q_s0, P0_s0, s_u0 = _extract_self_component_matrices(self_params[0])
    dim_0 = 4 + lag
    A_0 = np.zeros((dim_0, dim_0), dtype=np.float64)
    A_0[:4, :4] = A_s0
    A_0[4, 0] = 1.0 / s_u0
    A_0[4, 2] = 1.0 / s_u0
    for k in range(1, lag):
        A_0[4 + k, 4 + k - 1] = 1.0

    Q_0 = np.zeros((dim_0, dim_0), dtype=np.float64)
    Q_0[:4, :4] = Q_s0

    # Remaining channels 1..N-1
    P0_delay = np.eye(lag, dtype=np.float64)
    P0_0 = linalg.block_diag(P0_s0, P0_delay)

    A_blocks = [A_0]
    Q_blocks = [Q_0]
    P0_blocks = [P0_0]
    s_u_list = [s_u0]

    for i in range(1, N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_blocks.append(A_s)
        Q_blocks.append(Q_s)
        P0_blocks.append(P0_s)
        s_u_list.append(s_u)

    A_mv = linalg.block_diag(*A_blocks)
    Q_mv = linalg.block_diag(*Q_blocks)
    P0_mv = linalg.block_diag(*P0_blocks)
    dim_mv = dim_0 + 4 * (N - 1)

    C_mv = np.zeros((N, dim_mv), dtype=np.float64)
    # Channel 0
    C_mv[0, 0] = 1.0 / s_u0
    C_mv[0, 2] = 1.0 / s_u0

    # Channel 1: observes lagged leader + follower
    C_mv[1, 4 + lag - 1] = lead_coeff
    rem_coeff = np.sqrt(max(0.0, 1.0 - lead_coeff**2))
    off_ch1 = dim_0
    C_mv[1, off_ch1] = rem_coeff / s_u_list[1]
    C_mv[1, off_ch1 + 2] = rem_coeff / s_u_list[1]

    # Channels 2..N-1
    for i in range(2, N):
        off_i = dim_0 + 4 * (i - 1)
        C_mv[i, off_i] = 1.0 / s_u_list[i]
        C_mv[i, off_i + 2] = 1.0 / s_u_list[i]

    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)
    oracle_mv = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C_mv, R_mv, P0_mv, name="G5_MV_Oracle")

    # Univariate oracles:
    # Target 0: standard AR(2) + seasonal
    A_s0, Q_s0, P0_s0, s_u0 = _extract_self_component_matrices(self_params[0])
    C_uv0 = np.zeros((1, 4), dtype=np.float64)
    C_uv0[0, 0] = 1.0 / s_u0
    C_uv0[0, 2] = 1.0 / s_u0
    R_uv0 = np.array([[sigma_eps**2]], dtype=np.float64)
    oracle_uv0 = ExactLinearGaussianKalmanOracle(A_s0, Q_s0, C_uv0, R_uv0, P0_s0, name="G5_UV_Oracle_tgt0")

    # Target 1: follower channel alone
    # In univariate, channel 1 observes delayed driver + self, state dim is dim_0 + 4
    A_uv1 = linalg.block_diag(A_0, _extract_self_component_matrices(self_params[1])[0])
    Q_uv1 = linalg.block_diag(Q_0, _extract_self_component_matrices(self_params[1])[1])
    P0_uv1 = linalg.block_diag(P0_0, _extract_self_component_matrices(self_params[1])[2])
    dim_uv1 = dim_0 + 4
    C_uv1 = np.zeros((1, dim_uv1), dtype=np.float64)
    C_uv1[0, 4 + lag - 1] = lead_coeff
    C_uv1[0, dim_0] = rem_coeff / s_u_list[1]
    C_uv1[0, dim_0 + 2] = rem_coeff / s_u_list[1]
    R_uv1 = np.array([[sigma_eps**2]], dtype=np.float64)
    oracle_uv1 = ExactLinearGaussianKalmanOracle(A_uv1, Q_uv1, C_uv1, R_uv1, P0_uv1, name="G5_UV_Oracle_tgt1")

    uv_oracles = [oracle_uv0, oracle_uv1]
    for i in range(2, N):
        A_si, Q_si, P0_si, s_ui = _extract_self_component_matrices(self_params[i])
        C_uvi = np.zeros((1, 4), dtype=np.float64)
        C_uvi[0, 0] = 1.0 / s_ui
        C_uvi[0, 2] = 1.0 / s_ui
        R_uvi = np.array([[sigma_eps**2]], dtype=np.float64)
        oracle_uvi = ExactLinearGaussianKalmanOracle(A_si, Q_si, C_uvi, R_uvi, P0_si, name=f"G5_UV_Oracle_tgt{i}")
        uv_oracles.append(oracle_uvi)

    return oracle_mv, uv_oracles


def build_g6_oracle(meta: Dict) -> Tuple[ExactLinearGaussianKalmanOracle, List[ExactLinearGaussianKalmanOracle]]:
    """Builds exact Multivariate and Univariate oracles for G6 Null (Independent) model."""
    for key in ["N", "sigma_eps", "self_params"]:
        if key not in meta:
            raise KeyError(f"G6 metadata missing required key: '{key}'")

    N = int(meta["N"])
    sigma_eps = float(meta["sigma_eps"])
    self_params = meta["self_params"]

    A_blocks = []
    Q_blocks = []
    P0_blocks = []
    s_u_list = []

    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        A_blocks.append(A_s)
        Q_blocks.append(Q_s)
        P0_blocks.append(P0_s)
        s_u_list.append(s_u)

    A_mv = linalg.block_diag(*A_blocks)
    Q_mv = linalg.block_diag(*Q_blocks)
    P0_mv = linalg.block_diag(*P0_blocks)
    dim_mv = 4 * N

    C_mv = np.zeros((N, dim_mv), dtype=np.float64)
    for i in range(N):
        off_ar = 4 * i
        C_mv[i, off_ar] = 1.0 / s_u_list[i]
        C_mv[i, off_ar + 2] = 1.0 / s_u_list[i]

    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)
    oracle_mv = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C_mv, R_mv, P0_mv, name="G6_MV_Oracle")

    uv_oracles = []
    for i in range(N):
        A_s, Q_s, P0_s, s_u = _extract_self_component_matrices(self_params[i])
        C_uv = np.zeros((1, 4), dtype=np.float64)
        C_uv[0, 0] = 1.0 / s_u
        C_uv[0, 2] = 1.0 / s_u
        R_uv = np.array([[sigma_eps**2]], dtype=np.float64)
        oracle_uv = ExactLinearGaussianKalmanOracle(A_s, Q_s, C_uv, R_uv, P0_s, name=f"G6_UV_Oracle_tgt{i}")
        uv_oracles.append(oracle_uv)

    return oracle_mv, uv_oracles


# ==============================================================================
# Unified Evaluation Function (Task 1 Item 5: Scale-Consistent Evaluation)
# ==============================================================================

def compute_exact_kalman_oracles(
    ctx: np.ndarray,
    target: np.ndarray,
    target_idx: int,
    meta: Dict,
) -> Tuple[float, float]:
    """
    Unified entrypoint for Kalman oracle evaluation.
    Builds the exact state-space model matching generator meta, executes
    Kalman filtering and forecasting, and computes (mse_uv_star, mse_mv_star)
    evaluated on the exact same scale against identical ground truth.
    Guaranteed: E[mse_mv_star] <= E[mse_uv_star].
    """
    g_type = meta.get("type", None)
    if g_type is None:
        raise KeyError("Generator metadata missing 'type'")

    N, T = ctx.shape
    H = target.shape[1] if target.ndim > 1 else len(target)
    y_true = target[target_idx, :H] if target.ndim > 1 else target[:H]

    if "G1" in g_type:
        oracle_mv, uv_oracles = build_g1_oracle(meta)
    elif "G2" in g_type:
        oracle_mv, uv_oracles = build_g2_oracle(meta)
    elif "G3" in g_type:
        oracle_naive, oracle_aware, uv_oracles = build_g3_oracles(meta)
        oracle_mv = oracle_naive
    elif "G5" in g_type:
        oracle_mv, uv_oracles = build_g5_oracle(meta)
    elif "G6" in g_type:
        oracle_mv, uv_oracles = build_g6_oracle(meta)
    elif "G7" in g_type:
        raise ValueError("G7 (Lorenz-96) is nonlinear and does not support Kalman oracle.")
    else:
        raise ValueError(f"Unknown generator type: {g_type}")

    oracle_uv = uv_oracles[target_idx]

    # Forecasts
    pred_mv_all = oracle_mv.filter_and_forecast_mv(ctx, horizon=H)
    pred_mv = pred_mv_all[target_idx]

    pred_uv = oracle_uv.filter_and_forecast_uv(ctx[target_idx], horizon=H, target_idx=0)

    mse_mv_star = float(np.mean((pred_mv - y_true) ** 2))
    mse_uv_star = float(np.mean((pred_uv - y_true) ** 2))

    return mse_uv_star, mse_mv_star


# Backward-compatible alias
compute_factor_oracles = compute_exact_kalman_oracles
