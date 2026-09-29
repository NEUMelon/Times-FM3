import numpy as np
import scipy.linalg as linalg
from probe.synth.generators import generate_g5_lead_lag
from probe.synth.oracle import (
    ExactLinearGaussianKalmanOracle,
    _extract_self_component_matrices,
)
from probe.models.timesfm3 import TimesFM3Wrapper

wrapper = TimesFM3Wrapper(device="cuda")

N = 8
lag = 16
lead_coeff = 0.8
T_total = 512 + 96 + 50

for w in range(5):
    X, meta = generate_g5_lead_lag(N=N, T=T_total, lag=lag, lead_coeff=lead_coeff, seed=42 + w*100)
    ctx = X[:, :512]
    target = X[:, 512:512+96]
    self_params = meta["self_params"]
    sigma_eps = float(meta["sigma_eps"])

    # Build G5 state space with P0
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

    # Delay line covariance: each element is u_0, which has variance 1.0
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
    C_mv[0, 0] = 1.0 / s_u0
    C_mv[0, 2] = 1.0 / s_u0
    C_mv[1, 4 + lag - 1] = lead_coeff
    rem_coeff = np.sqrt(max(0.0, 1.0 - lead_coeff**2))
    off_ch1 = dim_0
    C_mv[1, off_ch1] = rem_coeff / s_u_list[1]
    C_mv[1, off_ch1 + 2] = rem_coeff / s_u_list[1]

    for i in range(2, N):
        off_i = dim_0 + 4 * (i - 1)
        C_mv[i, off_i] = 1.0 / s_u_list[i]
        C_mv[i, off_i + 2] = 1.0 / s_u_list[i]

    R_mv = (sigma_eps**2) * np.eye(N, dtype=np.float64)

    # UV oracle for channel 1
    A_uv1 = linalg.block_diag(A_0, _extract_self_component_matrices(self_params[1])[0])
    Q_uv1 = linalg.block_diag(Q_0, _extract_self_component_matrices(self_params[1])[1])
    P0_uv1 = linalg.block_diag(P0_0, _extract_self_component_matrices(self_params[1])[2])
    dim_uv1 = dim_0 + 4
    C_uv1 = np.zeros((1, dim_uv1), dtype=np.float64)
    C_uv1[0, 4 + lag - 1] = lead_coeff
    C_uv1[0, dim_0] = rem_coeff / s_u_list[1]
    C_uv1[0, dim_0 + 2] = rem_coeff / s_u_list[1]
    R_uv1 = np.array([[sigma_eps**2]], dtype=np.float64)

    oracle_mv = ExactLinearGaussianKalmanOracle(A_mv, Q_mv, C_mv, R_mv, P0_mv, name="G5_MV_Oracle")
    oracle_uv1 = ExactLinearGaussianKalmanOracle(A_uv1, Q_uv1, C_uv1, R_uv1, P0_uv1, name="G5_UV_Oracle_tgt1")

    f_mv = oracle_mv.filter_and_forecast_mv(ctx, horizon=96)
    f_uv = oracle_uv1.filter_and_forecast_uv(ctx[1], horizon=96, target_idx=0)
    pred_m, _ = wrapper.forecast(ctx, horizon=96, univariate=False, use_symmetric_averaging=True, return_quantiles=False)

    mse_mv = np.mean((f_mv[1] - target[1])**2)
    mse_uv = np.mean((f_uv - target[1])**2)
    mse_m = np.mean((pred_m[1] - target[1])**2)

    print(f"Window {w+1}: UV*={mse_uv:.5f}, MV*={mse_mv:.5f}, Model={mse_m:.5f}, Diff(Model - MV*)={mse_m - mse_mv:+.5f}")
