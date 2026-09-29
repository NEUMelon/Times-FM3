import numpy as np
import torch
from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch.evaluator import TimesFM3Evaluator
from probe.config import MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(str(MODEL_DIR), device=device)

# Dummy test with 4 channels, length 512
rng = np.random.RandomState(42)
ctx = rng.normal(0, 1.0, size=(4, 512)).astype(np.float32)
horizon = 96

# Method 1: All 4 channels as target contexts
ctx_list = [ctx[i] for i in range(4)]
outs_m1 = list(forecaster.predict_batch(
    contexts=ctx_list,
    horizon=horizon,
    past_only_covariates=None,
    use_symmetric_averaging=False,
    make_positive=False,
))
pred_m1 = outs_m1[0].forecast

# Method 2: Channel 0 as target, channels 1..3 as past_only_covariates
covs = ctx[1:]  # shape [3, 512]
outs_m2 = list(forecaster.predict_batch(
    contexts=[ctx[0]],
    horizon=horizon,
    past_only_covariates=[covs],
    use_symmetric_averaging=False,
    make_positive=False,
))
pred_m2 = outs_m2[0].forecast

diff = float(np.max(np.abs(pred_m1 - pred_m2)))
print(f"Max abs difference between Method 1 (Target) and Method 2 (Covariates): {diff:.10e}")
print(f"B6 condition (diff < 1e-5): {diff < 1e-5}")
