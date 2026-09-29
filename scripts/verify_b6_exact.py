import numpy as np
import torch
from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch.evaluator import TimesFM3Evaluator
from probe.config import MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(str(MODEL_DIR), device=device)

rng = np.random.RandomState(42)
ctx = rng.normal(0, 1.0, size=(4, 512)).astype(np.float32)
horizon = 96

# Method 1: All 4 channels as target channels in a single multivariate window
# contexts is a list of 2D arrays: [ [4, 512] ]
outs_m1 = list(forecaster.predict_batch(
    contexts=[ctx],
    horizon=horizon,
    past_only_covariates=None,
    use_symmetric_averaging=False,
    make_positive=False,
))
# outs_m1[0].forecast has shape [4, horizon]
pred_m1 = outs_m1[0].forecast[0]

# Method 2: Channel 0 as target [1, 512], channels 1..3 as past_only_covariates [3, 512]
outs_m2 = list(forecaster.predict_batch(
    contexts=[ctx[0:1]],
    horizon=horizon,
    past_only_covariates=[ctx[1:]],
    use_symmetric_averaging=False,
    make_positive=False,
))
# outs_m2[0].forecast has shape [1, horizon]
pred_m2 = outs_m2[0].forecast[0]

diff = float(np.max(np.abs(pred_m1 - pred_m2)))
print(f"Corrected Method 1 vs Method 2 max abs diff: {diff:.10e}")
print(f"B6 condition (diff < 1e-5): {diff < 1e-5}")

# Also test with Evaluator
evaluator = TimesFM3Evaluator(str(MODEL_DIR), device=device)
outs_e1 = list(evaluator.predict_batch(
    contexts=[ctx],
    horizon=horizon,
    univariate=False,
    past_only_covariates=None,
    use_symmetric_averaging=False,
    make_positive=False,
))
pred_e1 = outs_e1[0].forecast[0]

outs_e2 = list(evaluator.predict_batch(
    contexts=[ctx[0:1]],
    horizon=horizon,
    univariate=False,
    past_only_covariates=[ctx[1:]],
    use_symmetric_averaging=False,
    make_positive=False,
))
pred_e2 = outs_e2[0].forecast[0]

diff_e = float(np.max(np.abs(pred_e1 - pred_e2)))
print(f"Evaluator Method 1 vs Method 2 max abs diff: {diff_e:.10e}")
print(f"Evaluator B6 condition (diff < 1e-5): {diff_e < 1e-5}")
