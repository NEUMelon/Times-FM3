import numpy as np
import torch
from timesfm3.torch.evaluator import TimesFM3Evaluator
from probe.config import MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
evaluator = TimesFM3Evaluator.from_pretrained(str(MODEL_DIR), device=device)

rng = np.random.RandomState(42)
ctx = rng.normal(0, 1.0, size=(4, 512)).astype(np.float32)
horizon = 96

# Evaluator Method 1
outs_e1 = list(evaluator.predict_batch(
    contexts=[ctx],
    horizon=horizon,
    univariate=False,
    past_only_covariates=None,
    use_symmetric_averaging=False,
    make_positive=False,
))
pred_e1 = outs_e1[0].forecast[0]

# Evaluator Method 2
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
