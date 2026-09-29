import numpy as np
from timesfm3 import TimesFM3Evaluator
from probe.config import MODEL_DIR

evaluator = TimesFM3Evaluator(checkpoint_path=MODEL_DIR, device="cuda")

# 4 variates, context length 256
ctx = np.random.randn(4, 256).astype(np.float32)

# Pass [ctx] as a single multivariate series
print("Calling predict_batch with contexts=[ctx] and univariate=False...")
res_mv = list(evaluator.predict_batch(
    contexts=[ctx],
    horizon=32,
    univariate=False,
    make_positive=False,
    use_symmetric_averaging=True
))[0]
print("MV forecast shape:", res_mv.forecast.shape)

print("Calling predict_batch with contexts=[ctx] and univariate=True...")
res_uv = list(evaluator.predict_batch(
    contexts=[ctx],
    horizon=32,
    univariate=True,
    make_positive=False,
    use_symmetric_averaging=True
))[0]
print("UV forecast shape:", res_uv.forecast.shape)

diff = np.max(np.abs(res_mv.forecast - res_uv.forecast))
print(f"Difference between MV and UV: {diff:.6f}")
assert diff > 1e-4, "MV and UV should now differ because variate attention is active!"
print("TEST PASSED! Variate attention is now actively distinguishing MV from UV!")
