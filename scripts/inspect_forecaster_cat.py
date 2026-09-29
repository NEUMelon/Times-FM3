import inspect
from timesfm3.torch import timesfm3_forecaster

src = inspect.getsource(timesfm3_forecaster.TimesFM3Forecaster.predict_batch)
# Find how past_only_covariates are concatenated with context
lines = src.split("\n")
for i, l in enumerate(lines):
    if "covariate" in l or "torch" in l or "stack" in l or "cat" in l or "decode" in l:
        print(f"{i:>4}: {l}")
