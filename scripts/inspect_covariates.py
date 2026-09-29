import inspect
from timesfm3.torch import timesfm3_forecaster

src = inspect.getsource(timesfm3_forecaster.TimesFM3Forecaster.predict_batch)
# Find references to past_only_covariates
lines = [l for l in src.split("\n") if "past_only_covariates" in l or "covariate" in l]
for l in lines[:30]:
    print(l)
