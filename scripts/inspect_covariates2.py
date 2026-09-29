import inspect
from timesfm3.torch import timesfm3_forecaster

src = inspect.getsource(timesfm3_forecaster.TimesFM3Forecaster.predict_batch)
print("=== predict_batch source lines around covariates ===")
for i, line in enumerate(src.split("\n")):
    if "past_only" in line or "covariate" in line or "po_torch" in line:
        print(f"{i:>4}: {line}")
