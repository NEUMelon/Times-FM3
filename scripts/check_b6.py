import inspect
from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch.evaluator import TimesFM3Evaluator

print("TimesFM3Forecaster public methods:")
for m in dir(TimesFM3Forecaster):
    if not m.startswith("_"):
        print("  ", m)

print("\nTimesFM3Evaluator public methods:")
for m in dir(TimesFM3Evaluator):
    if not m.startswith("_"):
        print("  ", m)

if hasattr(TimesFM3Forecaster, "forecast"):
    print("\nTimesFM3Forecaster.forecast signature:", inspect.signature(TimesFM3Forecaster.forecast))

if hasattr(TimesFM3Forecaster, "predict_batch"):
    print("\nTimesFM3Forecaster.predict_batch signature:", inspect.signature(TimesFM3Forecaster.predict_batch))

if hasattr(TimesFM3Evaluator, "forecast"):
    print("\nTimesFM3Evaluator.forecast signature:", inspect.signature(TimesFM3Evaluator.forecast))
