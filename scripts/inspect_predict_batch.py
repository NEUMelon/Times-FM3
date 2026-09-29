import inspect
from timesfm3.torch import TimesFM3Forecaster
from timesfm3.torch.evaluator import TimesFM3Evaluator

print("=== TimesFM3Evaluator.predict_batch ===")
print(inspect.getsource(TimesFM3Evaluator.predict_batch))

print("\n=== TimesFM3Forecaster.predict_batch ===")
print(inspect.getsource(TimesFM3Forecaster.predict_batch))
