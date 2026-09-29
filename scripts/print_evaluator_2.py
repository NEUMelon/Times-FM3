import inspect
from timesfm3.torch.evaluator import TimesFM3Evaluator

src = inspect.getsource(TimesFM3Evaluator.predict_batch)
lines = src.splitlines()
for i in range(80, min(160, len(lines))):
    print(f"{i+1:3d}: {lines[i]}")
