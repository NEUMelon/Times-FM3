import inspect
from timesfm3.torch.evaluator import TimesFM3Evaluator

src = inspect.getsource(TimesFM3Evaluator.predict_batch)
lines = src.splitlines()
print(f"Total lines: {len(lines)}")
for i in range(min(80, len(lines))):
    print(f"{i+1:3d}: {lines[i]}")
