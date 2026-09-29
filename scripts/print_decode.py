import inspect
from timesfm3.torch import TimesFM3Forecaster
from probe.config import MODEL_DIR

forecaster = TimesFM3Forecaster.from_pretrained(str(MODEL_DIR), device="cpu")
src = inspect.getsource(forecaster.model.decode)
lines = src.splitlines()
print(f"Total lines in decode: {len(lines)}")
for i in range(min(70, len(lines))):
    print(f"{i+1:3d}: {lines[i]}")
