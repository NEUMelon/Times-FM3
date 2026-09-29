import inspect
from timesfm3.torch import timesfm3_forecaster
from timesfm3.torch import TimesFM3Forecaster
from probe.config import MODEL_DIR

forecaster = TimesFM3Forecaster.from_pretrained(str(MODEL_DIR), device="cpu")
src = inspect.getsource(forecaster.model.decode)
print(src[:2000])
