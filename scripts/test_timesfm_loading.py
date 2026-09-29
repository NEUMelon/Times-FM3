import os
import torch
from timesfm3.torch import TimesFM3Forecaster

def test_loading():
    model_path = "/root/autodl-tmp/models/timesfm-3.0-pytorch"
    print(f"Loading TimesFM-3.0 from {model_path} onto CUDA...")
    forecaster = TimesFM3Forecaster(checkpoint_path=model_path, device="cuda")
    print("Successfully instantiated TimesFM3Forecaster!")
    print(f"Device: {forecaster.device}")
    
    # Test a dummy forecast
    import numpy as np
    # 2 variables, context length 64
    ctx = np.random.randn(2, 64).astype(np.float32)
    print("Running a dummy forecast with batch shape:", ctx.shape)
    
    res = forecaster.predict_batch(
        ctx,
        horizon=16,
        return_quantiles=True,
        make_positive=False,
        use_symmetric_averaging=True
    )
    items = list(res)
    print(f"Generator yielded {len(items)} item(s)")
    first = items[0]
    print("Item type:", type(first))
    print("Item fields:", [k for k in dir(first) if not k.startswith("_")])
    print("forecast shape:", first.forecast.shape)
    print("quantiles shape:", first.quantiles.shape)
    print("forecast content:", first.forecast)
    print("quantiles content shape:", first.quantiles.shape)
    print("Dummy inference succeeded cleanly!")

if __name__ == "__main__":
    test_loading()
