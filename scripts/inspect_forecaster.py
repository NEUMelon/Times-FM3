import inspect
import timesfm3.torch.timesfm3_forecaster as tf

with open(inspect.getsourcefile(tf)) as f:
    for idx, l in enumerate(f, 1):
        if idx >= 640 and idx <= 725:
            print(f"{idx}: {l}", end="")
