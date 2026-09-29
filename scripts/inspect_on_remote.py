import inspect
import os
import torch
import timesfm3
import timesfm3.torch.model as m
import timesfm3.torch.timesfm3_forecaster as tf
import timesfm3.torch.evaluator as ev
from timesfm3.torch.evaluator import TimesFM3Evaluator

print("=" * 60)
print("1. timesfm3 package location:")
print("FILE:", timesfm3.__file__)
print("DIR:", os.path.dirname(timesfm3.__file__))

print("\n" + "=" * 60)
print("2. TimesFM3Torch attributes and decode method:")
print("Classes in model:", [x for x in dir(m) if "TimesFM" in x])
target_cls = getattr(m, "TimesFM3Torch", None) or getattr(m, "TimesFM_Torch", None)
if target_cls:
    print("Methods in target class:", [x for x in dir(target_cls) if not x.startswith("__")])
    if hasattr(target_cls, "decode"):
        src_file = inspect.getsourcefile(target_cls.decode)
        lines, start_line = inspect.getsourcelines(target_cls.decode)
        print(f"DECODE FILE: {src_file}:{start_line}")
        print("".join(lines[:45]))
        for idx, l in enumerate(lines):
            if any(k in l for k in ["for ", "while ", "horizon", "patch", "autoregressive"]):
                print(f"L{start_line + idx}: {l.strip()}")
    if hasattr(target_cls, "forward"):
        src_file = inspect.getsourcefile(target_cls.forward)
        lines, start_line = inspect.getsourcelines(target_cls.forward)
        print(f"FORWARD FILE: {src_file}:{start_line}")
        print("".join(lines[:35]))

print("\n" + "=" * 60)
print("3. Check try_gc in timesfm3_forecaster.py:")
src_tf = inspect.getsourcefile(tf.try_gc)
lines_gc, start_gc = inspect.getsourcelines(tf.try_gc)
print(f"FILE: {src_tf}:{start_gc}")
print("".join(lines_gc))

print("\n" + "=" * 60)
print("4. Where is try_gc called in timesfm3_forecaster.py:")
with open(src_tf, "r") as f:
    for idx, l in enumerate(f, 1):
        if "try_gc" in l:
            print(f"L{idx}: {l.strip()}")

print("\n" + "=" * 60)
print("5. Check evaluator and forecaster default per_core_batch_size:")
src_fc = inspect.getsourcefile(tf.TimesFM3Forecaster)
lines_fc, start_fc = inspect.getsourcelines(tf.TimesFM3Forecaster.__init__)
print(f"TimesFM3Forecaster.__init__: {src_fc}:{start_fc}")
for idx, l in enumerate(lines_fc):
    if any(k in l for k in ["batch_size", "per_core", "config"]):
        print(f"L{start_fc + idx}: {l.strip()}")

# Inspect default config
if hasattr(tf, "TimesFM3Config"):
    print("TimesFM3Config fields:")
    lines_cfg, start_cfg = inspect.getsourcelines(tf.TimesFM3Config)
    print(f"TimesFM3Config: {src_fc}:{start_cfg}")
    for idx, l in enumerate(lines_cfg):
        if any(k in l for k in ["batch_size", "per_core"]):
            print(f"L{start_cfg + idx}: {l.strip()}")

print("\n" + "=" * 60)
print("6. Torch Dtype and TF32 settings:")
print("torch.backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("torch.backends.cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32)
print("torch.get_float32_matmul_precision():", torch.get_float32_matmul_precision())

# Check evaluator model dtype
try:
    ev_obj = TimesFM3Evaluator(checkpoint_path="/root/autodl-tmp/DASFAA/models/timesfm-3.0-200m-pytorch", device="cpu")
    print("Model parameter dtype on CPU:", next(ev_obj.model.parameters()).dtype)
    print("Forecaster config per_core_batch_size:", getattr(ev_obj.forecaster.config, "per_core_batch_size", None))
    print("Forecaster config:", ev_obj.forecaster.config)
except Exception as e:
    print("Error checking evaluator:", e)
