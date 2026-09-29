import inspect
import os
import sys
import torch
import timesfm3.torch.timesfm3_forecaster as tf
import timesfm3.torch.model as tm
import timesfm3.torch.evaluator as te

print("==================== TASK 1: CODE VERIFICATION ====================")

# a. Decode in timesfm3.torch.model
print("\n--- 1a. decode in model.py ---")
decode_file = inspect.getsourcefile(tm.TimesFM3Torch.decode)
decode_lines, decode_start = inspect.getsourcelines(tm.TimesFM3Torch.decode)
print(f"Location: {decode_file}:{decode_start}")
print("Docstring:")
for l in decode_lines[1:15]:
    print("  ", l.rstrip())

# Check for any loops in decode
loops = [(idx + decode_start, l.rstrip()) for idx, l in enumerate(decode_lines) if l.strip().startswith(("for ", "while "))]
print(f"Loops in decode(): {loops}")

# b. Our wrapper and run_e1_ltsf.py
print("\n--- 1b. Batch size and wrapper configuration ---")
# Check default config in timesfm3_forecaster
fc_file = inspect.getsourcefile(tf)
with open(fc_file, "r") as f:
    fc_src = f.readlines()

for idx, l in enumerate(fc_src, 1):
    if "per_core_batch_size" in l or "class _ModelConfig" in l or "class TimesFM3Config" in l:
        print(f"  timesfm3_forecaster.py:{idx} | {l.rstrip()}")

# Check TimesFM3Evaluator.__init__ and predict_batch
ev_file = inspect.getsourcefile(te)
with open(ev_file, "r") as f:
    ev_src = f.readlines()

for idx, l in enumerate(ev_src, 1):
    if "per_core_batch_size" in l or "class TimesFM3Evaluator" in l or "def predict_batch" in l:
        print(f"  evaluator.py:{idx} | {l.rstrip()}")

# c. try_gc details
print("\n--- 1c. try_gc location & callers ---")
gc_lines, gc_start = inspect.getsourcelines(tf.try_gc)
print(f"Definition: {fc_file}:{gc_start}")
for l in gc_lines:
    print("  ", l.rstrip())

print("Callers of try_gc in timesfm3_forecaster.py:")
for idx, l in enumerate(fc_src, 1):
    if "try_gc(" in l:
        print(f"  timesfm3_forecaster.py:{idx} | {l.rstrip()}")

# d. Dtype and TF32
print("\n--- 1d. Dtype and TF32 settings ---")
print("torch.backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("torch.backends.cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32)
print("torch.get_float32_matmul_precision():", torch.get_float32_matmul_precision())

# Load model weights on CPU or check config to see dtype
checkpoint_dir = "/root/autodl-tmp/DASFAA/models/timesfm-3.0-200m-pytorch"
print("Model checkpoint files in", checkpoint_dir, ":")
try:
    print(os.listdir(checkpoint_dir))
    # Read config.json
    import json
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as jf:
            print("config.json torch_dtype:", json.load(jf).get("torch_dtype"))
except Exception as e:
    print("Error reading checkpoint:", e)
