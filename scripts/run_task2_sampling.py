import subprocess
import time
import pandas as pd
import numpy as np
from safetensors import safe_open

print("==================== TASK 1d: SAFETENSORS DTYPE ====================")
safetensors_path = "/root/autodl-tmp/models/timesfm-3.0-pytorch/model.safetensors"
with safe_open(safetensors_path, framework="pt") as f:
    sample_key = next(iter(f.keys()))
    t = f.get_tensor(sample_key)
    print(f"Sample tensor '{sample_key}' shape: {t.shape}, dtype: {t.dtype}")

print("\n==================== TASK 2b: PY-SPY PROFILING (PID 7161) ====================")
# Test py-spy dump on PID 7161
try:
    res = subprocess.run(["py-spy", "dump", "--pid", "7161"], capture_output=True, text=True, timeout=15)
    print("--- py-spy dump output (top 40 lines) ---")
    dump_lines = res.stdout.split("\n")
    for l in dump_lines[:40]:
        print(l)
except Exception as e:
    print("Error running py-spy dump:", e)

# Also run a 10-second py-spy profile summary if supported
try:
    res = subprocess.run(["py-spy", "record", "--pid", "7161", "--duration", "10", "--format", "speedscope", "-o", "/root/autodl-tmp/results/profile.json"], capture_output=True, text=True, timeout=25)
    print("py-spy record 10s finished.")
except Exception as e:
    print("Error running py-spy record:", e)

print("\n==================== TASK 2a: NVIDIA-SMI 60-SEC SAMPLING ====================")
csv_file = "/root/autodl-tmp/results/gpu_sample_60s.csv"
cmd = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
    "--format=csv,nounits,noheader",
    "-l", "1"
]
print("Starting 60s nvidia-smi sampling...")
with open(csv_file, "w") as out_f:
    p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    time.sleep(60)
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

print("Sampling complete. Reading stats from", csv_file)
try:
    df = pd.read_csv(csv_file, header=None, names=["gpu_util", "mem_util", "power_w", "mem_used_mib"])
    print(f"Total samples collected: {len(df)}")
    print("--- GPU Utilization (%) ---")
    print(f"  Mean: {df['gpu_util'].mean():.2f}%, Min: {df['gpu_util'].min()}%, Max: {df['gpu_util'].max()}%")
    print("--- Memory Controller Utilization (%) ---")
    print(f"  Mean: {df['mem_util'].mean():.2f}%, Min: {df['mem_util'].min()}%, Max: {df['mem_util'].max()}%")
    print("--- Power Draw (W) ---")
    print(f"  Mean: {df['power_w'].mean():.2f} W, Min: {df['power_w'].min():.2f} W, Max: {df['power_w'].max():.2f} W")
    print("--- Memory Used (MiB) ---")
    print(f"  Mean: {df['mem_used_mib'].mean():.2f} MiB, Min: {df['mem_used_mib'].min():.2f} MiB, Max: {df['mem_used_mib'].max():.2f} MiB")
except Exception as e:
    print("Error parsing csv:", e)
