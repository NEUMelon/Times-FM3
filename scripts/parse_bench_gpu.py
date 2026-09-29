import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("/root/autodl-tmp/results/gpu_bench_log.csv", header=None, names=["ts", "util", "pwr", "mem"])
# Parse timestamp
# nvidia-smi timestamp format: YYYY/MM/DD HH:MM:SS.mmm
df["epoch"] = df["ts"].apply(lambda x: datetime.strptime(x.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp())

markers = []
with open("/root/autodl-tmp/results/benchmark_markers.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("EXPERIMENT"):
            continue
        parts = line.split(",")
        markers.append((parts[0], float(parts[1]), float(parts[2])))

for name, t_start, t_end in markers:
    sub = df[(df["epoch"] >= t_start - 0.5) & (df["epoch"] <= t_end + 0.5)]
    if len(sub) > 0:
        mean_u = sub["util"].mean()
        mean_p = sub["pwr"].mean()
        max_m = sub["mem"].max()
        print(f"{name}: Samples={len(sub)}, MeanUtil={mean_u:.1f}%, MeanPwr={mean_p:.1f}W, PeakMem={max_m:.1f}MiB")
    else:
        print(f"{name}: No GPU samples found between {t_start} and {t_end}")
