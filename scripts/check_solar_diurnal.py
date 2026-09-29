import numpy as np
import pandas as pd
from probe.data import load_dataset

ds = load_dataset('solar')
raw_tr = ds.raw_values[:ds.border2s[0]]
N = raw_tr.shape[1]
steps_per_day = 144
num_days = len(raw_tr) // steps_per_day
trimmed = raw_tr[:num_days * steps_per_day].reshape((num_days, steps_per_day, N))
mean_curve = np.mean(trimmed, axis=(0, 2))
print('Max step in day:', int(np.argmax(mean_curve)), 'value:', float(np.max(mean_curve)))
print('Min step in day:', int(np.argmin(mean_curve)), 'value:', float(np.min(mean_curve)))

print('\nHour by hour (each hour = 6 steps of 10-min):')
for h in range(24):
    step_start = h * 6
    val = float(np.mean(mean_curve[step_start : step_start + 6]))
    print(f'Hour {h:02d} (steps {step_start:>3}-{step_start+5:>3}): {val:8.3f}')
