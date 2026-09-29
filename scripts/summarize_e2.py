import pandas as pd
import numpy as np

path = '/root/autodl-tmp/results/e2_interventions_summary.csv'
try:
    df = pd.read_csv(path)
    print("=" * 80)
    print("E2 INTERVENTIONS METRICS SUMMARY (ETTh1 & weather):")
    print("=" * 80)
    summary = df.groupby(['dataset', 'intervention'])[['Reliance_R', 'Sensitivity_S', 'MSE_MV', 'MSE_PI']].mean().reset_index()
    print(summary.to_string(index=False))
    print("=" * 80)
except Exception as e:
    print("Error reading E2 summary:", e)
