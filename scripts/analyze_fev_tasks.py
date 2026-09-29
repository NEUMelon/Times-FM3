import pandas as pd
import ast
import numpy as np

df = pd.read_csv("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv")
print(f"Loaded {len(df)} tasks from fev_bench_results.csv")

def parse_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except:
        return []

df["known_cov_list"] = df["known_dynamic_columns"].apply(parse_list)
df["past_cov_list"] = df["past_dynamic_columns"].apply(parse_list)
df["has_cov"] = df["known_cov_list"].apply(lambda x: len(x) > 0) | df["past_cov_list"].apply(lambda x: len(x) > 0)

# Check number of targets / variates
# In fev-bench, num_forecasts or id_column can tell us, or target column
print("\nUnique targets column types:", df["target"].value_counts())
print("\nTasks with known dynamic cov:", df["known_cov_list"].apply(lambda x: len(x) > 0).sum())
print("Tasks with past dynamic cov:", df["past_cov_list"].apply(lambda x: len(x) > 0).sum())
print("Tasks with ANY dynamic cov:", df["has_cov"].sum())

# Categorize into layers a, b, c, d
# Layer c: single target with covariates
# Layer a: multi-target without covariates
# Layer b: multi-target with covariates
# Layer d: pure univariate without covariates

# Let's inspect a few task definitions
print("\nSample task info:")
sample_cols = ["task_name", "dataset_config", "horizon", "num_forecasts", "target", "known_cov_list", "SQL", "MASE", "WQL"]
print(df[sample_cols].head(10))
