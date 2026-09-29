import pandas as pd

df_fev = pd.read_csv("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv")
print("=== fev_bench_results.csv ===")
print("Shape:", df_fev.shape)
print("Columns:", list(df_fev.columns))
print(df_fev.head(2).to_dict(orient="records"))

df_gift = pd.read_csv("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/gift_eval/all_results.csv")
print("\n=== all_results.csv (GIFT-Eval) ===")
print("Shape:", df_gift.shape)
print("Columns:", list(df_gift.columns))
print(df_gift.head(2).to_dict(orient="records"))
