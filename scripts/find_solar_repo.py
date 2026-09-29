from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
print("Searching repos for solar_AL...")
# Search datasets with solar in name or query
results = api.list_datasets(search="solar_AL")
for r in results:
    print("Found repo:", r.id)
    try:
        files = api.list_repo_files(r.id, repo_type="dataset")
        print(" - files:", [f for f in files if "solar" in f.lower()])
    except Exception as e:
        print(" - error:", e)

# Also check timeseries benchmark dataset repos
for cand in ["m4-competition/time-series", "huggingface/datasets", "unit8/darts", "salesforce/GiftEvalPretrain", "ServiceNow/multi-variate-time-series"]:
    try:
        files = api.list_repo_files(cand, repo_type="dataset")
        matches = [f for f in files if "solar" in f.lower()]
        if matches:
            print(f"Cand {cand}:", matches)
    except Exception:
        pass
