from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
datasets = api.list_datasets(search="solar")
for d in datasets:
    if any(k in d.id.lower() for k in ["ts", "time", "ltsf", "informer", "autoformer", "patchtst", "benchmark"]):
        print("Match:", d.id)

# Also check datasets like 'aleks/LTSF' or similar
for name in ["autonlab/timeseries-datasets", "salesforce/GiftEvalPretrain", "gpar-datasets/ltsf"]:
    try:
        files = api.list_repo_files(repo_id=name, repo_type="dataset")
        print(f"Files in {name}: {[f for f in files if 'solar' in f.lower()]}")
    except Exception as e:
        pass
