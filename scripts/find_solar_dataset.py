from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
datasets = api.list_datasets(search="solar-energy")
print("Search results for solar-energy:")
for d in datasets:
    print(" -", d.id)
