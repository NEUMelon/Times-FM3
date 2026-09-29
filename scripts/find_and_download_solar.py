from huggingface_hub import HfApi
import urllib.request
import os

api = HfApi(endpoint="https://hf-mirror.com")
print("Searching HuggingFace for solar_AL...")
# Search datasets that contain solar_AL in files
dataset_list = api.list_datasets(search="solar", limit=100)
found_file = None
for d in dataset_list:
    try:
        files = api.list_repo_files(d.id, repo_type="dataset")
        for f in files:
            if "solar_al" in f.lower() or (f.startswith("solar") and f.endswith((".csv", ".txt"))):
                print(f"Found match: repo={d.id}, file={f}")
                found_file = (d.id, f)
                break
        if found_file:
            break
    except Exception:
        pass

if found_file:
    repo_id, fname = found_file
    from huggingface_hub import hf_hub_download
    import shutil
    print(f"Downloading {fname} from {repo_id}...")
    local_p = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", endpoint="https://hf-mirror.com")
    dest = "/root/autodl-tmp/data/ltsf/solar.csv"
    shutil.copy(local_p, dest)
    print(f"Saved to {dest}, size: {os.path.getsize(dest)/1024/1024:.2f} MB")
else:
    print("No direct solar file found on HF yet.")
