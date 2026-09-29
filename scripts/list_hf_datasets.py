from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
files = api.list_repo_files(repo_id="thuml/Time-Series-Library", repo_type="dataset")
print("Files in thuml/Time-Series-Library:")
for f in files:
    if f.endswith(".csv"):
        print(" -", f)
