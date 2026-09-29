import os
import shutil
from huggingface_hub import hf_hub_download

target_dir = "/root/autodl-tmp/data/ltsf"
os.makedirs(target_dir, exist_ok=True)

files = [
    ("ETT-small/ETTh1.csv", "ETTh1.csv"),
    ("ETT-small/ETTh2.csv", "ETTh2.csv"),
    ("ETT-small/ETTm1.csv", "ETTm1.csv"),
    ("ETT-small/ETTm2.csv", "ETTm2.csv"),
    ("weather/weather.csv", "weather.csv"),
    ("electricity/electricity.csv", "electricity.csv"),
    ("traffic/traffic.csv", "traffic.csv"),
]

for src_name, dst_name in files:
    dst_path = os.path.join(target_dir, dst_name)
    if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        print(f"Downloading {src_name} via hf_hub_download...")
        cached_file = hf_hub_download(
            repo_id="thuml/Time-Series-Library",
            filename=src_name,
            repo_type="dataset",
            endpoint="https://hf-mirror.com"
        )
        shutil.copy(cached_file, dst_path)
        print(f"Saved {dst_name}: {os.path.getsize(dst_path)/1024/1024:.2f} MB")
    else:
        print(f"{dst_name} already exists: {os.path.getsize(dst_path)/1024/1024:.2f} MB")

print("\nDownloaded datasets in", target_dir)
for f in os.listdir(target_dir):
    p = os.path.join(target_dir, f)
    print(f" - {f}: {os.path.getsize(p)/1024/1024:.2f} MB")
