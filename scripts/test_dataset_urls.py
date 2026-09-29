import urllib.request
import os

test_urls = [
    ("ETTm1 via raw.githubusercontent.com", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTm1.csv"),
    ("ETTm1 via jsdelivr", "https://cdn.jsdelivr.net/gh/zhouhaoyi/ETDataset@main/data/ETT/ETTm1.csv"),
    ("ETTm1 via ghproxy", "https://ghproxy.net/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTm1.csv"),
    ("HuggingFace TSL", "https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv"),
]

for name, url in test_urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            head = resp.read(100)
            print(f"SUCCESS: {name} ({len(head)} bytes read, status {resp.status})")
    except Exception as e:
        print(f"FAILED: {name} -> {e}")
