import urllib.request
import os

target = "/root/autodl-tmp/data/ltsf/solar.csv"
if os.path.exists(target) and os.path.getsize(target) > 1000000:
    print("solar.csv already exists!")
else:
    urls = [
        "https://ghproxy.net/https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/dataset/solar-energy/solar_AL.txt",
        "https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/dataset/solar-energy/solar_AL.txt",
        "https://ghproxy.net/https://raw.githubusercontent.com/thuml/Autoformer/master/dataset/solar-energy/solar_AL.txt",
        "https://raw.githubusercontent.com/thuml/Autoformer/master/dataset/solar-energy/solar_AL.txt",
    ]
    for u in urls:
        print(f"Trying to fetch solar from {u}...")
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 100000:
                    with open(target, "wb") as f:
                        f.write(data)
                    print(f"Successfully downloaded solar to {target}: {len(data)/1024/1024:.2f} MB")
                    break
        except Exception as e:
            print("Failed:", e)

if os.path.exists(target):
    import pandas as pd
    # check shape
    df = pd.read_csv(target, header=None)
    print("solar.csv shape:", df.shape)
