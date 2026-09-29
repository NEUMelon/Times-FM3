import urllib.request
import gzip
import shutil
import os
import pandas as pd

url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
gz_path = "/root/autodl-tmp/data/ltsf/solar_AL.txt.gz"
dest_csv = "/root/autodl-tmp/data/ltsf/solar.csv"

print("Downloading canonical solar_AL.txt.gz...")
urllib.request.urlretrieve(url, gz_path)
print(f"Downloaded gz: {os.path.getsize(gz_path)/1024/1024:.2f} MB")

print("Decompressing to solar.csv...")
with gzip.open(gz_path, 'rb') as f_in:
    with open(dest_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove(gz_path)
print(f"Decompressed to {dest_csv}: {os.path.getsize(dest_csv)/1024/1024:.2f} MB")

# Verify shape: should be (52560, 137)
df = pd.read_csv(dest_csv, header=None)
print("Solar dataset verified shape:", df.shape)
assert df.shape[1] == 137, f"Expected 137 variates, got {df.shape[1]}"
print("Solar dataset downloaded and verified successfully!")
