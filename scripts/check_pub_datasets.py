import os
import glob

print("Checking /root/autodl-pub for LTSF datasets...")
matches = []
for root, dirs, files in os.walk("/root/autodl-pub"):
    for f in files:
        if any(name in f.lower() for name in ["ett", "weather", "electricity", "traffic", "solar"]):
            matches.append(os.path.join(root, f))
    if len(matches) > 30:
        break

print(f"Found {len(matches)} matches in /root/autodl-pub:")
for m in matches[:20]:
    print(" -", m)
