with open("/root/autodl-tmp/timesfm/src/timesfm3/torch/evaluator.py") as f:
    lines = f.readlines()

for i, l in enumerate(lines[50:180]):
    print(f"{i+51}: {l}", end="")
