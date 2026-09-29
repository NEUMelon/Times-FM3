import json

with open("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/fev_bench/fev_bench_timesfm3.ipynb") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        print(f"=== Cell {idx} ===")
        print(src[:200])
        print("...")
