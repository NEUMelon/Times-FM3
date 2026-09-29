import json
with open("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/fev_bench/fev_bench_timesfm3.ipynb") as f:
    nb = json.load(f)
print("Cell 0/1 content:")
print("".join(nb["cells"][0]["source"]))
print("---")
print("".join(nb["cells"][1]["source"]))
