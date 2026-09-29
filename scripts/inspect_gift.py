with open("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/gift_eval/evaluate_all_gift.py") as f:
    lines = f.readlines()
print("First 40 lines of evaluate_all_gift.py:")
for l in lines[:40]:
    print(l, end="")
