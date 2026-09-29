with open("/root/autodl-tmp/timesfm/timesfm3-usage/benchmarks/gift_eval/evaluate_all_gift.py") as f:
    lines = f.readlines()
for l in lines[40:100]:
    print(l, end="")
