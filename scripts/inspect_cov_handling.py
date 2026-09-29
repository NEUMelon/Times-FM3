with open("/root/autodl-tmp/timesfm/src/timesfm3/torch/timesfm3_forecaster.py") as f:
    code = f.read()

import re
matches = [m.start() for m in re.finditer(r"past_only_covariates", code)]
print(f"Found {len(matches)} occurrences of past_only_covariates")
for pos in matches[:5]:
    start = max(0, pos - 100)
    end = min(len(code), pos + 400)
    print("--- SNIPPET ---")
    print(code[start:end])
