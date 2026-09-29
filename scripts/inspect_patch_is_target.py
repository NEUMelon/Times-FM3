with open("/root/autodl-tmp/timesfm/src/timesfm3/torch/model.py") as f:
    code = f.read()

import re
matches = [m.start() for m in re.finditer(r"patch_is_target", code)]
print(f"Found {len(matches)} occurrences of patch_is_target in model.py")
for pos in matches[:3]:
    start = max(0, pos - 200)
    end = min(len(code), pos + 400)
    print("--- SNIPPET ---")
    print(code[start:end])
