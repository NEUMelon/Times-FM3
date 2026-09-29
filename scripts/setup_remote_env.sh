#!/usr/bin/env bash
set -e

echo "=== [1/5] Initializing Directory Structure ==="
mkdir -p /root/autodl-tmp/DASFAA
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/ltsf
mkdir -p /root/autodl-tmp/cache/l1
mkdir -p /root/autodl-tmp/cache/l2
mkdir -p /root/autodl-tmp/results

echo "=== [2/5] Installing Base Dependencies ==="
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

python3 -m pip install -q --upgrade pip $PIP_INDEX
python3 -m pip install -q \
    transformers \
    accelerate \
    einops \
    scipy \
    pandas \
    numpy \
    pyarrow \
    fastparquet \
    huggingface_hub \
    tqdm \
    statsmodels \
    scikit-learn \
    $PIP_INDEX

echo "=== [3/5] Setting up TimesFM Repository (commit e31dadd) ==="
cd /root/autodl-tmp/timesfm
git checkout e31dadd || true
echo '/root/autodl-tmp/timesfm/src' > /root/miniconda3/lib/python3.12/site-packages/timesfm.pth

echo "=== [4/5] Downloading TimesFM-3.0-PyTorch Model Checkpoint ==="
python3 -c "
import os
import timesfm
print('TimesFM successfully imported from:', timesfm.__file__)
"

python3 -c "
import os
from huggingface_hub import snapshot_download

model_dir = '/root/autodl-tmp/models/timesfm-3.0-pytorch'
os.makedirs(model_dir, exist_ok=True)
if not os.path.exists(os.path.join(model_dir, 'config.json')):
    print('Downloading google/timesfm-3.0-pytorch from hf-mirror.com...')
    snapshot_download(
        repo_id='google/timesfm-3.0-pytorch',
        local_dir=model_dir,
        endpoint='https://hf-mirror.com',
        max_workers=4
    )
    print('Model download complete!')
else:
    print('Model already exists at:', model_dir)
"

echo "=== [5/5] Environment Setup Verification ==="
python3 -c "
import torch
import transformers
import timesfm
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('TimesFM:', timesfm.__file__)
"

echo "=== Setup Completed Successfully! ==="
