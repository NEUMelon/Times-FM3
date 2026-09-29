import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import remote_client

commands = [
    # 1. locate package
    "python3 -c 'import timesfm3; print(\"TIMESFM3_PATH:\", timesfm3.__file__)'",
    
    # 2. check decode in model.py
    "python3 -c 'import timesfm3.torch.model as m; import inspect; print(inspect.getsource(m.TimesFM3Model.decode)[:1000])'",
    
    # 3. check try_gc in timesfm3_forecaster.py
    "python3 -c 'import timesfm3.torch.timesfm3_forecaster as tf; import inspect; print(inspect.getsource(tf.try_gc))'",
    
    # 4. check default per_core_batch_size in config
    "python3 -c 'import timesfm3.torch.evaluator as ev; import inspect; print(inspect.getsource(ev.TimesFM3Evaluator.__init__))'",
    
    # 5. check dtype and TF32 settings
    "python3 -c 'import torch; print(\"TF32_MATMUL:\", torch.backends.cuda.matmul.allow_tf32); print(\"TF32_CUDNN:\", torch.backends.cudnn.allow_tf32); print(\"MATMUL_PRECISION:\", torch.get_float32_matmul_precision())'",
    
    # 6. check py-spy availability
    "which py-spy 2>/dev/null || echo 'py-spy not installed'",
]

for cmd in commands:
    print(f"\n==================== RUN: {cmd} ====================")
    remote_client.run_remote(cmd)
