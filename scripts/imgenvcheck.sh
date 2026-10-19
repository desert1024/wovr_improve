#!/bin/bash
echo "=== IMAGE_OK ==="
echo "host: $(hostname)"
echo "default python: $(python -V 2>&1) at $(command -v python)"
echo "switch_env: $(command -v switch_env || echo MISSING)"
echo "--- /opt ---"
ls /opt 2>/dev/null | tr "\n" " "; echo
echo "--- switch_env source (head) ---"
head -40 "$(command -v switch_env)" 2>/dev/null
echo "--- activate openvla-oft ---"
source switch_env openvla-oft 2>&1 && echo "switch rc=0"
echo "python after switch: $(python -V 2>&1) at $(command -v python)"
python - <<"PY" 2>&1 | tail -30
import importlib.metadata as m
for n in ["torch","torchvision","torchaudio","transformers","tensorflow","flash-attn","openvla-oft","rlinf","ray","diffusers","hydra-core","deepspeed","libero"]:
    try: print(n, m.version(n))
    except Exception: print(n, "NOT INSTALLED")
PY
echo "--- import test ---"
python - <<"PY" 2>&1 | tail -10
try:
    import torch; print("torch ok, cuda", torch.cuda.is_available(), torch.cuda.device_count())
except Exception as e: print("torch FAIL", e)
try:
    import rlinf; print("rlinf", rlinf.__file__)
except Exception as e: print("rlinf FAIL", e)
try:
    import transformers; print("transformers", transformers.__version__, transformers.__file__)
except Exception as e: print("transformers FAIL", e)
PY
echo "--- /data/wovr mounted? ---"
ls /data/wovr 2>&1 | head
echo "=== DONE ==="
