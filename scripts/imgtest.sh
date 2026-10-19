#!/bin/bash
echo "=== IMAGE_OK ==="
echo "host: $(hostname)"
echo "python: $(python -V 2>&1)"
echo "which python: $(command -v python)"
echo "switch_env: $(command -v switch_env || echo MISSING)"
echo "--- versions ---"
python - <<'PY' 2>&1 | tail -20
import importlib.metadata as m
for n in ["torch","protobuf","tensorboard","transformers","gymnasium","numpy","pyarrow"]:
    try: print(n, m.version(n))
    except Exception: print(n, "NOT INSTALLED")
PY
echo "--- ls / ---"
ls / | tr '\n' ' '; echo
echo "--- ls /opt ---"
ls /opt 2>/dev/null | tr '\n' ' '; echo
echo "--- /data mounted? ---"
ls /data/wovr 2>&1 | head
echo "=== DONE ==="
