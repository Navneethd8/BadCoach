#!/usr/bin/env bash
# Pin isocourt-mmpose deps for nd17: numpy 1.26 (torch 2.1) + headless opencv.
#
# Run after ANY pip install in isocourt-mmpose — mmcv/mmdet/opencv keep breaking the stack.
#
# Usage:
#   conda activate isocourt-mmpose
#   ./scripts/cluster/ensure_mmpose_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate your conda env first: conda activate isocourt-mmpose" >&2
  exit 1
fi

echo "=== numpy 1.26.4 (torch 2.1 expects numpy 1.x) ==="
python -m pip install --no-cache-dir 'numpy==1.26.4'

echo "=== opencv headless ==="
"${SCRIPT_DIR}/ensure_opencv_headless.sh"

echo "=== verify torch + cv2 ==="
python - <<'PY'
import numpy as np
import torch
import cv2
ver = getattr(cv2, "__version__", None) or getattr(cv2, "getVersionString", lambda: "?")()
print("numpy", np.__version__)
print("torch", torch.__version__)
print("cv2", ver, cv2.__file__)
assert hasattr(cv2, "VideoCapture")
PY
