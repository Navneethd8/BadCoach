#!/usr/bin/env bash
# Force opencv-python-headless in the active conda env (nd17 has no libGL).
#
# mmcv/mmdet/mmpose often reinstall opencv-python, which breaks cv2 import with:
#   ImportError: libGL.so.1: cannot open shared object file
#
# Usage (after conda activate isocourt-mmpose):
#   ./scripts/cluster/ensure_opencv_headless.sh

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate your conda env first, e.g.: conda activate isocourt-mmpose" >&2
  exit 1
fi

OPENCV_VER="${ISOCOURT_OPENCV_HEADLESS_VERSION:-4.9.0.80}"

python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null || true
# Stale cv2 dirs survive pip uninstall and shadow a clean reinstall.
rm -rf "${CONDA_PREFIX}/lib/python3."*/site-packages/cv2 \
       "${CONDA_PREFIX}/lib/python3."*/site-packages/opencv_python*.dist-info \
       "${CONDA_PREFIX}/lib/python3."*/site-packages/opencv_python_headless*.dist-info 2>/dev/null || true

python -m pip install --no-cache-dir "opencv-python-headless==${OPENCV_VER}"

python - <<'PY'
import cv2
assert hasattr(cv2, "COLOR_BGR2RGB"), f"broken cv2 at {cv2.__file__}"
assert hasattr(cv2, "VideoCapture")
ver = getattr(cv2, "__version__", None) or getattr(cv2, "getVersionString", lambda: "?")()
print("cv2 OK", ver, "(headless)", cv2.__file__)
PY
