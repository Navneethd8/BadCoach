#!/usr/bin/env bash
# Install MMPose for BST collate in a SEPARATE conda env.
#
# The main isocourt env (torch 2.10+cu128) has no reliable prebuilt mmcv wheels
# and source builds are fragile. Use this env only for bst_prep_mmpose; train
# TemPose/ST-GCN back in isocourt as usual.
#
# Usage (cluster GPU node, repo root):
#   chmod +x scripts/cluster/install_bst_mmpose.sh
#   ./scripts/cluster/install_bst_mmpose.sh
#
# Then:
#   export ISOCOURT_PYTHON="$HOME/miniconda3/envs/isocourt-mmpose/bin/python"
#   ./scripts/cluster/run_train_tmux.sh bst_prep_mmpose ...
#
# Env overrides:
#   ISOCOURT_MMPOSE_ENV     conda env name (default: isocourt-mmpose)
#   ISOCOURT_MMPOSE_PYTHON  python version (default: 3.10)
#   SKIP_CONDA_CREATE=1     reuse existing env (only pip install)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${ISOCOURT_MMPOSE_ENV:-isocourt-mmpose}"
PY_VER="${ISOCOURT_MMPOSE_PYTHON:-3.10}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"

die() { echo "ERROR: $*" >&2; exit 1; }

if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found — run on a GPU node."
fi

if ! command -v conda &>/dev/null; then
  if [[ -x "${CONDA_BASE}/bin/conda" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    die "conda not found; set CONDA_BASE or install miniconda."
  fi
fi

if [[ "${SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' already exists — reusing (set SKIP_CONDA_CREATE=1 to silence)."
  else
    echo "Creating conda env ${ENV_NAME} (python=${PY_VER}) ..."
    conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
  fi
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

PYTHON_EXE="$(which python)"
PY_MINOR="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_EXE}" != *"/envs/${ENV_NAME}/"* ]]; then
  die "Wrong python: ${PYTHON_EXE}. You are NOT in ${ENV_NAME}. Run: conda activate ${ENV_NAME}"
fi
if [[ "${PY_MINOR}" != "3.10" && "${PY_MINOR}" != "3.11" ]]; then
  die "Need python 3.10/3.11 in ${ENV_NAME}, got ${PY_MINOR} at ${PYTHON_EXE}"
fi
echo "Using ${PYTHON_EXE} (python ${PY_MINOR})"

python -m pip install --upgrade pip wheel Cython ninja psutil
# Keep setuptools new for most packages; pin old only inside xtcocotools source build.
python -m pip install --upgrade 'setuptools>=68'

echo "=== torch 2.1.0 + cu121 + numpy 1.26 (mmcv 2.1.0 wheels on torch2.1 index) ==="
python -m pip install --no-cache-dir 'numpy==1.26.4'
python -m pip install --no-cache-dir \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

python <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("torch", torch.__version__, "cuda", torch.version.cuda, "OK")
PY

install_xtcocotools() {
  echo "=== xtcocotools (numpy 1.26 + source build for torch 2.1 stack) ==="
  python -m pip uninstall -y xtcocotools 2>/dev/null || true
  python -m pip install --no-cache-dir 'numpy==1.26.4' 'setuptools==69.5.1' matplotlib Cython
  python -m pip install --no-cache-dir --no-build-isolation --no-binary xtcocotools \
    'xtcocotools==1.14.3'
  python -c "import numpy; import xtcocotools._mask; print('xtcocotools OK, numpy', numpy.__version__)"
  python -m pip install --upgrade 'setuptools>=68'
}

install_xtcocotools

echo "=== base deps (headless opencv — cluster has no libGL) ==="
# opencv 5.x requires numpy>=2; torch 2.1 + mmcv need numpy 1.26.
python -m pip install --no-cache-dir 'numpy==1.26.4'
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null || true
python -m pip install --no-cache-dir 'opencv-python-headless==4.9.0.80' tqdm pandas matplotlib scipy
python -c "import numpy; import cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)"

echo "=== openmim + mmengine ==="
python -m pip install --no-cache-dir openmim mmengine

install_mmcv() {
  echo "=== mmcv 2.1.0 ==="
  python -m pip uninstall -y mmcv 2>/dev/null || true

  PY_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-${PY_TAG}-${PY_TAG}-manylinux1_x86_64.whl"
  echo "Trying prebuilt wheel: ${WHEEL_URL}"
  python -m pip install --no-cache-dir "${WHEEL_URL}" 2>/dev/null \
    || python -m pip install --no-cache-dir --only-binary mmcv mmcv==2.1.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

  if python - <<'PY' 2>/dev/null
import torch
from mmcv.ops import nms
boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
nms(boxes, scores, 0.5)
print("mmcv CUDA ops OK")
PY
  then
    python -c "import mmcv; print('mmcv', mmcv.__version__, '(prebuilt wheel)')"
    return 0
  fi

  echo "Prebuilt mmcv CUDA ops failed on this GPU — compiling mmcv 2.1.0 from source (~20-40 min) ..."
  python -m pip uninstall -y mmcv 2>/dev/null || true
  rm -rf /tmp/mmcv
  git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv
  (
    cd /tmp/mmcv
    export FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS="${MAX_JOBS:-8}"
    python -m pip install -r requirements/runtime.txt
    python -m pip install -v --no-build-isolation .
  )
  python -c "import mmcv; print('mmcv', mmcv.__version__, '(source build)')"
}

install_mmcv

install_mmpose() {
  echo "=== mmdet + mmpose (skip broken chumpy — not needed for inference) ==="
  python -m pip install --no-cache-dir "mmdet>=3.2.0,<3.4.0"

  # chumpy 0.70 is abandoned and breaks pip on py3.10+. mmpose still lists it but
  # MMPoseInferencer / RTMPose collate does not import it.
  if python -m pip install --no-cache-dir "mmpose>=1.3.0,<1.4.0"; then
    echo "mmpose installed (with deps)"
    return 0
  fi

  echo "mmpose pip failed (usually chumpy); installing --no-deps ..."
  python -m pip install --no-cache-dir "mmpose>=1.3.0,<1.4.0" --no-deps
  python -m pip install --no-cache-dir \
    json-tricks munkres pyyaml yapf rich termcolor colorama
}

install_mmpose

# mmdet/mmpose may reinstall the broken xtcocotools wheel — rebuild again.
install_xtcocotools

echo "=== re-pin mmpose env (numpy 1.26 + headless opencv) ==="
"${SCRIPT_DIR}/ensure_mmpose_env.sh"

echo "=== verify MMPoseInferencer ==="
python -c "from mmpose.apis import MMPoseInferencer; print('mmpose OK')"

ENV_PYTHON="$(conda run -n "${ENV_NAME}" which python)"
cat <<EOF

Done. MMPose env: ${ENV_NAME}
Python: ${ENV_PYTHON}

Run collate prep (from ~/IsoCourt):

  export ISOCOURT_PYTHON="${ENV_PYTHON}"
  export CUDA_VISIBLE_DEVICES=0
  export ISOCOURT_TMUX_SESSION=isocourt-bst-prep-mmpose-16
  export ISOCOURT_TMUX_REPLACE=1
  ./scripts/cluster/run_train_tmux.sh bst_prep_mmpose \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output-dir backend/data/bst_finebadminton_collated_mmpose_16 \\
    --sequence-length 16 \\
    --pose-style JnB_bone

Train TemPose/ST-GCN in isocourt as usual (output .npy is env-agnostic).
EOF
