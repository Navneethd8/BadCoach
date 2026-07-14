#!/usr/bin/env bash
# Cluster GPU node (e.g. nd17): venv + CUDA PyTorch + backend/requirements.txt.
# Assumes NVIDIA drivers are already installed by the host admins.
#
# Usage (from repo clone on the cluster):
#   chmod +x scripts/cluster/bootstrap_cluster.sh
#   ./scripts/cluster/bootstrap_cluster.sh
#
# Env:
#   TORCH_CUDA   PyTorch wheel tag: cu121 (default) or cu124
#   SKIP_APT=1   Skip apt install (you already installed GL/EGL deps)
#   VENV_DIR     Default: <repo>/.venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"

die() { echo "ERROR: $*" >&2; exit 1; }

if [[ ! -f "${REPO_ROOT}/backend/requirements.txt" ]]; then
  die "backend/requirements.txt not found; REPO_ROOT=${REPO_ROOT}"
fi

if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found. Ask cluster admins to install NVIDIA drivers on this node."
fi

driver_major() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1
}

DM="$(driver_major)"
[[ -n "${DM}" ]] || die "Could not read NVIDIA driver version from nvidia-smi"

case "${TORCH_CUDA}" in
  cu121) MIN_D=525 ;;
  cu124) MIN_D=550 ;;
  *) die "Unsupported TORCH_CUDA=${TORCH_CUDA} (use cu121 or cu124)" ;;
esac

if (( DM < MIN_D )); then
  die "Driver ${DM}.x is below minimum ${MIN_D} for PyTorch ${TORCH_CUDA}. Set TORCH_CUDA to match your driver."
fi

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if [[ -f /etc/debian_version ]]; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
      python3-venv python3-pip \
      libgl1 libgles2 libegl1 libegl-mesa0 \
      libglib2.0-0 libsm6 libxext6 libxrender-dev \
      build-essential libffi-dev \
      tmux rsync
  elif [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    . /etc/os-release
    if [[ "${ID:-}" == "amzn" ]]; then
      sudo dnf install -y python3 python3-pip tmux rsync gcc gcc-c++ make python3-devel \
        mesa-libGL mesa-libEGL libgomp || true
    else
      echo "WARN: Non-Debian OS (${ID:-unknown}); skipping package install. Install python3, tmux, rsync, and OpenCV/MediaPipe deps manually."
    fi
  else
    echo "WARN: Unknown OS; skipping package install. Install python3-venv, tmux, rsync, and OpenCV/MediaPipe system libs manually."
  fi
fi

python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "Installing torch/vision/audio from ${TORCH_INDEX} ..."
python -m pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX}"

python -m pip install --no-cache-dir -r "${REPO_ROOT}/backend/requirements.txt"

echo "Verifying CUDA from Python..."
python <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False (wrong torch build or driver?)")
x = torch.randn(1, device="cuda")
print("device tensor OK:", x.device)
PY

python <<'PY'
import cv2  # noqa: F401
import mediapipe  # noqa: F401
print("opencv + mediapipe import OK")
PY

if [[ "${INSTALL_BST_MMPOSE:-1}" == "1" && -f "${REPO_ROOT}/backend/requirements-bst-mmpose.txt" ]]; then
  echo "Installing BST MMPose stack (requirements-bst-mmpose.txt + mmcv via mim) ..."
  python -m pip install --no-cache-dir -r "${REPO_ROOT}/backend/requirements-bst-mmpose.txt"
  python -m pip install --no-cache-dir --force-reinstall xtcocotools
  mim install mmcv
  python <<'PY'
from mmpose.apis import MMPoseInferencer  # noqa: F401
print("mmpose OK")
PY
fi

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
