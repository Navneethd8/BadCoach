#!/usr/bin/env bash
# Start IsoCourt training inside a detached tmux session (survives SSH disconnect).
#
# Usage (repo root on EC2, after bootstrap):
#   export ISOCOURT_RSYNC_DEST="user@host:~/IsoCourt/"   # optional
#   export ISOCOURT_SHUTDOWN=1                          # optional: halt after success
#   export ISOCOURT_SHUTDOWN_ON_ERROR=1                 # optional: halt if training fails
#   ./scripts/ec2/run_train_tmux.sh MODEL [train script args...]
#
#   k_st_vit: v2 defaults conv3d vision + hit_centered, no early stopping; pass e.g.
#     --vision-backbone conv3d --resume-checkpoint backend/models/badminton_model_conv3d_pose.pth
#     --resume-k-st-vit backend/models/badminton_model_k_st_vit.pth --sampling hit_centered
#   (aliases: resnet50, full, conv3d_pose, skate)
#
# Env (export before running if you need them inside tmux):
#   ISOCOURT_TMUX_SESSION   tmux session name (default: isocourt-train)
#   ISOCOURT_TMUX_REPLACE   if 1, kill existing session with same name first
#   ISOCOURT_TRAIN_LOG      log file (default: repo/logs/train-UTC.log)
#   ISOCOURT_VENV           venv path (default: repo/.venv)
#   ISOCOURT_RSYNC_DEST     after success: rsync models + mlruns to this ssh path
#   ISOCOURT_RSYNC_EXTRA    extra rsync args (one string)
#   ISOCOURT_SHUTDOWN           if 1, sudo shutdown -h now after success
#   ISOCOURT_SHUTDOWN_ON_ERROR  if 1, sudo shutdown -h now when training fails (non-zero exit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${ISOCOURT_VENV:-${REPO_ROOT}/.venv}"
SESSION="${ISOCOURT_TMUX_SESSION:-isocourt-train}"
LOG="${ISOCOURT_TRAIN_LOG:-}"

MODEL_RAW="${1:?Usage: $0 MODEL [train script args...]}"
shift
# Lowercase without ${var,,} — macOS /bin/bash is 3.2 and rejects that expansion.
MODEL="$(printf '%s' "${MODEL_RAW}" | tr '[:upper:]' '[:lower:]')"

case "${MODEL}" in
  cnn_lstm|resnet50|full) TRAIN_SCRIPT="backend/pipelines/training/train_full.py" ;;
  conv3d|conv3d_pose) TRAIN_SCRIPT="backend/pipelines/training/train_conv3d.py" ;;
  timesformer) TRAIN_SCRIPT="backend/pipelines/training/train_timesformer.py" ;;
  k_st_vit|k-st-vit|kstvit|kinematic_st_vit)
    TRAIN_SCRIPT="backend/pipelines/training/train_k_st_vit.py"
    ;;
  jvc_no_xattn|jvc-no-xattn|jvc_no_cross_attn|jvc)
    TRAIN_SCRIPT="backend/pipelines/training/train_jvc_no_xattn.py"
    ;;
  bst_prep) TRAIN_SCRIPT="backend/pipelines/training/prepare_bst_finebadminton_collated.py" ;;
  bst_baseline|bst) TRAIN_SCRIPT="backend/pipelines/training/train_bst_baseline.py" ;;
  *) echo "Unknown MODEL=${MODEL_RAW}" >&2; exit 2 ;;
esac

if [[ ! -x "${VENV}/bin/python" && ! -x "${VENV}/bin/python3" ]]; then
  echo "Missing venv at ${VENV}; run scripts/ec2/bootstrap_ec2.sh first." >&2
  exit 1
fi
PYTHON_BIN="${VENV}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${VENV}/bin/python3"
fi

if [[ -z "${LOG}" ]]; then
  mkdir -p "${REPO_ROOT}/logs"
  LOG="${REPO_ROOT}/logs/train-$(date -u +%Y%m%dT%H%M%SZ).log"
else
  LOG="${LOG/#\~/$HOME}"
  mkdir -p "$(dirname "${LOG}")"
fi
LOG="$(cd "$(dirname "${LOG}")" && pwd)/$(basename "${LOG}")"

if [[ "${ISOCOURT_TMUX_REPLACE:-0}" == "1" ]]; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists. Attach: tmux attach -t ${SESSION}" >&2
  echo "Or re-run with ISOCOURT_TMUX_REPLACE=1" >&2
  exit 1
fi

INNER="${REPO_ROOT}/logs/.isocourt_tmux_inner.sh"
mkdir -p "${REPO_ROOT}/logs"

{
  echo "#!/usr/bin/env bash"
  echo "set -eo pipefail"
  echo "cd $(printf '%q' "${REPO_ROOT}")"
  echo "mkdir -p $(printf '%q' "$(dirname "${LOG}")")"
  echo "set +u"
  echo "source $(printf '%q' "${VENV}/bin/activate")"
  echo "set -u"
  echo "export PYTHONUNBUFFERED=1"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES}")"
  fi
  echo "export MLFLOW_TRACKING_URI=\"\${MLFLOW_TRACKING_URI:-file:$(printf '%q' "${REPO_ROOT}")/backend/mlruns}\""
  echo "echo \"=== \$(date -u -Iseconds) start ===\" | tee -a $(printf '%q' "${LOG}")"
  echo -n "$(printf '%q' "${PYTHON_BIN}") $(printf '%q' "${REPO_ROOT}/${TRAIN_SCRIPT}")"
  for a in "$@"; do echo -n " $(printf '%q' "$a")"; done
  echo " 2>&1 | tee -a $(printf '%q' "${LOG}")"
  echo "code=\${PIPESTATUS[0]}"
  echo "echo \"=== \$(date -u -Iseconds) exit \${code} ===\" | tee -a $(printf '%q' "${LOG}")"
  echo "if [[ \"\${code}\" -eq 0 ]]; then"
  echo "  if [[ -n \"\${ISOCOURT_RSYNC_DEST:-}\" ]]; then"
  echo "    rsync -avz --partial \${ISOCOURT_RSYNC_EXTRA:-} $(printf '%q' "${REPO_ROOT}/backend/models/") \"\${ISOCOURT_RSYNC_DEST%/}/backend/models/\" || true"
  echo "    if [[ -d $(printf '%q' "${REPO_ROOT}/backend/mlruns") ]]; then"
  echo "      rsync -avz --partial \${ISOCOURT_RSYNC_EXTRA:-} $(printf '%q' "${REPO_ROOT}/backend/mlruns/") \"\${ISOCOURT_RSYNC_DEST%/}/backend/mlruns/\" || true"
  echo "    elif [[ -d $(printf '%q' "${REPO_ROOT}/mlruns") ]]; then"
  echo "      rsync -avz --partial \${ISOCOURT_RSYNC_EXTRA:-} $(printf '%q' "${REPO_ROOT}/mlruns/") \"\${ISOCOURT_RSYNC_DEST%/}/mlruns/\" || true"
  echo "    fi"
  echo "  fi"
  echo "  if [[ \"\${ISOCOURT_SHUTDOWN:-0}\" == \"1\" ]]; then"
  echo "    sudo shutdown -h now"
  echo "  fi"
  echo "else"
  echo "  if [[ \"\${ISOCOURT_SHUTDOWN_ON_ERROR:-0}\" == \"1\" ]]; then"
  echo "    sudo shutdown -h now"
  echo "  fi"
  echo "  echo \"Training failed (exit \${code}). Log: $(printf '%q' "${LOG}")\" >&2"
  echo "  sleep 30"
  echo "  exit \"\${code}\""
  echo "fi"
} > "${INNER}"
chmod +x "${INNER}"

export ISOCOURT_RSYNC_DEST="${ISOCOURT_RSYNC_DEST:-}"
export ISOCOURT_RSYNC_EXTRA="${ISOCOURT_RSYNC_EXTRA:-}"
export ISOCOURT_SHUTDOWN="${ISOCOURT_SHUTDOWN:-0}"
export ISOCOURT_SHUTDOWN_ON_ERROR="${ISOCOURT_SHUTDOWN_ON_ERROR:-0}"

tmux new-session -ds "${SESSION}" "${INNER}"

echo "Started tmux session: ${SESSION}"
echo "Log file: ${LOG}"
echo "Attach: tmux attach -t ${SESSION}"
