#!/usr/bin/env bash
# Start model test eval in a detached tmux session (no MLflow).
#
# Skeleton (collated npy):  bst, tempose, stgcn
# Native (video + pose cache): cnn_lstm, conv3d, timesformer, jvc
#
# Usage (repo root on cluster):
#   export ISOCOURT_VENV="$HOME/miniconda3/envs/isocourt"
#   export CUDA_VISIBLE_DEVICES=3
#   export ISOCOURT_CHECKPOINT=backend/models/badminton_model_conv3d_pose_....pth
#   export ISOCOURT_TMUX_SESSION=isocourt-conv3d-test
#   export ISOCOURT_TMUX_REPLACE=1
#   ./scripts/cluster/run_eval_tmux.sh conv3d --per-class
#
# Env:
#   ISOCOURT_TMUX_SESSION    tmux session name (default: isocourt-eval-MODEL)
#   ISOCOURT_TMUX_REPLACE    if 1, kill existing session first
#   ISOCOURT_EVAL_LOG        log file (default: logs/eval-MODEL-UTC.log)
#   ISOCOURT_VENV / ISOCOURT_PYTHON
#   ISOCOURT_CHECKPOINT      checkpoint .pth (required unless --checkpoint passed)
#   ISOCOURT_COLLATED_ROOT   skeleton only (default: bst_finebadminton_collated_mmpose_16)
#   CUDA_VISIBLE_DEVICES     GPU index (e.g. 3 for 4th GPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${ISOCOURT_VENV:-${REPO_ROOT}/.venv}"
PYTHON="${ISOCOURT_PYTHON:-${VENV}/bin/python}"
COLLATED_DEFAULT="${ISOCOURT_COLLATED_ROOT:-backend/data/bst_finebadminton_collated_mmpose_16}"

MODEL_RAW="${1:?Usage: $0 MODEL [eval args...]}"
shift
MODEL="$(printf '%s' "${MODEL_RAW}" | tr '[:upper:]' '[:lower:]')"

SKELETON=0
case "${MODEL}" in
  bst|bst_baseline) MODEL="bst"; SKELETON=1 ;;
  tempose|tempose_baseline) MODEL="tempose"; SKELETON=1 ;;
  stgcn|stgcn_baseline) MODEL="stgcn"; SKELETON=1 ;;
  cnn_lstm|full|resnet50) MODEL="cnn_lstm" ;;
  conv3d|conv3d_pose) MODEL="conv3d" ;;
  timesformer) MODEL="timesformer" ;;
  jvc|k_st_vit|k-st-vit|kstvit|kinematic_st_vit) MODEL="jvc" ;;
  *)
    echo "Unknown MODEL=${MODEL_RAW}" >&2
    echo "Skeleton: bst, tempose, stgcn" >&2
    echo "Native: cnn_lstm, conv3d, timesformer, jvc" >&2
    exit 2
    ;;
esac

if [[ "${SKELETON}" == "1" ]]; then
  EVAL_SCRIPT="backend/scripts/eval_skeleton_baseline_checkpoint.py"
  CHECKPOINT="${ISOCOURT_CHECKPOINT:-backend/models/badminton_model_${MODEL}.pth}"
else
  EVAL_SCRIPT="backend/scripts/eval_native_baseline_checkpoint.py"
  case "${MODEL}" in
    cnn_lstm) CHECKPOINT="${ISOCOURT_CHECKPOINT:-backend/models/badminton_model_cnn_lstm.pth}" ;;
    conv3d) CHECKPOINT="${ISOCOURT_CHECKPOINT:-backend/models/badminton_model_conv3d_pose.pth}" ;;
    timesformer) CHECKPOINT="${ISOCOURT_CHECKPOINT:-backend/models/badminton_model_timesformer.pth}" ;;
    jvc) CHECKPOINT="${ISOCOURT_CHECKPOINT:-backend/models/badminton_model_jvc.pth}" ;;
  esac
fi

SESSION="${ISOCOURT_TMUX_SESSION:-isocourt-eval-${MODEL}}"
LOG="${ISOCOURT_EVAL_LOG:-}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing python at ${PYTHON}; set ISOCOURT_VENV or ISOCOURT_PYTHON." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/${EVAL_SCRIPT}" ]]; then
  echo "Missing ${EVAL_SCRIPT}; rsync latest code to the cluster." >&2
  exit 1
fi

has_arg() {
  local flag="$1"
  shift
  for a in "$@"; do
    if [[ "${a}" == "${flag}" ]]; then
      return 0
    fi
  done
  return 1
}

_RESOLVED_CKPT="${CHECKPOINT}"
if has_arg --checkpoint "$@"; then
  _next=0
  for a in "$@"; do
    if [[ "${_next}" == "1" ]]; then
      _RESOLVED_CKPT="${a}"
      break
    fi
    if [[ "${a}" == "--checkpoint" ]]; then
      _next=1
    fi
  done
fi
if [[ ! -f "${REPO_ROOT}/${_RESOLVED_CKPT}" ]]; then
  echo "Checkpoint not found: ${REPO_ROOT}/${_RESOLVED_CKPT}" >&2
  echo "Set ISOCOURT_CHECKPOINT or pass --checkpoint explicitly." >&2
  exit 1
fi

EVAL_ARGS=(--model "${MODEL}" --split test)
if ! has_arg --checkpoint "$@"; then
  EVAL_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${SKELETON}" == "1" ]] && ! has_arg --collated-root "$@"; then
  EVAL_ARGS+=(--collated-root "${COLLATED_DEFAULT}")
fi
EVAL_ARGS+=("$@")

if [[ -z "${LOG}" ]]; then
  mkdir -p "${REPO_ROOT}/logs"
  LOG="${REPO_ROOT}/logs/eval-${MODEL}-$(date -u +%Y%m%dT%H%M%SZ).log"
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

if [[ "${SKELETON}" == "1" ]]; then
  SPLIT_DESC="split=test (held-out test clips from collated npy)"
else
  SPLIT_DESC="split=test (video-level 70/10/20 held-out test videos)"
fi

INNER="${REPO_ROOT}/logs/.isocourt_eval_tmux_inner.sh"
mkdir -p "${REPO_ROOT}/logs"

{
  echo "#!/usr/bin/env bash"
  echo "set -eo pipefail"
  echo "cd $(printf '%q' "${REPO_ROOT}")"
  echo "mkdir -p $(printf '%q' "$(dirname "${LOG}")")"
  echo "export PYTHONUNBUFFERED=1"
  echo "export PYTHONPATH=$(printf '%q' "${REPO_ROOT}/backend"):\${PYTHONPATH:-}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES}")"
  fi
  echo "echo \"=== \$(date -u -Iseconds) test eval start (${MODEL}) ===\" | tee -a $(printf '%q' "${LOG}")"
  echo "echo \"python=$(printf '%q' "${PYTHON}")\" | tee -a $(printf '%q' "${LOG}")"
  echo "echo \"${SPLIT_DESC}\" | tee -a $(printf '%q' "${LOG}")"
  echo "echo \"checkpoint=$(printf '%q' "${_RESOLVED_CKPT}")\" | tee -a $(printf '%q' "${LOG}")"
  echo -n "$(printf '%q' "${PYTHON}") $(printf '%q' "${REPO_ROOT}/${EVAL_SCRIPT}")"
  for a in "${EVAL_ARGS[@]}"; do echo -n " $(printf '%q' "$a")"; done
  echo " 2>&1 | tee -a $(printf '%q' "${LOG}")"
  echo "code=\${PIPESTATUS[0]}"
  echo "echo \"=== \$(date -u -Iseconds) exit \${code} ===\" | tee -a $(printf '%q' "${LOG}")"
  echo "if [[ \"\${code}\" -ne 0 ]]; then"
  echo "  echo \"Eval failed (exit \${code}). Log: $(printf '%q' "${LOG}")\" >&2"
  echo "  sleep 30"
  echo "  exit \"\${code}\""
  echo "fi"
} > "${INNER}"
chmod +x "${INNER}"

tmux new-session -ds "${SESSION}" "${INNER}"

echo "Started tmux session: ${SESSION}"
echo "Log file: ${LOG}"
echo "Checkpoint: ${_RESOLVED_CKPT}"
if [[ "${SKELETON}" == "1" ]]; then
  echo "Collated root: ${COLLATED_DEFAULT}"
fi
echo "Split: test"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "Attach: tmux attach -t ${SESSION}"
