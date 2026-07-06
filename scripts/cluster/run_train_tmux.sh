#!/usr/bin/env bash
# Start IsoCourt training inside a detached tmux session (survives SSH disconnect).
#
# Usage (repo root on cluster, after bootstrap_cluster):
#   export ISOCOURT_RSYNC_DEST="user@host:~/IsoCourt/"   # optional post-success pull
#   ./scripts/cluster/run_train_tmux.sh MODEL [train script args...]
#
#   k_st_vit: v2 defaults conv3d vision + hit_centered, no early stopping; pass e.g.
#     --vision-backbone conv3d --resume-checkpoint backend/models/badminton_model_conv3d_pose.pth
#     --resume-k-st-vit backend/models/badminton_model_k_st_vit.pth --sampling hit_centered
#   (aliases: resnet50, full, conv3d_pose, sttr, official_st_tr, st_tr_official, st_tr_collate, skate)
#
# Env (export before running if you need them inside tmux):
#   ISOCOURT_TMUX_SESSION   tmux session name (default: isocourt-train)
#   ISOCOURT_TMUX_REPLACE   if 1, kill existing session with same name first
#   ISOCOURT_TRAIN_LOG      log file (default: repo/logs/train-UTC.log)
#   ISOCOURT_VENV           venv or conda env dir (default: repo/.venv)
#   ISOCOURT_PYTHON         python binary (default: ${ISOCOURT_VENV}/bin/python)
#   ISOCOURT_DISABLE_MLFLOW skip MLflow (default 1 on cluster; set 0 to enable)
#   CUDA_VISIBLE_DEVICES      GPU index for this session (e.g. 1 or 2)
#   ISOCOURT_RSYNC_DEST     after success: rsync models + mlruns to this ssh path
#   ISOCOURT_RSYNC_EXTRA    extra rsync args (one string)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${ISOCOURT_VENV:-${REPO_ROOT}/.venv}"
PYTHON="${ISOCOURT_PYTHON:-${VENV}/bin/python}"
SESSION="${ISOCOURT_TMUX_SESSION:-isocourt-train}"
LOG="${ISOCOURT_TRAIN_LOG:-}"
DISABLE_MLFLOW="${ISOCOURT_DISABLE_MLFLOW:-1}"

MODEL_RAW="${1:?Usage: $0 MODEL [train script args...]}"
shift
MODEL="$(printf '%s' "${MODEL_RAW}" | tr '[:upper:]' '[:lower:]')"

case "${MODEL}" in
  cnn_lstm|resnet50|full) TRAIN_SCRIPT="backend/pipelines/training/train_full.py" ;;
  conv3d|conv3d_pose) TRAIN_SCRIPT="backend/pipelines/training/train_conv3d.py" ;;
  timesformer) TRAIN_SCRIPT="backend/pipelines/training/train_timesformer.py" ;;
  st_tr|sttr) TRAIN_SCRIPT="backend/pipelines/training/train_st_tr.py" ;;
  gcn_st_tr|official_st_tr|st_tr_official)
    TRAIN_SCRIPT="backend/pipelines/training/train_gcn_st_tr.py"
    ;;
  st_tr_vit|st_tr_vit_fusion)
    TRAIN_SCRIPT="backend/pipelines/training/train_st_tr_vit.py"
    ;;
  skateformer|skate|skate_former)
    TRAIN_SCRIPT="backend/pipelines/training/train_skateformer.py"
    ;;
  skateformer_b|skateformer-b|skate_b|skateformer_b_fusion)
    TRAIN_SCRIPT="backend/pipelines/training/train_skateformer_b.py"
    ;;
  gv_xattn|gv-xattn|gv_x_attn|graph_vision_xattn)
    TRAIN_SCRIPT="backend/pipelines/training/train_gv_xattn.py"
    ;;
  k_st_vit|k-st-vit|kstvit|kinematic_st_vit)
    TRAIN_SCRIPT="backend/pipelines/training/train_k_st_vit.py"
    ;;
  jvc_no_xattn|jvc-no-xattn|jvc_no_cross_attn|jvc)
    TRAIN_SCRIPT="backend/pipelines/training/train_jvc_no_xattn.py"
    ;;
  qwen3_vl_8b|qwen3-vl-8b|qwen_vl_8b|qwen8b_vlm)
    TRAIN_SCRIPT="backend/pipelines/vlm/qwen-8b/train_qwen3_vl_8b.py"
    ;;
  st_tr_prep|st_tr_collate|st_tr_collated)
    TRAIN_SCRIPT="backend/pipelines/training/prepare_st_tr_collated.py"
    ;;
  bst_prep) TRAIN_SCRIPT="backend/pipelines/training/prepare_bst_finebadminton_collated.py" ;;
  bst_baseline|bst) TRAIN_SCRIPT="backend/pipelines/training/train_bst_baseline.py" ;;
  *) echo "Unknown MODEL=${MODEL_RAW}" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing python at ${PYTHON}; set ISOCOURT_VENV (conda env dir) or ISOCOURT_PYTHON." >&2
  echo "Example: export ISOCOURT_VENV=\$HOME/miniconda3/envs/isocourt" >&2
  exit 1
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
  echo "export PYTHONUNBUFFERED=1"
  echo "export PYTHONPATH=$(printf '%q' "${REPO_ROOT}/backend"):\${PYTHONPATH:-}"
  echo "export ISOCOURT_DISABLE_MLFLOW=$(printf '%q' "${DISABLE_MLFLOW}")"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES}")"
  fi
  if [[ "${DISABLE_MLFLOW}" != "1" && "${DISABLE_MLFLOW,,}" != "true" && "${DISABLE_MLFLOW,,}" != "yes" ]]; then
    echo "export MLFLOW_TRACKING_URI=\"\${MLFLOW_TRACKING_URI:-file:$(printf '%q' "${REPO_ROOT}")/backend/mlruns}\""
  fi
  echo "echo \"=== \$(date -u -Iseconds) start ===\" | tee -a $(printf '%q' "${LOG}")"
  echo "echo \"python=$(printf '%q' "${PYTHON}")\" | tee -a $(printf '%q' "${LOG}")"
  echo -n "$(printf '%q' "${PYTHON}") $(printf '%q' "${REPO_ROOT}/${TRAIN_SCRIPT}")"
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
  echo "else"
  echo "  echo \"Training failed (exit \${code}). Log: $(printf '%q' "${LOG}")\" >&2"
  echo "  sleep 30"
  echo "  exit \"\${code}\""
  echo "fi"
} > "${INNER}"
chmod +x "${INNER}"

export ISOCOURT_RSYNC_DEST="${ISOCOURT_RSYNC_DEST:-}"
export ISOCOURT_RSYNC_EXTRA="${ISOCOURT_RSYNC_EXTRA:-}"

tmux new-session -ds "${SESSION}" "${INNER}"

echo "Started tmux session: ${SESSION}"
echo "Log file: ${LOG}"
echo "Attach: tmux attach -t ${SESSION}"
