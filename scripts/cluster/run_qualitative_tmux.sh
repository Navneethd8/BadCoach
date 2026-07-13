#!/usr/bin/env bash
# Generate qualitative val figures in a detached tmux session (survives SSH disconnect).
#
# Usage (repo root on cluster, after rsync_push_code):
#   ./scripts/cluster/run_qualitative_tmux.sh jvc
#   ./scripts/cluster/run_qualitative_tmux.sh --gpu 0 jvc
#   ./scripts/cluster/run_qualitative_tmux.sh --gpu 1 jvc_no_xattn_4stream
#   ./scripts/cluster/run_qualitative_parallel.sh          # one model per GPU (0,1,2)
#
# Models:
#   jvc | k_st_vit              -> badminton_model_k_st_vit.pth
#   jvc_no_xattn_4stream        -> badminton_model_jvc_no_xattn_20260705T041121Z.pth
#   jvc_no_xattn_1stream        -> badminton_model_jvc_no_xattn_20260706T044917Z.pth (3ch / single stream)
#   all                         -> run all three sequentially on one GPU
#
# Env:
#   ISOCOURT_TMUX_SESSION   override tmux name (default: isocourt-qual-MODEL[-gpuN])
#   ISOCOURT_TMUX_REPLACE   if 1, kill existing session with same name first
#   ISOCOURT_QUAL_LOG       log file (default: logs/qualitative-MODEL[-gpuN]-UTC.log)
#   ISOCOURT_VENV           default: repo/.venv
#   ISOCOURT_PYTHON         default: ${ISOCOURT_VENV}/bin/python
#   CUDA_VISIBLE_DEVICES    GPU index (same as --gpu)
#   ISOCOURT_GPU            alias for --gpu when flag omitted
#   ISOCOURT_QUAL_EXTRA     override figure args (default: 2 correct, 1 error, 16 frames, crop)
#   ISOCOURT_RSYNC_DEST     after success: rsync docs/figures/qualitative_* to laptop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${ISOCOURT_VENV:-${REPO_ROOT}/.venv}"
PYTHON="${ISOCOURT_PYTHON:-${VENV}/bin/python}"
LOG="${ISOCOURT_QUAL_LOG:-}"
FIG_SCRIPT="${REPO_ROOT}/docs/figures/generate_qualitative_predictions.py"
POSE_CACHE="${REPO_ROOT}/backend/models/pose_cache_span_linspace.pt"
BATCH_SIZE="${ISOCOURT_QUAL_BATCH_SIZE:-16}"
QUAL_EXTRA=(--num-correct 2 --num-errors 2 --contact-frames 16 --frame-stride 4 --cell-size 220 --figure-scale 0.68 --faithful)
if [[ -n "${ISOCOURT_QUAL_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  QUAL_EXTRA=(${ISOCOURT_QUAL_EXTRA})
fi
GPU="${CUDA_VISIBLE_DEVICES:-${ISOCOURT_GPU:-}}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|-g)
      GPU="${2:?--gpu requires an index (e.g. 0, 1, 2)}"
      shift 2
      ;;
    --batch-size|-b)
      BATCH_SIZE="${2:?--batch-size requires a value}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,24p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "Usage: $0 [--gpu N] MODEL" >&2
  echo "  MODEL: jvc | jvc_no_xattn_4stream | jvc_no_xattn_1stream | all" >&2
  exit 2
fi

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

MODEL_RAW="${POSITIONAL[0]}"
MODEL="$(printf '%s' "${MODEL_RAW}" | tr '[:upper:]' '[:lower:]')"

if [[ -z "${ISOCOURT_TMUX_SESSION:-}" ]]; then
  SESSION="isocourt-qual-${MODEL}"
  if [[ -n "${GPU}" ]]; then
    SESSION="${SESSION}-gpu${GPU}"
  fi
else
  SESSION="${ISOCOURT_TMUX_SESSION}"
fi

_run_one() {
  local ckpt="$1"
  local label="$2"
  local out="$3"
  printf '%q ' "${PYTHON}" "${FIG_SCRIPT}" \
    "--checkpoint" "${ckpt}" \
    "--pose-cache" "${POSE_CACHE}" \
    "--model-label" "${label}" \
    "--out" "${out}" \
    "--batch-size" "${BATCH_SIZE}" \
    "${QUAL_EXTRA[@]}"
  echo
}

case "${MODEL}" in
  jvc|k_st_vit|k-st-vit|jvc_full)
    INNER_CMDS="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_k_st_vit.pth" \
      "JVC (K-STViT, 80.6%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_val.png")"
    ;;
  jvc_no_xattn_4stream|jvc-no-xattn-4stream|no_xattn_4stream)
    INNER_CMDS="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_jvc_no_xattn_20260705T041121Z.pth" \
      "JVC no-xattn four-stream (79.9%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_no_xattn_4stream.png")"
    ;;
  jvc_no_xattn_1stream|jvc-no-xattn-3ch|jvc-no-xattn-1stream|no_xattn_1stream|3ch)
    INNER_CMDS="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_jvc_no_xattn_20260706T044917Z.pth" \
      "JVC no-xattn single-stream (80.5%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_no_xattn_1stream.png")"
    ;;
  all)
    INNER_CMDS=""
    INNER_CMDS+="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_k_st_vit.pth" \
      "JVC (K-STViT, 80.6%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_val.png")"
    INNER_CMDS+="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_jvc_no_xattn_20260705T041121Z.pth" \
      "JVC no-xattn four-stream (79.9%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_no_xattn_4stream.png")"
    INNER_CMDS+="$(_run_one \
      "${REPO_ROOT}/backend/models/badminton_model_jvc_no_xattn_20260706T044917Z.pth" \
      "JVC no-xattn single-stream (80.5%)" \
      "${REPO_ROOT}/docs/figures/qualitative_jvc_no_xattn_1stream.png")"
    ;;
  *)
    echo "Unknown MODEL=${MODEL_RAW}" >&2
    echo "Use: jvc | jvc_no_xattn_4stream | jvc_no_xattn_1stream | all" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  fi
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing python at ${ISOCOURT_PYTHON:-${VENV}/bin/python}; set ISOCOURT_VENV or ISOCOURT_PYTHON." >&2
  echo "Example: export ISOCOURT_PYTHON=\$(which python)  # with conda activate isocourt" >&2
  exit 1
fi

if [[ ! -f "${FIG_SCRIPT}" ]]; then
  echo "Missing ${FIG_SCRIPT}; run ./scripts/cluster/rsync_push_code.sh from laptop." >&2
  exit 1
fi

if [[ ! -f "${POSE_CACHE}" ]]; then
  echo "Missing pose cache: ${POSE_CACHE}" >&2
  exit 1
fi

if [[ -z "${LOG}" ]]; then
  mkdir -p "${REPO_ROOT}/logs"
  LOG_STEM="qualitative-${MODEL}"
  if [[ -n "${GPU}" ]]; then
    LOG_STEM="${LOG_STEM}-gpu${GPU}"
  fi
  LOG="${REPO_ROOT}/logs/${LOG_STEM}-$(date -u +%Y%m%dT%H%M%SZ).log"
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

INNER="${REPO_ROOT}/logs/.isocourt_qual_${SESSION}.sh"
mkdir -p "${REPO_ROOT}/logs"

{
  echo "#!/usr/bin/env bash"
  echo "set -eo pipefail"
  echo "cd $(printf '%q' "${REPO_ROOT}")"
  echo "export PYTHONUNBUFFERED=1"
  echo "export PYTHONPATH=$(printf '%q' "${REPO_ROOT}/backend"):$(printf '%q' "${REPO_ROOT}/docs/figures"):\${PYTHONPATH:-}"
  GL_ENV="${REPO_ROOT}/scripts/cluster/mediapipe_gl.env"
  if [[ -f "${GL_ENV}" ]]; then
    echo "set -a"
    echo "# shellcheck disable=SC1090"
    echo "source $(printf '%q' "${GL_ENV}")"
    echo "set +a"
    echo "echo \"Loaded mediapipe_gl.env for live pose\" | tee -a $(printf '%q' "${LOG}")"
  else
    echo "echo \"WARNING: mediapipe_gl.env missing — run ./scripts/cluster/setup_mediapipe_gl.sh for live pose\" | tee -a $(printf '%q' "${LOG}")"
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES}")"
  fi
  echo "code=0"
  echo "echo \"=== \$(date -u -Iseconds) start ===\" | tee $(printf '%q' "${LOG}")"
  echo "echo \"python=$(printf '%q' "${PYTHON}")\" | tee -a $(printf '%q' "${LOG}")"
  echo "$(printf '%q' "${PYTHON}") --version 2>&1 | tee -a $(printf '%q' "${LOG}")"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\" | tee -a $(printf '%q' "${LOG}")"
    echo "nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null | tee -a $(printf '%q' "${LOG}") || true"
  fi
  while IFS= read -r cmd; do
    [[ -z "${cmd}" ]] && continue
    qcmd="$(printf '%q' "${cmd}")"
    echo "echo \"--- ${qcmd} ---\" | tee -a $(printf '%q' "${LOG}")"
    echo "eval ${qcmd} 2>&1 | tee -a $(printf '%q' "${LOG}")"
    echo "rc=\${PIPESTATUS[0]}"
    echo "[[ \"\${rc}\" -ne 0 ]] && code=\${rc}"
  done <<EOF
${INNER_CMDS}
EOF
  echo "echo \"=== \$(date -u -Iseconds) exit \${code} ===\" | tee -a $(printf '%q' "${LOG}")"
  echo "if [[ \"\${code}\" -eq 0 ]]; then"
  echo "  ls -lh $(printf '%q' "${REPO_ROOT}/docs/figures/")/qualitative_* 2>/dev/null | tee -a $(printf '%q' "${LOG}") || true"
  echo "  if [[ -n \"\${ISOCOURT_RSYNC_DEST:-}\" ]]; then"
  echo "    rsync -avz --partial \${ISOCOURT_RSYNC_EXTRA:-} \\"
  echo "      $(printf '%q' "${REPO_ROOT}/docs/figures/")/qualitative_*.png \\"
  echo "      $(printf '%q' "${REPO_ROOT}/docs/figures/")/qualitative_*.pdf \\"
  echo "      \"\${ISOCOURT_RSYNC_DEST%/}/docs/figures/\" || true"
  echo "  fi"
  echo "else"
  echo "  echo \"Qualitative generation failed (exit \${code}). Log: $(printf '%q' "${LOG}")\" >&2"
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
echo "Pose cache: ${POSE_CACHE}"
if [[ -n "${GPU}" ]]; then
  echo "GPU: ${GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
fi
