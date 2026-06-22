#!/usr/bin/env bash
# Prepare 16-frame VLM JSONL + optional Qwen3-VL-8B LoRA train on cluster.
#
# From repo root (after bootstrap + data symlink):
#   ./scripts/cluster/prepare_vlm_16frame.sh
#   ./scripts/cluster/run_train_tmux.sh qwen3_vl_8b
#
# Uses conda when active (CONDA_PREFIX), or ISOCOURT_PYTHON / ISOCOURT_VENV, else .venv.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_resolve_python() {
  if [[ -n "${ISOCOURT_PYTHON:-}" ]]; then
    printf '%s\n' "${ISOCOURT_PYTHON}"
    return
  fi
  local venv="${ISOCOURT_VENV:-${REPO_ROOT}/.venv}"
  if [[ -x "${venv}/bin/python" ]]; then
    printf '%s\n' "${venv}/bin/python"
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "No python found. Activate conda (conda activate isocourt) or set ISOCOURT_PYTHON." >&2
  exit 1
}

PYTHON="$(_resolve_python)"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not executable: ${PYTHON}" >&2
  echo "Example: export ISOCOURT_PYTHON=\$(which python)  # with conda activate isocourt" >&2
  exit 1
fi
echo "Using python: ${PYTHON}" >&2

export PYTHONPATH="${REPO_ROOT}/backend${PYTHONPATH:+:${PYTHONPATH}}"

JSONL="${REPO_ROOT}/backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl"

echo "=== Extract 16-frame JPEGs (if needed) ==="
"${PYTHON}" "${REPO_ROOT}/backend/pipelines/vlm/common/prepare_finebadminton_20k.py" \
  --skip-download --extract-training-frames

echo "=== Build 16-frame JSONL ==="
"${PYTHON}" "${REPO_ROOT}/backend/pipelines/vlm/common/build_finebadminton_jsonl.py" \
  --mode 16frame \
  --data-root "${REPO_ROOT}/backend/data" \
  --output "${JSONL}"

if [[ ! -f "${REPO_ROOT}/backend/models/pose_cache_span_linspace.pt" ]]; then
  echo "WARNING: pose_cache_span_linspace.pt missing. Build via e.g.:" >&2
  echo "  ${PYTHON} ${REPO_ROOT}/backend/pipelines/training/build_full_pose_cache.py \\" >&2
  echo "    --data-root ${REPO_ROOT}/backend/data \\" >&2
  echo "    --list-file ${REPO_ROOT}/backend/data/transformed_combined_rounds_output_en_evals_translated.json \\" >&2
  echo "    --output ${REPO_ROOT}/backend/models/pose_cache_span_linspace.pt \\" >&2
  echo "    --sampling span_linspace" >&2
fi

echo "Done. JSONL: ${JSONL}"
