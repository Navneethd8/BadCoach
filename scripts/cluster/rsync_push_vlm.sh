#!/usr/bin/env bash
# Push only VLM 16-frame scaffold files to the cluster (not the full repo).
#
# Usage (repo root):
#   ./scripts/cluster/rsync_push_vlm.sh
#   ./scripts/cluster/rsync_push_vlm.sh user@host:~/IsoCourt
#
# Loads scripts/cluster.env for CLUSTER_HOST / KEY_FILE / REMOTE_REPO.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_ENV="${REPO_ROOT}/scripts/cluster.env"

if [[ -f "${CLUSTER_ENV}" ]]; then
  unset CLUSTER_HOST KEY_FILE REMOTE_REPO REMOTE_DATA
  set -a
  # shellcheck disable=SC1090
  source "${CLUSTER_ENV}"
  set +a
  echo "Using ${CLUSTER_ENV} (CLUSTER_HOST=${CLUSTER_HOST:-})" >&2
fi

RSYNC_OPTS=( -avz --partial --relative )
if [[ -n "${KEY_FILE:-}" ]]; then
  RSYNC_OPTS+=( -e "ssh -i ${KEY_FILE}" )
fi

DEST="${1:-}"
if [[ -z "${DEST}" ]]; then
  if [[ -z "${CLUSTER_HOST:-}" ]]; then
    echo "Set CLUSTER_HOST in scripts/cluster.env or pass user@host:path" >&2
    exit 1
  fi
  REMOTE_REPO="${REMOTE_REPO:-~/IsoCourt}"
  DEST="${CLUSTER_HOST}:${REMOTE_REPO}"
fi

# Paths relative to repo root (created on remote as needed).
VLM_PATHS=(
  backend/core/label_maps.py
  backend/core/split.py
  backend/scripts/eval_vlm_stroke_checkpoint.py
  backend/pipelines/vlm/common/vlm_stroke_protocol.py
  backend/pipelines/vlm/common/vlm_pose_cache.py
  backend/pipelines/vlm/common/vlm_eval_common.py
  backend/pipelines/vlm/common/vlm_qwen3_defaults.py
  backend/pipelines/vlm/common/load_dataset_jsonl.py
  backend/pipelines/vlm/common/build_finebadminton_jsonl.py
  backend/pipelines/vlm/common/vlm_processor_utils.py
  backend/pipelines/vlm/common/vlm_train_metrics.py
  backend/pipelines/vlm/common/vlm_pose.py
  backend/pipelines/vlm/common/requirements-unsloth-vlm.txt
  backend/pipelines/vlm/qwen-8b
  backend/pipelines/vlm/openai
  scripts/cluster/prepare_vlm_16frame.sh
  scripts/cluster/run_train_tmux.sh
  scripts/cluster/rsync_push_vlm.sh
  scripts/cluster/README.md
)

missing=()
for rel in "${VLM_PATHS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${rel}" ]]; then
    missing+=("${rel}")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing local paths (aborting):" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "rsync_push_vlm → ${DEST} (${#VLM_PATHS[@]} paths)" >&2
cd "${REPO_ROOT}"
# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '._*' \
  --exclude '.DS_Store' \
  ${RSYNC_EXTRA:-} \
  "${VLM_PATHS[@]}" \
  "${DEST%/}/"

echo "Done. On cluster: chmod +x scripts/cluster/prepare_vlm_16frame.sh && ./scripts/cluster/prepare_vlm_16frame.sh" >&2
