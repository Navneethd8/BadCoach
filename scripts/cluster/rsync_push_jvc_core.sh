#!/usr/bin/env bash
# Push only JVC core modules to the cluster (not the full repo).
#
# Usage (repo root):
#   ./scripts/cluster/rsync_push_jvc_core.sh
#   ./scripts/cluster/rsync_push_jvc_core.sh user@host:~/IsoCourt
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

RSYNC_OPTS=( -avz --partial )
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

CORE_FILES=(
  backend/core/jvc.py
  backend/core/gv_xattn.py
  backend/core/conv3d_pose.py
  backend/core/skateformer_b.py
  backend/core/vit_clip_encoder.py
)

for f in "${CORE_FILES[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
    echo "Missing ${REPO_ROOT}/${f}" >&2
    exit 1
  fi
done

echo "rsync_push_jvc_core → ${DEST%/}/backend/core/" >&2
# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  "${CORE_FILES[@]/#/${REPO_ROOT}/}" \
  "${DEST%/}/backend/core/"

echo "Done. On cluster, verify:" >&2
echo "  python -c \"from core.jvc import build_jvc; print('import ok')\"" >&2
