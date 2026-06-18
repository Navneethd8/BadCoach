#!/usr/bin/env bash
# Push FineBadminton-20K dataset + merged labels JSON to the cluster shared data volume.
# Loads scripts/cluster.env when present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_ENV="${REPO_ROOT}/scripts/cluster.env"
# shellcheck source=_cluster_rsync_lib.sh
source "${SCRIPT_DIR}/_cluster_rsync_lib.sh"

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
  REMOTE_DATA="${REMOTE_DATA:-/data/isocourt}"
  DEST="${CLUSTER_HOST}:${REMOTE_DATA}"
fi

REMOTE_DATA_PATH="$(cluster_dest_path "${DEST}")"
REMOTE_HOST="$(cluster_dest_host "${DEST}")"

if [[ "${REMOTE_DATA_PATH}" == ~* ]] || [[ "${REMOTE_DATA_PATH}" == "${HOME}"* ]]; then
  echo "WARN: REMOTE_DATA looks like a home path (${REMOTE_DATA_PATH})." >&2
  echo "      Use the cluster shared data volume (e.g. /data/isocourt), not ~/data/..." >&2
fi

LIST_FILE="${REPO_ROOT}/backend/data/transformed_combined_rounds_output_en_evals_translated.json"
DATA_DIR="${REPO_ROOT}/backend/data/FineBadminton-20K"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset dir: ${DATA_DIR}" >&2
  echo "Prepare locally or on-cluster with prepare_finebadminton_20k.py" >&2
  exit 1
fi
if [[ ! -f "${LIST_FILE}" ]]; then
  echo "Missing labels file: ${LIST_FILE}" >&2
  exit 1
fi

ensure_remote_dir "${REMOTE_HOST}" "${REMOTE_DATA_PATH}/FineBadminton-20K"

echo "rsync_push_data → ${DEST}" >&2
# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  ${RSYNC_EXTRA:-} \
  "${DATA_DIR}/" "${DEST%/}/FineBadminton-20K/"

# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  ${RSYNC_EXTRA:-} \
  "${LIST_FILE}" "${DEST%/}/"
