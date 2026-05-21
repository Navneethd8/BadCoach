#!/usr/bin/env bash
# Pull checkpoints + registry + mlruns from EC2 to local repo.
#
# Loads scripts/ec2.env from the repo root when present: EC2_HOST, KEY_FILE, etc.
#
# Usage:
#   ./scripts/ec2/rsync_pull.sh
#   ./scripts/ec2/rsync_pull.sh user@ec2-host:~/IsoCourt   # explicit override
#
# Env (optional; often set in scripts/ec2.env):
#   EC2_HOST     default remote when no argument is passed
#   KEY_FILE     PEM path for ssh -i
#   RSYNC_EXTRA  additional rsync options

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EC2_ENV="${REPO_ROOT}/scripts/ec2.env"
if [[ -f "${EC2_ENV}" ]]; then
  unset EC2_HOST KEY_FILE
  set -a
  # shellcheck disable=SC1090
  source "${EC2_ENV}"
  set +a
  echo "Using ${EC2_ENV} (EC2_HOST=${EC2_HOST})" >&2
fi

RSYNC_SSH=()
if [[ -n "${KEY_FILE:-}" ]]; then
  RSYNC_SSH=( -e "ssh -i ${KEY_FILE}" )
fi

SRC="${1:-}"
if [[ -z "${SRC}" ]]; then
  if [[ -z "${EC2_HOST:-}" ]]; then
    echo "Usage: $0 [user@host:path/to/IsoCourt]" >&2
    echo "Or create ${EC2_ENV} with EC2_HOST=ec2-user@x.x.x.x (and optional KEY_FILE)." >&2
    exit 1
  fi
  SRC="${EC2_HOST}:~/IsoCourt"
fi

mkdir -p "${REPO_ROOT}/backend/models" "${REPO_ROOT}/backend/mlruns"

# shellcheck disable=SC2086
rsync -avz --partial "${RSYNC_SSH[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/backend/models/" "${REPO_ROOT}/backend/models/"

# shellcheck disable=SC2086
rsync -avz --partial "${RSYNC_SSH[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/backend/mlruns/" "${REPO_ROOT}/backend/mlruns/" || true
rsync -avz --partial "${RSYNC_SSH[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/mlruns/" "${REPO_ROOT}/mlruns/" || true
