#!/usr/bin/env bash
# Shared helpers for cluster rsync scripts (ssh + remote paths).
set -euo pipefail

cluster_ssh() {
  if [[ -n "${KEY_FILE:-}" ]]; then
    ssh -i "${KEY_FILE}" "$@"
  else
    ssh "$@"
  fi
}

# Parse user@host:/path or user@host:~/path from rsync destination.
cluster_dest_path() {
  local dest="$1"
  dest="${dest#*:}"
  printf '%s' "${dest}"
}

cluster_dest_host() {
  local dest="$1"
  printf '%s' "${dest%%:*}"
}

ensure_remote_dir() {
  local host="$1"
  local dir="$2"
  echo "Ensuring remote directory exists: ${host}:${dir}" >&2
  cluster_ssh "${host}" "mkdir -p $(printf '%q' "${dir}")"
}
