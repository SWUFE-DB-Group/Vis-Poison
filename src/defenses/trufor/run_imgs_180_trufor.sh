#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE="${BASE:-$REPO_ROOT/dataset/trufor_workspace}"
out_root="$BASE/trufor_outputs"
log_root="$BASE/trufor_logs"
groups=(poison)

mkdir -p "$out_root" "$log_root"
for group in "${groups[@]}"; do
  src="$BASE/$group"
  dst="$out_root/$group"
  log="$log_root/$group.log"
  status="$log_root/$group.status"
  mkdir -p "$dst"
  {
    echo "[$(date -Iseconds)] start $group"
    docker run --rm \
      -v "$src:/data:ro" \
      -v "$dst:/data_out" \
      trufor:server -gpu -1 -in /data -out /data_out
    echo "[$(date -Iseconds)] done $group"
  } >"$log" 2>&1 && echo ok >"$status" || echo failed >"$status"
done
