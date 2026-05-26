#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${BASE:-$REPO_ROOT/dataset/trufor_workspace}"
PKG="${PKG:-$REPO_ROOT/third_party/trufor_server_pkg}"

cd "$PKG"
mkdir -p "$BASE/trufor_outputs" "$BASE/trufor_logs"
for group in poison clip_white_data_2 siglip_white_data_2; do
  in_dir="$BASE/$group"
  out_dir="$BASE/trufor_outputs/$group"
  mkdir -p "$out_dir"
  start=$(date +%s)
  echo "START $group $(date)" | tee "$BASE/trufor_logs/${group}.status"
  docker run --rm \
    -v "$in_dir:/data:ro" \
    -v "$out_dir:/data_out" \
    trufor:server -gpu -1 -in /data -out /data_out \
    > "$BASE/trufor_logs/${group}.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "EXIT $rc $group $(date) elapsed=$((end-start))s" | tee -a "$BASE/trufor_logs/${group}.status"
  find "$out_dir" -type f -name '*.npz' | wc -l | tee -a "$BASE/trufor_logs/${group}.status"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
