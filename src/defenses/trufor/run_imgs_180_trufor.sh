#!/usr/bin/env bash
set -euo pipefail

in_dir="/home/ubuntu/liangrj/imgs_180"
out_root="$in_dir/trufor_outputs"
log_root="$in_dir/trufor_logs"
groups=(poison clip_white_data_2 siglip_white_data_2)

mkdir -p "$out_root" "$log_root"
for group in "${groups[@]}"; do
  src="$in_dir/$group"
  dst="$out_root/$group"
  log="$log_root/$group.log"
  status="$log_root/$group.status"
  mkdir -p "$dst"
  {
    echo "[$(date -Iseconds)] start $group"
    docker run --rm -v "$src:/data:ro" -v "$dst:/data_out" trufor:server -gpu -1 -in /data -out /data_out
    echo "[$(date -Iseconds)] done $group"
  } >"$log" 2>&1 && echo ok >"$status" || echo failed >"$status"
done
