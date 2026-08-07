#!/usr/bin/env bash
set -Eeuo pipefail

INTERVAL="${1:-30}"
LOG_FILE="${2:-}"
while true; do
  clear || true
  date
  echo
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || true
  echo
  ps -eo pid,etime,%cpu,%mem,cmd | grep -E 'test.py|refine.py|eval_bop19_pose.py' | grep -v grep || true
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    echo
    echo "--- log tail: $LOG_FILE ---"
    tail -n 40 "$LOG_FILE"
  fi
  sleep "$INTERVAL"
done
