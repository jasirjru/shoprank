#!/usr/bin/env bash
# Phase D runs (section 18): 2 more baseline seeds + 3 typo-augmented seeds, one after another on the GPU.
# Log: data/processed/logs/phase_d.log. Re-running skips models that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=data/processed/logs/phase_d.log
mkdir -p data/processed/logs
COMMON=(--teacher deberta_oof --teacher-weight 1.0 --fields title,brand,color,bullets)
run() {  # name, extra args...
  local name=$1; shift
  if [ -f "data/processed/runs/$name.json" ]; then echo "skip $name (exists)" >> "$LOG"; return; fi
  echo "=== $name start $(date +%T)" >> "$LOG"
  uv run python -m shoprank.cross.finetune "${COMMON[@]}" --name "$name" "$@" >> "$LOG" 2>&1
  echo "=== $name end $(date +%T)" >> "$LOG"
}
run minilm_distil_w10_bullets_s43 --seed 43
run minilm_distil_w10_bullets_s44 --seed 44
run minilm_distil_typo30_s42 --seed 42 --typo-rate 0.3
run minilm_distil_typo30_s43 --seed 43 --typo-rate 0.3
run minilm_distil_typo30_s44 --seed 44 --typo-rate 0.3
echo "=== ALL DONE $(date +%T)" >> "$LOG"
