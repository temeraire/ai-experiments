#!/bin/bash
# Pre-configured launcher for Phase B: dialogue architecture on the no-bribery substrate.
# Only run if Phase A's analyze_run.py reports VISION SILENT (mean delta < 0.1).
#
# 300K steps × ~50 fps (dialogue is heavier than standard CNN) ≈ 100 min.
set -e
cd "$(dirname "$0")/../.."   # cd to repo root

python -m alien_baby.crawler.train_dialogue \
  --steps 300000 \
  --n-envs 4 \
  --mps \
  --learning-rate 1e-4 \
  --ent-coef 0.2 \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --consistency-lambda 0.05 \
  --spawn-cone-deg 360 \
  --max-steps 2000 \
  --velocity-bonus-scale 0.0 \
  --approach-reward-scale 0.0 \
  --run-tag mimo_dialogue_v1
