#!/bin/zsh
set -e
REPO="/Users/davidwolpe/Documents/DataScience/dev/ai_experiments"
cd "$REPO"
export PYTHONPATH="$REPO"
source "$REPO/.venv/bin/activate"
MODEL="alien_baby/results/decoy_v2_ext_s0_best/best_model.zip"
run() { python -u -m alien_baby.crawler.eval_prism_decoy --model "$MODEL" --eval-eps 100 "$@"; }
echo "########## SIGHTED offset=0 (control, expect ~63%) ##########"
run --prism-offset 0  --run-tag ext_s0_off0
echo "########## ABLATED offset=0 (floor, expect ~50%) ##########"
run --prism-offset 0  --ablate --run-tag ext_s0_off0_abl
echo "########## SIGHTED offset=15 ##########"
run --prism-offset 15 --run-tag ext_s0_off15
echo "########## SIGHTED offset=30 ##########"
run --prism-offset 30 --run-tag ext_s0_off30
echo "########## SIGHTED offset=45 ##########"
run --prism-offset 45 --run-tag ext_s0_off45
echo "########## DONE ##########"
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
