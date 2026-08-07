#!/bin/zsh
set -e
REPO="/Users/davidwolpe/Documents/DataScience/dev/ai_experiments"
cd "$REPO"; export PYTHONPATH="$REPO"; source "$REPO/.venv/bin/activate"
echo "##### EXT_S0 300ep #####"
python -u -m alien_baby.crawler.eval_prism_decoy --prism-offset 0 --eval-eps 300 \
  --model alien_baby/results/decoy_v2_ext_s0_best/best_model.zip --run-tag sym_ext_s0_300
echo "##### S2 300ep #####"
python -u -m alien_baby.crawler.eval_prism_decoy --prism-offset 0 --eval-eps 300 \
  --model alien_baby/results/decoy_v2_s2_best/best_model.zip --run-tag sym_s2_300
echo "##### DONE #####"; afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
