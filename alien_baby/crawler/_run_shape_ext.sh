#!/bin/zsh
set -e
REPO="/Users/davidwolpe/Documents/DataScience/dev/ai_experiments"
cd "$REPO"; export PYTHONPATH="$REPO"; source "$REPO/.venv/bin/activate"
M="alien_baby/results/decoy_v2_ext_s0_best/best_model.zip"
run() { python -u -m alien_baby.crawler.eval_decoy_shape --model $M --eval-eps 200 "$@"; }
echo "######## EXT_S0 shape-vocab robustness (red=X vs blue=sphere) ########"
run --red ellipsoid --blue sphere --run-tag ext_s0_G_ell_sph
run --red cylinder  --blue sphere --run-tag ext_s0_H_cyl_sph
echo "######## EXT_S0 full-agnostic (both = same non-sphere) ########"
run --red ellipsoid --blue ellipsoid --run-tag ext_s0_I_ell_ell
run --red capsule   --blue capsule   --run-tag ext_s0_J_cap_cap
echo "######## EXT_S0 hardest cross (red=box vs blue=capsule, both novel, different) ########"
run --red box --blue capsule --run-tag ext_s0_K_box_cap
echo "######## SHAPE EXT DONE ########"
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
