#!/bin/zsh
set -e
REPO="/Users/davidwolpe/Documents/DataScience/dev/ai_experiments"
cd "$REPO"; export PYTHONPATH="$REPO"; source "$REPO/.venv/bin/activate"
M_EXT="alien_baby/results/decoy_v2_ext_s0_best/best_model.zip"
M_S2="alien_baby/results/decoy_v2_s2_best/best_model.zip"
run() { python -u -m alien_baby.crawler.eval_decoy_shape --eval-eps 200 "$@"; }
echo "######## EXT_S0 CORE 2x2 (sighted) ########"
run --model $M_EXT --red sphere --blue sphere --run-tag ext_s0_A_sph_sph
run --model $M_EXT --red box    --blue sphere --run-tag ext_s0_B_box_sph
run --model $M_EXT --red sphere --blue box    --run-tag ext_s0_C_sph_box
run --model $M_EXT --red box    --blue box    --run-tag ext_s0_D_box_box
echo "######## EXT_S0 ablated floors ########"
run --model $M_EXT --red sphere --blue sphere --ablate --run-tag ext_s0_A_sph_sph_abl
run --model $M_EXT --red box    --blue sphere --ablate --run-tag ext_s0_B_box_sph_abl
echo "######## EXT_S0 capsule cross-check (sighted) ########"
run --model $M_EXT --red capsule --blue sphere  --run-tag ext_s0_E_cap_sph
run --model $M_EXT --red sphere  --blue capsule  --run-tag ext_s0_F_sph_cap
echo "######## S2 core 2x2 (sighted) ########"
run --model $M_S2 --red sphere --blue sphere --run-tag s2_A_sph_sph
run --model $M_S2 --red box    --blue sphere --run-tag s2_B_box_sph
run --model $M_S2 --red sphere --blue box    --run-tag s2_C_sph_box
run --model $M_S2 --red box    --blue box    --run-tag s2_D_box_box
echo "######## SHAPE BATTERY DONE ########"
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
