#!/bin/zsh
set -e
REPO="/Users/davidwolpe/Documents/DataScience/dev/ai_experiments"
cd "$REPO"; export PYTHONPATH="$REPO"; source "$REPO/.venv/bin/activate"
M="alien_baby/results/decoy_v2_ext_s0_best/best_model.zip"
ev() { python -u -m alien_baby.crawler.eval_decoy_shape --model $M --eval-eps 200 --red sphere --blue sphere "$@"; }
echo "######## RECOLOR MECHANISM TEST (approach-red vs avoid-blue), sphere/sphere ########"
# baseline red/blue already = A control (77%). Reward is always on target_geom.
ev --target-color green --decoy-color blue  --run-tag ext_s0_R1_Tgreen_Dblue   # avoid-blue -> high; approach-red -> low/chance
ev --target-color red   --decoy-color green --run-tag ext_s0_R2_Tred_Dgreen     # approach-red -> high; avoid-blue -> chance
ev --target-color blue  --decoy-color red   --run-tag ext_s0_R3_Tblue_Dred      # both -> low (color-specificity check)
ev --target-color green --decoy-color yellow --run-tag ext_s0_R4_Tgreen_Dyellow # neither trained color -> chance expected
echo "######## FINAL VIDEOS ########"
vid() { python -u -m alien_baby.crawler.render_decoy_shape --model $M --episodes 3 --max-steps 320 "$@"; }
vid --red box    --blue sphere --out alien_baby/results/videos/decoy_shape_Rbox_Bsphere.mp4
vid --red sphere --blue box    --out alien_baby/results/videos/decoy_shape_Rsphere_Bbox.mp4
vid --red box    --blue capsule --out alien_baby/results/videos/decoy_shape_Rbox_Bcapsule.mp4
echo "######## FOLLOWON DONE ########"
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
