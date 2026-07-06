#!/bin/bash
# Phase X overnight runbook — deepen the proprio generalizer via object variety.
# Three isolated 250K DroQ proprio runs, each followed by reach-eval (NOT training
# reward — per the Phase W methodological correction). Renders were gated before
# launch. Robust: an eval failure (|| true) never kills the chain.
#
#   Run 1  R46  SIZE variety   train {0.040,0.053,0.075}  held-out {0.047,0.090}
#   Run 2  R47  SHAPE variety  train {sphere,box,cylinder} held-out {ellipsoid,capsule}
#   Run 3  R48  SIZE+SHAPE     both varied; the deepest equivalence-class test
cd /Users/davidwolpe/Documents/DataScience/dev/ai_experiments || exit 1
source .venv/bin/activate 2>/dev/null
LOG=alien_baby/results
M="$LOG/phase_x_overnight_master.log"
echo "=== Phase X overnight START: $(date) ===" > "$M"

COMMON="--steps 250000 --n-envs 16 --max-steps 2000 --mps --droq --dropout-rate 0.01 --utd 4 \
 --cart-mode constant_velocity_bouncer --cart-speed 0.15 --ball-speed 0.0 \
 --random-ball-box 0.08,0.08 --hip-actuation off --memory-obs --strength-scale 1.0 \
 --ent-coef 0.5 --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
 --velocity-bonus-scale 0.10 \
 --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
 --checkpoint-interval 10000 --seed 42"

size_evals () {   # $1 = run-tag
  echo "--- SIZE evals $1: $(date) ---" >> "$M"
  python -m alien_baby.visualization.eval_generalization_battery --run-tag "$1" \
      --sizes 0.047,0.090 --episodes-per-bin 20 > "$LOG/${1}_eval_heldout_sizes.log" 2>&1 || true
  python -m alien_baby.visualization.eval_generalization_battery --run-tag "$1" \
      --sizes 0.040,0.047,0.053,0.075,0.090 --episodes-per-bin 20 > "$LOG/${1}_eval_full_sizes.log" 2>&1 || true
}
shape_evals () {  # $1 = run-tag
  echo "--- SHAPE evals $1: $(date) ---" >> "$M"
  for S in sphere box cylinder ellipsoid capsule; do
    python -m alien_baby.visualization.eval_phase_v --run-tag "$1" \
        --ball-shape "$S" --episodes-per-bin 20 > "$LOG/${1}_eval_shape_${S}.log" 2>&1 || true
  done
}
ecc_retention () { # $1 = run-tag  (default sphere/default size — no-regression check vs R45)
  echo "--- ECC retention $1: $(date) ---" >> "$M"
  python -m alien_baby.visualization.eval_phase_v --run-tag "$1" \
      --episodes-per-bin 20 > "$LOG/${1}_eval_ecc.log" 2>&1 || true
}

# ---------- Run 1: SIZE ----------
R1=phase_x_R46_droq_proprio_sizevariety
echo "=== RUN 1 $R1 train START: $(date) ===" >> "$M"
python -m alien_baby.crawler.train_crawler --run-tag "$R1" $COMMON \
    --random-ball-radius 0.040,0.053,0.075 > "$LOG/${R1}_run.log" 2>&1
echo "=== RUN 1 train DONE (exit $?): $(date) ===" >> "$M"
size_evals "$R1"; ecc_retention "$R1"

# ---------- Run 2: SHAPE ----------
R2=phase_x2_R47_droq_proprio_shapevariety
echo "=== RUN 2 $R2 train START: $(date) ===" >> "$M"
python -m alien_baby.crawler.train_crawler --run-tag "$R2" $COMMON \
    --random-ball-shape sphere,box,cylinder > "$LOG/${R2}_run.log" 2>&1
echo "=== RUN 2 train DONE (exit $?): $(date) ===" >> "$M"
shape_evals "$R2"; ecc_retention "$R2"

# ---------- Run 3: SIZE + SHAPE ----------
R3=phase_x3_R48_droq_proprio_size_and_shape
echo "=== RUN 3 $R3 train START: $(date) ===" >> "$M"
python -m alien_baby.crawler.train_crawler --run-tag "$R3" $COMMON \
    --random-ball-radius 0.040,0.053,0.075 --random-ball-shape sphere,box,cylinder > "$LOG/${R3}_run.log" 2>&1
echo "=== RUN 3 train DONE (exit $?): $(date) ===" >> "$M"
size_evals "$R3"; shape_evals "$R3"; ecc_retention "$R3"

echo "=== Phase X overnight COMPLETE: $(date) ===" >> "$M"
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
