#!/bin/bash
# Phase V launch — R43 (MICOA + vision, STATIC balls, wide random placement)
#                  R44 (proprio control, STATIC balls, wide random placement)
#
# Rationale: Phase IV (R41/R42) isolated the bottleneck — vision is now genuinely
# integrated (ablation 0.85) but was tested on moving balls @0.08, which Phase H
# proved is task-saturated (~2/20 ceiling for ANY policy) AND which drove MICOA
# encoder pathology (sigma collapse, kl_pred -> 638). Two confounds stacked.
#
# Phase V removes both confounds: go back to STATIC balls (R40's architecture is
# HEALTHY here — no pathology, vision load-bearing AND task solved) but jitter the
# ball placement per-episode with --random-ball-box so the blind forward-paddle
# "general approach" response under-covers, while each individual ball stays
# REACHABLE. Box = 0.08,0.08 on top of the +/-0.15 curriculum offset puts ball-x in
# ~[0.07,0.23] (within the graceful reach band; R30 got 14/20 at 0.20, R31 collapsed
# at 0.25) so few episodes are hopeless -> we avoid a floor of -2000 "couldn't get
# close" outcomes that would drown the vision signal, while still injecting real
# per-episode directional uncertainty that only vision can resolve. This is the
# Goldilocks difficulty between "static centered" (proprio solves alone, 15-18/20)
# and "moving @0.08" (nobody solves).
#
# Single variable changed vs R40's healthy baseline: ball placement distribution
# (fixed -> per-episode jitter). sigma clamp UNTOUCHED (known-healthy on static).
#
# Tests the generalization hypothesis: if the proprio "general approach" class is
# primary, it should generalize broadly across random placements; vision should
# become load-bearing precisely and ONLY at the eccentric placements the general
# class cannot reach. Eval (run tomorrow, supervised) bins ablation by eccentricity.
set -e
cd /Users/davidwolpe/Documents/DataScience/dev/ai_experiments
source .venv/bin/activate

# Guard: kill any stragglers from a prior (possibly half-killed) launch before we
# start, so we never end up with two trainings competing for the MPS device. The
# set -e in older runs did not catch a killed `... | tee` pipeline, which let a
# stale R44 spawn alongside a fresh R43. This makes the script idempotent on entry.
pkill -9 -f "train_crawler" 2>/dev/null || true
sleep 3

LOG_DIR=alien_baby/results
mkdir -p "$LOG_DIR"

echo "=== Phase V launch: $(date) ===" | tee -a "$LOG_DIR/phase_v_launch.log"

# R43 — MICOA + vision + STATIC balls + wide random placement
echo "--- R43 start: $(date) ---" | tee -a "$LOG_DIR/phase_v_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag phase_v_R43_micoa_vision_static_randbox \
    --steps 250000 --n-envs 8 --max-steps 2000 --mps \
    --vision --micoa --micoa-beta 0.0 --micoa-pred-beta 0.1 \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.0 \
    --random-ball-box 0.08,0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --checkpoint-interval 10000 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/phase_v_R43_micoa_vision_static_randbox.log"
echo "--- R43 done: $(date) ---" | tee -a "$LOG_DIR/phase_v_launch.log"

# R44 — matched proprio control (no vision, no MICOA)
echo "--- R44 start: $(date) ---" | tee -a "$LOG_DIR/phase_v_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag phase_v_R44_proprio_static_randbox \
    --steps 250000 --n-envs 16 --max-steps 2000 --mps \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.0 \
    --random-ball-box 0.08,0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --checkpoint-interval 10000 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/phase_v_R44_proprio_static_randbox.log"
echo "--- R44 done: $(date) ---" | tee -a "$LOG_DIR/phase_v_launch.log"

echo "=== Phase V training complete: $(date) ===" | tee -a "$LOG_DIR/phase_v_launch.log"
echo "Evals intentionally deferred to a supervised session (eval_phase_h.py has the" | tee -a "$LOG_DIR/phase_v_launch.log"
echo "hardcoded-path bug noted in FINDINGS; run eval_phase_i.py with --run-tag tomorrow)." | tee -a "$LOG_DIR/phase_v_launch.log"
