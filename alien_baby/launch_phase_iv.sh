#!/bin/bash
# Phase IV launch — R41 (MICOA + vision, moving balls, 0.08 m/s)
#                 R42 (proprio control, moving balls, 0.08 m/s)
set -e
cd /Users/davidwolpe/Documents/DataScience/dev/ai_experiments
source .venv/bin/activate

LOG_DIR=alien_baby/results
mkdir -p "$LOG_DIR"

echo "=== Phase IV launch: $(date) ===" | tee -a "$LOG_DIR/phase_iv_launch.log"

# Pre-flight (CLAUDE.md mandate: render before training >50K)
echo "--- Pre-flight render: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"
python -m alien_baby.visualization.phase_h_preflight_render \
    --ball-speed 0.08 --steps 1200 --seed 0 \
    2>&1 | tee -a "$LOG_DIR/phase_iv_preflight.log"

# R41 — MICOA + vision + moving balls @ 0.08 m/s
echo "--- R41 start: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag phase_iv_R41_micoa_vision_speed08 \
    --steps 250000 --n-envs 8 --max-steps 2000 --mps \
    --vision --micoa --micoa-beta 0.0 --micoa-pred-beta 0.1 \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --checkpoint-interval 10000 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/phase_iv_R41_micoa_vision_speed08.log"
echo "--- R41 done: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"

# R42 — matched proprio control (no vision, no MICOA)
echo "--- R42 start: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag phase_iv_R42_proprio_speed08 \
    --steps 250000 --n-envs 16 --max-steps 2000 --mps \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --checkpoint-interval 10000 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/phase_iv_R42_proprio_speed08.log"
echo "--- R42 done: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"

# Post-training evals
echo "--- Evals start: $(date) ---" | tee -a "$LOG_DIR/phase_iv_launch.log"

python -m alien_baby.visualization.eval_phase_i \
    --run-tag phase_iv_R41_micoa_vision_speed08 \
    --ball-speed 0.08 --offset 0.15 \
    2>&1 | tee "$LOG_DIR/phase_iv_R41_eval_speed08.log"

python -m alien_baby.visualization.eval_phase_i \
    --run-tag phase_iv_R41_micoa_vision_speed08 \
    --ball-speed 0.0 --offset 0.15 \
    2>&1 | tee "$LOG_DIR/phase_iv_R41_eval_static.log"

python -m alien_baby.visualization.eval_phase_h \
    --run-tag phase_iv_R42_proprio_speed08 \
    --ball-speed 0.08 --offset 0.15 \
    2>&1 | tee "$LOG_DIR/phase_iv_R42_eval_speed08.log"

python -m alien_baby.visualization.eval_phase_h \
    --run-tag phase_iv_R42_proprio_speed08 \
    --ball-speed 0.0 --offset 0.15 \
    2>&1 | tee "$LOG_DIR/phase_iv_R42_eval_static.log"

echo "=== Phase IV complete: $(date) ===" | tee -a "$LOG_DIR/phase_iv_launch.log"
