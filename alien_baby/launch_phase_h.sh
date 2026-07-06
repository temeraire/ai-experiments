#!/bin/bash
# Phase H overnight launch script — runs R32, R33, R34, R35 in sequence.
# Each run waits for the previous to finish. Done-chime fires between runs
# via train_crawler.py's afplay call.
#
# Usage: nohup bash alien_baby/launch_phase_h.sh > /dev/null 2>&1 &

set -e
cd /Users/davidwolpe/Documents/DataScience/dev/ai_experiments
source .venv/bin/activate

LOG_DIR=alien_baby/results

echo "=== Phase H launch: $(date) ===" | tee -a "$LOG_DIR/phase_h_launch.log"

# R32 — proprio control, ball-speed=0.08
echo "--- R32 start: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag mimo_phase_h_R32_proprio_speed08 \
    --steps 80000 --n-envs 16 --max-steps 2000 --mps \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/mimo_phase_h_R32_proprio_speed08.log"
echo "--- R32 done: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"

# R33 — vision, ball-speed=0.08
echo "--- R33 start: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag mimo_phase_h_R33_vision_speed08 \
    --steps 80000 --n-envs 8 --max-steps 2000 --mps \
    --vision \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.08 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/mimo_phase_h_R33_vision_speed08.log"
echo "--- R33 done: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"

# R34 — proprio control, ball-speed=0.05
echo "--- R34 start: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag mimo_phase_h_R34_proprio_speed05 \
    --steps 80000 --n-envs 16 --max-steps 2000 --mps \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.05 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/mimo_phase_h_R34_proprio_speed05.log"
echo "--- R34 done: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"

# R35 — vision, ball-speed=0.05
echo "--- R35 start: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"
python -m alien_baby.crawler.train_crawler \
    --run-tag mimo_phase_h_R35_vision_speed05 \
    --steps 80000 --n-envs 8 --max-steps 2000 --mps \
    --vision \
    --cart-mode constant_velocity_bouncer --cart-speed 0.15 \
    --ball-speed 0.05 \
    --hip-actuation off --memory-obs \
    --strength-scale 1.0 \
    --ent-coef 0.5 \
    --ent-anneal-end-val 0.2 --ent-anneal-start-step 15000 --ent-anneal-end-step 30000 \
    --velocity-bonus-scale 0.10 \
    --curriculum --curriculum-warmup 2000 --curriculum-ramp-end 15000 --curriculum-final-offset 0.15 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/mimo_phase_h_R35_vision_speed05.log"
echo "--- R35 done: $(date) ---" | tee -a "$LOG_DIR/phase_h_launch.log"

echo "=== Phase H complete: $(date) ===" | tee -a "$LOG_DIR/phase_h_launch.log"
