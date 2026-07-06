#!/bin/bash
# Watcher: wait for Phase V training to finish, then run the full eval +
# generalization battery on BOTH runs so results are ready in the morning.
#
# Signal: launch_phase_v.sh prints "Phase V training complete" to phase_v_launch.log
# when both R43 and R44 finish. We also require no train_crawler process to be
# alive (belt-and-suspenders) before starting evals, to avoid MPS contention.
cd /Users/davidwolpe/Documents/DataScience/dev/ai_experiments
source .venv/bin/activate
LOG_DIR=alien_baby/results
LAUNCH_LOG="$LOG_DIR/phase_v_launch.log"
WLOG="$LOG_DIR/phase_v_evals_watcher.log"

R43=phase_v_R43_micoa_vision_static_randbox
R44=phase_v_R44_proprio_static_randbox

echo "=== watcher start: $(date) ===" > "$WLOG"

# Wait up to ~10h for the completion marker AND a quiet trainer.
i=0
while [ $i -lt 7200 ]; do
  if grep -q "Phase V training complete" "$LAUNCH_LOG" 2>/dev/null; then
    if ! pgrep -f "train_crawler" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 5
  i=$((i+1))
done
echo "=== training detected complete: $(date) (waited ~$((i*5))s) ===" >> "$WLOG"
sleep 10   # let MuJoCo/file handles settle

run_evals () {
  TAG=$1
  echo "--- eval_phase_v $TAG: $(date) ---" >> "$WLOG"
  python -m alien_baby.visualization.eval_phase_v \
      --run-tag "$TAG" --episodes-per-bin 20 \
      > "$LOG_DIR/phase_v_${TAG}_eval.log" 2>&1
  echo "    eval_phase_v exit=$?" >> "$WLOG"
  echo "--- battery $TAG: $(date) ---" >> "$WLOG"
  python -m alien_baby.visualization.eval_generalization_battery \
      --run-tag "$TAG" --episodes-per-bin 15 \
      > "$LOG_DIR/phase_v_${TAG}_battery.log" 2>&1
  echo "    battery exit=$?" >> "$WLOG"
}

# Proprio control first (faster, no rendering), then vision.
run_evals "$R44"
run_evals "$R43"

echo "=== watcher DONE: $(date) ===" >> "$WLOG"
# audible cue, same convention as train_v8 done-sound
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
