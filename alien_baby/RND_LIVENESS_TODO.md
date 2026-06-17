# RND + zero step-penalty run, and the "realm of possibility" liveness gate

Goal: make the creature MOVE. Add an intrinsic curiosity drive (RND), stop
punishing existence (zero step penalty), and enforce a hard gate that voids any
run where the creature never moved ("how to die" is not tolerated).

## Todo
- [x] 1. RND intrinsic reward — new `alien_baby/crawler/rnd_wrapper.py` (VecEnvWrapper, flat Box obs).
- [x] 2. Free-body env (`mimo_crawler_env.py`): make STEP_COST configurable (`step_cost` param, default unchanged) + expose `body_motion` in info.
- [x] 3. Cart env (`mimo_crawler_cart_env.py`): expose `body_motion` in info.
- [x] 4. `train_crawler.py`: `LivenessGateCallback` + CLI flags; wrap train env with RND; thread `step_cost`.
- [x] 5. CLAUDE.md: add "Realm of possibility" rule under training-run workflow.
- [x] 6. Smoke-test imports + tiny run; confirm the gate fires and prints a motion number.

## Review

What changed (smallest-footprint edits; all defaults preserve existing runs bit-identical):
- **`rnd_wrapper.py` (new)**: `RNDRewardWrapper(VecEnvWrapper)`. Fixed random target net + trained
  predictor net over the obs vector; per-step prediction error → normalized novelty bonus added to
  reward. Trains the predictor one step per env step. Flat Box obs only (guards vision/HER).
- **`mimo_crawler_env.py`**: `STEP_COST` → configurable `step_cost` ctor param (default -0.05
  unchanged); reward uses `self.step_cost`. Added `body_motion` (= hip speed + mean joint speed) to info.
- **`mimo_crawler_cart_env.py`**: added `body_motion` (= mean joint speed + AB-commanded cart speed
  in steer mode only) to info. Step penalty already configurable via hunger flags.
- **`train_crawler.py`**: new `LivenessGateCallback`; CLI flags `--rnd`, `--rnd-coef`, `--step-cost`,
  `--liveness-gate-step`, `--liveness-min-motion`, `--no-liveness-gate`; wraps train env with RND via
  `model.set_env`; threads `step_cost` into both make_env calls. Gate ON by default.
- **`CLAUDE.md`**: new governing rule "Realm of possibility — behavior that leads to dying is not
  tolerated", plus the curiosity/motivation knobs are explicitly sanctioned.
- **GLOSSARY.md / FLASHCARDS.md**: added "Freeze attractor" and "RND" entries.

Verified (tiny SAC runs, free-body position_offset + cart env step):
- RND wraps and trains; with `--step-cost 0` + RND, ep_rew_mean is positive (not the dead-flat
  negative of a freeze).
- Liveness gate PASS path: prints measured `mean body_motion=0.835` at the gate step, continues.
- Liveness gate VOID path: stops training at the gate, prints "Nothing happened", writes
  `RUN_VOID_NO_MOVEMENT.txt`.
- Cart env emits `body_motion` (5.01 on a random action); hunger-zeroed reward = 0.0.

Calibration note: `--liveness-min-motion` default 0.05 is a starting point. Untrained free-body
position_offset already shows ~0.835, so a real freeze would be far below; tune from printed values.

Not done (next): pick the run config, render an untrained preflight video (per CLAUDE.md), then
launch. Recommended first run is in the response.
