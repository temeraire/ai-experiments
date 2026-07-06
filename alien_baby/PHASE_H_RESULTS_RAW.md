# Phase H Raw Results — Audit Trail

Date: 2026-05-20
Eval script: alien_baby/visualization/eval_phase_h.py (inline version)
Eval output file: alien_baby/results/phase_h_eval_output.txt

---

## Training Log Summary

### R32 — proprio, ball_speed=0.08
- n_envs=16, max_steps=2000, seed=42, strength_scale=1.0
- ep_rew_mean progression:
  - t=2208: 412
  - t=4272: 416
  - t=5936: 426
  - t=7264: 430
  - t=8608: 431
  - t=11872: 433
  - t=13216: 436
  - t=14112: 436
  - t=16592: 437 (peak rollout)
  - t=40608: 337
  - t=44624: 165
  - t=45952: 33.5
  - t=48592: -80.9
  - EVAL t=50000: -1872.60 ± 536.46 (ep_len=1901 ± 431) — NEW BEST
  - t=72608: -195
  - t=76624: -279
  - t=77632: -325
  - t=78992: -398
- Final rollout: -398
- Checkpoint saved: 50K eval (only eval triggered)
- Semaphore leak warning at shutdown: cosmetic only

### R33 — vision, ball_speed=0.08
- n_envs=8, max_steps=2000, seed=42, strength_scale=1.0
- ep_rew_mean progression:
  - t=3040: 422
  - t=4648: 432
  - t=6232: 442
  - t=7520: 440
  - t=9112: 439
  - t=9872: 439
  - t=10368: 437
  - t=13560: 438
  - t=14328: 439 (peak rollout)
  - t=29560: 295
  - t=30232: 156
  - t=44960: 53.6
  - t=46232: -93
  - t=46632: -172
  - EVAL t=50000: -2051.38 ± 202.58 (ep_len=2000 ± 0) — NEW BEST
  - t=61888: -235
  - t=62632: -312
  - t=77888: -398
  - t=78632: -463
- Final rollout: -463
- Note: ep_len=2000 ± 0 at eval means EVERY seed ran to max_steps without both-touching

### R34 — proprio, ball_speed=0.05
- n_envs=16, max_steps=2000, seed=42, strength_scale=1.0
- ep_rew_mean progression:
  - t=5136: 439
  - t=5424: 442
  - t=5920: 440
  - t=8432: 441
  - t=11440: 442
  - t=13216: 442
  - t=14064: 443 (peak rollout)
  - t=15008: 442
  - t=17472: 442
  - t=42464: 394
  - t=45536: 216
  - t=46560: 91.1
  - t=49264: -33.5
  - EVAL t=50000: -1450.64 ± 786.54 (ep_len=1706 ± 699) — NEW BEST
  - t=74464: -147
  - t=77216: -196
  - t=77872: -227
  - t=79008: -310
- Final rollout: -310

### R35 — vision, ball_speed=0.05
- n_envs=8, max_steps=2000, seed=42, strength_scale=1.0
- ep_rew_mean progression:
  - t=3752: 432
  - t=5520: 437
  - t=6848: 441
  - t=8728: 442
  - t=9848: 441
  - t=10624: 439
  - t=12528: 439
  - t=14624: 440 (peak rollout)
  - t=28008: 386
  - t=30624: 181
  - t=44008: 23.3
  - t=46624: -127
  - t=47160: -203
  - EVAL t=50000: -1955.25 ± 264.77 (ep_len=2000 ± 0) — NEW BEST
  - t=62624: -268
  - t=63160: -366
  - t=78624: -418
  - t=79160: -492
- Final rollout: -492
- Note: ep_len=2000 ± 0 at eval means EVERY seed ran to max_steps without both-touching

---

## Touch-Counting Eval — Raw Output

All cells: 20 seeds, deterministic, max_steps=2000, env matching training config.
Source: alien_baby/results/phase_h_eval_output.txt

### Primary (training ball_speed, offset=0.15)

R32_proprio_speed08 | ball_speed=0.08 | offset=0.15
  mean=-1857.3 ± 540.7 | both=1/20 | one=15/20 | neither=4/20

R33_vision_speed08 | ball_speed=0.08 | offset=0.15
  mean=-1991.1 ± 219.5 | both=0/20 | one=12/20 | neither=8/20

R34_proprio_speed05 | ball_speed=0.05 | offset=0.15
  mean=-1837.2 ± 516.1 | both=1/20 | one=19/20 | neither=0/20

R35_vision_speed05 | ball_speed=0.05 | offset=0.15
  mean=-1773.8 ± 537.6 | both=1/20 | one=15/20 | neither=4/20

### Static ball (speed=0.0, offset=0.15)

R32_proprio_speed08 | ball_speed=0.00 | offset=0.15
  mean=-312.5 ± 1026.9 | both=14/20 | one=6/20 | neither=0/20

R33_vision_speed08 | ball_speed=0.00 | offset=0.15
  mean=-511.3 ± 791.8 | both=8/20 | one=12/20 | neither=0/20

R34_proprio_speed05 | ball_speed=0.00 | offset=0.15
  mean=-12.9 ± 742.3 | both=18/20 | one=2/20 | neither=0/20

R35_vision_speed05 | ball_speed=0.00 | offset=0.15
  mean=-736.5 ± 825.5 | both=6/20 | one=14/20 | neither=0/20

### Reach curve (training ball_speed, offset=0.10)

R32_proprio_speed08 | ball_speed=0.08 | offset=0.10
  mean=-1871.7 ± 546.0 | both=1/20 | one=14/20 | neither=5/20

R33_vision_speed08 | ball_speed=0.08 | offset=0.10
  mean=-2017.4 ± 232.0 | both=0/20 | one=10/20 | neither=10/20

R34_proprio_speed05 | ball_speed=0.05 | offset=0.10
  mean=-1493.5 ± 944.7 | both=4/20 | one=16/20 | neither=0/20

R35_vision_speed05 | ball_speed=0.05 | offset=0.10
  mean=-1887.0 ± 187.1 | both=0/20 | one=17/20 | neither=3/20

### Reach curve (training ball_speed, offset=0.05)

R32_proprio_speed08 | ball_speed=0.08 | offset=0.05
  mean=-1809.7 ± 755.3 | both=2/20 | one=10/20 | neither=8/20

R33_vision_speed08 | ball_speed=0.08 | offset=0.05
  mean=-1803.3 ± 767.1 | both=2/20 | one=8/20 | neither=10/20

R34_proprio_speed05 | ball_speed=0.05 | offset=0.05
  mean=-1494.5 ± 957.9 | both=4/20 | one=14/20 | neither=2/20

R35_vision_speed05 | ball_speed=0.05 | offset=0.05
  mean=-1881.5 ± 564.4 | both=1/20 | one=11/20 | neither=8/20

---

## Vision Ablation — Raw Output

### R33 — vision, ball_speed=0.08, offset=0.15

(a) Single-step L2 action delta (200 steps × 10 seeds = 2000 delta measurements):
  mean  = 0.0244
  median = 0.0187

Phase G references for comparison:
  R14 ablation: 0.023 (same dead zone)
  R17 ablation: 0.009 (same dead zone)
  Random action (action space ±1, 25-dim): ~1–3 expected

(b) Episode-level, 20 seeds each condition:
  pixels NORMAL: both-touched = 0/20
  pixels ZEROED: both-touched = 0/20
  delta = 0 (identical outcome)

### R35 — vision, ball_speed=0.05, offset=0.15

(a) Single-step L2 action delta (200 steps × 10 seeds = 2000 delta measurements):
  mean   = 0.0207
  median = 0.0202

(b) Episode-level, 20 seeds each condition:
  pixels NORMAL: both-touched = 1/20
  pixels ZEROED: both-touched = 1/20
  delta = 0 (identical outcome)

---

## Computed Gaps vs Proposal Success Criteria

Speed 0.08: Vision (R33) 0/20 vs Proprio (R32) 1/20
  Relative gap = (0-1)/1 = -100% (vision WORSE)
  Threshold for Promising: +25%; Threshold for Marginal: +5–25%
  Grade: REFUTED

Speed 0.05: Vision (R35) 1/20 vs Proprio (R34) 1/20
  Relative gap = (1-1)/1 = 0%
  Grade: REFUTED (within noise)

Ablation sensitivity R33: 0.0244 (threshold Promising: ≥0.20; Marginal: 0.05–0.20)
  Grade: REFUTED (<0.05)

Ablation sensitivity R35: 0.0207
  Grade: REFUTED (<0.05)

Per-speed gradient (proprio 0.05 → 0.08):
  1/20 → 1/20 = flat (no drop, so vision cannot "compensate for a drop that doesn't exist")
  The gradient test is vacuous at floor performance

---

## Historical Context

Phase G ablation results (Phase H's predecessor) for comparison:
  R14 ablation: mean delta 0.023, episode zeroed 14/20 vs normal 14/20 (identical)
  R17 ablation: mean delta 0.009, episode zeroed 17/20 vs normal 16/20 (IMPROVED when blind)

Phase H ablation results:
  R33: mean delta 0.0244, episode zeroed 0/20 vs normal 0/20 (identical at floor)
  R35: mean delta 0.0207, episode zeroed 1/20 vs normal 1/20 (identical at floor)

The ablation sensitivity range (0.009–0.024) is identical across 7 vision experiments.
The pixel pathway has never moved outside this dead zone under SAC + cart substrate.

---

## Checkpoint Paths

| Run | best_model.zip path |
|---|---|
| R32 | alien_baby/results/mimo_phase_h_R32_proprio_speed08_best/best_model.zip |
| R33 | alien_baby/results/mimo_phase_h_R33_vision_speed08_best/best_model.zip |
| R34 | alien_baby/results/mimo_phase_h_R34_proprio_speed05_best/best_model.zip |
| R35 | alien_baby/results/mimo_phase_h_R35_vision_speed05_best/best_model.zip |

All best_models are the t=50K checkpoint (the only eval point fired during training).
vec_normalize.pkl in corresponding run directories (not _best subdirectory).

---

## Notes on Eval Methodology

- All evals used fixed_ball_positions matching training offset, deterministic predict.
- Vision model evaluation: obs shape (1, 6215) = 71 proprio + 6144 stereo pixels.
- Proprio model evaluation: obs shape (1, 71).
- ball_speed in eval matched training ball_speed (not zero) for primary and reach cells.
- Episodes ran to max_steps=2000 or until both-touched (whichever first).
- Seeds 0–19 used deterministically for all 20-seed cells.
- Ablation (a): both full and blind actions computed at same timestep from same state.
- Ablation (b): separate 20-episode runs (normal first, then zeroed).

---

## Pre-flight Verification (from PHASE_H_PREFLIGHT.md)

All 5 checks PASSED:
1. Balls move at correct speed, bounce off all 4 edges, visible in head_cam — PASS
2. Touch detection fires correctly for moving balls, ball removed on contact — PASS
3. Ball velocity NOT in proprio (obs length 71 at both ball_speed=0.0 and 0.08) — PASS
4. Stationary arm both-touched = 2/20 (10%) at speed=0.08 — ball motion breaks blanket-sweep — PASS
5. --ball-speed 0.0 bit-identical to Phase G (max diff = 0.00e+00) — PASS
