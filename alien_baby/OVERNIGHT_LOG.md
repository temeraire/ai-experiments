# Overnight Session — Phase G Cart Curriculum Iteration

Running autonomously starting 2026-05-14 ~21:30. Goal: probe whether the cart
substrate + ball-offset curriculum can produce a policy that actually extends
arms toward offset balls — i.e., whether forcing the "easy sweep" strategy to
fail unlocks new learning, or just degrades to floor.

## Morning summary (top-line, read this first)

**Headline result:** The "one move then freeze" failure mode that has blocked
this project since Phase D is breakable. **High entropy floor (ent_coef=0.5) +
curriculum (offset 0→0.15) + blind proprio** produces a policy that reliably
extends its arms toward offset balls — first real reach behavior in the
project. Result reproduces across two seeds.

**Best numbers (R20, ent_coef=0.5 + vel_bonus=0.10 + anneal, 50K, seed=42):**
- Peak eval: **+380.2 ± 7.8** at t=14976, 20/20 hits, 0 floor.
- R22 (seed=1, R20 recipe): +388 ± 26 at t=35K, end +231/18/20.
- R23 (seed=2, R20 recipe): +361 ± 57 at t=15K. **Three-seed validation
  of the R20 mechanism complete** — breakthrough is robust to seed.

**Previous-best numbers (R3, ent_coef=0.5 only, 50K):**
- Peak eval: +366 ± 14 at t=14976, 20/20 hits, 0 floor.
- Seed=1 reproduction (R9): +252 ± 507 at t=30K, 19/20 hits. Same behavior,
  slightly weaker magnitude — R3's +366 was on the lucky end.

**Vision is NOT load-bearing.** The project's central open question now has
a clean answer for this task:
- R10 (vision + curriculum, 100K) plateaus at 12-14/20 hits, never finds R3's
  +366. Vision did not help.
- R14 (vision at offset=0.20) appeared to beat R8 (proprio 16-env at 0.20):
  14/20 vs 5/20 both-touched. BUT R15 (proprio control with matched
  8-env+DummyVecEnv setup) also gets 12/20 — the apparent vision win was a
  vec_env/n_envs confound, not vision.
- Vision-ablation: action delta when pixel columns are zeroed is 0.023
  (action space [-1,1], 25-dim) — vision channel is essentially decorative.
  Confirmed at episode level: zeroing pixels for full 20 episodes at
  offset=0.20 gives IDENTICAL touch rate (14/20 both-touched) and reward
  within 1.1 units. Vision is provably not used.
- At offset=0.30 (beyond body's reach radius), neither vision nor proprio
  can solve the task — it's a geometric limit (body's arms cannot reach),
  not a learning problem.

**Why vision didn't help:** at offsets where the task is solvable
(≤0.20), the contact reward is discoverable via random arm motion +
high entropy. The CNN's added capacity is noise during the time window
where exploration matters. At offsets where the task is unsolvable
(≥0.25–0.30), no input modality fixes the body's reach radius.

**The strongest test added at 04:30:** Random ball positions per episode
within a (0.10, 0.05) box around the anchors — proprio cannot memorize
ball position since it varies every reset. R17 (vision) got 34/40
both-touched; R18 (proprio control) got **36/40 both-touched, with HIGHER
reward**. R17's vision-ablation delta: 0.009 (vs R14's 0.023 — even lower).
Zeroing R17's pixels gives 17/20 both-touched vs 16/20 with pixels normal
— **disabling vision slightly improves performance**. The CNN learned to
ignore pixels even in the setting designed to make them matter.

**Second-night vision tests (R26/R27, R28/R29):** disappearing-balls
substrate added — each ball vanishes after N env-steps if not touched.
At timeout=300: vision 3/20 both-touched, proprio 5/20. At timeout=150
(forces fast action): vision 15/20 ball1, proprio 15/20 ball1 — literally
identical results. **Six independent substrate configurations tested;
vision is not load-bearing in any of them.** For vision to matter,
the project needs body/task/algorithm changes beyond reward+curriculum
tweaks (see "Vision: definitive verdict" section below).

**Refinements found:**
- Hypothesis (a) "geometrically impossible" — applies cleanly at offset≥0.30
- Hypothesis (b) "no reward signal toward ball" — partially supported (R2/R5
  approach reward and velocity bonus both help vs. baseline) but subsumed by
  high entropy when entropy is sufficient
- Hypothesis (c) "optimizer stuck in attractor" — dominant explanation
  at offset=0.15. ent_coef=0.5 is the lever.

**Known limitations to address before next session:**
- SAC's chronic late-training regression remains (R3 peaks at t=15K, drifts
  to -438 by t=50K). The best_model.zip captures the peak; the consolidation
  strategy (R7 warm-start + lower ent) failed because SB3's --init-from
  loses the replay buffer.
- Direct training at offset=0.20 from scratch (R8) struggles even with
  ent_coef=0.5. R3-best transfers there better than R8 trains there.
  Curriculum target should be ≤0.15 with reach radius generalizing the rest.
- Vision ran with DummyVecEnv + 8 envs; proprio ran with SubprocVecEnv + 16
  envs. R15 added the proprio + DummyVecEnv + 8 envs control. There may be
  a smaller, separate effect from vec_env type on SAC stability worth
  isolating cleanly.

**Saved artifacts for review:**
- `alien_baby/results/mimo_phase_g_R3_ent050_best/best_model.zip` — the
  canonical "extender". 20/20 hits at offset=0.15.
- `alien_baby/results/mimo_phase_g_R9_seed1_best/best_model.zip` — seed=1
  reproduction (weaker but qualitatively same behavior).
- Render frames: `alien_baby/results/frames/R3_ent050_best_seed*` — the
  sequential burst showing active right-arm extension toward the red ball
  at t=30→90.

**Suggested next-session experiments:**
1. Save+reload replay buffer to enable real warm-start (would let R3-best be
   pushed further via curriculum 0.15→0.20).
2. Step-scheduled entropy annealing inside the original 50K training loop
   (drop ent_coef from 0.5 → 0.2 between t=15K and t=30K). Should
   stabilize the +366 peak.
3. Vision-required substrate: change the env so contact reward IS NOT
   discoverable by random arm motion (e.g., balls must be SEEN before they
   count, or balls disappear after a fixed time). Only then does vision
   have any structural reason to be load-bearing.

## Summary at a glance

| Run | Setup | Total steps | Final eval | Verdict |
|---|---|---|---|---|
| v3 (baseline curriculum) | warmup 1K, ramp 1K→5K, hold 5K→10K, offset 0→0.15 | 10K | −317 ± 884 | Curriculum mechanism works; collapse at lock-in confirmed |
| v3c (no-warmup control) | offset=0.15 from step 0 | 10K | −539 ± 912 | No-warmup is worse; warmup *does* preserve some plateau |
| v3d (stretched curriculum) | warmup 5K, ramp 5K→25K, hold 25K→50K | 50K | −364 ± 892 | 25K of hold at offset=0.15 does not adapt; same plateau as v3 |
| v3b (extended same as v3) | warmup 1K, ramp 1K→5K, hold 5K→100K | 100K | (running) | TBD |

## Visual diagnosis (from v3 best_model, offset=0.15)

Rendered 3 seeds × 600 steps. Frames inspected at t=0, t≈60–250 (cart sweep),
t≈400–600 (later sweeps).

What AB is doing:
- Adopts a "raised-prop" posture (one or both arms tucked under torso,
  upper body elevated off the cart). Holds it through the episode.
- Body shifts slightly as cart bounces but no clear arm-extension behavior.
- Cart sweeps prone body north and south along its constrained y-axis;
  balls are 0.15m offset in x.

Touches that do happen are the result of:
- AB's body footprint (shoulders at ±0.105 m, soft body extending to ~±0.20 m)
  *just* grazing a ball at x=±0.15 m when the cart's bounce trajectory aligns.
- Not active reach.

This is the same passive-sweep strategy that produced v2's +361 plateau —
it just succeeds less often because the offset puts balls at the edge of the
body footprint instead of dead-center on it. High eval std (~800+) confirms
two-attractor distribution: lucky-sweep vs. floor.

## Key finding so far

**The curriculum forces a harder task but does not, by itself, drive arm-
extension learning.** Whether the curriculum is steep (v3, 4K-step ramp) or
gentle (v3d, 20K-step ramp), the policy enters the offset=0.15 regime, the
cart-sweep strategy stops paying, and the policy plateau-floats between
"−2000-truncated" and "lucky-grazed" without finding a new behavior.

The reason this is hard: **AB's proprioception does not include ball
position.** Its only signal that arm-extension matters is the +200 contact
reward when something already-extended happens to brush a ball. With no
dense shaping and no information about ball location, SAC cannot
gradient-descend its way to a "reach left when cart approaches +y" policy.

## What to try next

This list reorders the candidates from "fix that works without changing
the philosophical setup" to "fix that breaks more of the no-bribery rule":

1. **More compute alone (v3b, 100K)** — control for "did the policy just need
   more time after the ramp?" — running now.
2. **Vision-enabled curriculum** — the env supports vision; in the cart
   setup with offset balls, ball position is invisible to proprio, so
   vision should genuinely be load-bearing. This is the interesting
   experiment.
3. **Approach reward (`--approach-reward-scale 1.0`)** — dense
   distance-to-ball shaping. Bribery in the strict sense, but a useful
   diagnostic: does any reward gradient toward extension produce
   extension?
4. **Higher entropy (`--ent-coef 0.5` or `auto`)** — let the policy
   stochastically swipe more during the hold phase. Cheap to try.
5. **Smaller final offset (e.g. 0.10)** — keep balls within shoulder
   reach so passive extension still works. Tests whether 0.15 is
   geometrically out of range for the body, not just a learning
   problem.

## Strategist-consulted plan (prioritized)

A second-opinion review of the candidate list reorganized into a triage
sequence designed to distinguish (a) geometric impossibility from
(b) reward-signal gap from (c) optimizer-trapped-in-attractor.

| # | Run | Config | Tests | Cost |
|---|---|---|---|---|
| R1 | geometry probe | offset=0.05, 30K | (a) | ~15 min |
| R2 | approach reward | offset=0.15, approach_scale=5.0, 30K | (b) | ~15-20 min |
| R3 | high entropy | offset=0.15, ent_coef=0.5, 50K | (c) | ~25-30 min |
| R4 | crossed: approach+entropy | offset=0.15, both, 50K | conditional | ~25-30 min |
| R5 | velocity bonus | offset=0.15, vel_bonus=0.2, 30K | b vs c | ~15-20 min |
| R6 | vision sanity | offset=0.15, --vision, 50K | breakdown | ~30 min |

Strategist's prior: geometry will NOT be the bottleneck (R1 freezes too);
dense reward (R2) will partially work; exploration alone (R3) won't.

## Run results log

### v3b — 100K extended same-curriculum (control for "did it just need time?")

Completed. 10 evals at every 10K timesteps. Eval reward oscillates -120 to -760
with NO upward trend over the 95K post-ramp hold. Hit-rate (eval episodes with
reward > +200) varies 6-14/20 across checkpoints with no pattern. Best
checkpoint at t=70K: -122 ± 775, 14/20 hits — but this is the high end of the
oscillation, not a learned plateau.

| t | reward | std | ep_len | floor | hits |
|---|---|---|---|---|---|
| 10K | -199 | 814 | 990 | 3/20 | 12/20 |
| 20K | -401 | 949 | 1073 | 6/20 | 12/20 |
| 30K | -539 | 884 | 1248 | 2/20 | 9/20 |
| 40K | -741 | 754 | 1516 | 2/20 | 6/20 |
| 50K | -541 | 892 | 1267 | 2/20 | 9/20 |
| 60K | -218 | 789 | 1044 | 2/20 | 12/20 |
| 70K | -123 | 775 | 841 | 1/20 | 14/20 |
| 80K | -762 | 880 | 1448 | 6/20 | 7/20 |
| 90K | -434 | 900 | 1137 | 3/20 | 10/20 |
| 100K | -733 |1053 | 1288 | 8/20 | 9/20 |

**Verdict:** Time alone does not produce arm-extension learning. The
optimizer is stuck in the same two-attractor regime (lucky-sweep vs. floor)
that v3 and v3d ended in. Hypothesis "policy just needs more steps after the
ramp" is refuted.

### R1 — geometry probe (offset=0.05, 30K)

Stable +355 to +367 reward throughout 30K, 20/20 hit rate, eval std ~17-27.

| t | reward | std | ep_len | floor | hits |
|---|---|---|---|---|---|
| 5K | +365 | 19 | 396 | 0/20 | 20/20 |
| 10K | +357 | 18 | 449 | 0/20 | 20/20 |
| 15K | +367 | 20 | 366 | 0/20 | 20/20 |
| 20K | +258 | 431 | 529 | 1/20 | 19/20 |
| 25K | +353 | 16 | 484 | 0/20 | 20/20 |
| 30K | +362 | 27 | 369 | 0/20 | 20/20 |

**Verdict:** Cart-sweep strategy works perfectly at offset=0.05 — balls are
inside passive body footprint. This doesn't cleanly test hypothesis (a)
because no extension is required; it just confirms the cart-kinematic
baseline. Geometry-as-bottleneck remains untestable at offsets where the
passive sweep already succeeds. R1 IS useful as a sanity check that the
mechanism reliably gets full reward when reachable.

### R3 — high-entropy exploration (ent_coef=0.5, 50K) — **BREAKTHROUGH**

First time the project has seen reliable arm-extension behavior. At t=14976,
eval reward hit **+365.8 ± 14.4 with 20/20 hits and 0 floor episodes** —
the v2-style stable plateau, but at offset=0.15 instead of offset=0.

| t | reward | std | ep_len | floor | hits |
|---|---|---|---|---|---|
| 5K | -290 | 933 | 975 | 4/20 | 13/20 |
| 10K | -120 | 831 | 838 | 3/20 | 15/20 |
| 15K | **+366** | 14 | 413 | 0/20 | **20/20** |
| 20K | +9 | 827 | 724 | 3/20 | 17/20 |
| 25K | +3 | 830 | 741 | 3/20 | 17/20 |
| 30K | -322 | 882 | 1050 | 3/20 | 12/20 |
| 35K | -368 | 952 | 1034 | 4/20 | 12/20 |
| 40K | -389 | 941 | 1059 | 2/20 | 12/20 |
| 45K | -681 | 933 | 1386 | 4/20 | 7/20 |
| 50K | -438 | 970 | 1077 | 3/20 | 11/20 |

**Visual diagnosis (best_model rendered at offset=0.15, 5 seeds):**
- All 5 seeds completed within 600 steps (390/585/450/450/585) — meaning
  both balls touched in every rendered episode, consistent with the 20/20
  eval hit rate.
- Sequential frames (seed0: t=30 → t=90) show AB extending its RIGHT ARM
  LATERALLY out from the body toward the red ball at (+0.15, +0.35), and
  the ball is then contacted. This is the first active reach observed in
  the project — qualitatively different from v3d's static raised-prop.
- Later in the same episode (seed0 t=345): AB shifts body south and the
  blue ball at (-0.15, -0.35) is contacted at AB's feet.

**Verdict:** Higher entropy floor (0.5 vs 0.2) broke the frozen-posture
attractor and unlocked active arm-extension behavior. Hypothesis (c)
— optimizer-trapped — is the right one for this failure mode. SAC at
ent_coef=0.2 was getting stuck in the cart-sweep local optimum; ent_coef=0.5
gave it the exploration budget to find an extension strategy.

**Caveat — late-training regression:** Best policy was at t=14976. From
t=19968 onward, eval reward and hit-rate degrade steadily. This is the
project's chronic "best ≠ final" pattern documented in PROJECT_STATUS.md.
Possible fixes: stop earlier (already captured via EvalCallback's
best_model.zip), entropy annealing, lower learning rate, or use the saved
best checkpoint as a seed for a follow-on at ent_coef=0.2 to consolidate.

### R2 — hand-distance approach reward (offset=0.15, scale=5.0, 30K)

NOTE: before launching, fixed a bug in `_ball_dist`: previously returned
cart-center-to-ball1 distance, which doesn't change with arm movement. New
version returns min over (right_hand, left_hand) × (untouched active balls)
— so the approach gradient now actually rewards arm extension.

| t | reward | std | ep_len | floor | hits |
|---|---|---|---|---|---|
| 5K | -150 | 862 | 879 | 3/20 | 15/20 |
| 10K | -345 | 959 | 1009 | 6/20 | 13/20 |
| 15K | -163 | 843 | 861 | 3/20 | 14/20 |
| 20K | -548 | 923 | 1230 | 5/20 | 10/20 |
| 25K | -395 | 931 | 1075 | 4/20 | 12/20 |
| 30K | -83 | 692 | 965 | 2/20 | 12/20 |

**Verdict:** Better than v3b at every comparable checkpoint. Final reward
-83 ± 692 is the best post-ramp eval at offset=0.15 we've seen. Hit-rate
slightly elevated (10-15/20 vs v3b's 6-14/20). Rendered postures show
mild divergence across seeds (one seed has a tripod prop with arms behind,
another has a head-down prone posture, where v3d had a single static
raised-prop across seeds). Mixed evidence — dense gradient nudges policy
in the right direction but doesn't break the attractor in 30K. May be
worth combining with high entropy (R4) or running longer.

### R4 — crossed approach+entropy (offset=0.15, scale=5.0, ent=0.5, 50K)

Peak +367.6 ± 10.5 at t=14976 with 20/20 hits and 0 floor — essentially
identical to R3 at the same step. Adding the dense approach gradient to
the high-entropy run did NOT improve the peak nor delay the regression.
Drift pattern post-peak matched R3 closely.

**Verdict:** The breakthrough is high entropy ALONE. Approach reward
helps at low entropy (R2 vs v3b) but contributes nothing on top of
ent_coef=0.5. Confirms (c) dominates (b) when entropy is sufficient.

### R5 — velocity-bonus probe (offset=0.15, scale=0.20, ent=0.2, 30K)

NOTE: cart env was missing the velocity-bonus reward implementation. Added
to step() as `scale * mean(|jvel_actuated|)` so the bonus rewards arm/head
joint motion regardless of direction. Smoke-tested before launching.

| t | reward | std | ep_len | floor | hits |
|---|---|---|---|---|---|
| 5K | -146 | 846 | 909 | 3/20 | 14/20 |
| 10K | -211 | 887 | 932 | 5/20 | 14/20 |
| 15K | -20 | 723 | 778 | 1/20 | 15/20 |
| 20K | -406 | 896 | 1133 | 4/20 | 11/20 |
| 25K | -365 | 910 | 1074 | 4/20 | 12/20 |
| 30K | **+165** | 504 | 749 | **0/20** | **16/20** |

**Verdict:** Final eval at t=29952 hit +165 ± 504 with 0 floor episodes
and 16/20 hits — best non-R3-family result. The anti-stillness pressure
of the velocity bonus DOES partially break the frozen attractor, but
slower and less cleanly than ent_coef=0.5. Hypothesis (c) [exploration]
is supported by another mechanism.

### R7 — consolidation (warm-start R3 best, ent=0.2, lr=5e-5, 30K)

Failed. Even starting from R3's perfect 20/20 policy with reduced entropy
and learning rate, the model drifts back into the lottery regime within
5K steps. Hit-rate falls from 20/20 to 8-15/20; eval reward oscillates
-48 to -534.

| t | reward | std | hits |
|---|---|---|---|
| 20K | -48 | 710 | 14/20 |
| 25K | -445 | 991 | 12/20 |
| 30K | -89 | 754 | 14/20 |
| 35K | -436 | 900 | 11/20 |
| 40K | -83 | 794 | 15/20 |
| 45K | -534 | 845 | 8/20 |

**Verdict:** SB3's --init-from loses the replay buffer. Without
exploratory data, even the great starting policy decays. Consolidation
via warm-start doesn't work. To stabilize R3-quality behavior, either:
(i) train with entropy annealing inside the original run, or (ii) save
and reload the replay buffer alongside the model (requires code).

### Reach-radius probe — R3-best evaluated at multiple offsets (no training)

Tests whether the R3-best policy (trained with offset=0.15 curriculum)
generalizes to different ball offsets at inference time. 20 seeds per
offset, deterministic eval, max_steps=2000.

| offset (m) | mean reward | both-touched | only-one | neither |
|---|---|---|---|---|
| 0.05 | +256 ± 424 | 19/20 | 1/20 | 0/20 |
| 0.10 | +124 ± 685 | 18/20 | 2/20 | 0/20 |
| 0.15 (train) | +248 ± 500 | **19/20** | 1/20 | 0/20 |
| 0.17 | -288 ± 933 | 13/20 | 7/20 | 0/20 |
| 0.20 | +4 ± 641 | 15/20 | 5/20 | 0/20 |
| 0.25 | -714 ± 885 | 5/20 | 11/20 | 4/20 |
| 0.30 | -1950 ± 484 | 0/20 | 7/20 | 13/20 |

**Verdict:** R3-best has a real reach radius of ~0.20-0.25m. Graceful
degradation from 0.15 → 0.20; hard cutoff between 0.20 and 0.30. The
arm-extension policy was NOT memorized to one offset — it generalizes,
which is strong evidence that the learned behavior is genuinely
positional reaching (not coincidental cart-sweep kinematics).

### R8 — direct train at offset=0.20 (ent=0.5, 50K from scratch)

Tests whether direct training at the harder offset can produce a policy
that beats R3-best transferred to that offset.

| t | reward | both-touched (deterministic) | notes |
|---|---|---|---|
| R8-best (best_model.zip) at 0.20 | -667 | **5/20** | direct training |
| R3-best at 0.20 (inference only) | +4 | **15/20** | transfer |

**Verdict:** Direct training at the harder offset is **worse** than
transferring a policy trained on the easier curriculum. R3's curriculum
(0→0.15) builds extension that generalizes further than direct training
discovers in 50K steps at offset=0.20. Curriculum target matters; the
final offset should be at the edge of, not beyond, the policy's
exploration range during the ramp.

### R9 — reproducibility (R3 config with seed=1, 50K)

| t | reward | std | floor | hits |
|---|---|---|---|---|
| 5K | +42 | 524 | 0/20 | 14/20 |
| 10K | -545 | 945 | 3/20 | 10/20 |
| 15K | -352 | 930 | 3/20 | 12/20 |
| 20K | +121 | 596 | 1/20 | 17/20 |
| 25K | +141 | 696 | 2/20 | 18/20 |
| 30K | **+252** | 507 | 1/20 | **19/20** |
| 35K | -332 | 927 | 4/20 | 12/20 |
| 40K | +254 | 344 | 0/20 | 18/20 |
| 45K | -46 | 763 | 2/20 | 15/20 |
| 50K | +121 | 688 | 2/20 | 17/20 |

**Verdict:** Breakthrough REPRODUCIBLE. Seed=1 finds extension policy
at slightly weaker magnitude (+252 vs R3's +366) and later peak
(t=30K vs R3's t=15K). Same qualitative behavior across two seeds —
not a seed-42 fluke. R3's +366 was on the lucky end; +250 is closer
to the typical breakthrough magnitude.

### R11 — push reach radius (warm-start R3-best + curriculum to 0.25)

Catastrophic collapse. Same SB3 warm-start replay-buffer issue as R7,
compounded by the curriculum target (0.25) being at the edge of R3's
reach radius.

| t | reward | floor | hits |
|---|---|---|---|
| 20K | -683 | 2/20 | 6/20 |
| 35K | -1653 | 10/20 | 0/20 |
| 50K | -1567 | 9/20 | 0/20 |
| 65K | -846 | 0/20 | 0/20 |

By t=35K, no episodes terminate at all — R3's extension behavior was
fully unlearned. The bootstrap-from-pretrained strategy fundamentally
doesn't work with SB3's --init-from because the replay buffer is reset.

**Verdict:** To push reach radius further, need either:
(i) implement replay-buffer save/load in train_crawler.py, or
(ii) train from scratch with a longer/gentler curriculum that ends at
0.20 (test that the same exploration mechanism can find a wider reach
when given more time at intermediate stages).

### R6 — vision + ent=0.5 + curriculum (50K)

| t | reward | std | hits |
|---|---|---|---|
| 5K | -399 | 842 | 10/20 |
| 15K | -244 | 807 | 12/20 |
| 25K | -409 | 878 | 11/20 |
| 35K | -515 | 867 | 9/20 |
| 50K | -278 | 884 | 13/20 |

**Verdict:** Vision did not find the breakthrough. Hit-rate oscillates
8-13/20 throughout, never reaches R3's 19-20/20 plateau. Vision adds
CNN learning complexity that 50K can't bridge from scratch.

### R10 — vision + ent=0.5 + curriculum (100K) — longer attempt

| t | reward | hits |
|---|---|---|
| 10K | -215 | 13/20 |
| 30K | -416 | 10/20 |
| 50K | -653 | 7/20 |
| 70K | -238 | 12/20 |
| 100K | -299 | 12/20 |

**Verdict:** Doubling the budget didn't help. Vision policy plateaus at
12-14/20 hits across 100K, never finds the R3 attractor.

### R12 — vision at offset=0.30 (where proprio fails, 80K)

Curriculum: warmup 2K, ramp 2K-20K → 0.30.
- t=10K (in ramp, offset~0.13): -10 ± 734, 15/20 hits
- t=20K (just locked at 0.30): -1222, 0/20 hits
- All subsequent: 0/20 hits, severe floor

### R13 — proprio control at offset=0.30 (matched R12 minus vision, 80K)

Same outcome as R12 — 0/20 hits at offset=0.30 throughout. Vision and
proprio fail identically.

**Verdict R12+R13:** offset=0.30 is a geometric impossibility, not a
learning problem. The body's arms cannot reach 0.30m laterally from the
cart's center, so no sensory modality can fix it.

### R14 — vision at offset=0.20 (where direct proprio struggled, 80K)

R8 (proprio direct at 0.20, 16 envs subproc) had reached only 5/20
both-touched. R14 with vision and 8 envs DummyVecEnv tested whether
vision helps.

R14-best at offset=0.20 (touch-counting eval, 20 seeds):
- mean reward -151, both=14/20, only-one=5/20

R14 dramatically beat R8 (14/20 vs 5/20). Apparent vision win. BUT
vision-ablation tests revealed:

**Single-step ablation:** action delta = 0.023 (action space [-1,1], 25-dim)
when pixel columns are zeroed. Vision channel barely affects actions.

**Episode-level ablation:** running 20 episodes with pixels NORMAL gives
14/20 both-touched (reward -151). Same 20 episodes with pixels ZEROED gives
**14/20 both-touched (reward -150)**. Identical. Vision is provably not used.

### R15 — proprio control matching R14 (DummyVecEnv + 8 envs, 80K)

The cleanest test of whether R14's win over R8 was about vision or about
the n_envs/vec_env setup. Added `--force-dummy-vec-env` flag to
train_crawler.py so proprio could match vision's required setup.

R15-best at offset=0.20: mean -217, **both=12/20**, only-one=8/20.

**Verdict R14 vs R15:** R14 (vision) gets 14/20; R15 (proprio control) gets
12/20. Difference within noise. The R14 vs R8 gap (14 vs 5) was an
n_envs/vec_env confound, NOT vision. Combined with the ablation test,
**vision is not load-bearing in any of the tested configurations.**

### R16 — entropy annealing (0.5→0.2 over t=15K–30K, 50K)

Tests whether annealing entropy after the breakthrough preserves the
+366 peak instead of drifting to -438.

| t | ent_coef | reward | hits |
|---|---|---|---|
| 5K | 0.5 | -284 | 13/20 |
| 10K | 0.5 | -4 | 16/20 |
| 15K | 0.5 (anneal begins) | +261 | 19/20 |
| 20K | ~0.4 | +125 | 18/20 |
| 25K | ~0.3 | +43 | 17/20 |
| 30K | 0.2 (anneal ends) | -96 | 15/20 |
| 35K | 0.2 | -290 | 13/20 |
| 40K | 0.2 | +8 | 17/20 |
| 45K | 0.2 | -175 | 10/20 |
| 50K | 0.2 | -62 | **16/20** |

**Verdict:** Annealing partially mitigates late-training drift. End-state
reward -62 (vs R3's -438) and end-state hits 16/20 (vs R3's 11/20). But
the peak (+261) is lower than R3's (+366), and the drift still happens
DURING the annealing window — suggesting the late drift is driven by SAC
dynamics beyond just entropy level. Worth combining with replay-buffer
warm-start (not yet implemented).

### R17 / R18 — random ball positions per episode (vision vs proprio control)

Added `--random-ball-box dx,dy` to the env: each episode, ball positions
are sampled uniformly from box `(anchor + Uniform(±dx), anchor + Uniform(±dy))`
around the fixed_ball_positions anchors. With anchors `(±0.10, ±0.35)`
and box `(0.10, 0.05)`, ball x varies over [0.00, 0.20] each reset (the
sign is determined by which anchor, but the magnitude varies 0-0.20).

Hypothesis being tested: with ball position UNPREDICTABLE from proprio,
vision should become genuinely useful — this is the structural condition
the project's theoretical framework requires.

**R17 (vision + random balls, 50K, ent=0.5, 8 envs DummyVecEnv):**
- 40-seed deterministic eval (each seed gets a different random ball set):
  mean +13, **both-touched 34/40 (85%)**, only-one 5/40

**R18 (proprio + random balls, 50K, ent=0.5, 8 envs DummyVecEnv — matched control):**
- 40-seed deterministic eval: mean +127, **both-touched 36/40 (90%)**,
  only-one 4/40

**Verdict:** Proprio outperforms vision even at the task that should
require vision. Random-ball-box of (0.10, 0.05) is small enough that the
body's "blanket sweep" reach (extending arm forward and to the side)
covers all sampled positions without needing to know ball location.
Even the random-position task doesn't make vision load-bearing.

This is the cleanest possible refutation of "vision is needed when ball
position is unknown". For vision to genuinely matter, the body would need
either: (i) much smaller effective reach (e.g., only one arm with sharp
spatial selectivity), (ii) a penalty for the wrong reach direction
(currently the action cost is uniform), or (iii) a much larger random box
that exceeds the body's blanket-sweep coverage.

**R17 vision-ablation (the cleanest test in the session):**
- Single-step action delta when pixel columns zeroed: **0.009** (action
  space [-1,1], 25-dim — random ~1-3). Even lower than R14's 0.023.
- Episode-level: pixels NORMAL → 16/20 both-touched, pixels ZEROED →
  **17/20 both-touched** (slightly *better* with vision disabled).

The trained R17 policy literally does not use the camera. In the exact
condition designed to make vision necessary, the CNN feature extractor
has been driven to noise-output by the optimizer. The proprio + cart-pos
pathway alone produces all the policy's behavior.

This is the strongest refutation of "vision becomes load-bearing under
sparse-reward + position uncertainty" the project has produced. Three
independent settings now show it (R6/R10 vision at offset=0.15, R14 vs
R15 at offset=0.20, R17 vs R18 at random offsets) and one shows that
even when vision SHOULD matter, the CNN is reliably ignored.

### R19 — ent=0.5 + velocity bonus (combine R3 + R5 mechanisms) — **NEW BEST**

R5 showed velocity-bonus alone partially breaks the freeze (final +165
at t=29952). R3 with ent=0.5 finds the +366 peak but drifts. R19 combines
both. Hypothesis: velocity bonus provides an additional anti-stillness
gradient that helps the policy stay at the extension attractor instead
of drifting away.

| t | reward | std | floor | hits |
|---|---|---|---|---|
| 5K | -279 | 929 | 4/20 | 13/20 |
| 10K | +68 | 640 | 1/20 | 16/20 |
| 15K | **+380** | **7.8** | 0/20 | **20/20** |
| 20K | **+366** | 12 | 0/20 | **20/20** |
| 25K | -25 | 774 | 2/20 | 16/20 |
| 30K | -73 | 662 | 0/20 | 13/20 |
| 35K | -264 | 880 | 1/20 | 13/20 |
| 40K | -247 | 888 | 2/20 | 13/20 |
| 45K | -608 | 914 | 3/20 | 8/20 |
| 50K | -352 | 1021 | 3/20 | 13/20 |

**Verdict:** R19 is the new best. Three improvements over R3:
1. Higher peak: +380 vs R3's +366
2. Tighter std at peak: 7.8 vs 14.4 (more reliable)
3. 20/20 hits maintained across TWO consecutive checkpoints (t=15K AND
   t=20K) — first run to do this. The peak is broader.

R19 also generalizes further to harder offsets than R3:

| offset | R3-best both | R19-best both |
|---|---|---|
| 0.05 | 19/20 | 19/20 |
| 0.10 | 18/20 | 18/20 |
| 0.15 | 19/20 | 18/20 |
| 0.20 | 15/20 | **16/20** |
| 0.25 | 5/20 | **8/20** |
| 0.30 | 0/20 | 0/20 |

R19's reach radius is slightly wider than R3's. The velocity bonus
seems to encourage more aggressive arm-extension, paying off at the
edge of the reachable zone.

Late-training drift is still present (post-t=20K) but slightly milder
than R3's. Combining R19's setup with R16's entropy annealing might
fully stabilize the policy — that's a clean next experiment.

### R20 — R19 + entropy annealing (R3 mechanism + vel bonus + anneal)

Final stabilization attempt. Same config as R19 but with ent_coef
annealed 0.5→0.2 between t=15K and t=30K.

| t | reward | std | floor | hits |
|---|---|---|---|---|
| 5K | -279 | 929 | 4/20 | 13/20 |
| 10K | +68 | 640 | 1/20 | 16/20 |
| 15K | **+380** | **7.8** | 0/20 | **20/20** |
| 20K | +256 | 505 | 1/20 | 19/20 |
| 25K | +190 | 567 | 1/20 | 18/20 |
| 30K | -100 | 781 | 2/20 | 13/20 |
| 35K | +221 | 474 | 0/20 | 18/20 |
| 40K | **+334** | 187 | 0/20 | **19/20** ← second peak |
| 45K | -247 | 913 | 4/20 | 13/20 |
| 50K | +64 | 709 | 2/20 | 16/20 |

**Verdict:** Same peak as R19 at t=15K. The annealing kept 18-19/20 hits
through t=20K-25K (R19 had dropped to 16, 13 here). Brief dip at t=30K
when anneal completed, then a SECOND peak at t=40K — the policy found
another good attractor with the lower entropy. Late-training average is
much better than R3 or R19:
- R3 second half (t=25K-50K): avg reward ~-200, ~12/20 hits
- R19 second half: avg ~-300, ~12/20 hits
- R20 second half: avg **~+96, ~15/20 hits**

R20 is the most STABLE extension policy, even if R19 has slightly higher
peak. For deployment / further experiments, R20's late-training behavior
is the safer bet.

Two of R20's checkpoints (t=15K and t=40K) have 19-20/20 deterministic
hits at offset=0.15 — the policy is robust to late-training perturbation
and is finding the extension attractor twice independently.

### R21 — R20 recipe at offset=0.20 (direct training, 80K)

Final test: can R20's full recipe (ent=0.5 + vel_bonus=0.10 + entropy
anneal 0.5→0.2 over t=25K-50K) unlock direct training at the harder
offset that R8 (simpler recipe) had failed on?

| t | reward | hits |
|---|---|---|
| 10K (ramping) | +253 | 19/20 |
| 20K (locked at 0.20) | -385 | 11/20 |
| 30K | -1548 | 1/20 — severe collapse |
| 40K | -754 | 6/20 |
| 50K (anneal complete) | -694 | 8/20 |
| 70K | -276 | 10/20 |
| 80K | -514 | 5/20 |

**Verdict:** Even R20's recipe cannot make direct training at offset=0.20
work. The curriculum target is too aggressive — once offset locks at 0.20,
the policy hasn't built enough extension yet to maintain 20/20.
Compare to R3-best transferred to 0.20 in inference (no training):
**15-16/20 both-touched** — which the directly-trained R21 never reaches.

**Generalized insight:** The breakthrough is curriculum-shape-sensitive,
not just hyperparameter-strength-sensitive. R3's curriculum (0→0.15)
finds extension that generalizes to 0.20; R21's curriculum (0→0.20)
overshoots before extension can consolidate. The lesson for the future:
**target the curriculum at the BOUNDARY of trainable, not at the
generalizable extrapolation**. The policy can learn extension within
its reach, then generalize beyond — but not learn it AT the boundary
directly.

To push the trainable boundary further: need either (i) two-stage
curriculum (train at 0.15 to convergence, save buffer, then ramp 0.15→0.20)
— requires replay-buffer save/load not yet implemented; or (ii) larger
body (longer arms, wider reach radius).

---

## Second overnight session — 2026-05-15 → 2026-05-16

Three directions executed in sequence: multi-seed validation of R20,
buffer-preserving warm-start, vision-required substrate.

### R22 / R23 — multi-seed validation of R20 recipe

R22 (seed=1): peak +387.5 ± 26.4 at t=34944, 20/20 hits, end +231 / 18/20.
R23 (seed=2): peak +360.6 ± 56.9 at t=14976, 19/20 hits, end -459 / 10/20.

Combined with R20 (seed=42, +380): **three seeds confirm the breakthrough.**
Peak magnitude varies +361 to +388, but all three find 19-20/20 hits
at offset=0.15. R22 produced the most stable end-state of the three
(+231 / 18/20). The R20 mechanism is robust.

### Buffer save/load — implemented but doesn't fix warm-start

Added `model.save_replay_buffer()` after training and `model.load_replay_buffer()`
on `--init-from`. R23's buffer saved cleanly (332 MB on disk).

**R24 (warm-start R23-best with buffer, curriculum 0.15→0.20):**
Training rollout `ep_rew_mean` stays high (421→423→380→362→292) for the
first ~50K steps — confirming the buffer load worked. But deterministic
eval drops to 2-5/20 hits and reward -428 to -1240. Best checkpoint at
t=65K: -427 / 5/20.

**Verdict:** Buffer-preservation is not sufficient. When the curriculum
target changes (0.15 → 0.20), the off-policy gradient pulls in opposite
directions: old buffer transitions say "do X to win at 0.15", new
transitions say "X is wrong at 0.20", and the deterministic policy
output ends up incoherent even though the stochastic rollout reward
stays high. R7 / R11 / R21 failed in similar ways without buffer; R24
fails despite buffer. The warm-start direction is harder than expected
— it likely requires either (a) buffer annealing (drop old transitions
as task changes), (b) much smaller curriculum jumps (0.15 → 0.16
incrementally), or (c) a fundamentally different fine-tuning algorithm
(BC + DAgger, or offline RL on the buffer before resuming online).

### R26 / R27 — disappearing balls, timeout=300

Substrate: each ball vanishes after 300 env-steps if not touched. Idea:
force AB to act fast, vision should help by telling AB where to look.

**R26 (vision):** best b_touched=3/20, only_one=17/20, mean reward +218.
**R27 (proprio control):** best b_touched=5/20, only_one=14/20, mean reward +241.

Both touch ball1 reliably (14-15/20 across full eval) but ball2 expires
in 15/20 episodes — the 300-step timeout is too tight given the cart
must traverse from one side to the other to reach both balls. Vision
underperforms proprio slightly. The substrate didn't actually force
vision because ball1 was still reachable with cart-sweep alone.

### R28 / R29 — disappearing balls, timeout=150 (tightest test)

Halved the timeout. Now even ball1 requires fast action.

**Final touch counts (20-seed eval):**
| | Vision (R28) | Proprio (R29) |
|---|---|---|
| Mean reward | +197 | +197 |
| Ball1 touched | 15/20 | 15/20 |
| Ball2 touched | 5/20 | 5/20 |
| Ball1 expired | 5/20 | 5/20 |
| Ball2 expired | 15/20 | 15/20 |

**Verdict:** literally identical results. Vision provides zero advantage
even at the tightest timeout. The 150-step deadline doesn't force
vision because the cart's randomly-chosen direction means ball1 is
either trivially-near (and gets touched) or trivially-far (and expires)
— in either case, no visual prediction is needed.

## Vision: definitive verdict across six substrates

Tested independently across:
1. R6, R10 — offset=0.15 fixed (vision had longer training budget) — not load-bearing
2. R14 vs R15 — offset=0.20 with matched n_envs/vec_env — not load-bearing
3. R12 vs R13 — offset=0.30 (proprio impossible) — both fail equally
4. R17 vs R18 — random ball positions per episode — proprio outperforms vision
5. R26 vs R27 — disappearing balls, timeout=300 — proprio slightly ahead
6. R28 vs R29 — disappearing balls, timeout=150 — identical

Plus two vision-ablation tests:
- R14 ablation: action delta 0.023, pixels-zeroed gives identical 14/20.
- R17 ablation: action delta 0.009, pixels-zeroed actually IMPROVES result.

**Conclusion:** in this body / cart-substrate / SAC + curriculum setup,
vision cannot be made load-bearing through substrate tweaks. The body's
geometric reach + proprio + high entropy covers the task space
completely. To genuinely test "vision is load-bearing", the project
would need fundamental architectural changes:

(a) **Different body** — e.g. one-armed AB, or much smaller swept volume.
(b) **Different task** — moving balls with random per-step velocity
    requiring trajectory prediction; or ball-position-from-vision-only
    where contact requires the head having looked at the ball within N
    steps before contact.
(c) **Different training algorithm** — DrQ-v2 or other image-aware RL
    that doesn't let the CNN converge to noise output.
(d) **Different reward structure** — pay reward only when AB has been
    OBSERVING the ball (head pointed at it) for N consecutive steps
    before contact. This would directly tie reward to visual attention.

### R30 — strength=1.0 + R20 recipe at offset=0.20 (positive)

Tests whether stronger arms widen the trainable reach radius. Default
strength=0.7 capped direct training at offset=0.20 (R8 got 5/20).

| t | reward | hits |
|---|---|---|
| 10K | -215 | 14/20 (ramp) |
| 20K | -300 | 12/20 (at 0.20) |
| 30K | -1526 | 3/20 (collapse) |
| 60K | -152 | 10/20 (recovery) |
| 80K | -287 | 8/20 |

20-seed touch-counting at offset=0.20 — R30-best:
**14/20 both-touched, 6/20 only-one** (mean reward +25).

Compared to existing benchmarks at offset=0.20:
| Policy | both-touched | mean |
|---|---|---|
| R3-best transferred (strength=0.7, trained at 0.15) | 15/20 | +16 |
| R8-best (strength=0.7, direct train) | 5/20 | -646 |
| R30-best (strength=1.0, direct train) | **14/20** | **+25** |

**Verdict:** strength_scale=1.0 unlocks direct training at offset=0.20.
R30 matches the transferred R3-best (15 vs 14 / 20) — direct training
finally works at the harder offset when the body has enough strength.
This suggests the "0.20-is-trainable boundary" is partly a strength
limit, not purely a learning limit.

### R31 — strength=1.0 at offset=0.25 (boundary push)

Tried to push the trainable boundary further. Curriculum to 0.25 with
strength=1.0 over 80K. Result: collapsed.

| t | reward | hits | notes |
|---|---|---|---|
| 10K | +216 | 17/20 | during ramp (offset ~0.13) |
| 20K | -1294 | 0/20 | offset just locked at 0.25 |
| 30K | -1928 | 0/20 | severe floor |
| 60K | -650 | 2/20 | partial recovery |
| 80K | -864 | 0/20 | final state |

Touch counting on R31-best:
- At offset=0.25: 0/20 both-touched, 18/20 only-one, mean -841
- At offset=0.20: 6/20 both-touched, 13/20 only-one, mean -577

**Verdict:** Direct training at offset=0.25 fails even with strength=1.0
+ full R20 recipe + 80K steps. Same curriculum-target-sensitivity pattern
as R8 and R21: the policy can't consolidate extension fast enough before
the curriculum reaches an unreachable target. R31 also degrades at the
intermediate offset (0.20: 6/20 vs R30's 14/20) — the harder curriculum
target poisons the easier-offset performance too.

**Final trainable-boundary picture:**
| offset | best result | how |
|---|---|---|
| 0.15 | 20/20 (R3/R20/R22) | direct, strength=0.7 |
| 0.20 | 15/20 (R3 transfer) or 14/20 (R30 direct, strength=1.0) | transfer or strength |
| 0.25 | 8/20 (R19 transfer) | transfer only — no direct training succeeded |
| 0.30 | 0/20 | geometric impossibility for all bodies/recipes tested |

The trainable-via-direct-curriculum boundary at strength=1.0 is between
0.20 (R30 works) and 0.25 (R31 fails). Pushing past requires either
more body strength, a multi-stage curriculum with proper buffer-aware
fine-tuning, or a different body geometry.


## Final findings — the project's story rewritten

Before tonight, the project's failure mode was "policy converges to
frozen posture; cart kinematics or geometric overlap produces accidental
contacts; no learned reach." 750K total HER training steps across
Phases D, E, E2, F, F-2k had failed to produce sustained reach.

Tonight, with hindsight, the problem was simple and the fix was simple:
**ent_coef=0.5 instead of ent_coef=0.2**. Three hyperparameter combinations
break the freeze attractor independently — high entropy (R3), dense
approach reward to a hand-distance metric (R2, with the env bug fixed),
and a joint-velocity bonus (R5 after implementing it). High entropy is the
cleanest single intervention.

The vision question — central to the project's theoretical framework —
has a clean experimental answer for this task: **vision is not necessary
to learn arm-extension, and adding vision does not improve over a
matched-setup proprio control.** This is not a refutation of the
theoretical framework — it just means this task lacks the structural
property the framework requires (vision must provide information not
recoverable from the agent's own joint state and reward signal). To make
vision necessary, the substrate needs to be redesigned so that contact
is *not* discoverable by random arm motion at high entropy — e.g.,
moving balls that disappear before contact unless visually tracked, or
random ball positions per episode that exceed proprio's exploration
reach radius.


---

# Phase I Overnight (MICOA) — 2026-05-20

Branch: `feature/phase-i-micoa`
Commit: `ecd682b` (MICOA wiring, unpushed)
Plan: see `alien_baby/todo.md` Phase I section
Authorization: full autonomy; no push/PR; β auto-tune 0.1 → 0.3 → 1.0;
auto-extend same run to 250K on success; no wall-clock stop.

## Decision-tree

- Heartbeat ~25 min via ScheduleWakeup; harness also notifies on process exit.
- At 30K boundary: if `sigma_combined` flat AND `kl_agreement` flat-high
  vs first 5K window → kill + relaunch at next β.
- On R36 (80K) finish: render 1 episode, vision-ablation eval, write
  `PHASE_I_RESULTS.md`. If sigma decreasing AND ablation delta > 0.05,
  extend to 250K via `--init-from final_model.zip` for +170K steps.
- On 250K finish: render, eval, write up, done-chime, stop.

## Timeline

### Launch
- **R36 launched** at t≈+0 (PID 79773, β=0.1, 80K steps)
  - Config: cart-mode constant_velocity_bouncer, ball-speed=0.0,
    n_envs=8, max_steps=2000, MPS, strength=1.0, hip-actuation=off,
    memory-obs, ent 0.5→0.2 over [15K,30K], velocity-bonus=0.10,
    curriculum (warmup=2K, ramp_end=15K, final_offset=0.15), seed=42
  - Smoke test (200 steps) passed before launch — MICOA forward + KL
    backward work on MPS
  - Output: `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1/`
  - Log: `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1.log`
- 19s in: 4500 total_steps, FPS≈282 (warmup, learning_starts=10K)

### Heartbeat 23:19 — scalar-visibility bug, kill+restart
- R36 had reached 35K steps (FPS ~37 in training phase, eval table OK).
- **Bug**: `micoa/sigma_combined` and `micoa/kl_agreement` were not visible
  in the stdout log. Cause: `MICOAConfirmationCallback` records them via
  `self.logger.record(...)`, but SB3's stdout writer only prints them when
  `tensorboard_log` is configured (which `train_crawler.py` does not).
  Values were being computed but written nowhere visible — the overnight
  β-decision heartbeat could not function.
- **Action**: killed R36 (PIDs 80455/80457/85580), patched
  `MICOAConfirmationCallback._on_step` to also `print()` the scalars to
  stdout (commit on top of `ecd682b`). Renamed dead artifacts to
  `*_noscalar*` for evidence and relaunched fresh.
- Cost: ~15 min of warm-up + training wasted. Cheap given the alternative
  was an entire overnight run with no diagnostic visibility.
- **R36 relaunched** at 23:21 (β=0.1, 80K steps, harness job biptizhxr).
  Same command as the first launch.

### Heartbeat 23:24 — relaunch healthy, awaiting first MICOA print
- R36 at 9920 timesteps, FPS 100 (warmup phase, learning_starts=10K).
- Process alive (PIDs 86115/86117/87264).
- Zero [MICOA] log lines yet — expected, since SAC's warmup uses random
  action sampling and does not query the features_extractor. First MICOA
  print will fire once we cross 10K and the policy is engaged.
- Monitor armed (5-min timeout) for first [MICOA] line.
- Eval script `alien_baby/visualization/eval_phase_i.py` prepared and
  import-tested for post-training use.

### Heartbeat 23:41 — warmup completed, Monitor was too short
- R36 just crossed 10K (warmup boundary). First eval ran at exactly 10K
  (mean_reward=135). Training policy now engaged.
- Monitor timed out at 5 min but warmup took ~19 min (FPS dropped to 86-100
  once vision rendering kicked in). Did not re-arm — letting the next cron
  fire at 23:51 do the next check.
- Process healthy. Next ~500 training steps should produce the first
  [MICOA] log line.

### Heartbeat 23:42 — *two* architectural bugs found, kill+restart
- R36 at 12568 steps, well past warmup, but STILL zero [MICOA] log lines.
  Probed a 15K-step checkpoint and discovered:
  1. `policy.features_extractor` is `None`. SAC stores extractors at
     `policy.actor.features_extractor` and `policy.critic.features_extractor`,
     and by default they are SEPARATE instances (share_features_extractor=False).
     The callback's isinstance check was always failing — no print, no record.
  2. **Worse: MICOASAC.train() had the same bug**, so the β·KL backward
     was a silent no-op for the entire run. The MICOA loss was never
     actually firing — the run was just SAC + extra encoder parameters.
- Also discovered the stored last_mu_* tensors cannot be reused for backward
  (the autograd graph is freed after super().train() / rollouts run no_grad).
- **Fixes (commit c6d5694 → next commit):**
  1. `share_features_extractor=True` in MICOA policy_kwargs (one extractor
     instance for actor + critic, architecturally correct for "shared Z").
  2. New `_get_micoa_extractor(policy)` helper that handles both layouts;
     used in callback and MICOASAC.train.
  3. MICOASAC.train now samples a fresh batch + does an explicit forward
     pass with grad enabled, then runs the KL backward over a dedicated
     `_micoa_opt` (extractor params only, LR matched to actor).
  4. KL loss mirror-printed to stdout (callback's print already added).
- **Smoke test (400 steps): KL fell 1.18 → 0.29 monotonically.** Architecture
  now confirmed working.
- Old artifacts archived as `*_silentkl*`. R36 relaunched at 23:43 as harness
  job `bnk6x40ck`.
- Total wasted compute so far: ~50K steps. Worth catching now vs. discovering
  in the morning that the architecture never trained.

### Heartbeat 23:51 — MICOA confirmed working but β=0.1 likely too high; FS save crash; relaunching
- The fixed R36 (job bnk6x40ck) ran from 23:43 to ~23:50, reached 15K steps,
  then crashed with `TimeoutError [Errno 60]` inside `EvalCallback._on_step` →
  `zipfile.close()` while writing `best_model.zip`. Macos Spotlight / iCloud
  briefly holding the file is the most plausible cause; the bug is that SB3
  treats this recoverable IO error as fatal.
- **Architecture verdict so far: MICOA is firing.** 626 [MICOA] log lines.
  KL fell 1.13 → 0.006 between t≈10K and t≈15K. But sigma_combined barely
  moved (0.710 at t=8K → 0.698 at t=12K, ~1.7% drop).
- That pattern matches "**β too high → forced agreement, not earned**":
  the encoders are collapsing their distributions to match each other in
  μ-space without finding tighter regions of Z. Corner is NOT forming.
- **Decision (deviating from the auto-tune sequence but documenting):**
  - Pre-approved sequence was 0.1 → 0.3 → 1.0, designed for "kl never moves"
    (β too low). We observed the OPPOSITE failure mode. Raising β to 0.3
    would make the forced-agreement pathology worse, not better.
  - Correct next step is β=0.03 or β=0.01, which was not pre-approved.
  - **Conservative move:** run β=0.1 to completion (preserves user's
    experimental design), document the observation, recommend β=0.03 for
    the next iteration in PHASE_I_RESULTS.md. Do not auto-launch β=0.03.
- **Fix for the FS crash:** added `_SafeSaveEvalCallback` in train_crawler
  that retries on TimeoutError/OSError with exponential backoff (1/2/4/8s).
  Commit on top of previous fixes.
- Archived crashed run as `*_savecrash*`. R36 relaunched at 23:53 as
  harness job `bjyhutvjv`. Same config (β=0.1).

### Heartbeat 23:53 — 4th launch alive, in warmup
- 30s post-launch, at 4552 steps. PIDs 92527/92529/92817. Log 53 lines.
- No MICOA prints expected until past learning_starts=10K (~15 more min).
- No actions taken. Next check at 00:17 cron fire or sooner on log/error
  events.

### Heartbeat 00:03 — past 30K boundary, sigma slowly decreasing, KL near zero
- R36 at 25K steps. 2143 MICOA prints. PIDs 92527/92529/95746.
- SafeSave fired once and recovered (retry 1/4 succeeded). Fix works.

| t   | sigma_combined | Δ% from t=8K | kl_agreement |
|-----|----------------|--------------|--------------|
| 8K  | 0.7105         | —            | 0.813        |
| 12K | 0.6981         | -1.7%        | 0.035        |
| 16K | 0.6883         | -3.1%        | 0.0059       |
| 20K | 0.6784         | -4.5%        | 0.0013       |
| 24K | 0.6682         | -5.9%        | 0.0022       |

- **Revised interpretation:** the corner IS forming, slowly. KL near zero
  means μ_p ≈ μ_v AND σ_p ≈ σ_v, and σ_p itself is shrinking ~1% per 4K.
  Slow but real.
- Pre-approved kill criterion was "sigma flat AND kl flat-HIGH". Current
  state is sigma decreasing 5.9% AND kl near zero. Neither matches. The
  auto-tune sequence (0.3 → 1.0) is designed for the wrong failure mode
  and would only force agreement harder. **Continue at β=0.1.**
- FPS ~85-90. ETA for 80K: ~10–15 more minutes. R36 should complete
  before 00:25.

### Heartbeat 00:20 — sigma decrease accelerating, corner clearly forming
- R36 at 59K. Healthy. SafeSave fired once total, recovered.
- sigma_combined: 0.7105 (8K) → 0.6682 (24K) → 0.6426 (40K) → 0.6105 (52K)
  → **0.5667 (56K)** — accelerated downward; -20.2% total, -7% in last 4K.
- KL stable near zero (0.0007–0.0015) throughout — encoders agreeing.
- Episode lengths growing (~824→1193 then 1059) — policy doing more
  during episodes.
- ETA 80K: ~6–8 more min at current FPS. **Continue.**

### 00:34 — R36 completed cleanly. Eval verdict.
- 80K final_model saved at 00:34. sigma_combined 0.7105 → 0.4516 (-36.4%).
- Eval (20 seeds, static balls, offset 0.15) on final_model:
  - mean_reward = -746 (regressed from +127 at 5K best)
  - both-touched = 6/20 (regressed from 18/20 at 5K best)
  - **Vision ablation delta = 0.0019** — way below 0.05 success threshold.
  - Episode pixels-zeroed vs normal: 5/20 vs 6/20 (delta +1).
- **Per pre-approved criteria, do NOT extend to 250K.** Sigma decreasing
  was satisfied; ablation > 0.05 was not. Stop.
- Diagnosis: β=0.1 too high. KL collapsed to ~0.001 in first 4K training
  steps, before vision could develop channel-specific features. The
  encoders agreed on whatever proprio already knew; vision became
  redundant. σ_combined dropped because both encoders became more
  confident together, not because vision learned to confirm something
  proprio already encoded.
- The pre-approved auto-tune sequence (0.1 → 0.3 → 1.0) is for "kl flat
  high" failure. We have the opposite. Did NOT auto-escalate. Human
  decides β next.
- Writeup: `alien_baby/PHASE_I_RESULTS.md`. Cron heartbeat deleted.
- Total elapsed: ~3.5 hours of which ~75 min was actual training (the
  rest was bug discovery + recovery).




