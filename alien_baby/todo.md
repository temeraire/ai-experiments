# Alien Baby: Rebuild Plan

## Goal
Rebuild from scratch with three fixes before attempting vision integration:
1. Head motion is too wild (position servo snaps instantly, no damping)
2. Creature barely moves (no incentive to cover ground)
3. Vision and proprio trained serially — replace with joint training + dialogue architecture

---

## Phase 1: Fix head physics (XML change) ✅ DONE 2026-05-07

- [x] In `platform_creature.xml`, reduce head actuator `kp` from 30 → 5
- [x] Apply the same changes to `platform_creature_v9.xml`
- [x] Smoke-render an untrained policy to confirm no crash — renders cleanly
- Note: joint damping left at default (2.0) — with kp=5 the system is already
  heavily overdamped; additional damping is not needed

## Phase 2: Fix locomotion reward (env change) ✅ DONE 2026-05-07

- [x] Added `VELOCITY_BONUS_SCALE = 0.02` constant at top of env file
- [x] Added velocity bonus in `step()`: `VELOCITY_BONUS_SCALE * ||torso_qvel[0:3]||`
- [x] Added `_velocity_bonus_sum` tracking (init, reset, step)
- [x] Logged as `velocity_bonus_sum` in `_info()`
- [x] All existing rewards unchanged

## Phase 3: Train new stage1 with fixes

- [ ] Train new stage1 from scratch (~500K steps) using fixed XML + env
  - Use `--run-tag stage1_v9_headfix_velbonus` (or similar)
  - Same MPS + 16 envs defaults
- [ ] After training, render 5 episodes deterministically and verify:
  - [ ] Head moves smoothly (no panic oscillation)
  - [ ] Creature covers ground actively
  - [ ] Touch rate is at least as good as old stage1 (≥ 60% blind)
- [ ] Record new blind baseline touch rate

## Phase 4: New joint architecture — "dialogue" model

Design principle: vision and proprio are equal peers that must agree on an action.
Neither has structural priority. Disagreement is explicitly measurable.

Architecture:
```
proprio_encoder  →  action_proprio   (full action proposal)
vision_encoder   →  action_vision    (full action proposal)
agreement_weight (learned, context-dependent scalar or vector)
final_action     =  w * action_proprio + (1-w) * action_vision
disagreement     =  ||action_proprio - action_vision||
```

- [ ] Write `DialogueSAC` class in a new file `alien_baby/agents/train_v9_dialogue.py`
  - Both encoders train jointly from scratch (no freeze, no weight transfer)
  - Agreement weight `w` is learned (e.g. output of a small MLP on concatenated features)
  - Disagreement is logged each step as a diagnostic signal
  - A small consistency loss `λ * disagreement` encourages convergence without forcing it
- [ ] Keep the policy head (actor/critic) standard SAC — only the encoder structure changes
- [ ] Vision ablation is trivial: run with pixels zeroed, compare `action_vision` to baseline

## Phase 5: Train joint model

- [ ] Train `DialogueSAC` from scratch on the fixed env (~500K steps)
- [ ] Render after 150K, 300K, 500K steps
- [ ] Success criteria (both must be true):
  - Touch rate ≥ 60% (matches blind baseline)
  - Disagreement is nonzero AND decreases as training progresses
    (vision and proprio converging on the same answer = integration happening)
- [ ] Measure vision ablation sensitivity at each checkpoint

---

## Phase I: MICOA — Product of Experts + KL agreement loss  ← NEXT

**Problem diagnosed by Phase H (REFUTED verdict):**
Vision has never been load-bearing across 7 experiments. Ablation delta ~0.02
in every run: vision is inert. SAC always finds proprio first and the pixel
encoder gets no useful gradient.

**Architectural diagnosis (the "tubes vs. corners" insight):**
Concatenating proprio + pixels into one flat vector and feeding SAC is welding
two tubes end-to-end — it makes a longer rod, not a box. There is no component
in the network whose *job* is to detect when two independent streams are
constraining the same world-state simultaneously. That detector is the
missing piece.

Earlier mitigations all leave the architecture flat:
- Proprio-dropout, staged training, vision-only fine-tuning — none of these
  build a corner; they just change which tube carries the load.

**The fix: structurally enforce confirmation via Product of Experts.**

Each stream encodes to a Gaussian over a shared latent Z:
```
proprio → encoder_p → (μ_p, σ_p)
vision  → encoder_v → (μ_v, σ_v)
```
The PoE fuses them with parameter-free math:
```
μ_combined  = (μ_p/σ_p² + μ_v/σ_v²) / (1/σ_p² + 1/σ_v²)
σ_combined² = 1 / (1/σ_p² + 1/σ_v²)
```
When the streams agree, `σ_combined` becomes small — that is the corner. When
they disagree, `σ_combined` stays large and the policy represents its own
uncertainty. The policy acts on Z sampled from the combined distribution.

**KL agreement loss closes the escape hatch.**
Without an explicit penalty, vision can output a wide `σ_v` (high uncertainty)
and the PoE just lets proprio carry the signal — same failure as today. So we
add symmetric KL between the two encoder distributions to the actor loss:
```
total_loss = sac_loss + β * symmetric_KL(p, v)
```
Now vision must actually find the same Z as proprio, from pixels alone.
Start with β = 0.1.

### Architecture file (DONE)

`alien_baby/agents/micoa_architecture.py` contains:
- `ProprioEncoder`, `VisionEncoder` (output μ, σ each)
- `product_of_experts()` — parameter-free fusion
- `encoder_agreement_loss()` — symmetric KL
- `MICOAExtractor` (SB3 `BaseFeaturesExtractor`, outputs `[z_combined | μ_p | μ_v]`,
  features_dim = `LATENT_DIM * 3`)
- `MICOASAC` — SAC subclass that adds `β * KL` to the actor optimizer step
- `MICOAConfirmationCallback` — logs `micoa/sigma_combined`, `micoa/kl_agreement`

### Implementation tasks (wiring)

- [ ] Add `--micoa` CLI flag to `train_crawler.py` (default False)
- [ ] Add `--micoa-beta` CLI flag (default 0.1)
- [ ] When `--micoa` is set and `--vision` is set, in the policy_kwargs branch
      (`train_crawler.py:303-311`) swap `StereoCrawlerCNN` for `MICOAExtractor`
      with kwargs `(proprio_dim=PROPRIO_DIM, cam_h, cam_w, in_channels, latent_dim=64)`
- [ ] When `--micoa` is set, build `MICOASAC(..., micoa_beta=args.micoa_beta)`
      instead of `SAC(...)` at line 340. HER + MICOA is out of scope for R36.
- [ ] When `--micoa` is set, append `MICOAConfirmationCallback(log_freq=500)` to
      `callback_list` (after the existing CheckpointCallback at line 343)
- [ ] Smoke-render an untrained MICOA policy via `sanity_render.py` to confirm
      cameras + physics still sane (per CLAUDE.md training-run workflow rule)
- [ ] Write `launch_phase_i.sh`:
  - **R36 (the diagnostic):** `--micoa --vision`, static balls (speed=0.0),
    80K steps, β=0.1
- [ ] Verify `--micoa` off path is bit-identical to current train_crawler

### Success criteria (R36, 80K steps)

This is a diagnostic run, not a perf run. We're asking: *does a corner form?*

| Metric | Pass | Fail mode |
|---|---|---|
| `micoa/sigma_combined` mean over last 10K | decreasing vs first 10K | flat → corner never forming |
| `micoa/kl_agreement` mean over last 10K | moderate (1–5), trending down | instant collapse → β too high; flat high → β too low |
| Ablation delta (pixels zeroed) at end | > 0.05 (vs 0.02 in all Phase G/H) | < 0.03 → vision still inert |
| Eval both-touched at static balls | ≥ 50% | catastrophic regression vs R34 |

If `sigma_combined` shrinks but `kl_agreement` collapses instantly, lower β.
If `sigma_combined` is flat and `kl_agreement` is flat-high, raise β to 0.3.

### Connection to Phase 4 (dialogue architecture) and Phase II (anticipation)

MICOA is the structural prerequisite for Phase 4's dialogue model — it gives
the architecture a real notion of "agreement" measured in latent space, not
just in action space.

Phase II (anticipatory vision) is the natural follow-on once R36 shows the
corner is forming: vision learns to predict proprio's next-step distribution,
i.e. to *see contact before feeling it*. That requires storing (μ_v, σ_v) in
the replay buffer. Defer until R36 succeeds.

If R36 fails (sigma_combined flat AND ablation < 0.03), the conclusion is that
even with structural confirmation pressure, vision cannot ground in this env
from RL alone — and the next step is the temporal predictive loss (Phase II)
rather than abandoning the architecture.

---

## Notes / open questions

- The drive/hunger question: does velocity_bonus_scale of 0.02 create purposeful search
  or just random wandering? Tune after watching Phase 3 videos.
- The "dialogue reaching agreement" loss: λ too high → forces agreement too early,
  vision never develops its own signal. λ too low → they never integrate.
  Start at λ=0.05 and watch disagreement curve.
- The old breakthrough checkpoint (micoa_freeze_zeroinit_cone30 30K) is from the serial
  approach and will not carry over to the dialogue architecture. It is preserved on disk.

---

## Rung 0 (2026-06-15): Can AB make ONE directed reach to a SENSED fixed target?

### Why
This session established AB is *blind to the ball* on proprio (obs = own body state +
2 touch bits; no target position) AND the cart runs set `--approach-reward-scale 0.0`,
so the hand-based approach reward was OFF. So `both_HAND=0` (never reaches with a hand)
is doubly explained: no perception of the target, no reward for moving a hand toward it.
Rung 0 = the missing bottom rung of David's stable-world curriculum: prove AB can learn a
directed HAND reach when it can sense the target. Decided scope: **reach-to** (seated AB,
arm extends), NOT travel-to (no locomotion).

### Design: a matched pair (the contrast IS the result)
- TEST   : target-in-obs ON  -> predict hand-touch rate climbs over training
- CONTROL: target-in-obs OFF -> predict hand-touch stays ~0 (reproduces the blindness)
If TEST succeeds and CONTROL fails, perception enables reaching. Clean, decisive.

### Setup (reuse cart env, seated)
- `--cart-speed 0` : static base; AB sits and reaches with an arm (cart kinematic, no walk)
- single fixed target (one `--fixed-ball-positions` entry -> ball2 inactive)
- target randomized in a small box each episode (so CONTROL can't memorize a constant)
- placed within hand reach but off body-center, so a hand (not a body sweep) is required
- `--approach-reward-scale 2.0` : already hand-based (_ball_dist = nearest-hand->ball)
- velocity bonus on (anti-freeze), hunger on (deadline)
- success/score = HAND touch (`hand_touched_ball*`)

### Code changes (small, opt-in, default-off = existing behavior unchanged)
- [ ] env: `target_obs` flag -> append egocentric target vector to proprio (mirror memory_obs)
- [ ] env: `hand_success` flag -> contact reward + termination use hand-touch (default off)
- [ ] env: pin the target so a bump can't move it (weld ball, or static cube geom)
- [ ] train_crawler: wire `--target-obs`, `--hand-success`
- [ ] eval: hand-touch rate vs training checkpoint (developmental curve) + held-out positions

### Pre-flight (training-run rules)
- [ ] render ONE episode from an UNTRAINED model with this config: AB seated upright,
      target within a hand's reach, physics/cameras sane. Fix before training.

### Decisions made (2026-06-15)
- A) sense = hand->target error vector, BOTH hands (6 dims). [DONE]
- B) pinned ball (pin_targets). [DONE]
- C) single target. [DONE]
- difficulty = short horizon + wide target; metric = steps-to-touch (+ hit-rate). [DONE]

### Code changes — DONE + smoke-tested
- [x] env: target_obs (6-dim hand->target both hands), hand_success, pin_targets
- [x] train_crawler: --target-obs / --hand-success / --pin-targets wired
- [x] smoke test: obs=75, pin drift 0.00000 m, hand_success terminates, no crash

### Pre-flight findings (the de-risking that mattered)
- Naive single-touch is TRIVIAL: random policy hits 100% in ~75 steps (workspace small
  enough that flailing sweeps it). Fix: wide target + short horizon, score steps-to-touch.
- Reach envelope (random rollout): x in [-0.25,0.21], y in [0.09,0.40], reach<=0.40 m.
- Targets at y<0.20 spawn under AB's body (occluded/trivial) -> push workspace forward.

### VALIDATED final config
- cart-speed 0 (seated); fixed-ball-positions "0.0,0.30"; random-ball-box 0.18,0.08
  (target x in [-0.18,0.18], y in [0.22,0.38] -- wide, in front of body, reachable)
- max-steps 50 (short horizon); hand-success; pin-targets; approach-reward-scale 2.0;
  velocity-bonus-scale 0.05; hip-actuation off
- random baseline at this config: 50% hit / ~31 steps-to-touch -> headroom for TEST
- PAIR: TEST = --target-obs ; CONTROL = (omit --target-obs), else identical, seed-matched

### TODO
- [ ] write steps-to-touch eval (mean steps-to-hand-touch + hit-rate@50, vs checkpoint)
- [ ] launch TEST + CONTROL (250K DroQ, checkpoints @10K to plot the learning curve)
- [ ] result: does TEST reach fast/reliably while CONTROL stays near baseline?

### RUN LOG
- rung0_TEST_targetobs (v1: UTD=4, ent 0.5->0.2 anneal, 250K) -> COLLAPSED.
  Best 36% hit@50 (< 50% random); 100K-250K det. eval = 0%; ep_rew fell 200->~5,
  ep_len->50 (stillness). Kinematics: hand DOES move (0.1-0.38 m path) toward target,
  ends 0.086-0.19 m away -> perception->approach partly works, but imprecise + collapsed.
  Verdict: optimization-stability failure (DroQ UTD=4 + entropy anneal), NOT a capability
  finding. Reward design fine (CONTACT_REWARD=200 dominates).
- rung0_TEST_v2_stable (UTD=1, ent auto, 150K) -> RUNNING. Anti-collapse retry.

### RUN LOG (cont.) — embodiment + the collapse finding
- rung0_h15_TEST vs rung0_h15_CONTROL (horizon 15): IDENTICAL (both 0% touch / 8% range
  / ~7.7 steps). Target signal INERT even when the task requires it. But horizon 15 also
  near-infeasible (body too slow), so partly a feasibility wall.
- rung0_motor_fixed (passive hands, ONE fixed target, pure motor): best 53% touch (random 40%),
  median final 0.077 m. Even reaching ONE fixed spot is barely learnable; floppy hand caps it.
- EMBODIMENT FINDING: AB's wrists (hand1/2/3) + fingers are PASSIVE (no actuators). Arm IS
  actuated (shoulder x3 + elbow per side). So: a reaching arm ending in a dead floppy paddle;
  cannot orient hand or grasp. (Confirms David's "no functioning hands -> can't learn".)
- ADDED actuated hands: new body mimo_crawler_cart_hands.xml (+8 motors, action 25->33,
  proprio 69->85), --actuate-hands flag. Original body + all checkpoints untouched.
- rung0_hands_motor_fixed (actuated, fixed target): best 0% touch (WORSE than passive 53%;
  actuated random 60%). rung0_hands_TEST (actuated + perception + wide): best 7% touch / 18%
  range (passive TEST was 42%; actuated random wide 40%). ACTUATING HANDS MADE IT WORSE.
- ROOT CAUSE (robust across ALL ~9 runs): SAC COLLAPSES every time. ep_rew_mean 200 -> ~7-10,
  ep_len -> full horizon, regardless of hands/perception/horizon/entropy/UTD. More DOF
  (actuated hands) collapses HARDER. The bottleneck is the TRAINING OPTIMISATION, not
  embodiment or perception. Prime suspect: reward magnitude (CONTACT_REWARD=200 sparse +
  hunger to -2000) destabilises the critic. Can't test the hands hypothesis until collapse
  is fixed.
- NEXT: fix the collapse first. Reduce CONTACT_REWARD (200 -> ~5) + hunger scale so the
  critic has a sane value range; retest passive AND actuated reach.

### COLLAPSE-MAGNITUDE TEST (refuted) + the deep invariant
- rung0_lowR_motor_fixed (passive, fixed target, CONTACT_REWARD 200->20, near-bonus 1.0):
  best 53% touch, median final 0.077 m -- IDENTICAL to contact=200. Reward MAGNITUDE is NOT
  the collapse cause. ep_rew still declined (22 -> ~4).
- DEEP INVARIANT across ~10 runs (hands/no-hands, perception/blind, horizon 12-50, ent
  0.5-anneal vs auto, UTD 1 vs 4, contact 20 vs 200): the reach CAPS at ~53% on a FIXED
  target (random 40%), median hand-to-target ~0.077 m, and training COLLAPSES every time.
  No knob moved it. Perception never helped; hand actuation made it worse.
- HONEST CONCLUSION: the bottleneck is NOT perception, NOT reward, NOT a single hyperparam.
  This MIMo body under SAC torque control cannot learn a reliable, stable reach even to one
  fixed spot. It is a foundational control/optimisation wall -- the same wall as the project's
  chronic stillness-collapse / can't-crawl history, now reproduced at the simplest task.
- STRATEGIC OPTIONS (for David): (a) different control scheme -- position/PD control instead
  of raw torque (likely the biggest lever); (b) simpler body / fewer DOF for the reach
  primitive; (c) different algorithm or scripted/imitation bootstrap; (d) step back and
  question whether this body+sim is the right substrate at all.
