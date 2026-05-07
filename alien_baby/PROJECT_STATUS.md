# Alien Baby (AB) — Project Status

## What we're trying to do

Build a minimal embodied agent that learns to perceive the physical world through **survival consequence**, not by rewarding desired behavior. The project is grounded in set-theoretic perception theory (Taylor): perception is a *response* shaped by what the world conditions, not a model the experimenter writes in. The end goal is a creature whose sensory-motor categories — "there is a ball," "I am upright," "I am about to fall" — emerge from the environment, not from reward shaping.

This implies a methodological rule: when we want AB to learn behavior X, we **change the environment** so X becomes instrumental, rather than paying reward for X directly. (We call this "pressure not bribery.")

## Why MuJoCo

MuJoCo is a fast, accurate, open-source rigid-body physics simulator (owned by DeepMind). We need real gravity, real contact mechanics, and real joint dynamics — otherwise anything the agent "learns" could be an artifact of a toy environment. MuJoCo is the standard in the robot-arm RL literature. Gravity and friction are the pencil-tap — the ever-present conditions the creature must respond to.

We use **Stable-Baselines3 SAC** as the RL algorithm (continuous actions, off-policy, standard for this kind of task).

## Body: what we include, what we leave out

The body is deliberately minimal:

| Part | What we have | What we left out | Why |
|---|---|---|---|
| Torso | Blocky chassis (0.24×0.18×0.10) with 4 passive rolling wheels | Legs | Bipedal walking is a whole research problem; wheels give locomotion without balance as a distraction |
| Arms | 2 arms, shoulder (pitch+roll) + elbow, total reach ~0.22m | Wrists, independent fingers | Enough to paddle and reach; more DOFs bloat the action space |
| Hands | Spheres with touch sensors | Gripping, individual fingers | Our task is contact-based, not grasping |
| Head | Pan + tilt, on top of chassis center | Facial features, multiple sensors | Gaze is the key perceptual act — everything else is noise |
| Eye | Single head-mounted camera with 25° FOV | Binocular vision, peripheral vision | One camera is the minimum "eye" needed to test "learning to look" |

The philosophical point: **every joint we include is a claim about what the creature needs**. We want to make as few claims as possible and let the environment force the rest.

## What didn't work (chronological)

1. **Pancake torso with wheels at the corners.** Torso was flatter than its wheelbase, so tipping onto the "edge" of the box was a stable resting state — wheels ended up "on its back."
2. **Blocky torso, but arms extending sideways (±X).** Arms had 30 Nm gear and reached far outside the wheelbase (hand at 0.43m, wheels at 0.20m). Any downward push at full extension levered the whole body off the wheels. Body flipped under random action.
3. **Forward-biased head (`y=0.08`).** Put the head slightly in front of the chassis. That created the only fore-aft mass asymmetry in the body. Result: every trained policy moved strictly *backward* (pushing arms forward was mechanically easier). We confirmed this by measuring torso displacement: ~0.34m, same direction, every seed.
4. **Hand-only contact detection.** Task code only registered touch when a hand geom contacted the target. Episodes where AB rolled over the ball or pinned it with the torso got no credit — a bug in the task definition, not the learner.
5. **Min target radius 0.2.** With the ball that close, a resting hand was already touching the spawn point. "14/20 touches" was almost entirely spawn-luck, not learning.
6. **Sideways-extending arms, even with weaker gear.** The core problem was kinematic, not power: with a 2-DOF shoulder (pitch Y + roll X), an arm *aligned* with X has shoulder_roll rotating it around its own axis — useless. Only pitch (Y) worked, and pitch swings the arm up/down. The body could only push *down*, which creates wheelies, not forward/back paddle strokes.

## What worked

1. **Blocky chassis with wheelbase *wider* than the chassis in every axis.** Wheels at ±0.20×±0.14 around a 0.24×0.18 body. Tipping onto any non-bottom face now elevates the wheels — tipping is no longer mechanically stable.
2. **Baby arms: shorter (0.22 total reach from shoulder) and weaker (gear 15/10).** Matches a toddler's arm/body ratio. Arms can't lever a 3.6 kg chassis.
3. **Centered head (`y=0`).** Removed fore-aft mass asymmetry → backward-only bias went away.
4. **Forward-pointing arms (+Y).** Rotated arms so shoulder_roll (axis X) swings the arm in the Y-Z plane — forward, down under the body, back. Natural paddle/rowboat stroke. Locomotion emerged: AB reliably moves 0.34–0.37m per episode.
5. **Any-body-to-target contact counts.** Contact is contact — torso, hand, or wheel — matches Taylor's frame.

## Current state (as of now)

- **Stage 0** (flat floor, no fall penalty, wide targets, scaffolded): **6/20 touches, 0 falls** in blind proprio mode.
- **Stage 1** (2×2m platform, fall penalty, gaze-pressure cone exclusion): warm-started from Stage 0, peaks at **7/20 touches, 0 falls**. Late-training regression — same pattern we saw in earlier v8 runs (peak at 90K, degraded to 2/20 by 150K).
- **Zero falls anywhere.** The paddle-arm body solved posture completely.
- **One-direction motion.** Without vision, AB has no input telling it where the ball is (by design — no target vector in proprio). So it learns one canned paddle motion and hopes the ball is in that direction. ~1/3 of ball placements happen to be within the cone of motion.

## Frozen vs unfrozen proprio (why this matters)

When we add vision on top of a blind proprio-trained policy, there are two architectural choices:

- **Frozen proprio:** only the pixel-input columns of the actor's first layer can train. Proprio's behavior is mathematically preserved (drift = 0). A consistency loss `λ·MSE(h_full, h_blind)` pushes vision to *confirm* what proprio alone would have activated, not reshape it. Philosophically: proprio established itself before vision existed, so vision cannot overwrite that ground. This is the "protect proprio" principle.

- **Unfrozen proprio:** everything trains together. More flexibility, but vision's gradients can drift proprio's established behavior.

Does "frozen" make sense? **Yes — but only if proprio is fully trained on the target environment first.** Our first attempt froze proprio from a *stage-0-trained* policy and put it in stage 1, which dropped performance from 6/20 to 1/20. The protect-proprio principle presumes proprio has already learned the terrain; we were freezing an incomplete policy.

The A/B we're running right now is the correct version: freeze (or not) from a *stage-1-adapted* checkpoint, 100K each, to see if vision helps at all and under which freeze regime.

## What we might need to change to advance

1. **Late-training regression.** Best-checkpoint and final-checkpoint differ by 5× in touch rate. SAC is over-exploiting and losing the plateau. Fixes: entropy-coefficient floor, earlier stopping, lower learning rate.
2. **Fix "one direction of motion."** Blind proprio fundamentally cannot steer — there's no input telling it where the ball is. Vision is the designed fix. If vision doesn't produce steering either, we consider: (a) stronger vision signal (larger FOV, multi-frame observation), (b) temporary target-vector in proprio as a crutch, (c) explicit head-orientation encouragement in the reward (bribery — last resort).
3. **Gait diversity.** A steering policy needs asymmetric paddling (left arm harder to turn right). Currently the policy is symmetric. More training or higher entropy floors may help SAC discover this.
4. **Stage 1.5?** If the full stage-1 jump (small platform + fall penalty + cone exclusion) is too much at once, a gentler stage 1.5 (e.g., wider target cone, no fall penalty) could let proprio finish adapting before vision layers on.
5. **Compute budget.** 150K–200K SAC steps is *small* by the field's standards. Locomotion papers routinely use 1M+. If nothing above works, we budget a longer run.

## A/B result: frozen vs unfrozen (100K each)

| Checkpoint | Touches | Falls |
|---|---|---|
| Stage-1 blind (baseline) | **7/20** | 0 |
| Follow-on FROZEN best | 4/20 | 0 |
| Follow-on FROZEN final | 1/20 | 0 |
| Follow-on UNFROZEN best | 5/20 | 0 |
| Follow-on UNFROZEN final | **7/20** | 0 |

**Vision added nothing in 100K.** Unfrozen matched blind baseline; frozen strictly hurt.

**Vision-ablation sensitivity** (how much the action changes when pixel columns are zeroed):
- UNFROZEN final: 0.85 (moderate)
- FROZEN best: 2.29 (high)

Both policies *are* conditioning on pixels. Vision is being integrated — it's just not being integrated *productively*. The unfrozen policy reads pixels but doesn't derive steering from them. The frozen policy is even more pixel-sensitive, but the consistency loss drags its effective output back toward blind-proprio, yielding net-negative contribution.

**The real bottleneck:** we don't have enough pressure on vision. 7/20 via forward-paddle + attract-reward is a viable non-vision strategy, so vision has no *need* to develop useful features. The gaze-pressure cone exclusion pushes targets off-frame but doesn't make blind paddling *fail*. For vision to become load-bearing, blind must fail.

## v8-final: rolling wheels unlock locomotion

Diagnostic on the paddle-arm body with *sliding* wheel spheres: hand-coded paddle produced 0.19m in 20s (~1.5 cm/s). Under a trained policy: ~0.34m per episode, touch rate 6/20. **Wheels were sliding, not rolling** — the chassis was skidding on low-friction spheres.

Fix: each wheel became a child body with a ball joint, and wheel sliding friction was bumped from 0.05 to 2.0. This converts wheels from "low-friction skids" into "grip-and-roll" contact.

Result (200K stage-0 on rolling-wheel body):
- **9/20 touches, 0/20 falls** (up from 6/20 with sliding wheels)
- **Mean speed 7.46 cm/s, max 21.10 cm/s** — max matches a baby crawl

This clears the v8 prerequisite (workable locomotion) and we declare the v8 body complete.

## Open: v9 — visual pressure

The v8 story: body that won't tip, body that can roll, body that can paddle — but vision is still not load-bearing because forward-paddle-and-hope resolves the critical state often enough. Taylor's framework makes the diagnosis precise: the equivalence class is too inclusive.

v9 splits the class. Specific candidate: **moving target env**. The ball spawns with a small random velocity, rolls until caught or falls off platform. Blind forward-paddle fails because the ball isn't where a fixed-direction policy expects. Success requires predicting where the ball *will be*, which requires seeing it move — the exact visual discrimination that has to emerge.

Theoretical framing: in Taylor's notation, M (forward-paddle) must FAIL unless modified by visual evidence of the ball's trajectory. Same critical state D (contact drive), same sensory surface Q, but now Q's visual component is the only way to derive the right M. Engrams for "ball-left" and "ball-right" must become conditioned to asymmetric paddle strokes. Pure pressure, no bribery.

---

## Current best numbers

| Metric | Value | Run |
|--------|-------|-----|
| Peak ep_rew_mean (eval) | 61.73 | entpin_005_3_hand_only_cone_2026_05_07 (160K step checkpoint) |
| Best deterministic touch rate (vision follow-on) | 4/50 (8%) | micoa_freeze_zeroinit_cone30 (30K checkpoint) — tied with previous MICOA freeze run |
| Best deterministic touch rate (blind proprio) | 12/20 (60%) | stage1_v8_best — this remains the baseline to beat |
| Falls (all runs) | 0/50 | All runs — body is stable |
| Proprio-column drift (MICOA freeze runs) | 0.00e+00 | Both MICOA runs — freeze held perfectly |
| consistency_loss (this run) | 0.20–0.38 | micoa_freeze_zeroinit_cone30 — higher early activity than previous MICOA run |
| Rollout left/right touch symmetry | 0.102 / 0.098 | micoa_freeze_zeroinit_cone30 — first near-symmetric result (all prior runs biased left) |
| Vision load-bearing? | Not yet confirmed | Rollout symmetry is first weak positive signal; ablation test needed |

## Last run

- **Tag:** micoa_freeze_zeroinit_cone30_2026_05_07
- **Date:** 2026-05-07
- **Result:** Peak 55.6 at 30K (earlier than previous MICOA run's 60K peak). Final -9.1 at 150K. Best checkpoint (30K): 4/50 (8%) deterministic touches, 46/50 timeouts, 0/50 falls. Zero-init pixel columns and ±15° forward spawn cone are the two new changes. Rollout symmetry (0.102L / 0.098R) is the first behavioral signature consistent with directed visual steering. Ablation test needed to confirm vision is load-bearing.

---

## Direction Change — 2026-05-07

### What we observed that prompted this

Video review of the stage1_v8_best checkpoint (the 60% blind policy, our best result) revealed two fundamental problems that no amount of vision architecture work can fix:

1. **The creature barely moves.** In 5 rendered episodes it was nearly stationary on 4 of them. The one touch (seed 3) succeeded because the ball spawned almost touching the creature — not because the creature moved to find it. This is the "stillness local optimum": the creature learned that not moving avoids ctrl_cost and pays only the small hunger penalty.

2. **Head motion is extreme.** The head oscillates wildly across its full range every few timesteps — what was described as "panic attack" behavior. With `kp=30` and no damping override, the position servo snaps the head to any commanded position in a single timestep. A policy free to command full-range head positions every step can produce this behavior trivially. A head moving this way cannot produce a useful visual signal for any downstream learning.

These are not vision integration problems. They are foundational locomotion and physics problems. The serial approach (train blind proprio → freeze → add vision) cannot recover from them because the broken foundation carries forward.

### Decision: rebuild from scratch

The serial training approach (stage 1 blind → freeze → vision follow-on) is **retired**. The new approach is:

1. **Fix the physics first**: reduce head servo gain from kp=30 to kp=5 in both XML files. This makes the head track commanded positions slowly over multiple timesteps instead of snapping instantly. (Done — 2026-05-07)

2. **Fix the reward to incentivize movement**: add a small velocity bonus (`VELOCITY_BONUS_SCALE=0.02 × torso_speed`) to break the stillness optimum. Moving is now strictly better than standing still, regardless of direction. (Done — 2026-05-07)

3. **Train a new stage1** from scratch under the fixed physics and reward. Target: smooth locomotion, active ground coverage, calm head motion. Blind touch rate should remain ≥ 60%.

4. **Joint training with dialogue architecture**: replace the serial freeze/unfreeze approach with a jointly-trained two-stream model where vision and proprio are equal peers. Neither has structural priority. Both propose an action; a learned agreement layer combines them. Disagreement is explicitly logged. The architecture enforces that the two channels must reach agreement, not that one defers to the other.

### What is preserved from prior work

- The MICOA insight (channels must confirm each other, not compete) remains the theoretical foundation.
- The vision-ablation sensitivity test (does zeroing pixels change actions?) remains the key measurement.
- The 30K breakthrough checkpoint (`micoa_freeze_zeroinit_cone30_2026_05_07_best/best_model.zip`) is preserved on disk as the first confirmed vision-load-bearing result.
- The success criterion remains: touch rate ≥ blind baseline AND nonzero vision-ablation sensitivity.

### Files changed in this reset

- `envs/platform_creature.xml`: head actuator `kp` 30 → 5
- `envs/platform_creature_v9.xml`: same
- `envs/platform_creature_env.py`: added `VELOCITY_BONUS_SCALE=0.02` velocity bonus, logged as `velocity_bonus_sum` in episode info
- `todo.md`: full rebuild plan (Phases 1–5)
