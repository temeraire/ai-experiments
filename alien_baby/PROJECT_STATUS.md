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

**UPDATE — Phase H result (2026-05-20):** The moving-ball hypothesis was tested as Phase H (R32–R35, ball speeds 0.05 and 0.08 m/s, cart substrate). The result was REFUTED — see Current Best Numbers and Last Run sections below. Moving balls at these speeds make the task too hard for any policy rather than creating pressure that only vision can relieve. The hypothesis that ball motion forces visual learning is now tested and closed.


**UPDATE — Phase I/II/III result (2026-05-21):** After Phase H closed the substrate-level avenue, the project pivoted to an architectural intervention: MICOA (Multimodal Inference via Coupled Objectives Across modalities) — wire proprio and vision encoders as a Product-of-Experts Gaussian and add a KL-based agreement loss between them. Phase I (R36/R37) confirmed MICOA worked mechanically, but vision remained inert (ablation in the 0.002–0.023 dead zone). Phase II (R38) replaced symmetric agreement with a one-sided *predictive* KL — vision pulled toward proprio(t+1).detach() — and **broke the ablation barrier for the first time**: single-step action delta 0.66, 350× R36, 13× threshold. Vision was *active* but not *productive* (6/20 with or without pixels) and σ_p instability hurt task performance. Phase III R39 changed three knobs at once (multi-horizon + tighter σ clamp + higher β) and regressed on load-bearing. **R40 isolated the σ clamp variable** (single-horizon t+1 like R38, tightened σ clamp like R39) and is the first run where vision is load-bearing AND the task is mostly solved (ablation 0.481, 15/20 both-touched, mean reward −234). Goldilocks reading: σ_p ∈ [0.018, 7.4] (R38) too permissive; σ_p ∈ [0.135, 7.4] (R40) prevents extreme precision while still letting vision broadcast useful signal. Remaining limit: action-level ablation is big (0.48) but outcome-level delta is only +1 episode (R40 15 vs R38-substrate 14) — proprio is still good enough to solve mostly alone. Next: re-introduce ball-speed > 0 to give vision a job proprio cannot do, now from an architecture where vision is actually integrated.

---

## Current best numbers

| Metric | Value | Run |
|--------|-------|-----|
| Peak ep_rew_mean (stage1, eval) | 176.9 at 270K | stage1_headfix_velbonus_2026_05_07 (v1) |
| Best deterministic touch rate (blind proprio) | **17/20 (85%)** | stage1_headfix_velbonus2_cone180_600steps_2026_05_07 — AWAITING HUMAN VIDEO REVIEW |
| Previous best deterministic touch rate (blind proprio, v1) | 9/20 (45%) | stage1_headfix_velbonus_2026_05_07 — pending video confirmation |
| Best deterministic touch rate (vision follow-on) | 4/50 (8%) | micoa_freeze_zeroinit_cone30 (30K checkpoint) — built on old invalid stage1 |
| Mean ground coverage (blind proprio) | 0.417 mean_dist_mean | stage1_headfix_velbonus_2026_05_07 (consistent across both velbonus runs) |
| Falls (all runs) | 0/20 | stage1_headfix_velbonus2_cone180_600steps_2026_05_07 — body stable under new physics |
| Vision load-bearing? | Not yet confirmed | Phases E/E2 are blind proprio; vision not yet tested under corrected wrapper |
| Best HER ep_rew_mean | −1899 ± 0.34 (Phase E2, 250K) | First time any HER run has left the −2000 floor |
| Eval std (HER runs) | 0.30–0.42 (Phase E2) | First non-zero eval std in months; prior phases all returned std = 0.00 |
| Wrapper bug status | FIXED (her_wrapper.py line ~150) | Phase D and E both ran with effective velocity bonus = 0; Phase E2 is first valid test |
| **Phase G best (cart substrate, ent=0.5 + vel_bonus=0.10)** | **+380 ± 7.8, 20/20 hits** | **R19 (mimo_phase_g_R19_ent05_velbonus) at t=14976; also 20/20 maintained at t=19968** |
| Phase G previous best (cart substrate, blind proprio + ent=0.5) | +366 ± 14, 20/20 hits | R3 (mimo_phase_g_R3_ent050) at t=14976 |
| Reach radius of R3-best | ~0.20-0.25m (graceful 0.15→0.20, hard cutoff 0.25→0.30) | R3-best generalizes 15/20 both-touched at offset=0.20, 0/20 at 0.30 |
| Vision load-bearing? | **No** — vision-ablation delta = 0.023; episode ablation produces identical 14/20 → 14/20 | R14 vs R15 ablation comparison, R10/R12 longer vision attempts |
| Vision under random ball positions? | **Still no** — R17 (vision) 34/40 both-touched, R18 (proprio control) 36/40; pixel-zero ablation IMPROVES R17 from 16/20 to 17/20 | R17/R18 random_ball_box test, 04:30 |
| Vision under disappearing balls (timeout=300)? | **Still no** — R26 vision 3/20 both-touched, R27 proprio 5/20 | 2nd overnight, R26/R27 |
| Vision under disappearing balls (timeout=150)? | **Still no** — R28 vision 15/20 ball1 + 5/20 ball2; R29 proprio identical | 2nd overnight, R28/R29 |
| Three-seed validation of R20 mechanism | Robust: peak 361-388 across seeds 42, 1, 2 | R20, R22, R23 |
| Buffer-preserving warm-start works? | **No** — R24 (with buffer load) still collapses on curriculum 0.15→0.20 (off-policy gradient inconsistency) | R24 |
| Direct training at offset=0.20 with strength=1.0? | **Yes** — R30 gets 14/20 both-touched (matches R3 transfer's 15/20) | R30 |
| Direct training at offset=0.25? | **No** — even strength=1.0 + R20 recipe collapses (R31: 0/20 at 0.25) | R31 |
| **Phase H best vision policy (moving balls)** | **0/20 both-touched (speed=0.08), 1/20 (speed=0.05)** | **R33 (vision, 0.08), R35 (vision, 0.05) — see PHASE_H_RESULTS_RAW.md** |
| **Phase H best proprio policy (moving balls)** | **1/20 both-touched (both speeds)** | **R32 (proprio, 0.08), R34 (proprio, 0.05)** |
| **Vision under moving balls (speed 0.05/0.08)?** | **No — REFUTED.** Ablation 0.020–0.024 (dead zone, same as Phase G). Vision and proprio equally destroyed by ball motion. | **Phase H, R32–R35, 2026-05-20** |
| Phase I best (MICOA PoE + symmetric KL agreement) | Vision still inert: ablation 0.0019 (R36), 0.0023 (R37); task preserved | R36 (β=0.10), R37 (β=0.03) |
| Phase II best (MICOA + temporal predictive KL, β_pred=0.10) | **First ablation-barrier break**: action L2 delta 0.6563 (350× R36, 13× threshold). Vision active but not productive: 6/20 with or without pixels. Task regressed (−757, 6/20). | R38 |
| **Phase III R40 (single-horizon t+1 + tightened σ clamp)** | **Ablation 0.481, both-touched 15/20, mean reward −234. First run with vision load-bearing AND task solved.** | **R40, 2026-05-21** |
| Phase III R39 (multi-horizon t+1/t+5/t+25/t+50 + tightened σ clamp) | Regressed on load-bearing; confounded (3 knobs changed at once). 3× total β likely deformed actor gradient; possible cross-horizon interference. | R39 |
| **Phase IV R41 (MICOA+vision, moving balls @ 0.08)** | **Ablation 0.8511 (highest ever). Task: 1/20 both-touched moving, 9/20 static. MICOA encoder pathology: kl_pred_k1 exploded to 638, sigma_combined collapsed to 0.10. Vision over-integrated but behaviorally harmful.** | **R41, 2026-05-22** |
| **Phase IV R42 (proprio only, moving balls @ 0.08)** | **2/20 both-touched moving, 18/20 static — best proprio result yet on moving balls. Outperformed matched MICOA+vision run on both conditions.** | **R42, 2026-05-22** |
| **Vision load-bearing under MICOA + moving balls?** | **Action-level YES (ablation 0.85). Outcome-level NO — R41 worse than proprio control on both metrics. High ablation ≠ useful encoding.** | **Phase IV R41 vs R42, 2026-05-22** |
| **Vision load-bearing (architectural intervention)?** | **Yes (action level): R40 ablation 0.481 ≫ 0.05 threshold.** Episode-outcome lift small (+1 episode vs proprio-dominant baseline) — proprio still does most of the work. | **Phase III R40, 2026-05-21** |
| **Phase V R43 (MICOA+vision, static reachable, eccentricity sweep)** | **Ablation 0.62–1.04 across eccentricity bins (integrated). Task: 20/20 at ecc=0.00, 0/20 at ecc=0.25. Identical to proprio control at every bin. Vision integrated but non-directional.** | **R43, 2026-05-30** |
| **Phase V R44 (proprio only, static reachable, eccentricity sweep)** | **20/20 at ecc=0.00, 0/20 at ecc=0.25. Distance law r=+0.96, extrapolates outside training band. Speed retention ~0.50 (graceful).** | **R44, 2026-05-30** |
| **Phase W R45 (DroQ proprio validation, 150K steps)** | **20/20 at ecc=0.05, 20/20 at ecc=0.10 (beats R44 16/20). Matches R44 at all other bins. DroQ infra validated; no degradation.** | **R45, 2026-06-14** |
| **Methodological finding (Phase W)** | **eval mean_reward and critic_loss are INVALID health metrics on this substrate. Only deterministic eccentricity sweep (eval_phase_v.py) is valid.** | **Phase W, 2026-06-14** |
| **Vision load-bearing on cleanest static reachable task?** | **Action-level YES (abl_L2 0.62–1.04). Outcome-level NO — R43 ≈ R44 at every eccentricity bin. Strongest null result in project history. Vision bound globally, not recruited where it carries directional information.** | **Phase V R43 vs R44, 2026-05-30** |

## Last run

- **Tag:** Phase W R45 (DroQ critic-stabilization infrastructure validation, proprio only)
- **Date:** 2026-06-14
- **Setup:** Identical to Phase V R44 (proprio only, cart constant_velocity_bouncer speed 0.15, ball_speed=0.0, box jitter +/-0.08, hip off, memory obs, strength 1.0, entropy anneal 0.5->0.2, vel-bonus 0.10, curriculum warmup 2000/ramp 15000/offset 0.15, seed 42, n_envs=16). Single change: --droq flag (LayerNorm + Dropout rate 0.01 on critic hidden layers; UTD=4). 150K steps.
- **Key results:**
  - ep_rew_mean: +413 (start) -> -115 (end)
  - Eval mean_reward at 150K: -486 +/- 857 (negative throughout second half, same as R44 -- see methodological finding)
  - Eccentricity sweep (20 eps/bin) at 150K vs R44 at 250K: R45 matches or beats R44 at every bin; ecc=0.10 notably 20 vs 16
  - critic_loss: baseline ~4-11 with spikes to 40-145 -- identical profile to R44 (spikes are contact-event TD artifacts, not pathology)
  - **METHODOLOGICAL FINDING:** Pre-registered pass/fail criteria (eval reward must be positive, critic_loss must stay below 10) are invalid discriminators on this task. R44 -- the confirmed-best generalizer -- fails both checks too. The only valid metric is the deterministic eccentricity sweep.
  - DroQ safe to use as opt-in; sample-efficiency gain tentative (needs matched-step check)
- See FINDINGS.md Phase W entry for full eval table and verified numbers.

### Previous run (Phase V R43/R44, 2026-05-30)
- R43 (MICOA+vision, static reachable): 20/20 at ecc=0.00, 0/20 at ecc=0.25. Ablation 0.62-1.04. Vision integrated but non-directional.
- R44 (proprio, static reachable): 20/20 at ecc=0.00, 0/20 at ecc=0.25. Distance law r=+0.96. Speed retention 0.50 (graceful).
- See FINDINGS.md Phase V entry for full tables.
---
