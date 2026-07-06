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

**UPDATE — Phase VIII result (2026-05-20):** The moving-ball hypothesis was tested as Phase VIII (R32–R35, ball speeds 0.05 and 0.08 m/s, cart substrate). The result was REFUTED — see Current Best Numbers and Last Run sections below. Moving balls at these speeds make the task too hard for any policy rather than creating pressure that only vision can relieve. The hypothesis that ball motion forces visual learning is now tested and closed.


**UPDATE — Phase IX/X/XI result (2026-05-21):** After Phase VIII closed the substrate-level avenue, the project pivoted to an architectural intervention: MICOA (Multimodal Inference via Coupled Objectives Across modalities) — wire proprio and vision encoders as a Product-of-Experts Gaussian and add a KL-based agreement loss between them. Phase IX (R36/R37) confirmed MICOA worked mechanically, but vision remained inert (ablation in the 0.002–0.023 dead zone). Phase X (R38) replaced symmetric agreement with a one-sided *predictive* KL — vision pulled toward proprio(t+1).detach() — and **broke the ablation barrier for the first time**: single-step action delta 0.66, 350× R36, 13× threshold. Vision was *active* but not *productive* (6/20 with or without pixels) and σ_p instability hurt task performance. Phase XI R39 changed three knobs at once (multi-horizon + tighter σ clamp + higher β) and regressed on load-bearing. **R40 isolated the σ clamp variable** (single-horizon t+1 like R38, tightened σ clamp like R39) and is the first run where vision is load-bearing AND the task is mostly solved (ablation 0.481, 15/20 both-touched, mean reward −234). Goldilocks reading: σ_p ∈ [0.018, 7.4] (R38) too permissive; σ_p ∈ [0.135, 7.4] (R40) prevents extreme precision while still letting vision broadcast useful signal. Remaining limit: action-level ablation is big (0.48) but outcome-level delta is only +1 episode (R40 15 vs R38-substrate 14) — proprio is still good enough to solve mostly alone. Next: re-introduce ball-speed > 0 to give vision a job proprio cannot do, now from an architecture where vision is actually integrated.

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
| Vision load-bearing? | Not yet confirmed | Phases V/V.2 are blind proprio; vision not yet tested under corrected wrapper |
| Best HER ep_rew_mean | −1899 ± 0.34 (Phase V.2, 250K) | First time any HER run has left the −2000 floor |
| Eval std (HER runs) | 0.30–0.42 (Phase V.2) | First non-zero eval std in months; prior phases all returned std = 0.00 |
| Wrapper bug status | FIXED (her_wrapper.py line ~150) | Phase IV and V both ran with effective velocity bonus = 0; Phase V.2 is first valid test |
| **Phase VII best (cart substrate, ent=0.5 + vel_bonus=0.10)** | **+380 ± 7.8, 20/20 hits** | **R19 (mimo_phase_g_R19_ent05_velbonus) at t=14976; also 20/20 maintained at t=19968** |
| Phase VII previous best (cart substrate, blind proprio + ent=0.5) | +366 ± 14, 20/20 hits | R3 (mimo_phase_g_R3_ent050) at t=14976 |
| Reach radius of R3-best | ~0.20-0.25m (graceful 0.15→0.20, hard cutoff 0.25→0.30) | R3-best generalizes 15/20 both-touched at offset=0.20, 0/20 at 0.30 |
| Vision load-bearing? | **No** — vision-ablation delta = 0.023; episode ablation produces identical 14/20 → 14/20 | R14 vs R15 ablation comparison, R10/R12 longer vision attempts |
| Vision under random ball positions? | **Still no** — R17 (vision) 34/40 both-touched, R18 (proprio control) 36/40; pixel-zero ablation IMPROVES R17 from 16/20 to 17/20 | R17/R18 random_ball_box test, 04:30 |
| Vision under disappearing balls (timeout=300)? | **Still no** — R26 vision 3/20 both-touched, R27 proprio 5/20 | 2nd overnight, R26/R27 |
| Vision under disappearing balls (timeout=150)? | **Still no** — R28 vision 15/20 ball1 + 5/20 ball2; R29 proprio identical | 2nd overnight, R28/R29 |
| Three-seed validation of R20 mechanism | Robust: peak 361-388 across seeds 42, 1, 2 | R20, R22, R23 |
| Buffer-preserving warm-start works? | **No** — R24 (with buffer load) still collapses on curriculum 0.15→0.20 (off-policy gradient inconsistency) | R24 |
| Direct training at offset=0.20 with strength=1.0? | **Yes** — R30 gets 14/20 both-touched (matches R3 transfer's 15/20) | R30 |
| Direct training at offset=0.25? | **No** — even strength=1.0 + R20 recipe collapses (R31: 0/20 at 0.25) | R31 |
| **Phase VIII best vision policy (moving balls)** | **0/20 both-touched (speed=0.08), 1/20 (speed=0.05)** | **R33 (vision, 0.08), R35 (vision, 0.05) — see PHASE_H_RESULTS_RAW.md** |
| **Phase VIII best proprio policy (moving balls)** | **1/20 both-touched (both speeds)** | **R32 (proprio, 0.08), R34 (proprio, 0.05)** |
| **Vision under moving balls (speed 0.05/0.08)?** | **No — REFUTED.** Ablation 0.020–0.024 (dead zone, same as Phase VII). Vision and proprio equally destroyed by ball motion. | **Phase VIII, R32–R35, 2026-05-20** |
| Phase IX best (MICOA PoE + symmetric KL agreement) | Vision still inert: ablation 0.0019 (R36), 0.0023 (R37); task preserved | R36 (β=0.10), R37 (β=0.03) |
| Phase X best (MICOA + temporal predictive KL, β_pred=0.10) | **First ablation-barrier break**: action L2 delta 0.6563 (350× R36, 13× threshold). Vision active but not productive: 6/20 with or without pixels. Task regressed (−757, 6/20). | R38 |
| **Phase XI R40 (single-horizon t+1 + tightened σ clamp)** | **Ablation 0.481, both-touched 15/20, mean reward −234. First run with vision load-bearing AND task solved.** | **R40, 2026-05-21** |
| Phase XI R39 (multi-horizon t+1/t+5/t+25/t+50 + tightened σ clamp) | Regressed on load-bearing; confounded (3 knobs changed at once). 3× total β likely deformed actor gradient; possible cross-horizon interference. | R39 |
| **Phase XII R41 (MICOA+vision, moving balls @ 0.08)** | **Ablation 0.8511 (highest ever). Task: 1/20 both-touched moving, 9/20 static. MICOA encoder pathology: kl_pred_k1 exploded to 638, sigma_combined collapsed to 0.10. Vision over-integrated but behaviorally harmful.** | **R41, 2026-05-22** |
| **Phase XII R42 (proprio only, moving balls @ 0.08)** | **2/20 both-touched moving, 18/20 static — best proprio result yet on moving balls. Outperformed matched MICOA+vision run on both conditions.** | **R42, 2026-05-22** |
| **Vision load-bearing under MICOA + moving balls?** | **Action-level YES (ablation 0.85). Outcome-level NO — R41 worse than proprio control on both metrics. High ablation ≠ useful encoding.** | **Phase XII R41 vs R42, 2026-05-22** |
| **Vision load-bearing (architectural intervention)?** | **Yes (action level): R40 ablation 0.481 ≫ 0.05 threshold.** Episode-outcome lift small (+1 episode vs proprio-dominant baseline) — proprio still does most of the work. | **Phase XI R40, 2026-05-21** |
| **Phase XIII R43 (MICOA+vision, static reachable, eccentricity sweep)** | **Ablation 0.62–1.04 across eccentricity bins (integrated). Task: 20/20 at ecc=0.00, 0/20 at ecc=0.25. Identical to proprio control at every bin. Vision integrated but non-directional.** | **R43, 2026-05-30** |
| **Phase XIII R44 (proprio only, static reachable, eccentricity sweep)** | **20/20 at ecc=0.00, 0/20 at ecc=0.25. Distance law r=+0.96, extrapolates outside training band. Speed retention ~0.50 (graceful).** | **R44, 2026-05-30** |
| **Phase XIV R45 (DroQ proprio validation, 150K steps)** | **20/20 at ecc=0.05, 20/20 at ecc=0.10 (beats R44 16/20). Matches R44 at all other bins. DroQ infra validated; no degradation.** | **R45, 2026-06-14** |
| **Methodological finding (Phase XIV)** | **eval mean_reward and critic_loss are INVALID health metrics on this substrate. Only deterministic eccentricity sweep (eval_phase_v.py) is valid.** | **Phase XIV, 2026-06-14** |
| **Vision load-bearing on cleanest static reachable task?** | **Action-level YES (abl_L2 0.62–1.04). Outcome-level NO — R43 ≈ R44 at every eccentricity bin. Strongest null result in project history. Vision bound globally, not recruited where it carries directional information.** | **Phase XIII R43 vs R44, 2026-05-30** |
| **Phase XV R46 (DroQ proprio, size variety, 250K)** | **Held-out sizes: 17/20 (interpolation 0.047) and 16/20 (extrapolation 0.090). Held-out range 0.17 = INVARIANT. Full sweep range 0.67 = SENSITIVE, driven by 0.075 anomaly (DIAGNOSED 2026-06-15: a [0.070-0.080] knock-away band artifact, not a generalization hole). Ecc: 19,19,18,15,11,0.** | **R46, 2026-06-15** |
| **Phase XV R47 (DroQ proprio, shape variety, 250K)** | **Held-out ellipsoid: 20/20 at ecc=0.00. Held-out capsule: 17/20 at ecc=0.00, 15/20 at ecc=0.05. Shape-only training slightly reduces high-ecc reach vs size-variety. Ecc (sphere): 20,20,19,8,2,0.** | **R47, 2026-06-15** |
| **Phase XV R48 (DroQ proprio, combined size+shape, 250K)** | **Held-out ellipsoid 20/20, capsule 20/20 at ecc=0.00. Ecc: 20,20,19,13,5,0 (+3 at ecc=0.15 vs R44). Held-out sizes: 19/20 (0.047), 15/20 (0.090). 0.075 anomaly persists (6/20) — DIAGNOSED 2026-06-15 as a [0.070-0.080] knock-away band artifact.** | **R48, 2026-06-15** |
| **Object variety zero-shot transfer (Phase XV)** | **CONFIRMED for BROAD-touch: held-out sizes and held-out shapes transferred at center (15-20/20 both-touched). Combined variety (R48) improved ecc=0.15 margin vs R44/R45 baseline. BUT see 2026-06-15 diagnostic below: the task is never COMPLETED with the hands (both_HAND 0), so read this as transfer of "be in a posture the swept ball intersects", weaker than reach generalization.** | **Phase XV R46/R47/R48, 2026-06-15** |
| **0.075 "anomaly" — DIAGNOSED (2026-06-15)** | **Not a point spike but a dead BAND ~[0.070-0.080]; real and seed-independent (original 6/20,10/20 were low-side noise, true rate ~0.45-0.55 at 30 eps). Mechanism (on video): glancing blow knocks the mid-size ball to the wall, fixed-base creature can't recover it. A contact-dynamics ARTIFACT, not a size-generalization hole. Geometric (ball mass unchanged by radius).** | **diag_0075_anomaly.py / render_0075_diag.py, 2026-06-15** |
| **Hand-touch finding (2026-06-15)** | **both_HAND = 0 across every size AND every ecc bin, on the WHOLE cart line (R44/R45/R46/R48) — the two-ball task is never completed with two hands. But hands DO engage for one ball (any_HAND up to 21/30), rising with eccentricity where a real reach is needed. The cart carries AB's BODY into the ball (cart geom is a non-colliding visual marker, never pushes the ball). Reframe: body-bump = valid unconditioned "world-is-consistent" signal; hand-reach = intentional/conditioned signal. both_HAND=0 is a DEVELOPMENTAL STAGE, not a bug. Track hand-touch fraction over training as the real progress metric.** | **rescore_hand_touch.py + rescore_hand_touch_ecc.py, 2026-06-15** |
| **Phase XVI R49 (MICOA+vision + aux ball-position decode loss, 250K)** | **REPRESENTATION FIX FAILED. Lateral R² = 0.010 (chance; below proprio control 0.043 and below R43 ~0.08 baseline; well under 0.30 success bar). Forward R² = 0.157 (decode loss caught distance, not direction). Ecc sweep: 20,19,19,8,4,0 — roughly on par with R43, no gain. Ablation high (1.29–2.04) but confounded by kl_pred_k1 explosion (244; healthy 0.5–2) and sigma_combined near-collapse (0.117). Ablation-profile flip (ecc=0 lower, off-center high-flat) auto-flagged as "recruited at margin" — NOT a positive finding; this is the known high-ablation-under-encoder-pathology trap (cf. R41). REFRAME: cart substrate makes vision directionally pointless because AB cannot steer; fix is locomotion under position-offset control.** | **R49, 2026-06-16** |
| **rnd_propulsion_400k (propulsion-affordance test, position_offset, 400K, seed 0)** | **TRANSLATION NOT ACHIEVABLE (Verdict B). Eval mean_reward peaked 64.46 ± 22.44 at 310K then collapsed to 11.14 at final. ep_len_mean=600 on all 40 evals = zero ball contacts throughout. Velocity_bonus (×2.0) fired on in-place CoM oscillation (rocking), not net floor translation. Peak-then-regression pattern confirms no stable gait was found. Video (3 seeds, all TIMEOUT): seeds 0 and 1 collapse/fall prone, ball stays stationary, no floor-crossing. Option-3 "widen the reward" from rnd_directed_250k is tested and closed.** | **rnd_propulsion_400k, 2026-07-01** |

| **crawl_minimal_400k (wide XML + arms_fwd pose, 400K, seed 0)** | **FIRST NON-ZERO CONTACT RATE in any free-body crawler run. Eval reward: 0.03 (10K) → peak 41.12 ± 80.87 (370K). Deterministic eval: 5/30 contacts (16.7%), 0/30 tip-terminated, 25/30 timed out. Mean CoM displacement 0.298 m (best single episode 0.646 m, exceeds hand-drive ceiling 0.225 m). Mean dist change = 0.000 m: locomotion discovered but undirected (ball position not in proprio). Video confirms real belly-down body translation in touch episodes. Verdict: (B) PARTIAL.** | **crawl_minimal_400k, 2026-07-02** |

## Last run

- **Tag:** crawl_minimal_400k — first non-zero contact rate in any free-body crawler run
- **Date:** 2026-07-02
- **Setup:** Wide body (mimo_crawler_pos_wide.xml) + arms_fwd crawl pose + tip-termination (50°, −5 penalty). `--action-mode position_offset --velocity-bonus-scale 0.0 --approach-reward-scale 10.0 --step-cost 0.0 --rnd --rnd-coef 0.1 --spawn-radius 0.70 0.80`, 400K steps, 16 envs, seed 0.
- **Key results:**
  - Eval reward: 0.03 (10K) → peak 41.12 ± 80.87 (370K, best_model) → 20.29 ± 60.73 (400K final)
  - Deterministic eval (30 eps from best_model.zip): **5/30 contacts (16.7%)**, 0/30 tip-terminated, 25/30 timed out
  - Mean CoM displacement all: 0.298 m; touch episodes: 0.379 m; best single episode: 0.646 m (EXCEEDS hand-drive ceiling 0.225 m)
  - Mean dist change to ball = **+0.000 m** — locomotion is undirected (ball XY not in proprio obs)
  - Liveness gate: PASS at 10K (body_motion=1.277)
  - Video (seed 56 / ep8, seed 63 / ep9): belly-down body translation confirmed toward ball; (seed 0,1): spinning/rolling without contact
  - **Verdict: (B) PARTIAL** — real locomotion discovered, contacts genuine, but steering not yet learned. Ball position must be added to proprio obs for homing to emerge.
- See FINDINGS.md crawl_minimal_400k entry for full tables, displacement table, video paths, eval trajectory, and comparison to prior runs.

### Previous run (rnd_propulsion_400k, 2026-07-01)
- Propulsion-affordance test: 0 contacts after 400K, eval reward peaked 64.46 at 310K then collapsed. Velocity_bonus fired on rocking not translation. Option-3 closed.
- See FINDINGS.md rnd_propulsion_400k entry for full details.
---
