# Phase G Proposal — Cart Substrate (`mimo_phase_g_cart_v1`)

## 1. Tag / title
**`mimo_phase_g_cart_v1`** — AB rides a motorized cart; locomotion is given, not learned.

## 2. One-sentence summary
This experiment removes the "creature must locomote" prerequisite from the dependency chain by mounting AB on a constant-velocity cart that traverses the platform autonomously, and tests whether — when the visual sensorium is populated with consequences regardless of policy action — vision finally becomes load-bearing for ball contact, mediated only by arms, head, and a non-stationary hunger penalty that makes freezing globally costly.

## 3. Hypothesis
**Single testable claim:** Under cart-induced locomotion, AB will learn a policy whose deterministic eval (a) achieves a touch rate ≥ 25% across 20 episodes, and (b) shows vision-ablation sensitivity (mean L2 action delta when pixels zeroed) ≥ 0.3 on the best checkpoint.

- **Confirmed** if both (a) and (b) hold: vision is finally load-bearing and is producing reach behavior.
- **Refuted** if (b) < 0.1 even when (a) ≥ 25%: the creature is solving the task by proprio-grope timing against the cart's predictable trajectory and the eye is still inert; the cart substrate alone does not force the eye to matter.
- **Also refuted** if (a) < 5%: the cart destabilizes AB physically, the arm action space is too narrow to produce any contact, or hunger is mis-tuned and the policy freezes anyway.

## 4. Substrate changes from Phase F-2k

**Cart added to env.** A new rigid body, ~0.4 m × 0.4 m × 0.05 m, slides along the platform's XY plane at a constant velocity vector `(vx, vy)` initialized at episode reset to magnitude ~0.15 m/s with a random initial angle drawn from `Uniform[0, 2π)`. The cart is *not* an actuator and not a policy output; it is a kinematic prop driven directly by `qvel`-write in `env.step()`, with four lines reversing `vx` (or `vy`) when the cart's center approaches a platform edge minus a margin. Pong on a rectangular table. AB is rigidly mounted on top via a weld or zero-strength free-joint freeze; the cart inherits AB's mass for stability but is light enough that the constant-velocity drive dominates. The cart is decorative-from-the-policy's-perspective: AB does not observe it, does not control it, and cannot detach from it.

**AB's action space: option (a) — arms + head only, hips disabled.** The simplest possible commitment consistent with "every joint we include is a claim about what the creature needs." We are testing whether the visual sensorium becomes load-bearing when motion is *given*; allowing AB to attempt locomotion at the same time would reintroduce the very failure mode the cart was designed to bypass. Option (b) — let hips actuate but be overridden — adds an unobservable confound (hip torque burning energy invisibly). Option (c) — hips can override the cart — invites AB to learn to "wriggle off" the cart, which is the wrong test for Phase G. Hip joint torque limits are set to zero in the action space construction; the hip degrees of freedom remain in proprio so AB knows its own pose, but the policy cannot command them. Arms (4 dims per arm × 2 arms = 8 dims if shoulder pitch/roll/yaw + elbow are kept; the existing MIMo action layout already exposes these) plus head pan/tilt (2 dims) give a ~10-dim action space, which is smaller than the existing 25-DOF MIMo space and should make critic learning easier.

**Hunger curve: option (b), convex (quadratic) ramp.** Per-step cost
```
per_step_cost = -BASE_COST - HUNGER_RATE × (steps_since_last_contact / SCALE)^2
```
with initial parameters `BASE_COST = 0.05` (matches the existing step cost), `HUNGER_RATE = 0.20`, `SCALE = 500`. At step 0 the cost is −0.05/step; at step 500 it is −0.05 − 0.20 = −0.25/step; at step 1000 it is −0.85/step; at step 2000 (worst case) it is −3.25/step. A touch resets `steps_since_last_contact` to zero, so the late-episode cost vanishes only by making contact. The "longing" language the human researcher used is convex — late steps are *much* worse than early ones — and quadratic gets that asymmetry without the numerical fragility of exponential blow-up. With two balls in the environment the worst-case per-episode total under freeze is approximately −3300 (sum of the integral of the quadratic over 2000 steps), against a touch reward of +200; the policy must produce ~17 touches to break even on a frozen episode, which is impossible without active reaching — exactly the pressure we want. Exponential (option c) was rejected because it can saturate floating-point math at late steps and because we already saw in Phases D–E2 that abrupt reward cliffs interact badly with SAC's critic. Linear (option a) was rejected because it punishes early and late steps equivalently, which is precisely the "freezing is mildly bad" signal the prior phases already had and was insufficient.

**Drop HER.** HER's value was as a sparse-reward bootstrap when reaching was hard and trajectory endpoints were varied. With the cart bringing balls into and out of FOV regardless of policy action, every episode now has rich, varied state coverage by construction; sparse-reward bootstrap is no longer the binding constraint. HER also brings the well-documented `compute_reward()`-pure constraint that made the velocity bonus survive only 1/5 of relabeled samples in Phases D–E2 (the 4/5 dilution dilemma), and the wrapper-bug audit risk noted in the methodological lesson. The Phase G reward is the standard SAC dense-reward composition: `reward = CONTACT_REWARD × (touched_this_step ? 1 : 0) + per_step_cost(steps_since_last_contact)`. This is `MultiInputPolicy` not needed, `HerReplayBuffer` not needed, `HERCrawlerWrapper` not used.

**Memory flags: keep.** Two binary `touched_ball1` / `touched_ball2` flags in proprio. With two fixed balls and the hunger reset behavior, the policy must know which ball it has already contacted; without memory flags a stateless SAC has no way to distinguish "I am still looking for ball 1" from "I have touched ball 1 and am now looking for ball 2." This is enabling structure, not bribery.

**Random start orientation: drop.** This was useful in Phase E2 only because we needed to know whether the policy was orientation-invariant (the answer, surfaced by F-2k's std=99.48, was no, in a damning way). With AB rigidly mounted on the cart and the cart's initial heading randomized, the equivalent diversity in initial conditions is supplied by the cart's random initial angle. Holding AB's prone-quaternion fixed on the cart removes one confound dimension.

**Ball positions: fixed, two balls, asymmetric placement.** Keep `(0.7, 0.0)` and `(0.0, 0.7)` as in Phases C through F-2k. Fixed positions give the agent a stable spatial structure to model; the user's "two-balls-define-a-line" hypothesis from the overnight session remains live and the cart finally enables it to be tested.

**Vision: stereo CNN, both eyes.** The project has alternated; the dialogue/Phase B run used stereo, the recent MIMo HER chain has been mono-camera-in-env but blind-policy. Stereo is the right call here because the visual flow induced by the cart's translation is the central learnable signal — depth/parallax cues are real, and the stereo extractor (`StereoCrawlerCNN`) is already implemented. The 15° downward head tilt from Phase A stays; balls resting on the platform should be in the FOV as the cart passes them. The action space includes head pan/tilt so AB can choose where to look.

**ent_coef:** pinned at `0.2` for the whole run, matching Phase E2. The cart removes the freeze attractor at the substrate level, so we no longer need entropy-pinning as the primary defense against collapse, but pinning is cheap insurance and the E2 evidence is that it doesn't hurt critic convergence.

## 5. Launch command

```bash
python -m alien_baby.crawler.train_crawler \
    --run-tag mimo_phase_g_cart_v1 \
    --steps 250000 \
    --n-envs 16 \
    --max-steps 2000 \
    --mps \
    --vision \
    --cart-mode constant_velocity_bouncer \
    --cart-speed 0.15 \
    --hunger-mode quadratic \
    --hunger-base 0.05 \
    --hunger-rate 0.20 \
    --hunger-scale 500 \
    --hip-actuation off \
    --fixed-ball-positions "0.7,0.0;0.0,0.7" \
    --memory-obs \
    --approach-reward-scale 0.0 \
    --velocity-bonus-scale 0.0 \
    --ent-coef 0.2 \
    --learning-rate 1e-4 \
    --buffer-size 500000 \
    --learning-starts 10000 \
    --checkpoint-interval 50000 \
    --seed 42
```

Flags not present in current `train_crawler.py` that need to be added by the training engineer: `--cart-mode`, `--cart-speed`, `--hunger-mode`, `--hunger-base`, `--hunger-rate`, `--hunger-scale`, `--hip-actuation`. These thread through to `MimoCrawlerEnv.__init__` and require (a) the cart body in `mimo_crawler.xml` (single new `<body>` element with a weld constraint to AB's root and four-line bounce logic in `env.step()`) and (b) replacing `HUNGER_PENALTY` with the steps-since-contact-aware quadratic. Existing flags (`--vision`, `--mps`, `--memory-obs`, `--fixed-ball-positions`, `--ent-coef`) are reused as-is. HER is omitted (no `--her` flag).

## 6. Success criteria

| Metric | Promising | Exhausted | Abandon |
|---|---|---|---|
| Touch rate (det. eval, best ckpt, 20 eps) | ≥ 25% | 5–25% | < 5% |
| Eval mean reward | ≥ −500 | −500 to −2000 | < −2000 (worse than freeze) |
| Eval std (across 20 deterministic eps) | > 50, < 500 | < 50 (single attractor) or > 1000 (chaotic) | std=0 (degenerate) |
| Per-step reward at best checkpoint | ≥ −0.20 | −0.20 to −1.00 | < −1.00 |
| Vision-ablation sensitivity (mean L2 action delta, 200 steps) | ≥ 0.30 | 0.10–0.30 | < 0.05 (vision inert) |
| ent_coef | held at 0.2 throughout | drifts | runaway |
| Falls / cart departures | 0 | < 5/20 | ≥ 5/20 |

The vision-ablation metric is the decisive number. Touch rate alone could be inflated by proprio-grope against the cart's predictable trajectory (the cart hits a ball periodically by geometry regardless of what AB does with its arms). Touch rate ≥ 25% AND ablation ≥ 0.30 together would be the first run in this project's history that simultaneously achieves the locomotion-link prerequisite (via the cart) and the perception-link goal (via a load-bearing eye).

## 7. Budget
**250K steps.** Default per the project rules. At 16 parallel envs on MPS this is approximately 4–5 hours. CheckpointCallback every 50K means we have four mid-run checkpoints to render from. Extend to 1M only if 250K shows touch rate ≥ 25% and ablation ≥ 0.30; otherwise the failure mode tells us what to redesign rather than how long to wait.

## 8. Pre-flight checks (per CLAUDE.md training workflow)

1. **Sanity-render a fresh (untrained) episode.** Spawn the cart at a random initial heading, advance the env for 600 steps with `np.random.uniform(-1, 1, action_dim)` per step, and render an overhead + ringside video. Confirm: cart traverses the platform smoothly, bounces correctly at all four edges, AB stays mounted, the hunger penalty accrues quadratically (log `info["steps_since_last_contact"]` and the per-step cost to verify), and the stereo camera frames show the balls drifting through the FOV as the cart passes them. The balls and platform must be inside the lens; if AB's downward 15° tilt clips them out, raise the head or reduce the tilt before launch.

2. **Smoke-test the hunger curve.** Run a 2000-step episode with zero contact and a separate 2000-step episode with one contact at step 500 and another at step 1500. Log the total reward and verify it matches the quadratic integral. Off-by-one mistakes in `steps_since_last_contact` are the equivalent of the her_wrapper bug from Phases D–E and would silently invalidate the run.

3. **Confirm hip actuation is off.** With `--hip-actuation off`, write a policy that outputs `[0]*hip_dim + uniform(action_other)` and one that outputs `uniform(hip_dim) + [0]*action_other`. Verify the second produces no torso motion (the cart still drives it, but the hip torques don't bias it). The action-space mask must clamp hip inputs to zero before they reach MuJoCo.

4. **No HER wrapper.** Confirm `--her` is NOT passed; the wrapper composition bug from D/E was a one-line oversight and verifying the absence is cheap.

## 9. Risks / open questions

**Cart-mount physical instability.** If the cart accelerates faster than AB's joints can damp (e.g., on edge-bounces), AB's torso may flap or detach. Mitigation: the constant-velocity drive means accelerations are zero except at bounce instants; ensure bounce direction-reversals are instantaneous in `qvel` rather than ramped, so there is no transient acceleration. If AB still flaps, increase the weld constraint stiffness, or lower the cart's speed.

**Hunger too aggressive → freeze.** If the quadratic hunger ramps fast enough to dominate the contact reward by step ~200, the policy may give up entirely (return to the Phase D total-stillness attractor, just in a moving cart). Conversely if it ramps too slowly the cart drives around for 2000 steps and the policy never feels pressure to reach. Diagnostic: monitor the per-step cost at evaluation milestones. If `per_step_cost` at step 1000 is already < −2.0 by the first eval (50K), reduce `HUNGER_RATE` to 0.10.

**Vision-load-bearing assumption may not hold.** The cart guarantees that balls *enter and leave the FOV* regardless of action — but the policy can still choose to ignore the pixel stream if proprio plus memory flags plus cart-time-encoding is sufficient to solve the task by timing. Phase A's vision-ablation 0.0226 on the flat-concat StereoCrawlerCNN is the cautionary data point: the architecture can stay inert even when vision is the only sensible signal. If the ablation comes back < 0.10 at 250K, the diagnosis is architectural (flat-concat fails again) and Phase H should pair the cart substrate with the dialogue architecture from Phase B (with the gate-floor fix).

**Cart speed mismatch.** 0.15 m/s is a guess. Too slow: AB never sees a ball during its first 50K of warmup, no learning signal. Too fast: AB can't react in time; the visual flow blurs and the arm action takes longer than a ball is in reach. The pre-flight render at 0.15 m/s should be timed — count how many ball passes occur per 2000-step episode at random heading. If that number is < 4, lower the cart speed or shrink the platform.

**Hip-disabled action space may be too narrow.** With only arms and head, AB has ~10 actuated DOFs. If the arms cannot extend far enough to reach a ball passing 30 cm to the side of the cart, the task is geometrically unsolvable regardless of perception. Mitigation: the pre-flight render should include a hand-coded "extend right arm down to reach" trajectory across multiple cart passes to verify reachability. If contact is geometrically impossible, widen the cart or grow the arms in the XML before launch.

## 10. What this experiment DOES NOT do

This run does not test whether AB can sustain its own locomotion (deferred — that is the entire F-* family of candidates and is the question Phase G-cart sidesteps rather than answers). It does not test the F-tipped-init, F-curiosity, F-sustained, or F-info-velocity proposals; those remain on the candidate list. It does not address the her_wrapper.py 4/5 relabel dilution from Phases D/E2 because HER is dropped entirely — that hygiene fix is recommended as a separate small task before any future HER run. It does not test whether what AB learns under cart-locomotion transfers to self-locomotion later; that transfer question is its own subsequent experiment (Phase H or later). And it does not test the dialogue architecture; this run uses the existing `StereoCrawlerCNN` + flat MLP head, on the principle that one structural change (substrate redirect) at a time.

## 11. Comparison to F-tipped-init explicitly

F-tipped-init was the leading Phase G candidate going into this proposal because it directly addresses both layers of the Phase F-2k failure mode — the "freeze after first action" attractor and the "initial-condition-dependence" lottery — within the existing substrate. It is conservative, narrow-scope, single-file. We are deferring it not because it is wrong but because the project has now spent 10+ phases trying to bootstrap link 2 of the dependency graph (`can move`) and has not extracted useful signal at link 5 (`vision integration`). F-tipped-init would, on its best day, produce a creature that locomotes; we would then need another follow-on run to test whether vision becomes load-bearing on top of that locomotion, and that follow-on has historically failed eight times in this project. The cart redirect tests link 5 (`vision becomes load-bearing`) directly by skipping link 2, mirroring the developmental scaffolding a real infant gets from a caregiver. If Phase G-cart succeeds, F-tipped-init becomes a parallel track for "self-locomotion later"; if Phase G-cart fails, F-tipped-init remains the next single-substrate intervention to try and we know more precisely what the cart did not fix.
