# Phase H Proposal — Moving Balls (`mimo_phase_h_moving_balls_v1`)

## 1. Tag / title
**`mimo_phase_h_moving_balls_v1`** — balls move; the policy must time the arm to where a ball *will be*, which proprio + memory cannot recover.

## 2. One-sentence summary
This experiment keeps the Phase G cart substrate that produced our first sustained arm-extension behavior and the R20+R30 reward recipe that stabilized it, but replaces the stationary targets with constant-velocity bouncing balls, so that the body's blanket-sweep coverage is no longer sufficient and the policy must consult the pixel stream to predict where to reach.

## 3. Hypothesis
**Single testable claim:** Under cart-induced locomotion with moving balls (constant-velocity, edge-bouncing, magnitude 0.05–0.08 m/s), the vision policy will achieve a both-touched rate at least 25% higher (relative) than the matched proprio-only control at the same ball speed, and a vision-ablation sensitivity (mean L2 action delta with pixels zeroed) of ≥ 0.20 on the best vision checkpoint.

- **Confirmed** if vision both-touched beats proprio control by ≥ 25% relative AND ablation ≥ 0.20: vision is finally load-bearing.
- **Marginal** if ablation lands in 0.05–0.20 or vision matches proprio within noise at speed=0.05 but pulls ahead at speed=0.08: the substrate is on the right track but the speed regime or training budget needs more probing.
- **Refuted** if vision and proprio remain identical-within-noise at both speeds AND ablation < 0.05: moving balls also fail to make the eye load-bearing in this body, and the project needs a structural change beyond the task (e.g., one-armed body, gaze-gated reward, or a different RL algorithm) — see the four candidates listed at OVERNIGHT_LOG.md lines 786–816.

## 4. Substrate changes from Phase G (R20 + R30 recipe combined)

**Inherited from Phase G unchanged.** Cart in constant-velocity-bouncer mode at 0.15 m/s; AB rigidly mounted; hip actuation off; arms + head only; stereo vision; two binary `touched_ball{1,2}` memory flags in proprio; `ent_coef=0.5`; entropy anneal 0.5 → 0.2 over `t=15K-30K` (R20's stabilizer); velocity bonus scale = 0.10 (R20's mid-training extender); approach reward off; `strength_scale=1.0` (R30's arm-power setting that widened the reachable radius without collapsing); ball-offset curriculum 0 → 0.15 over `t=2K-15K`. This is the R20+R30 union — the only combination in project history that produced 19–20/20 hits with stable late-training behavior across three seeds.

**NEW: ball motion.** Each ball receives a per-episode-reset velocity vector of magnitude `BALL_SPEED` with heading drawn from `Uniform[0, 2π)`. Balls integrate position each env step (`pos += vel × dt`) and reflect off platform edges using the same four-line pong logic the cart already uses (sign-flip on edge crossing, no transient acceleration, no per-step heading noise). Velocity persists per ball until touched; on touch the ball is removed from the active set, matching the existing two-ball touch-and-disappear behavior. The deterministic-given-init choice is load-bearing: a learnable trajectory is what makes the visual stream informative — random per-step jitter would saturate the signal into noise that no policy could exploit.

**NEW CLI flag:** `--ball-speed FLOAT`, default `0.0`. The default-zero value MUST trigger the existing stationary-ball code path bit-exactly (no integration call, no edge-reflection check, ball position written once at reset and never again). Backward compatibility with Phase G runs is non-negotiable; every prior cart result must reproduce when `--ball-speed` is omitted.

**Two speed settings tested in this overnight.** `0.05 m/s` and `0.08 m/s`. The cart is 0.15 m/s for reference; 0.08 m/s makes the relative velocity between the cart and a ball as large as ~0.23 m/s closing or as small as ~0.07 m/s closing depending on the angle, which puts the timing window inside the reaction time of an extended arm. 0.05 m/s is the gentler regime to confirm the gradient before pushing harder.

**Curriculum offset: do NOT stack.** Keep the R20 curriculum (0 → 0.15 over `t=2K-15K`); do not raise the offset target. Moving balls already inject novel difficulty and the R30 evidence is that 0.20 is at the edge of trainable even with stationary balls. Stacking moving-ball difficulty on top of an offset-extrapolation target is the kind of compound change Phase G's "one structural change at a time" rule was written to prevent.

**Ball velocity is NOT exposed to proprio.** This is the critical hygiene check. If `ball_vel_x` or `ball_vel_y` leak into the observation vector via the existing ball-position observation pathway, the proprio control gets the trajectory information for free and the comparison is invalidated. Pre-flight check 3 verifies this is air-gapped.

## 5. Launch command

Four runs in sequence (the all-nighter), launched from a single shell script that waits on each to finish and chimes between them:

```bash
# R32 — proprio control, ball-speed=0.08
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
    --seed 42

# R33 — vision, ball-speed=0.08 (vision auto-triggers DummyVecEnv per train_crawler.py:218)
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
    --seed 42

# R34 — proprio control, ball-speed=0.05  (same as R32 with --ball-speed 0.05 and run-tag _R34_proprio_speed05)
# R35 — vision,           ball-speed=0.05  (same as R33 with --ball-speed 0.05 and run-tag _R35_vision_speed05)
```

Flags not present in current `train_crawler.py` that need to be added by the training engineer: `--ball-speed`. This threads through to `MimoCrawlerCartEnv.__init__` and requires (a) a new `_step_balls()` call in `env.step()` that integrates ball positions and applies edge reflection, gated by `ball_speed > 0`; (b) a per-episode-reset hook in `env.reset()` that draws a fresh heading and writes `self._ball_vel[i]` for each active ball; (c) verification that touch detection (`touched_ball1` / `touched_ball2`) still fires from the same contact-geom path with the ball in motion. All other flags are reused from Phase G as-is.

## 6. Success criteria

| Metric | Promising | Marginal | Refuted |
|---|---|---|---|
| Vision both-touched vs. matched proprio control (same speed) | ≥ +25% relative | +5% to +25% | within noise (±5%) or worse |
| Vision-ablation sensitivity (mean L2 action delta, 200 steps, pixels zeroed) | ≥ 0.20 | 0.05–0.20 | < 0.05 (vision inert again) |
| Per-speed gradient: proprio both-touched at 0.08 vs. 0.05 | drops as speed rises | drops slightly | flat (motion doesn't bite) |
| Per-speed gradient: vision both-touched at 0.08 vs. 0.05 | flat or rises (vision compensates) | drops less than proprio | drops same as proprio |
| Eval mean reward (vision, best ckpt) | ≥ +200 | 0 to +200 | < 0 |
| Falls / cart departures | 0 | < 3/20 | ≥ 5/20 |

The decisive comparison is vision-vs-proprio at the same speed, paired with the per-speed gradient. If proprio's both-touched rate drops as ball-speed rises but vision's stays flat or improves, that's the signature we are looking for — the visual policy is using the pixel stream to compensate for the body's loss of blanket coverage. Vision-ablation ≥ 0.20 is the supporting evidence that the action distribution actually depends on pixels, not just on proprio-coincidence.

## 7. Budget
**80K steps per run × 4 runs = 320K steps total.** At ~16 parallel envs (proprio) and 8 envs (vision) on MPS, the four runs should complete in ~5–7 hours wall clock — a single overnight. CheckpointCallback every 25K gives three mid-run checkpoints per run. Extend any single run to 250K only if the 80K result is in "Promising" or "Marginal" — the prior R10 (100K vision at static balls) result is the warning that vision sometimes needs longer to find the attractor.

## 8. Pre-flight checks (per CLAUDE.md training workflow)

1. **Sanity-render an untrained episode at ball-speed=0.08.** Launch the moving-ball env, advance for 2000 steps with `np.random.uniform(-1, 1, action_dim)` per step, and render an overhead + ringside + head_cam video. Confirm: balls move at the requested speed, bounce off all four platform edges (env step dt = 0.005 × n_substeps=4 = 0.020 s; at ball_speed=0.08 m/s per-step travel is 1.6 mm, so the edge-cross window is small and tunneling is not a concern as long as the reflection check is inside the substep loop or uses an edge margin ≥ 5 mm), are visible from the head_cam as they pass, and are NOT clipped by the 15° downward head tilt. Both balls must remain on the platform for the full 2000 steps with zero touches.

2. **Smoke-test touch detection with balls in motion.** Run an episode where the cart is steered into a moving ball's path (hand-coded constant action) and confirm `info["touched_ball1"]` fires when the contact-geom registers, and that the ball is removed from the active set on touch. Run a second episode where the arm is hand-extended into a moving ball's path; confirm the same. Off-by-one mistakes in the touch path with a moving ball would silently break the comparison.

3. **Confirm ball velocity is NOT in proprio.** Print the observation dict at reset and step 1; confirm `obs["proprio"]` length is identical between `--ball-speed 0.0` and `--ball-speed 0.08`. If the lengths differ by 2 or 4, velocity leaked in. The pixel-only signal property is what the entire phase rests on.

4. **Hand-coded "stationary outstretched arm" check.** Write a policy that holds AB's arms in the maximum-reach pose for the entire episode (no head motion, no arm motion after step 0). Run 20 episodes at ball-speed=0.08. If both-touched rate exceeds 30%, the ball motion is too slow to break blanket-sweep — raise the speed to 0.10 or 0.12 before launching R32/R33. The whole phase is premised on motion breaking blanket-sweep; if blanket-sweep still wins, the substrate has not actually changed.

5. **`--ball-speed 0.0` reproduces Phase G bit-exactly.** Run a 5000-step deterministic eval of R30-best with `--ball-speed 0.0` explicit and with the flag omitted entirely; confirm the per-step reward stream is identical to within floating-point noise. If it isn't, the default-zero code path has a bug and every Phase G result is at risk.

## 9. Risks / open questions

**Ball motion may make the task geometrically harder than expected.** It is possible that both vision and proprio drop together — moving targets are harder for everyone, and the comparison may show "vision and proprio are both bad" rather than "vision pulls ahead." Mitigation: the per-speed gradient (proprio 0.05 vs. proprio 0.08) is the diagnostic. If proprio's both-touched is flat across speeds, the motion isn't biting and we need higher speed; if proprio's both-touched collapses, we may be in a regime where neither modality can solve the task and we need to back off speed.

**80K may be too short for vision.** R10 ran 100K with vision at static balls and never found the attractor; R33/R35 at 80K may be in the same trap. The four-run plan trades depth for breadth — we want to know whether *any* speed setting shows a vision-advantage signal before committing 250K to a single run. If R33 or R35 ends with `ep_rew_mean` still climbing on a non-collapsed trajectory, extend that single run rather than starting a new structural change.

**Default-zero code path bug.** The single highest-impact bug we could ship is a `--ball-speed` implementation whose `if ball_speed > 0:` gate has an off-by-one, a NaN propagation, or a side-effect that mutates ball state even when speed is zero. Pre-flight check 5 is the canary. If it fails, every prior Phase G result becomes suspect because we cannot rule out that the new code path is corrupting them.

**Vision-load-bearing assumption may still not hold.** The six prior vision tests (R6, R10, R14/R15, R17/R18, R26/R27, R28/R29) failed because in every substrate the body's blanket-sweep covered the box. Moving balls are our best remaining substrate-level hypothesis. If R32–R35 also show ablation < 0.05, the project has run out of substrate-only options and the next phase has to commit to a structural intervention (one-armed body, gaze-gated reward, DrQ-v2, or one of the four candidates in OVERNIGHT_LOG.md lines 786–816). That is a real possible outcome of this run and we should be ready to accept it.

## 10. What this experiment DOES NOT do

This run does not test gaze-gated reward, one-armed body, DrQ-v2, or any other structural intervention from the OVERNIGHT_LOG "definitive verdict" list. It does not extend the offset curriculum past 0.15 — strength=1.0 widened the reach already, and moving balls is the new variable. It does not test whether what AB learns at one ball-speed transfers to another speed. It does not test random per-step heading jitter (rejected: jittered trajectories are not learnable from vision). It does not change the body, the reward composition, the algorithm, or the cart speed. One structural change — ball motion — at a time.

## 11. Comparison to alternatives explicitly

**Gaze-gated reward (rejected as bribery-adjacent).** OVERNIGHT_LOG option (d): pay reward only when AB has been looking at the ball for N consecutive steps before contact. This would force vision to matter by construction, but it does so by tying reward directly to a visual variable, which is exactly the engineered-correlation pattern Greg's framework warns about. The eye would matter not because seeing the ball was the only way to solve the task, but because we paid for seeing. We want to test whether the *task structure* makes vision load-bearing, not whether we can bribe the policy into looking. Moving balls preserves that hands-off property.

**One-armed body (rejected as it eats the night on baseline re-establishment).** OVERNIGHT_LOG option (a): cut one arm off AB and re-run. This changes the body geometry, which means every Phase G baseline (R3, R19, R20, R30) has to be re-established before any vision-vs-proprio comparison is meaningful — that is a full overnight of baseline work before the comparison can even start. Moving balls keeps the body fixed, so the Phase G baselines remain the proprio controls and the comparison is direct from step zero. One-armed body remains the leading candidate for Phase I if Phase H is refuted.

**Static balls with longer vision budget (rejected as repetition).** The R10 100K vision run at static balls already exhausted this. Doing it again at 250K or 1M is the "re-running something that already failed" pattern the project's own rules prohibit without a clear "what is different this time" answer. Moving balls is the answer.
