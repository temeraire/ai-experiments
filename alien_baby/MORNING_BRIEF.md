# Overnight session — Morning brief

[This file will be filled in when training finishes. Below is the live status.]

## Quick status (auto-updated as the run progresses)

- Branch: `feature/mimo-dialogue-v1`
- Commits tonight: 2 (substrate + dialogue arch + diagnostic tools)
- Phase A: 250K substrate-only training (RUNNING)
- Phase B: 300K dialogue arch (CONDITIONAL — only if Phase A doesn't show vision integration)

## What was built tonight

**Substrate (no bribery):**
- 360° ball spawn (vs. previous 45°/180°)
- 2000-step episodes (vs. previous 600)
- Guardrails verified (already in XML — no fall termination)
- `velocity_bonus_scale` and `approach_reward_scale` both default-off via CLI
- Cameras tilted 15° downward in `mimo_crawler.xml` to clear prone-arm occlusion
  (anatomically realistic — a baby in prone position looks down, not horizontally)

**Diagnostic tooling:**
- `visualization/extract_frames.py` — pull PNGs from videos (even sweep + dense
  burst at 10fps for jitter detection); Claude reads frames directly.
- `visualization/vision_ablation.py` — measure L2 action delta when pixels zeroed.
  Calibrated against v10 (returns 0.0075 = "vision silent" — matches Sonnet's prior finding).
- `visualization/analyze_run.py` — combines render → frames → ablation → verdict
  in one command. Used at the 150K decision gate and at the end.

**Dialogue architecture (Phase B, ready to launch):**
- `crawler/dialogue_policy.py` — two-stream actor (proprio MLP + StereoCNN+MLP),
  learned scalar gate w ∈ [0,1], per-stream action means cached for consistency loss.
- `crawler/train_dialogue.py` — `DialogueSAC` subclass adding `λ · ||μ_p − μ_v||²`
  to the actor loss. All `dialogue/*` metrics flow into SB3 logger.
- `crawler/launch_phase_b.sh` — pre-configured 300K launch.
- Verified end-to-end on 250-step CPU smoke: consistency_loss recorded,
  gate_mean drifting from 0.5, disagreement nonzero. Architecture works.

## Decisions pre-committed before launch

Phase A stopping rule (at 250K):
- Touch rate > 60% AND vision-ablation sensitivity > 0.5 → vision IS load-bearing → Phase B skipped → declare result
- Touch rate < 60% OR ablation < 0.5 → substrate alone insufficient → launch Phase B (300K, dialogue)
- Either way: NO hyperparameter chase, NO bribery additions

## Hard limits I did NOT cross

- No git push, no PR
- No commit to main
- No deletion of prior results/checkpoints
- No bribery rewards added (velocity bonus = 0, approach reward = 0, no FOV reward)
- No runs longer than the pre-committed budgets

## Substrate verification (done before launch)

I rendered 5 init-state frames with the new substrate (random seeds 0, 1, 5, 10, 15) and read them as images. Findings:

1. **Prone start works.** Creature lies face-down, hips low. Anatomically correct.
2. **Guardrails work.** 20cm walls visible in ringside view, creature contained.
3. **360° spawn distribution confirmed.** Sampled bearings: -42°, +81°, +133°, -64°, +102° — full hemisphere coverage.
4. **Eye-camera occlusion is real.** The prone creature's own outstretched arms occupy the bottom ~40% of the camera frame. This blocks visibility of the ball when it's at low elevation (which is always, since the ball sits on the platform).
5. **Geometric check (programmatic).** Projected the ball into the camera frame for 3 seeds:
   - seed 0: bearing -42° in world → -61° from camera (camera is at world y=+0.46, forward of hip). Outside camera FOV by 16°.
   - seed 10, 15: ball is *behind* the camera entirely.
6. **Camera tilt fix applied.** Tilted cameras 15° downward in mimo_crawler.xml. This raises the ground in the camera frame and reduces "wasted" sky pixels, but does not solve arm-occlusion.

**Implication:** the substrate is hard. With 360° spawn, most resets put the ball outside the camera FOV at start. The creature has to either (a) turn its body to scan, or (b) move and let camera coincidentally capture the ball. Vision can provide signal only intermittently. This is the right kind of hard — it's what makes the no-bribery test meaningful — but it also means SAC may struggle to integrate vision in 250K steps.

The honest experimental prediction: Phase A probably comes back as vision-silent or vision-weak. Phase B (dialogue arch) is then the right next test. But Phase A is the necessary control; without it, any Phase B success could be attributed to the substrate fix rather than the dialogue arch.

## Phase A results — VISION SILENT (substrate alone insufficient)

Phase A finished and was analyzed. Full entry appended to `FINDINGS.md`. Quick numbers:

| Eval step | Mean reward | Touch est | Note |
|----------:|------------:|----------:|------|
| 50K  | −85.04 ± 65.20 | 1/20 (5%) | |
| 100K | **−60.01** ± 96.80 | 3/20 (15%) | peak |
| 150K | **−100.00** ± **0.00** | 0/20 | total collapse |
| 200K | −70.08 ± 89.75 | 2/20 (10%) | partial recovery |
| 250K | **−100.00** ± **0.00** | 0/20 | total collapse again |

- Best-checkpoint render (5 seeds × 600 steps): **0/5 touched**
- **Vision-ablation sensitivity: 0.0226** (15 seeds × 200 steps; threshold for "load-bearing" is 0.5)
- Frame inspection: creature flips onto its back, legs in air, never finds ball

The std=0.00 at 150K and 250K is diagnostic: every one of 20 deterministic episodes returned exactly −100.00 = 2000 × −0.05 step cost = zero contacts, zero motion. The deterministic policy collapsed to a literal no-action attractor — the SAC failure mode is *worse* without bribery than with it, because the policy has no gradient signal to maintain locomotion when vision is silent.

Pre-committed rule fired: launching Phase B (dialogue architecture) per the no-hyperparameter-chase discipline.

## Phase B status (live)

Started 07:55. Running 300K steps with the **same no-bribery substrate** but with the **dialogue architecture** (two-stream actor + learned gate + consistency loss). At 24K steps:
- gate_mean = 0.023 (97.7% weight on vision)
- disagreement_mean = 0.099
- consistency_loss = 0.016
- fps ~158 (full run estimated ~50 min)

Expected completion ~08:45. Then full analysis (render + ablation + frames).

The early gate value is the first thing to verify when results land: if the gate stays high-on-vision AND vision-ablation is large at the end, the dialogue architecture works as designed. If the gate stays high-on-vision but ablation is small at the end, the gate learned to weight vision *because the vision stream's actions happen to be quieter/smaller* (degenerate gate collapse), not because vision is informative. Those are very different outcomes.

## What's at the end of this file (when done)

- Phase B's evaluation history
- Phase B's vision-ablation result
- Phase B frame analysis
- Verdict and recommendation
- Session-summary copy-paste block

---

(Live status updated as Phase B completes...)
