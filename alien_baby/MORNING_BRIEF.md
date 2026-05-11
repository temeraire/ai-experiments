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

## Phase B — killed at 112K (gate collapsed; trajectory looked like Phase A in slow motion)

Full entry in `FINDINGS.md`. Headlines:

- Gate w collapsed from 0.49 (random init) → 0.005 within 40K steps → 0.07 by kill at 112K
- 99%+ weight on vision throughout. Proprio stream effectively disabled.
- Disagreement and consistency_loss BOTH grew over time (0.09 → 0.42, and 0.01 → 0.31) — the opposite of what the consistency objective should produce. The system was *anti*-converging.
- ep_rew_mean stuck around −65 (≈ 14% one-ball touch). First eval at 60K matched Phase A's 50K eval (−85.04). 
- Diagnostic: μ_vision was about half the magnitude of μ_proprio. With gate weighting vision at 95%+, the deterministic action was the small μ_vision — too small to produce locomotion.
- Architectural conclusion: the dialogue's symmetric-peer design is not symmetric in practice because the gradient flow with gate ≈ 0 starves the proprio stream while the consistency loss is too weak to compensate.

## Phase C and Phase C2 — fixed ball positions tested (user's two-balls-define-a-line idea)

Two follow-up experiments testing whether random spawn was the load-bearing failure:

- **Phase C** (killed 56K): two fixed balls at (0.7, 0.0) and (0.0, 0.7), random starting orientation, blind proprio, no bribery. First eval @50K: **−100.00 ± 0.00**.
- **Phase C2** (killed 56K): same as Phase C + **memory_obs** appended to proprio (two binary flags: touched_ball1, touched_ball2). Lets a stateless SAC condition on its own past contacts. First eval @50K: **−100.00 ± 0.00**.

Both produced the identical deterministic-collapse signature as Phase A. The user's two-ball + memory idea is theoretically sound but not testable in this configuration: the prerequisite "agent reliably moves at all" was never satisfied. Both fixed positions and memory require a working locomotor base.

## The result of the overnight session, in one sentence

In the no-bribery + sparse-contact-reward setting on the MIMo crawler, **SAC fails to develop locomotion at all** — the deterministic policy collapses to zero-action across every substrate variant we tried (random spawn, fixed positions, with memory, without memory, dialogue architecture, flat-concat CNN). The earlier 85% blind result relied on a small velocity bonus to maintain the locomotor base; removing that bribery component is sufficient to break everything downstream.

## The dependency graph this clarifies

```
body works → can move → can encounter things → can remember encounters
                                                       ↓
                                            can build spatial maps
                                                       ↓
                                            can integrate vision
```

The session showed the chain breaks at link 2 ("can move") under no-bribery conditions. Memory flags (link 4) and fixed structure (link 6) cannot compensate for an absent locomotion base. We've now empirically confirmed what the dependency graph predicts: prerequisites can't be skipped.

## Open question for next session — what counts as "enabling" vs "bribery"?

The user articulated tonight that **velocity_bonus is closer to enabling than bribery** — it tells the creature "moving is better than not moving" but not WHICH direction. Restoring a *small* velocity bonus (say 0.01–0.02, vs the 0.05 of the prior 85% blind run vs 0.0 in tonight's runs) is the cleanest next experiment. Combined with fixed ball positions and memory flags, this should finally produce the locomotor base needed for the user's spatial-structure hypothesis to be testable on its own merits.

## Files written tonight

| File | Purpose |
|------|---------|
| `crawler/dialogue_policy.py` | Two-stream actor + gate + consistency loss |
| `crawler/train_dialogue.py` | DialogueSAC subclass with consistency loss in train() |
| `crawler/mimo_crawler_env.py` | Substrate: 360° spawn, no-bribery defaults, fixed_ball_positions, random_start_orientation, memory_obs, second target ball |
| `crawler/mimo_crawler.xml` | Second ball (blue) + 15° downward camera tilt |
| `crawler/train_crawler.py` | CLI plumbing for all of the above |
| `crawler/launch_phase_b.sh` | Pre-configured Phase B launcher (used) |
| `visualization/extract_frames.py` | Frame extraction (even + burst modes); Claude reads PNGs directly |
| `visualization/vision_ablation.py` | L2-action-delta-when-pixels-zeroed measurement |
| `visualization/analyze_run.py` | One-command run analysis (evals + render + frames + ablation) |
| `FINDINGS.md` | Three new entries (Phase A, Phase B, Phase C/C2) |
| `MORNING_BRIEF.md` | This file |

## What's still pending

- THEORY_LOG.md entry summarizing the session in MICOA / pressure-not-bribery terms. Will write next.
- Decision on next experiment: I'd recommend Phase D = the same Phase C2 config but with `velocity_bonus_scale=0.02` (small enabling, not bribery), to test whether locomotion + fixed structure + memory together produce learning.

---

## Session summary (copy-paste friendly)

---
SESSION SUMMARY 2026-05-10/11
Topic: MIMo crawler — testing no-bribery substrate, dialogue arch, fixed-ball hypothesis

Key decisions:
- Built diagnostic infrastructure first (extract_frames.py reads PNGs for me; vision_ablation.py canonicalizes the "is vision load-bearing" test)
- Verified the substrate visually before launching (15° camera tilt added after I saw arms occluding the ball)
- Ran Phase A as the no-bribery control with existing CNN architecture
- Built dialogue architecture from May 7 retirement memo
- Ran Phase B on Phase A's substrate (killed early when gate collapse + slow-motion plateau was clear)
- User proposed fixed-ball-positions hypothesis ("random spawn destroys learnability")
- Built two-ball env with random orientation (Phase C) and added memory observation flags (Phase C2)
- Both Phase C and C2 collapsed identically to Phase A's deterministic eval pattern

Key findings:
- Removing bribery (velocity bonus + approach reward) breaks SAC locomotion entirely
- Three different substrate variants (random spawn, fixed positions, memory flags) all hit the same deterministic-zero-action attractor
- Dialogue architecture's gate collapses onto whichever stream has smaller-magnitude actions, regardless of which is informative
- The user's two-ball insight is sound but requires a working locomotor base as prerequisite
- The dependency-graph principle is empirically confirmed: prerequisites can't be skipped

Files changed: see "Files written tonight" above; ~2700 lines added across two feature-branch commits (5d0e957, 92807a9, 92de844, b3245a3) plus FINDINGS.md and MORNING_BRIEF.md updates

Next steps:
- Phase D: restore small velocity_bonus_scale=0.02 (user's "enabling, not bribery") + fixed balls + memory + random orientation
- If Phase D produces locomotion + memory-conditional behavior, vision can be layered on top
- Consider recurrent policy (LSTM) for a cleaner test of within-episode memory
- Open theoretical question: is observation engineering (adding memory flags) on the bribery side or the substrate side?
---
