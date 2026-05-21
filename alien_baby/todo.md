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
