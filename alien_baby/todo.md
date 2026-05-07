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

## Notes / open questions

- The drive/hunger question: does velocity_bonus_scale of 0.02 create purposeful search
  or just random wandering? Tune after watching Phase 3 videos.
- The "dialogue reaching agreement" loss: λ too high → forces agreement too early,
  vision never develops its own signal. λ too low → they never integrate.
  Start at λ=0.05 and watch disagreement curve.
- The old breakthrough checkpoint (micoa_freeze_zeroinit_cone30 30K) is from the serial
  approach and will not carry over to the dialogue architecture. It is preserved on disk.
