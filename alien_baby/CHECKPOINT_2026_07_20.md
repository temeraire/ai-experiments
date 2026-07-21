# CHECKPOINT / HANDOFF — 2026-07-20  (read this first to resume)

## One-paragraph state
The quadruped WALKER now works and its vision is load-bearing for a COLOR CHOICE. After an
8-version gait saga (the real blocker was an environment bug, not the learner — see below), we have
a frozen walking gait (`ant_gait_v10`) that walks upright, steps for real, and turns cleanly on
command, and a high-level vision "driver" (`vsteer_v10`) that, reusing the crawler's transplanted
frozen eyes, walks to a RED target over a BLUE decoy 100% of the time with eyes vs 50% (chance) on a
fair-blind control. This is the FIRST clean vision-load-bearing result in the project. It is SCOPED:
it is categorical COLOR discrimination, NOT the fine SPATIAL-BEARING (steer-to-a-single-target-by-
direction) capability that Phase V found vision failed at. **The agreed next task is the SPATIAL-
BEARING TEST (spec below).** Full write-up: `FINDINGS.md`, entry dated 2026-07-20.

## The key lesson from today (carry it forward)
Gait v3–v8 all failed and I iterated the reward SIX times (incl. the field-standard legged_gym recipe
via a reuse-first lit-scout). The real cause was an ENVIRONMENT bug I introduced: enlarging the target
balls to body-size made them ~105 kg, and the gait training env parked only ONE of the two balls, so
the creature spawned with its foot inside a 105 kg ball and got flung every reset. Fixing it (park both
balls) flipped the gait reward from stuck-negative (−30 at 12M steps) to +864 at 320K. **Lesson: after
ANY change to world geometry, smoke-test the ENVIRONMENT (spawn overlaps, contacts, masses) before
blaming the learner.** Also see the new CLAUDE.md rule "Diagnosis is a HYPOTHESIS until a control
confirms it" (added today from David's feedback about false-causality/going-in-circles).

## Key files & frozen models (all present, nothing mid-run)
- Gait (frozen "legs"): `alien_baby/results/ant_gait_v10_best/best_model.zip`. Trained by
  `alien_baby/crawler/train_ant_gait.py` (standard velocity-command recipe: saturating speed + turn
  tracking, feet-air-time, upright/smoothness regularizers; BOTH balls parked far in reset). v10 walks
  ~0.23 m/s, fastest at cmd_fwd≈0.6 (slows at 0.8), turns cleanly, 0 falls. Validate with
  `eval_gait_distance.py` (5 seeds; measures distance covered, not just heading).
- Vision driver: `alien_baby/results/vsteer_v10_best/best_model.zip`. Trained by
  `alien_baby/crawler/train_vision_steer.py` (VisionSteerEnv: 2 balls 2.5–4 m ahead, both in view,
  ≥1.2 m apart, random red/blue label; high-level obs = 9 proprio + stereo 32×32; action→`cmd`
  mapped `cmd_fwd=0.3*(a0+1)`∈[0,0.6], `cmd_turn=0.6*a1`∈[±0.6]; drives frozen v10 gait).
- Transplant source (crawler eyes): `alien_baby/results/mildhead_vis_s0_best/best_model.zip`
  (`--init-vision` loads its conv/proj/bearing, freezes conv trunk, retrains head).
- Choice eval: `alien_baby/crawler/eval_vsteer_choice.py` (sighted vs blind; render fixed to mimsave).
- Body: `alien_baby/crawler/quad_walker.xml` (Ant + stereo eyes; balls r=0.5, density=3 ≈ light).
- Glossary term added: `cmd` (the two-number command). CLAUDE.md rule added: hypothesis-gate.

## KNOWN caveats to fix in the next work
1. **Blind control:** hard-zero AND gray pixels collapse the policy to a fixed action (OOD artifact) →
   the FAIR blind is NOISE pixels (policy still varies, 50% red = chance). Make NOISE the standard blind
   in `eval_vsteer_choice.py` (currently it zeroes; the noise probe was a one-off).
2. **Single seed, n=80** — cheap to confirm vsteer_v10 with a 2nd training seed if wanted.

## THE NEXT TASK — SPATIAL-BEARING TEST (agreed 2026-07-20)
Question: does the walker's vision encode the DIRECTION to a target (spatial bearing), or only a
categorical color flag? Phase V found vision failed at direction (ablation peaked at ecc=0, flat
elsewhere). The color-choice win does NOT resolve this. Test whether the creature can steer to a
SINGLE target by WHERE it is, using vision.

Suggested build (reuse VisionSteerEnv; make a variant or a flag):
- ONE ball only (drop the decoy), spawned at a RANDOM bearing within the view cone (e.g. ±0.5 rad)
  and 2.5–4 m out, in view (verify, winnability). Reward = get closer / reach it (d<1.0). No color
  choice — success requires turning the correct WAY, which needs the ball's DIRECTION from vision.
- Train a driver (reuse transplanted frozen eyes, frozen v10 gait), then evaluate with the
  INSTRUMENT-VALIDATION the project requires (GAZE rule, CLAUDE.md / 2026-07-17 rule):
  **sighted-R²(true target bearing, creature heading/turn) must be HIGH and blind-R² LOW.**
  Reaching alone is NOT enough — a target-independent motor habit can reach a single ball; the R²
  check is what proves vision is setting the DIRECTION. Use the FAIR (noise) blind.
- Render + watch (render-every-experiment rule). Then theory-monitor before writing to FINDINGS.
- Expected per Phase V: this is the hard one; vision may fail to steer-by-direction. If it does fail,
  the identified fix is an auxiliary supervised "where's-the-ball" decode loss on the encoder (the
  field's proven lever; our earlier attempt was broken by an unrelated bug, not the technique).

## How to resume in the new session
CLAUDE.md auto-loads. Say: "Continue AB — do the spatial-bearing test per
alien_baby/CHECKPOINT_2026_07_20.md." Everything needed is in the files above; nothing is mid-run.
