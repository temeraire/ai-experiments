# Prism apparatus fix — make "offset" GAZE-RELATIVE, not world-anchored (proposal, 2026-07-18)

**STATUS 2026-07-18: CORE FIX IMPLEMENTED + VALIDATED (David approved).**
`mimo_crawler_env.py`: added `_update_prism_ghost()` (gaze-relative, cyclopean-eye, per-step),
called at reset (post-settle) and every step; removed the old world-origin placement. Validated in
the live env (`newprism_eye_0_30_60.png`): delivered retinal azimuth = +offset EXACTLY at 30/60,
elevation preserved (delta 0.0), ghost TRACKS the moving eye (stays +offset after 20 steps), offset-0
path unchanged (agent sees the real ball), `step()`/contacts intact. Old prism showed an EMPTY eye at
offset 30; new prism shows the ghost shifted laterally and in view.

DONE 2026-07-18 (David: "knock out i and iii"):
- **Gaze-frame spawn winnability** — opt-in `gaze_spawn=True` env flag + `_gaze_spawn_ball()`: after
  settling, re-places the ball so the VISIBLE target (ghost under a lens, real ball at offset 0) lands
  within +-spawn_cone/2 of the eye's GAZE, bearing from the eye, distance from the torso, clamped
  on-platform. Validated (30 resets x offset 0/30/60): visible-target gaze-az within +-75, ball 0.75 m
  from torso, on-platform 30/30. `gaze_spawn=False` (default) keeps the old world-frame spawn
  bit-identical. TODO wiring: pass `gaze_spawn=True` from the prism eval/train make_env when re-running.
- **Experimenter-only ghost marker** — `render_prism_decoy.py`: ghost renders as a TRANSLUCENT CUBE
  (vs solid SPHERE real ball) in the overhead/ringside panels, geom type restored before the agent's
  eye renders (agent still sees a normal sphere). Validated on `ghost_marker_check.png`.

RESOLVED 2026-07-18 (panel: strategist + lit-scout + Taylor local-book, all converged): the bearing
frame is NOT a body-vs-gaze choice — it is a GAIN-FIELD. Keep the displacement retinal (done), keep the
recalibration TARGET body/effector-frame (Taylor 8.12/9.20; Tsay/Ivry PReMo = shift lives in the
proprioceptive/effector map; VICES = effector-frame action), and carry head/eye posture as a
MULTIPLICATIVE gain on the retinal code (Pouget & Sejnowski 1997; Salinas & Abbott 2001). The pure-gaze
target is predicted to FAIL (Taylor's one non-adapter anchored to a head/gaze frame). The ~90deg
torso-vs-gaze misalignment IS the neck-proprioception term the gain field needs.

IMPLEMENTED + VALIDATED 2026-07-18:
- **Gain-field bearing head** (`crawler_cnn_extractor.py`, opt-in `gain_field=True`, `--gain-field`):
  FiLM-modulates the pixel latent by proprio (head-pose) before the bearing readout, so
  theta_vis = f(retinal, head-pose). Identity init (gamma=beta=0 => starts as the plain readout, learns
  the modulation). Validated: forward()/policy-input untouched; bearing depends on proprio once trained;
  default off = byte-identical. Full-path smokes pass (fresh AND warm-start from decoy_v2_ext_s0_best,
  non-strict missing=18/unexpected=0; mismatch aux trains the gain_mlp = 18 encoder tensors).
- **Gate-Zero winnability render: PASS** (`gatezero_agenteye_grid.png`): at offset 0/30/60 with
  `--gaze-spawn` the visible target is at the FOV edge at reset and brought to center by a head turn
  (offset 30 reached gAz~0 by t20). No empty-eye/off-screen case (the old-prism failure). Recommend
  narrowing the gaze-spawn cone to ~+-45-50deg for cleaner initial visibility (targets currently spawn
  as far as the +-75 edge).

=> The corrected-prism Stage-1 A/B is now simply **`--gain-field` OFF (plain body-frame readout,
control) vs ON (gain-field)**, both with the body-frame target and `--gaze-spawn` — this SUPERSEDES the
strategist's `--bearing-frame {body,gaze}` flag (never needed; the gaze target is the wrong map).

### CONFLICT surfaced during implementation — do NOT blindly move the training bearing to the gaze frame
`_ball1_ego_bearing` (the mismatch/training target) is deliberately in the BODY/root frame. Its
docstring records a 2026-07-12 matched-step A/B (theory-monitor verified): the HEAD/camera frame HURT
behavioral steering (67/43 sighted-vs-blind body-frame -> 53/50 head-frame), because the policy reads
the visual latent and steers the BODY, so a head-relative target needs a head-pose composition it does
not learn. So the earlier evidence favors the body frame for STEERING — but the prism displacement
lives in the GAZE frame. That is a genuine tension (prism in gaze frame, target in body frame, related
by a head-pose transform that varies as the head scans). It was measured under the OLD broken prism, so
it may not hold now. RESOLUTION: left the training bearing in body frame; flagged for the
experiment-strategist + a matched re-test under the corrected prism before changing it.

### Body frame is ~90 deg MISALIGNED from the functional forward (found 2026-07-18, David's probing)
Measured at the seed-7 settled pose: the ball is +97.8 deg in the torso's local frame but only +10.6
deg in the GAZE frame; the body's local +Y ("forward") points ~90 deg off the gaze/travel direction
because the torso is pitched into the prone pose (`topdown_geometry.png`). So "body-frame bearing" is
measured against a twisted axis that does NOT point where the creature looks or goes. This is almost
certainly a root cause of the head-vs-body-frame confusion, and an independent argument for defining
target geometry in the gaze frame (posture-invariant) once the steering-frame tension above is resolved.

---
(Original proposal follows.)

**Status: PROPOSAL. Nothing in the env changed yet. Awaiting David's review.**
Surfaced by David questioning "what is offset relative to?" + why the agent-eye view didn't match
the overhead. Diagnosis and the fix below are validated numerically (no env change, prototype only).

---

## 1. The problem (validated, not theoretical)

The current prism places the GHOST (what the agent sees) at the real ball's azimuth **measured from
the WORLD ORIGIN**, rotated by `prism_offset_deg`, once at reset:
```
th = arctan2(ball_x, ball_y) + offset      # azimuth about the WORLD ORIGIN
ghost = [r*sin(th), r*cos(th), ball_z]     # static world point
```
But the agent's eye is **not at the world origin**. It sits on the head, which reaches ~0.42 m
**forward** of the torso, and looks **forward-and-down ~18°**. So world-azimuth-from-origin is NOT
the angle the eye actually sees (the gaze-relative / retinal azimuth). Measured for the seed-7 frame,
offset 30°:

| quantity | retinal azimuth (what the eye sees) | in view? |
|---|---|---|
| REAL ball | +10.4° | yes |
| faithful +30° prism SHOULD show | +40.4° | yes |
| **CURRENT world-anchored ghost** | **+65.5°** | **NO — out of the ~±60° field** |

So the "30° prism" we have been training and testing on actually delivers a **+55° retinal
displacement that lands off-screen.** Two compounding errors:
1. **Wrong frame:** offset is about the world origin, but the eye is offset forward + tilted down, so
   a 30° world rotation ≠ 30° retinal shift (here it becomes 55°).
2. **Static, not continuous:** the ghost is a fixed world point placed once at reset. A real prism is
   fixed to the EYES and displaces the image continuously as the head/body moves. Ours does not.

Consequence: the eye has been asked to recalibrate to a displacement that (a) is the wrong size, (b)
is frequently off-screen, and (c) is not a constant retinal quantity. This is a **winnability
violation** (per CLAUDE.md): a failure to adapt is the apparatus failing to present a learnable
prism, not the creature failing. It plausibly explains much of the marginal/messy prism history.
Shown directly: at offset 30, seed 7, the agent's eye is **empty** (`eye_off0_vs_off30.png`) — real
ball hidden, ghost off-screen — so there is literally nothing to adapt to.

### The REAL ball has the same frame bug, in two other roles (David's Q1)
The proposal is ghost-heavy because the ghost is the agent's percept UNDER the lens, but the real
ball carries the identical world-origin frame error in two places that matter just as much:
- **As the recalibration reference (the mismatch "true bearing" target).** If the eye is trained to
  match a true bearing computed as world-azimuth-from-origin, the target is in the wrong frame too —
  so even a correct ghost would be taught against a mis-framed label. The true-bearing target must be
  the ball's **camera-frame azimuth**.
- **At offset 0 (the aftereffect test) there is NO ghost at all** — the agent sees the real ball, and
  the whole measurement is "does the eye aim off from the REAL ball." So the aftereffect reference
  (real-ball bearing) must be in the gaze frame as well.
=> The fix is not "reposition the ghost"; it is "**do all target geometry in the camera/gaze frame**"
— ghost placement, the training reference, and the measurement reference alike.

## 2. The fix — a faithful lateral prism in the GAZE frame (validated)

A real laterally-displacing prism shifts every visual ray by a constant angle **in the retinal
(camera) frame**, preserving elevation and range, regardless of pose. Implement exactly that: each
step, place the ghost by shifting the real ball's **camera-frame azimuth** by `+offset`, keeping
elevation and range.

```python
# camera (the eye) world pose, after mj_forward:
cam  = data.cam_xpos[left_eye_id]                 # eye position
M    = data.cam_xmat[left_eye_id].reshape(3,3)
right, up, fwd = M[:,0], M[:,1], -M[:,2]           # camera looks along -z

v  = ball_world - cam                              # eye -> real ball
vf, vr, vu = v@fwd, v@right, v@up
h  = hypot(vf, vr)                                 # horizontal magnitude in the fwd-right plane
az = atan2(vr, vf) + deg2rad(offset)               # shift RETINAL azimuth by +offset
ghost = cam + (h*cos(az))*fwd + (h*sin(az))*right + vu*up   # keep elevation (vu) and range
```

**Validated across seeds 7/11/23/42 x offsets 30/60:** delivered azimuth = `+offset` EXACTLY,
elevation change = `0.0` exactly. This is the correct, pose-invariant, range-preserving lateral prism.

### Implementation footprint
- Add `_update_ghost()` computing the above; **call it every step** (in `step()` after the physics
  substeps, before the agent's cameras are read), not just at reset. Same for the blue decoy ghost
  (`ghost2`) when `decoy_ball` is on.
- Reference eye: use the **head-center / cyclopean** frame (midpoint of left/right eyes) rather than
  one eye, so the two stereo views are symmetric. (A single world ghost can't be at exactly +offset
  in BOTH eyes — eye separation ~5 cm at ~0.7 m gives ~4° parallax. Cyclopean reference minimizes the
  asymmetry; it is a small, known limitation inherent to a single-object ghost.)
- **Reversibility:** offset 0 path is unchanged (no ghost; agent sees the real ball). Existing
  offset-0 checkpoints and evals are byte-unaffected.
- **BREAKING for offset != 0:** this changes what "offset 30" MEANS. Every prior prism-offset result
  (Stage-2 adaptation, the aftereffect probes, the 60deg dissociation) was under the OLD broken prism
  and is **not comparable** to new runs. Any adapted checkpoint must be **retrained** under the new
  prism. This is intended — the old prism was wrong.

## 3. Two connected issues this surfaced (decide scope with David)

**(a) The mismatch/recalibration target frame.** For the eye to learn a clean prism correction, its
supervised target (the "true bearing") should be the ball's **camera-frame azimuth** — the same frame
the displacement lives in. Then the eye's job is a single constant "subtract offset from retinal
azimuth," which is learnable. This is the perceptual half of the old head-frame-vs-body-frame muddle;
the transform from retinal to a body-frame ACTION is a separate learned mapping (gain-field / Taylor
interpenetration). Recommend revisiting the mismatch target frame alongside this fix.

**(b) Spawn geometry / camera FOV — winnability.** Even with the faithful prism, the prototype shows
real balls at large spawn angles are already out of view at reset (seed 23 real ball at −100° retinal,
seed 42 at −76°; seed 11 at +44° with the +60 ghost pushed to +104°). The spawn cone is 150° in WORLD
terms, but the usable eye view is much narrower and offset. So either the ball or its displaced ghost
is frequently unseeable. Options: (i) define the spawn cone in the GAZE frame and match it to the
usable FOV; (ii) widen / re-aim the camera; (iii) rely on head-search to bring targets in (but then
the displacement must stay faithful as the head turns — which the per-step gaze-relative fix gives).

**(c) Camera geometry for a prone body (David's point).** The 18°-down forward eye on the reached-out
head sees mostly a narrow floor patch ahead. A crawling infant's natural gaze is up-and-forward. Worth
deciding whether to re-aim the eye (less down / more forward) so more of the world — and the displaced
ghost — is visible without a large head turn. This is a body/camera design choice, separable from the
offset-frame fix.

**(d) Make the ghost visually distinguishable — EXPERIMENTER cameras ONLY (David's Q1).** In renders
it is currently hard to tell a ghost from a real ball (faint+shrunk is too subtle). Give the ghost an
obvious marker in the overhead/ringside debug views — a painted face, a wireframe, an "X", or a
distinct hue/checker. HARD CONSTRAINT: this must NOT appear in the agent's eye camera. A real prism
displaces a NORMAL-looking object; marking the ghost in the agent's pixels would hand the agent a
"this one is fake" cue and destroy the experiment. So the marking goes only in the display panels
(same pattern already used for the faint+shrunk ghost styling, which touches only the experimenter
cameras and restores before the agent's eye renders). Cheap; do it alongside the offset fix so future
renders are unambiguous.

## 4. Recommended order of operations
1. **Implement the gaze-relative offset** (Section 2) — the clear, validated fix.
2. **Fix spawn/FOV winnability** (3b) so the displaced target is seeable across the spawn set.
3. **Re-validate with renders**: agent-eye at offset 0/30/60, confirm the ghost appears at
   real_az+offset, in view, elevation preserved (reproduce the prototype numbers in the live env).
4. **Revisit the mismatch target frame** (3a) so recalibration is a clean constant subtraction.
5. **Only then** re-run Stage-1 -> Stage-2 prism training under the corrected prism, and re-attempt the
   aftereffect. All prior prism-offset results are superseded as baselines.

This supersedes the aftereffect-metric work from earlier today: those probes were measuring
recalibration to a prism that was the wrong size and often off-screen. Fixing the apparatus comes
first.

## 5. Validation artifacts (already produced, no env change)
- `results/videos/agenteye_off0.png` — agent sees the real ball at offset 0 (verified).
- `results/videos/agenteye_off30.png` — at offset 30 the agent sees nothing (ghost off-screen).
- `results/videos/eye_bearing_test.png` — balls at world-bearing -45/0/+45; only 0 visible (narrow
  effective view; the world-vs-gaze frame mismatch made concrete).
- Prototype numbers in this doc (Sections 1-2) reproduce from the seed-7 frame.
