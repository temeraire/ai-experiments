# CHECKPOINT — 2026-07-19 (end of a long session)

## Where we are, in one paragraph
The project now has TWO creatures. (1) The original MIMo infant CRAWLER, which by session start had a
grounded, vision-steering eye (~88% directed choice), a distance law that provably extrapolates, and
the prism-recalibration apparatus. (2) A NEW quadruped WALKER (standard MuJoCo Ant) that we built this
session so the creature can actually move where it looks — with a working command-steerable gait and a
first vision-steering "driver," reusing the crawler's trained eyes. The session's headline outcomes:
the distance-generalization result PASSED (qualified); the Stage-2 prism-recalibration test DID NOT
PASS (our rig failed, not AB); we gave the crawler visual MEMORY for motion parallax; David designed
the GREEK ROOM (a primitive colonnade for occlusion + perspective); and we pivoted locomotion to a
walker body. The one hard problem that persists across both bodies: making VISION genuinely
LOAD-BEARING (drive behavior), not just present. Best result so far is a weak-positive.

## The fixed target (unchanged)
Grounding: perception genuinely tied to the world through the creature's own moving and touching, as
the substrate meaning grows on. Prism recalibration is the sharpest test of world-coupled vision.
Governing rule reaffirmed all session: "faithful to the MECHANISM (self-produced movement drives
vision via cross-modal error; motor competence before vision is trusted), free on the IMPLEMENTATION
(body, gait, compressed lifetimes are means, not sacred)."

## What this session did (the arc)
1. DISTANCE GENERALIZATION — David's "compress years / densely sample" idea, refined (azimuth already
   dense; DISTANCE is the under-sampled axis; dense coverage masks memorization unless you hold out a
   region). Built `--spawn-radius-hole`, ran a held-out-gap experiment on the vision line.
   RESULT: QUALIFIED PASS — the vision policy reaches/steers at a NEVER-TRAINED middle distance
   (gap 0.55-0.75m: 97% contact, 72% choice), and degrades GRACEFULLY past the 0.85m trained max
   (contact 95->91->72->23% over 0.85->1.20m) with choice holding above floor = a distance LAW that
   extrapolates, not a lookup table. Prior art: cite (Kirk 2022 interpolation; DeLosh 1997
   rule-vs-lookup; Taylor's "automatic interpolation device" 6.11).
2. LEGIBILITY TOOLING — so we can SEE what the creature sees and trust it: gaze-ray overlay
   (render_gaze_check) + a numeric FOV validator (99.4% geometry-vs-pixels agreement); head-steadiness
   tuning (mild damping recovered far-distance steering to 85% after heavy damping dropped it to 59%);
   eyebrows/hair/nose/neck/ghost-opacity fixes so face direction and real-vs-ghost balls read clearly.
3. STAGE-2 PRISM RECALIBRATION (the north star) — overnight 2-arm x 2-seed run. DID NOT PASS. On the
   clean metric no arm recalibrated (all <= base); the gain-field arm INVERTED (reproducibly). Two rig
   failures diagnosed: the mismatch aux-loss never converged (the teaching engine never engaged), and
   the after-effect RULER is broken (fails its own sanity check twice). Recorded as a diagnosis of OUR
   setup, not AB's limit.
4. PARALLAX — gave the crawler visual MEMORY: strided frame-stacking so the eye can see self-motion
   (motion parallax is a memory-over-time cue a single frame can't carry). Warm-started by TILING the
   trained conv filters across the new time-slots (starts as the working policy). Decode probe: motion
   adds only a LITTLE distance info (+0.06 R2) and vision still < proprio for distance = capacity
   present but not yet load-bearing (no task pressure to use it).
5. GREEK ROOM — David's world design (columns as stacked cylinders, checkered floor, occlusion +
   perspective; partial visibility = doubt / atomization of meaning). Built a first primitive room,
   rendered the creature's eye-view: confirmed columns loom, corridor perspective, and the target
   partly OCCLUDED behind a column (see-but-must-move). This is the environment that would FORCE the
   move-to-see the memory was built for.
6. WALKING — decided to give the creature a WALKING body rather than teach the infant to walk. Route:
   QUADRUPED FIRST (stable, fast to get walking on our machine), then TRANSFER to a biped later (the
   learned vision + steering transfer because they output an abstract command; only the gait and the
   distance calibration are body-specific -- confirmed by Taylor 6.14: crawl->walk preserves the
   mechanism, only eye-height calibration re-tunes, additively/fast). Built the Ant + stereo eyes;
   camera preflight PASSED (eyes see the floor ball, no leg occlusion).
7. GAIT — trained a command-STEERABLE Ant gait (v3): walks straight on command, turns both directions
   correctly, stays upright. Proprio-only so it trains in ~2 min. This is the frozen "legs."
8. VISION-STEERING DRIVER — hierarchical: eyes -> (forward, turn) command -> frozen gait. Iterations,
   each fixing a real flaw:
   - s0: reached balls but BLIND matched sighted (task too easy - close ball, narrow cone).
   - s1: widened spawn cone -> still blind>=sighted (a blind sweep finds side balls).
   - s2: added a blue DECOY, but biased placement let blind exploit POSITION (73% by geometry).
   - s3: SYMMETRIC decoy -> FIRST positive gap (sighted 53% > blind 42%, blind pinned near 50%) but
     WEAK -- from-scratch eyes barely learned color in 500K steps.
   - PROCESS PIVOTS (David): (a) STOP re-deriving vision from scratch when the crawler already learned
     to see -- lit-scout reuse-first map says TRANSPLANT the crawler encoder (freeze conv trunk,
     retrain head; downloadable foundation encoders like VC-1 are NOT a safe win -- 224px-photo domain
     gap vs our 32px sim frames). (b) WINNABILITY -- a target the creature can't see is a useless test;
     MEASURE the facing every try and place the balls in front of where it ACTUALLY looks, never a
     blind spot; searching for out-of-view targets comes LATER.
   - s4/s5: transplanted+frozen crawler eyes + balls placed relative to MEASURED forward + a
     winnability measurement (balls-in-view% + how long the target stays in view) reported every run.
     s5 is RUNNING as of this checkpoint.

## Key numbers to remember
- Distance holdout: gap 97% contact / 72% choice; extrapolation contact 95->91->72->23% (0.85->1.2m).
- Head steadiness far-choice: floppy 76.8% -> heavy-damped 59.4% -> mild-damped 85.4% (mild wins).
- Gaze viewer validity: 99.4% geometry-vs-pixel agreement (the instrument is trustworthy).
- Stage-2 prism: no recalibration; aux-loss flat; after-effect ruler invalid (broken).
- Parallax: motion adds +0.06 R2 to distance decode; vision (0.50) < proprio (0.53) = not load-bearing.
- Ant gait v3: forward straight (heading -8deg), turn-left +16deg, turn-right -22deg, stays up.
- Vision-steering best so far: s3 sighted 53% vs blind 42% (weak positive). s5 pending.

## The central open problem
Making VISION LOAD-BEARING -- the policy actually USING what it sees to act -- across BOTH bodies. It
was the crawler's multi-phase struggle and it recurs on the walker. Current bet: reuse the crawler's
trained encoder (transplant+freeze) + guarantee winnable, in-view targets + (if still weak) a clean
auxiliary supervised "where's the ball" decode loss (the field's proven fix; our one prior attempt was
broken by an unrelated bug, not the technique).

## Process / communication rules locked into CLAUDE.md this session
- Always give results in PLAIN ENGLISH (technical alongside is fine), tie to Taylor in plain English.
- Drop the word "honest" as filler.
- When David ASKS what a term means, it ALWAYS gets a glossary (+ flashcard) entry.
- The "prism-on" cue is REQUIRED: whenever untouchable ghosts appear, AB gets a persistent signal it
  is wearing the lens (Taylor's glasses-are-on cue; a binary flag).
- Lit-scout REUSE-FIRST mandate: for any capability, return (1) what's been done + how, (2) what to
  DOWNLOAD that works, (3) the recipe if not -- never re-derive or discard reusable work.
- Winnability by MEASUREMENT: verify the target is seeable every try; never assume the geometry.

## What's running now
`vsteer_s5` -- walker vision-steering with the crawler's transplanted+frozen eyes, forward-relative
in-view ball placement, red decoy vs blue, reporting winnability (balls-in-view%, target-in-view%) and
choice (sighted vs blind) + an eye-view video. Watch: does sighted beat the ~50% blind floor, and does
it keep the target in view instead of turning away.

## Next steps (in order)
1. Read s5: if sighted >> blind AND it keeps the target in view -> vision is load-bearing on the walker
   (Phase 1 core met); if still weak -> add the clean aux-decode loss on the transplanted encoder.
2. Then: put the walker in the GREEK ROOM (occlusion forces move-to-see; the memory + parallax become
   useful), first steering to visible targets, then search for occluded ones.
3. The PRISM/grounding north star remains the ultimate target -- but it is BLOCKED on repairing the
   after-effect ruler and the non-converging mismatch engine (Stage-2 v2 protocol already drafted:
   fix ruler first (free), then engine; add the prism-on cue; then the awareness battery). Bring the
   walker's better mobility to it once vision-steering is solid.

## Key files/artifacts (this session)
- Env/training: mimo_crawler_env.py (spawn_radius_hole, frame_stack), crawler_cnn_extractor.py
  (frame-stack general), train_head_search.py (hole+frame-stack+tiled warm-start), train_ant_gait.py
  (steerable gait), train_vision_steer.py (hierarchical driver + transplant + winnability measures),
  quad_walker.xml (Ant + stereo eyes + decoy), mimo_crawler_greek.xml (Greek room).
- Tools: render_gaze_check.py, verify_gaze_fov.py, render_greek.py, render_quad_preflight.py,
  eval_depth_dissoc.py.
- Docs: FINDINGS.md + THEORY_LOG.md (distance holdout, prism-fail), PREREG_distance_generalization_
  holdout.md, PREREG_stage2_prism_v2.md, GLOSSARY.md (many new terms), this checkpoint.
- Models (results/, gitignored): mildhead_vis_s0 (best crawler eyes, the transplant source),
  ant_gait_v3 (frozen legs), vsteer_s3 (best vision-steering so far), parallax_mem_s0.
