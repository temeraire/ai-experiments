# Visual realignment program (prism) — staged & gated. Framing approved 2026-07-11.

GOAL (engineering, not a test): give AB the tools to LEARN genuine visual realignment — vision
swinging into line with reality via the sensorimotor loop — replicating the ESTABLISHED human prism
result. Winnability rule: if AB won't realign, our setup is missing a tool; that is not AB failing
and not a disproof of the phenomenon.

DIAGNOSIS (this session, theory-monitor-verified 3×): under a held +30° visual shift, AB DROPPED
vision and ran on a fixed motor habit — heading↔real-ball circular corr fell 0.38 → 0.00 (= blind).
Causes of the shortcut: (1) task winnable blind; (2) trained only under the shift, never reconciling
worlds; (3) NO seen-vs-felt error signal; (4) strong motor fallback.

THE TOOL: a seen-vs-CONTACTED mismatch signal. Eye's reported ball-bearing (theta_vis) vs the
direction the BODY actually met the ball at contact (theta_contact). David's call 2026-07-11:
contact-anchored / honest signal, NOT the sim oracle. Gradient trains the EYE ONLY (stop-grad on
the policy trunk), so the motor side cannot "solve" it by compensating. Sparse (updates at contact).

STAGED, GATED curriculum (David's hard constraint — do NOT be premature; each gate must pass before
advancing; if a gate fails, fix THAT stage, do not advance):
  S1 no lens: eye learns true ball-direction. GATE: heading↔real corr ≥0.30 (blind ~0) AND a
     blind-ablation control FAILS the task (vision-necessary) AND bearing-decode R² ≥0.30.
  S2 lens held +30°: mismatch drives re-align. GATE: under the lens, heading↔REAL corr recovers
     (≥~0.25, not merely "scores well") AND mismatch error re-minimizes (theta_vis shifts ~-30°).
  S3 lens alternates on/off: GATE: sight-steers in BOTH worlds; post-switch wrong-way error shrinks.

## SMALLEST FIRST STEP (build the new signal + de-risk it, NO lens) — IN PROGRESS (2026-07-12, overnight, autonomous)
- [x] env (mimo_crawler_env.py): `ball1_bearing` (true ego-bearing to real red) in info dict. Additive.
      NOTE: implemented as true ego-bearing; honesty enforced at the LEARNING level via contact-gating
      (eye trains only on episodes that reached the ball), not by withholding the geometry from info.
- [x] encoder (crawler_cnn_extractor.py): bearing head (sin,cos) → theta_vis off the pixel latent; shared
      `_pixel_latent` refactor so existing runs are byte-unaffected.
- [x] trainer (train_head_search.py): `--mismatch-coef` aux loss (unit-vec MSE of theta_vis vs true
      bearing) via a SEPARATE optimizer over the encoder; `MismatchPPO` freezes the encoder during
      PPO.train so reward trains heads only. `--no-gate-contact` toggles honesty off (default honest).
- [x] smoke test (2K steps): non-strict load OK (missing=6 = bearing head's 3 shared refs), aux loss
      falls 0.29→0.05, saved model LOADS for render (fixed an optimizer-surgery load bug via MismatchPPO).
- [x] render check: mm_smoke renders clean → cameras/physics intact under the new code path.
- [x] run 250K no-lens validation (stage1_ground_smoke_s0): DONE, no VOID. aux loss 0.29→0.078.
- [x] eval: **VALIDATION PASSED.** bearing readout circ_corr(theta_vis,theta_true)=**+0.54** (vs ~0
      untrained) → eye reads direction. Vision-necessary: sighted choice **0.67** vs ablated **0.48**
      (chance) → blind FAILS. Behavioral heading↔real corr +0.174 sighted vs −0.065 blind (modest,
      below the 0.30 full-S1 gate → strengthen with 1M). Caveat: MAE 59°, signed bias −22° = likely
      body-vs-head-frame artifact (cancels in the S2 pre/post recalibration difference).
- [x] render (stage1_ground_off0.mp4) watched (Gemini): crude rolling, approaches balls, cameras intact.
- [x] theory-monitor verdict on 250K validation: **PASSED, advance justified, no fatal problem.**
      Independently found MORE support: sign-concordance 66.7% sighted vs 43.3% blind (cleaner than the
      Pearson +0.174, which is dragged down by timeout-wander outliers → use a robust circular stat on
      1M); and the BLIND policy has a strong rightward motor bias (82% of blind headings +) that vision
      OVERRIDES (sighted back to balanced 42%) — independent non-choice evidence the eye steers. Two asks:
      (a) confirm bearing corr is per-timestep [CONFIRMED: 1364 per-step samples, not per-episode];
      (b) **fix theta_true to head/camera frame before S2** (rejected my "it cancels" argument: head yaws,
      so the offset won't cancel if head-scan differs under the lens).
- [x] FRAME FIX applied: `_ball1_ego_bearing` now computed in the HEAD body frame (camera is on body
      "head", id 13), not torso/root. Verified head vs body bearings differ 40–60° by head pose → the fix
      matters. Stopped the body-frame 1M (PID 54196).
- [~] **HEAD-FRAME S1 (1M) RELAUNCHED** from ext_s0 (fresh eye, learns head-frame target), PID 59317,
      run-tag stage1_headframe_s0. Early-checking readout at 50K ckpt (expect signed bias −22°→~0) before
      trusting the full run. GATE to S2: heading↔real corr ≥0.30 (report a robust circular stat too).
## STAGE-1 GATE RESOLVED (2026-07-12) — head-frame fix HURT behavior; reverted to body/action frame
The head-frame 1M run FAILED the gate (behavioral steering within noise). A matched-step A/B
(monitor-verified with z-tests) isolated the cause: **the head-frame target itself hurt behavioral
steering**, NOT step-count drift.
- body-frame 250K: sighted-vs-blind steering 67% vs 43% (z≈2.65, REAL); vision-necessary choice 0.67
  vs 0.48 (z≈2.03); blind at chance.
- head-frame ~200K (matched): 53% vs 50% (z≈0.36 — NO detectable effect); choice 0.55 vs 0.47.
- head-frame 1M: 65% vs 54% (z≈1.21, ns); blind rose to 0.55 (winnability drifted with extra steps).
PRINCIPLE (Taylor interpenetration): the eye's signal must be in the frame the policy ACTS in
(body), not the frame the pixels arrive in (head). Head frame = cleaner readout (MAE 40 vs 59) but
the policy reads the CNN latent and steers the BODY, so a head-relative signal needs a head-pose
composition it doesn't learn → steering collapses. Body frame = blurrier readout but action-aligned →
used directly.
RESOLUTION: reverted `_ball1_ego_bearing` to body/root frame. **The body-frame 250K checkpoint
(stage1_ground_smoke_s0_best) is the VALIDATED Stage-1 base** (significant steering + vision-necessary
+ blind-at-chance, by the robust sign-concordance/z metrics — the raw 0.30 circ_corr gate was too
strict; use the robust metric). Head-frame detour closed.
- GATE METRIC going forward: robust sign-concordance sighted-vs-blind (z-test) + vision-necessity, NOT
  raw circular corr ≥0.30.
- DEFERRED upgrade — head-pose into the VISUAL pathway (lit-scout verified 2026-07-12, cross-disciplinary):
  the transform "retinal + eye/head position -> body/spatiotopic location" is TEXTBOOK sensorimotor
  neuroscience (**gain fields**: Andersen & Mountcastle 1983; Zipser & Andersen 1988 = canonical net demo;
  **basis functions**: Pouget & Sejnowski 1997; Salinas & Abbott 2001). Reaching needs an effector/body-frame
  target (Batista 1999; Cohen & Andersen 2002; Buneo 2002). RL confirms action-frame reps help control
  (VICES, Martin-Martin 2019). -> FRAME OUR FINDING AS APPLY/CONFIRM, cite these; the modest genuine bit is
  the *measured* double-dissociation (accurate in head-frame, usable in body-frame, head-pose the bridge)
  via the decode-probe. TWO PITFALLS if we build the upgrade:
  (1) the transform ADDS noise — head-pose error corrupts the body-frame estimate (Sober & Sabes 2005); our
      body-frame blur is EXPECTED, not a bug -> reliability-WEIGHT the two frames, don't commit to one.
  (2) gain fields are MULTIPLICATIVE -> use FiLM-style gating of the visual latent by head-pose, NOT plain
      concatenation (a linear cat may not build the product term).
- Winnability: fine at ~250K (blind at chance); only drifts up with over-training → keep S1 ≤ ~250-400K.

- [ ] on S1 base (body-frame 250K) → **Stage 2**: continue under +30 lens FROM stage1_ground_smoke_s0_best;
      gate = heading↔real recovers under lens AND signed bias shifts ~−30° (theta_vis recalibrates),
      measured CONTROLLING for head pose. Build the embodiment cue first. Render + monitor each.

## LOCAL-BOOK-AGENT SWEEP of Taylor for prism/spectacles (2026-07-12) — what we'd MISSED
Full-corpus sweep (validated the new local-book-agent workflow). Beyond the Ch.9 items already logged:
- **Experiment III (prism-in-contact-lens):** adaptation happens with NO reaching — from SCANNING alone,
  IF the eye movements themselves are put in error. SHORT corrective movements adapt fast (line straight
  in ~20 s); long sweeps FAIL. Bit-by-bit, not global (bookcase bottom straightens, top still curved).
  Clean negative aftereffect, dissociable between the two eyes. → train on small corrective steps to near goals.
- **Taylor PROPOSES ~our task (9.22):** "hold a tray with a ball… tilt it to make the ball roll" — invented
  to INJECT a corrective DRIVE into a subsystem that has none. Principle: a shift is corrected ONLY where
  errors carry a consequence the agent is driven to minimize. → mis-reaching must have a cost in every
  region we want recalibrated (ties to winnability/liveness).
- **Head-first staged order WITH rationale (5.2–5.11):** head→trunk→arms→legs→stand→walk, each requires the
  prior stable; reason = multistable speed argument (adapt ONE subsystem at a time, FREEZE the rest — "head
  held rigidly during first efforts to stand"). Papert staged → true perception day 8 vs >2 wks for
  "be maximally active" Innsbruck. → theoretical backing for the staged curriculum + freezing.
- **Positive/negative feedback sign-flip (5.2):** a visual shift can FLIP a control loop's sign, making the
  naive controller destabilizing until relearned. Explains WHY AB dropped vision (under the shift vision
  became positive/harmful feedback → abandon it); recalibration = restoring negative feedback. Check our
  setup doesn't trap AB in a positive-feedback loop it must survive.
- **Timescale targets:** Papert day 8; Exp II 13 days; Exp III ~20 s to straighten a line; INTERMITTENT wear
  eliminates the aftereffect AND adapts faster than continuous.
- **CAVEAT 1 — a rigid self-anchored frame can BLOCK adaptation:** one subject wore reversing specs 71 DAYS
  and never adapted, anchoring to an invariant head-referenced frame. Flag on our head-frame target: it's
  right for removing yaw noise, but must still provide genuine error under the lens (it does — contact-
  confirmed true bearing still errs when pixels shift) and not become an invariant the system hides behind.
- **CAVEAT 2 — Taylor's cleanest recalibration data are SLOPE/SHAPE, not lateral POSITION** (our exact
  transform). He avoided pure lateral-displacement responses (walked defensively) and predicts little change
  in perceived lateral position. Mechanism support is strong; quantitative design-target support for a
  sideways-position shift is weaker. Weigh before leaning on Taylor numerically for our transform.
- Reusable principles surfaced: response-specificity, interpenetration, multistable subsystems walled by
  inactive part-functions, functional equivalence classes, (cue,1)/(cue,0) parameter, goal-gradient/START,
  negative-aftereffect=recalibration, feedback sign-flip, drive-prerequisite-for-correction.

## TAYLOR CH.9 REFRAME (2026-07-12, after reading the source — David directed) — changes the plan
Read Taylor "Behavioral Basis of Perception" Ch 9 (Experiments I reversing / II wedge-prism). Key points
that DIRECTLY reshape our design:
1. **The cue is the linchpin, and Taylor formalizes it.** Intermittent wearing works because s' (glasses on)
   carries "constant stimuli such as the pressure of the spectacles on the nose, narrowing of the visual
   field... an afferent function having only two values, 1 and 0." State = (s',1) with / (s,0) without. That
   bit lets the subject hold TWO mappings and switch with NO disruption, NO aftereffect (Papert biked,
   removed+replaced glasses mid-ride, no wobble). → my "no cue needed" was WRONG; the cue is the mechanism.
2. **Adaptation is RESPONSE-SPECIFIC (interpenetration).** Exp II: ground he WALKED on leveled; distant
   objects he only LOOKED at stayed distorted to the end ("stability for objects within reach... undiminished
   instability beyond that range"). Vision realigns ONLY through the specific corrective action. → VALIDATES
   contact-gating; predicts AB's realignment is specific to the ball-approach it practices, not global.
3. **Head-centered frame of reference** (8.11 Innsbruck subject) → independently vindicates the head-frame fix.
4. **Start the response correctly** (Schermann tapped the hand the instant it started wrong; roundabout paths
   that still reach the goal get reinforced) → weight the START of the approach, not just contact.

DESIGN CHANGES (implement before Stage 2):
- [ ] Add a LENS-STATE CUE to AB's obs — but it MUST be STATE/EMBODIMENT-grounded, NOT an abstract bit.
      **Literature-scout correction (2026-07-12):** "dual adaptation" is the correct standard term (Welch,
      Bridgeman, Anand & Browman 1993, Percept&Psychophys — origin; Lee & Schweighofer 2009; Kim et al.
      2021). BUT "requires a reliable cue" was OVERSTATED: the cue must be TASK/STATE/MOVEMENT-relevant to
      drive IMPLICIT dual adaptation — ARBITRARY cues (color, shape — and by analogy a bare on/off flag)
      largely FAIL / work only via explicit strategy (Woolley et al. 2015; Kim et al. 2021). Two mappings can
      even be held with NO explicit cue via fast/slow timescales (Smith et al. 2006). → So DON'T append an
      abstract "lens=1/0" bit (that's the color-cue failure mode). Model Taylor's actual cue: the "glasses"
      NARROW AB's field of view (and/or a proprioceptive/postural signal) — an embodiment cue, the kind that
      works. Taylor (1962) is NOT cited by the modern dual-adaptation lineage; Kravitz & Yaffe (1972, tone
      cue) is the historical bridge — and notably they used an arbitrary tone (the weaker kind).
- [ ] Add an IN-VIEW GATE to the mismatch loss (train the eye only on states where the ball is in the camera
      FOV / acted-upon) — technically fixes the head-frame out-of-view noise AND matches Taylor's response-
      specificity. (Head-frame S1 @50K shows eye tracks direction +0.586 but signed bias −41.7° from out-of-view
      contamination.)
- [ ] Reframe **Stage 3 = Taylor's Experiment I**: intermittent wearing WITH the cue → perceive correctly with
      AND without, no aftereffect. NOT "alternate blindly" (which is ill-posed without the cue).

## Stage-3 PREREQUISITE (from David, 2026-07-12) — AB needs a CUE to tell lens-on from lens-off
David's Q: "how does AB differentiate wearing the prism glasses vs not?" Current setup: NO cue — the
lens just shifts the pixels; nothing flags lens state. This is CORRECT and human-faithful for S1/S2 +
the basic after-miss (you don't need to know the lens is on; the after-miss IS the failure to instantly
differentiate — the recalibration persisting into the no-lens world). BUT Stage 3 (hold BOTH mappings,
alternating) is **dual adaptation**, which REQUIRES a reliable contextual cue signalling which mapping
is active (feeling/seeing the glasses in humans); without one, AB can only average the two → blurs.
→ Stage 3 is ill-posed until we give AB a differentiating cue (proprioceptive flag / visible lens edge /
visual context). What the cue should be = a real design decision, model it on what humans actually use.
Do NOT run S3 as "alternate with no cue" — that measures our setup's impossibility, not AB's ability.
  SUCCESS: corr climbing above blind, R² > ~0.2, mismatch trending down, ablated < full → greenlight S1 full (1M).
  FAIL: corr ~0 / R² flat → eye can't read 32px; fall back to AB_CAM_RES=64 or a distil pre-pass.
        Blind scores = full → task not vision-necessary; harden (wider cone / shorter horizon) first.
Decision: grounded eye built IN-PLACE during S1 (default); distil pre-pass is the fallback if it fails.

---

# Overnight run — 2026-07-08 → 07-09 (approved: "do #1-5 in order, keep running all night")

Chain of the five FINDINGS next-steps. Sequential (single MPS device → no parallel training).

- [x] **1. Aftereffect re-bin diagnostic** (no compute; `_rescore_aftereffect.py`).
      RESULT: s0 aftereffect real (near-phantom 78% / far 7%), ~uniform across bearing (leans global
      "subtract ~30°" bias), persistent (no washout 50–300K). s2 & frozen cells on disk show NO
      aftereffect → makes #2 the pivotal test.
- [ ] **2. Seed-replicate the aftereffect on s2.** Phase B: continue `decoy_v2_s2_best` under +30° prism,
      decoy task, 1M steps, seed 2 → `prism_adapt_s2_rep`. Phase C: prism-off aftereffect + −30 control
      + 300-ep symmetric baseline (eval_prism_decoy), then re-run `_rescore` for the new s2 cell.
- [ ] **3. Gentler adapter (clean Phase-B curve) on ext_s0.** Continue `decoy_v2_ext_s0_best` under +30°,
      `--ent-coef 0.003`, `--steps 2000000` → `prism_adapt_ext_s0_gentle`. (Note: `--lr` is ignored on the
      continue path; entropy + steps are the working knobs.)
- [ ] **4. Shape × displacement (eval-only).** Add additive `--offset` passthrough to eval_decoy_shape.py
      (safe/additive), then run shape battery at offset 30/45 on ext_s0.
- [ ] **5. Higher-res camera (64×64), from scratch.** CNN adapts automatically; must be from-scratch.
      Smoke-test first; apply resolution change revertibly so 32px models stay usable. Launch last.

## Progress log
- ~22:30 — Step 1 done (result above).
- ~22:35 — Step 2 Phase-B training LAUNCHED (~2h15m).
- ~02:17 — Step 2 DONE. **Aftereffect REPLICATES on s2: near-far +67 (s0 ref +71), pre-adapt baseline
  +12 symmetric.** Two-seed result. FINDINGS entry written. Frozen-enc cell (near9/far56) shows freezing
  abolishes it → recalibration needs a plastic encoder.
- ~02:20 — Step 3 (gentle adapter, 2M) LAUNCHED (~7.9h @ ~70fps, slower than est).
- ~02:25 — Step 4 code (--offset ghost-shape passthrough) added + smoke-verified during the wait.
- ~09:57 — Step 3 DONE. Curve evals + aftereffect run (~10:30). **NEGATIVE/informative:** gentler adapter
  (ent 0.003, 2M) did NOT clean the curve — it ABOLISHED recalibration. Under-prism curve drifts to chance
  (60.9→48%), prism-off aftereffect near-far −16 (wrong sign), discrimination eroded 78→48%. Bounds the
  recipe (flagship ent 0.01/~1M is the recalibrating regime) and shows the aftereffect TRACKS recalibration
  (present s0/s2, absent frozen-enc + over-gentle). FINDINGS entry written.
- ~10:35 — Step 4 shape×displacement battery. DONE. **Follow-the-ghost is SHAPE-INVARIANT** (off30 63/65/59%,
  off45 56/58/57%, all within CI; ablated floor 52.7% = chance). FINDINGS entry written. Closes the
  offset-0-only caveat.
- ~11:xx — GROUNDING_LLMS.md written (user requested); GLOSSARY + FLASHCARDS updated with 4 new terms.
- ~11:xx — Step 5 setup: made CAM resolution a REVERSIBLE env-var override (`AB_CAM_RES`, default 32 →
  existing 32px models untouched). 64px env verified to build (VISION_DIM 24576).
- ~11:xx — 64px smoke: **fps 121 — SAME as 32px** (bottleneck is physics/PPO, not conv). So full 2M
  64px run ≈ 4.5h, not the feared 10h+. Plumbing validated (warmstart loads, reward climbs, CNN adapts).
- ~12:xx — Step 5 FULL 64px run LAUNCHED (task bspb6m33d, decoy_64px_s0, curriculum, 2M).
- 20:49 — Step 5 COMPLETED all 2M steps (final saved). CORRECTION: an earlier turn wrongly read it as
  "interrupted at ~287K" — it actually kept running to 2M. Took ~9.9h (fps throttled 121→56, not ~4.5h).
- Step 5 EVAL done: **INCONCLUSIVE.** 64px control choice only 56% (< 32px seeds' 63–78%) → too weak a
  discriminator to run a clean shape-vs-color test. Box holds (62–65%), capsule drops to chance (46%,
  vs 32px 74–76%) — a possible shape effect but within noise. Needs a stronger 64px discriminator first.

## Review — overnight #1–5 chain (2026-07-09)
- **1 Aftereffect re-bin:** s0 aftereffect real, ~global bias, persistent. (done, no compute)
- **2 s2 replication:** ✅ HEADLINE — negative aftereffect REPLICATES on s2 (+67 vs s0 +71). Two-seed.
- **3 Gentler adapter:** ❌ informative null — over-gentling ABOLISHED recalibration (no aftereffect,
  discrimination eroded). Aftereffect tracks whether recalibration happened.
- **4 Shape × displacement:** ✅ follow-the-ghost is SHAPE-INVARIANT (box=sphere at every offset).
- **5 Higher-res 64px:** ⚠️ INCONCLUSIVE — discriminator too weak (56%) to test shape-vs-color.
- Net: the flagship recalibration result is now two-seed and better-characterized; one clean new
  invariance result; one useful null; one open follow-up (train a strong 64px discriminator).
- Code: `AB_CAM_RES` env-var (reversible res override), `eval_decoy_shape --offset` (shape×displacement).
- Separately: launched the LLM-grounding program (GROUNDING_LLMS.md + grounding/ probe) and set two
  standing rules (agent visibility; literature-scout prior-art check).
- All work uncommitted but safe on disk; not committed (awaiting go-ahead).

## Grounding program (GROUNDING_LLMS.md) — separate track, user-directed 2026-07-09
- Doc written; decisions locked: governor-first-then-see, lean foundation (both can work, foundation
  ceiling conditional on grounding breadth); probe is a multi-EMBEDDING ladder (RSA primary), not one model.
- Probe (a) AB-side DONE: `grounding/probe_ab_extract.py` → `results/grounding/ab_latents.npz`
  (N=400, latent128 + conv2592 + true bearing/dist). NEXT: LLM ladder + RSA alignment
  (all-mpnet + OpenAI embed + Qwen2.5-7B text vs Qwen2.5-VL contrast); filter to in-view |theta|<68.

## Command reference (verified — all paths exist; run from repo root, prefix `PYTHONPATH=<repo>`)
**Step 2 Phase B (running):** `train_head_search --init-model results/decoy_v2_s2_best/best_model.zip
  --decoy --prism-offset 30 --xml crawler/mimo_crawler_pos_wide_prism.xml --steps 1000000 --seed 2
  --run-tag prism_adapt_s2_rep`  → outputs `results/prism_adapt_s2_rep_final.zip`, `..._best/`.
**Step 2 Phase C (after B):**
  - `eval_prism_decoy --model results/prism_adapt_s2_rep_final.zip --prism-offset 0  --eval-eps 100 --run-tag aftereffect_s2_rep_off0`
  - `eval_prism_decoy --model results/prism_adapt_s2_rep_final.zip --prism-offset -30 --eval-eps 100 --run-tag aftereffect_s2_rep_offm30`
  - `eval_prism_decoy --model results/decoy_v2_s2_best/best_model.zip --prism-offset 0 --eval-eps 300 --run-tag sym_s2_rep_300` (pre-adapt symmetric baseline)
  - then re-run `_rescore_aftereffect.py` (add the new s2_rep cell) to check near≫far.
**Step 3 (gentler adapter):** `train_head_search --init-model results/decoy_v2_ext_s0_best/best_model.zip
  --decoy --prism-offset 30 --xml crawler/mimo_crawler_pos_wide_prism.xml --steps 2000000 --ent-coef 0.003
  --seed 0 --run-tag prism_adapt_ext_s0_gentle`  (~4h30m). Then adaptation-curve evals like Phase C above.
**Step 4 (shape×disp) — CODE DONE + smoke-verified (overhead frame shows red-BOX ghost).** eval_decoy_shape.py
  now takes `--offset`: at offset 0 overrides the REAL balls (as before); at offset≠0 overrides the seen
  GHOST shapes and keeps reals as spheres (contact physics held constant). Battery to run on
  `decoy_v2_ext_s0_best` (200 eps each), reusing eval_prism_decoy's sphere-ghost baselines (off30=62%, off45=53%):
    - off30: `--red box --blue sphere`, `--red sphere --blue box`, `--red box --blue box`
    - off45: same three
    - ablated floor check: one `--offset 30 --red box --blue sphere --ablate`
  Shape-invariant follow-the-ghost ⇒ box-ghost choice_vs_true matches sphere-ghost at each offset.
**Step 5 (64px):** set `CAM_H=CAM_W=64` in mimo_crawler_env.py (CNN adapts via runtime dummy-forward);
  MUST be from-scratch (no 32px ckpt loads). Smoke-test ~50K first. Apply revertibly (keep 32px usable).
  `train_head_search --decoy --curriculum --steps 2000000 --seed 0 --run-tag decoy_64px_s0`.

---

# Vision-as-reinforcement plan (A → B → C) — proposed 2026-07-04, AWAITING SIGN-OFF

## Goal
Demonstrate that what proprioception learned (the crawl gait + posture) is **reinforced, not
destroyed**, when vision is introduced — vision supplies the ball bearing that proprio lacks,
without degrading the motor substrate. Staged as three residual/adapter designs of increasing
ambition, all behind one shared camera fix.

## Measurement contract (applies to every stage)
Report competence as **substrate vs capability**, not one number:
- **Substrate (must be PRESERVED):** CoM displacement, mean speed, tip-rate, gait smoothness.
- **Capability (must CLIMB):** contact rate + mean-dist-toward-ball (bearing-dependent).
- **Reinforced** = substrate holds/improves AND capability climbs (12–32% → toward 57.5%) AND
  vision-ablation degrades *gracefully* to ≥ proprio baseline (never below).
- **Destroyed** = substrate falls when vision is added, or removing vision leaves policy < baseline.

## Prerequisite (GATES ALL THREE — shared, do once)
- [ ] P0. Fix head-cam FOV so the ball is a clear centred blob across the spawn cone
  (2026-07-02 preflight: only ±15° visible, need ±45°). Widen `left_eye`/`right_eye` fovy in
  `mimo_crawler_pos_wide.xml`; re-run the visibility preflight until it passes. No vision run
  launches until this passes (winnability rule).
- [ ] P1. Add substrate/capability logging to `eval_phase_v.py` (gait metrics beside contacts).

## Stage A — vision fills the slot (distilled) [GATE run, cheapest]
- [ ] A1. Roll out frozen `crawl_ppo_2M_best` in the vision env (target_obs + vision both ON;
  env already emits true bearing beside pixels) to collect (pixels → true ego-bearing) pairs.
- [ ] A2. Train CNN `g(pixels)→3-vector` by regression to the true bearing (reuse StereoCrawlerCNN).
- [ ] A3. Eval: inject `g(pixels)` into slot [69:72] (raw units → through the saved
  `vec_normalize.pkl`), motor policy frozen. Report decode-R² and substrate/capability.
- **Success:** decode-R² > 0.30 AND contacts recover toward 57.5% with substrate unchanged.
- **Gate:** weak R² here ⇒ camera/representation problem — STOP, fix before B/C.

## Stage B — vision fills the slot, grown by reward [only if A passes]
- [ ] B1. Same frozen motor policy; make `g` the only trainable module; train by PPO reward
  (gradient reward → frozen actor → g). Warm-start `g` from Stage A's weights. Let value
  head/log-std train; keep actor trunk frozen.
- [ ] B2. Eval: substrate/capability + vision-ablation graceful-degradation check.
- **Success:** capability climbs under reward alone (no bearing supervision) with substrate intact.

## Stage C — additive action residual on a blind base [own mini-project]
- [ ] C0. PREREQ: train a competent blind/proprio-only crawler (no target_obs) to freeze as base.
- [ ] C1. PPO `ActorCriticPolicy` subclass: action = frozen_blind_action + zero-init CNN residual
  (translate `dialogue_policy.py`'s dual-stream pattern from SAC → PPO).
- [ ] C2. Train residual by reward; eval substrate/capability + ablation.
- **Success:** vision-added steering lifts a policy that never had a bearing, substrate preserved.

## Notes / known integration details
- `StereoCrawlerCNN` infers `[proprio|memory|pixels]` and ignores a target_obs slot — fine for
  A/B (true slot not routed through it); needs a 3-line tweak for any integrated bearing+vision policy.
- If A/B/C land, the payoff experiment is the prism-ghost aftereffect (`PRISM_GHOST_PROPOSAL.md`).

---

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

---

## Rung 0 (2026-06-15): Can AB make ONE directed reach to a SENSED fixed target?

### Why
This session established AB is *blind to the ball* on proprio (obs = own body state +
2 touch bits; no target position) AND the cart runs set `--approach-reward-scale 0.0`,
so the hand-based approach reward was OFF. So `both_HAND=0` (never reaches with a hand)
is doubly explained: no perception of the target, no reward for moving a hand toward it.
Rung 0 = the missing bottom rung of David's stable-world curriculum: prove AB can learn a
directed HAND reach when it can sense the target. Decided scope: **reach-to** (seated AB,
arm extends), NOT travel-to (no locomotion).

### Design: a matched pair (the contrast IS the result)
- TEST   : target-in-obs ON  -> predict hand-touch rate climbs over training
- CONTROL: target-in-obs OFF -> predict hand-touch stays ~0 (reproduces the blindness)
If TEST succeeds and CONTROL fails, perception enables reaching. Clean, decisive.

### Setup (reuse cart env, seated)
- `--cart-speed 0` : static base; AB sits and reaches with an arm (cart kinematic, no walk)
- single fixed target (one `--fixed-ball-positions` entry -> ball2 inactive)
- target randomized in a small box each episode (so CONTROL can't memorize a constant)
- placed within hand reach but off body-center, so a hand (not a body sweep) is required
- `--approach-reward-scale 2.0` : already hand-based (_ball_dist = nearest-hand->ball)
- velocity bonus on (anti-freeze), hunger on (deadline)
- success/score = HAND touch (`hand_touched_ball*`)

### Code changes (small, opt-in, default-off = existing behavior unchanged)
- [ ] env: `target_obs` flag -> append egocentric target vector to proprio (mirror memory_obs)
- [ ] env: `hand_success` flag -> contact reward + termination use hand-touch (default off)
- [ ] env: pin the target so a bump can't move it (weld ball, or static cube geom)
- [ ] train_crawler: wire `--target-obs`, `--hand-success`
- [ ] eval: hand-touch rate vs training checkpoint (developmental curve) + held-out positions

### Pre-flight (training-run rules)
- [ ] render ONE episode from an UNTRAINED model with this config: AB seated upright,
      target within a hand's reach, physics/cameras sane. Fix before training.

### Decisions made (2026-06-15)
- A) sense = hand->target error vector, BOTH hands (6 dims). [DONE]
- B) pinned ball (pin_targets). [DONE]
- C) single target. [DONE]
- difficulty = short horizon + wide target; metric = steps-to-touch (+ hit-rate). [DONE]

### Code changes — DONE + smoke-tested
- [x] env: target_obs (6-dim hand->target both hands), hand_success, pin_targets
- [x] train_crawler: --target-obs / --hand-success / --pin-targets wired
- [x] smoke test: obs=75, pin drift 0.00000 m, hand_success terminates, no crash

### Pre-flight findings (the de-risking that mattered)
- Naive single-touch is TRIVIAL: random policy hits 100% in ~75 steps (workspace small
  enough that flailing sweeps it). Fix: wide target + short horizon, score steps-to-touch.
- Reach envelope (random rollout): x in [-0.25,0.21], y in [0.09,0.40], reach<=0.40 m.
- Targets at y<0.20 spawn under AB's body (occluded/trivial) -> push workspace forward.

### VALIDATED final config
- cart-speed 0 (seated); fixed-ball-positions "0.0,0.30"; random-ball-box 0.18,0.08
  (target x in [-0.18,0.18], y in [0.22,0.38] -- wide, in front of body, reachable)
- max-steps 50 (short horizon); hand-success; pin-targets; approach-reward-scale 2.0;
  velocity-bonus-scale 0.05; hip-actuation off
- random baseline at this config: 50% hit / ~31 steps-to-touch -> headroom for TEST
- PAIR: TEST = --target-obs ; CONTROL = (omit --target-obs), else identical, seed-matched

### TODO
- [ ] write steps-to-touch eval (mean steps-to-hand-touch + hit-rate@50, vs checkpoint)
- [ ] launch TEST + CONTROL (250K DroQ, checkpoints @10K to plot the learning curve)
- [ ] result: does TEST reach fast/reliably while CONTROL stays near baseline?

### RUN LOG
- rung0_TEST_targetobs (v1: UTD=4, ent 0.5->0.2 anneal, 250K) -> COLLAPSED.
  Best 36% hit@50 (< 50% random); 100K-250K det. eval = 0%; ep_rew fell 200->~5,
  ep_len->50 (stillness). Kinematics: hand DOES move (0.1-0.38 m path) toward target,
  ends 0.086-0.19 m away -> perception->approach partly works, but imprecise + collapsed.
  Verdict: optimization-stability failure (DroQ UTD=4 + entropy anneal), NOT a capability
  finding. Reward design fine (CONTACT_REWARD=200 dominates).
- rung0_TEST_v2_stable (UTD=1, ent auto, 150K) -> RUNNING. Anti-collapse retry.

### RUN LOG (cont.) — embodiment + the collapse finding
- rung0_h15_TEST vs rung0_h15_CONTROL (horizon 15): IDENTICAL (both 0% touch / 8% range
  / ~7.7 steps). Target signal INERT even when the task requires it. But horizon 15 also
  near-infeasible (body too slow), so partly a feasibility wall.
- rung0_motor_fixed (passive hands, ONE fixed target, pure motor): best 53% touch (random 40%),
  median final 0.077 m. Even reaching ONE fixed spot is barely learnable; floppy hand caps it.
- EMBODIMENT FINDING: AB's wrists (hand1/2/3) + fingers are PASSIVE (no actuators). Arm IS
  actuated (shoulder x3 + elbow per side). So: a reaching arm ending in a dead floppy paddle;
  cannot orient hand or grasp. (Confirms David's "no functioning hands -> can't learn".)
- ADDED actuated hands: new body mimo_crawler_cart_hands.xml (+8 motors, action 25->33,
  proprio 69->85), --actuate-hands flag. Original body + all checkpoints untouched.
- rung0_hands_motor_fixed (actuated, fixed target): best 0% touch (WORSE than passive 53%;
  actuated random 60%). rung0_hands_TEST (actuated + perception + wide): best 7% touch / 18%
  range (passive TEST was 42%; actuated random wide 40%). ACTUATING HANDS MADE IT WORSE.
- ROOT CAUSE (robust across ALL ~9 runs): SAC COLLAPSES every time. ep_rew_mean 200 -> ~7-10,
  ep_len -> full horizon, regardless of hands/perception/horizon/entropy/UTD. More DOF
  (actuated hands) collapses HARDER. The bottleneck is the TRAINING OPTIMISATION, not
  embodiment or perception. Prime suspect: reward magnitude (CONTACT_REWARD=200 sparse +
  hunger to -2000) destabilises the critic. Can't test the hands hypothesis until collapse
  is fixed.
- NEXT: fix the collapse first. Reduce CONTACT_REWARD (200 -> ~5) + hunger scale so the
  critic has a sane value range; retest passive AND actuated reach.

### COLLAPSE-MAGNITUDE TEST (refuted) + the deep invariant
- rung0_lowR_motor_fixed (passive, fixed target, CONTACT_REWARD 200->20, near-bonus 1.0):
  best 53% touch, median final 0.077 m -- IDENTICAL to contact=200. Reward MAGNITUDE is NOT
  the collapse cause. ep_rew still declined (22 -> ~4).
- DEEP INVARIANT across ~10 runs (hands/no-hands, perception/blind, horizon 12-50, ent
  0.5-anneal vs auto, UTD 1 vs 4, contact 20 vs 200): the reach CAPS at ~53% on a FIXED
  target (random 40%), median hand-to-target ~0.077 m, and training COLLAPSES every time.
  No knob moved it. Perception never helped; hand actuation made it worse.
- HONEST CONCLUSION: the bottleneck is NOT perception, NOT reward, NOT a single hyperparam.
  This MIMo body under SAC torque control cannot learn a reliable, stable reach even to one
  fixed spot. It is a foundational control/optimisation wall -- the same wall as the project's
  chronic stillness-collapse / can't-crawl history, now reproduced at the simplest task.
- STRATEGIC OPTIONS (for David): (a) different control scheme -- position/PD control instead
  of raw torque (likely the biggest lever); (b) simpler body / fewer DOF for the reach
  primitive; (c) different algorithm or scripted/imitation bootstrap; (d) step back and
  question whether this body+sim is the right substrate at all.

## 2026-07-13 (overnight) — reframed.docx assessment + prepared (NOT launched) 2nd-seed replication

### Done this session (analysis only, no compute burned)
- Assessed `documents/reframed.docx` at David's request. Verdict: direction right (already ours),
  but premise overstates results, and the 3 mechanisms are established prior art
  (SayCan/Grounded-Decoding, world-model-as-verifier, VLA/RT-2). Full memo:
  `documents/reframed_assessment.md`.
- theory-monitor: the doc's "theory is largely correct" CHALLENGES the record — true for proprio,
  negative-to-narrow for vision (R²=0.08-0.14 direction decode; ablation peaks at ecc=0).
- literature-scout: mechanisms are restatements; "weights as governor" doesn't type-check (fix =
  affordance-as-scalar, per SayCan). Thesis = Harnad/Barsalou, empirically Xu et al. 2025 (Nat Hum
  Behav). Folded citations into GROUNDING_LLMS.md appendix.
- Flagged GROUNDING_LLMS.md §3 as STALE (2026-07-09 draft, pre-07-12 correction): 2 of 5 asset
  pillars overstated (recalibration = mostly blind motor bias; "reference" = approach-red salience
  reflex). Added a dated ASSET-AUDIT banner at §3 pointing to the memo. Did NOT rewrite §3 (David's call).

### PREPARED, GATED — second Stage-2 seed (monitor's #1 next step). NOT launched (needs David greenlight).
Why not launched autonomously: a multi-hour run + a real design fork are David's call. But the
reproducibility blocker that gated this is now RESOLVED (2026-07-17, analysis only, no compute) —
see PARAM-PIN below.

## PARAM-PIN of the Stage-2 recipe (2026-07-17) — reproducibility gap closed on analysis alone
Recovered from `stage2_adapt_s0` checkpoints + trainer source; no compute spent.
- **PPO block, EXACT (read from the SB3 checkpoint `data` blob):** lr 3e-4, ent_coef 0.01,
  n_steps 1024, batch 512, n_epochs 10, gamma 0.99, gae_lambda 0.95, vf_coef 0.5, max_grad_norm 0.5.
- **Env, PINNED (obs-dim 6213 uniquely identifies it):** `mimo_crawler_pos_wide_prism.xml`,
  --prism-offset 30, --decoy, --spawn-cone-deg 136, --spawn-radius 0.70 0.80, --max-steps 1000
  (trainer defaults; match the surrounding runs). Confirm 6213 against the prism XML if paranoid.
- **The `--mismatch-coef` that the todo flagged `<UNCONFIRMED>` is NEARLY A NON-PARAMETER.** The aux
  optimizer is `torch.optim.Adam(fe.parameters(), lr=aux_lr)` and the loss enters as `coef * MSE` —
  a constant scale on the loss fed through Adam. Adam's update = grad / sqrt(second-moment), so a
  uniform `coef` multiplier cancels out of the step for any value above the eps floor. => for any sane
  coef (~0.05-10) the trained eye is ESSENTIALLY IDENTICAL; the real plasticity knob is aux-lr (3e-4,
  recovered). Only needed coef > 0 (signal on). Write coef=1.0 and document it; the value is immaterial.
- **Built-in reproduction check (makes the coef question moot regardless):** when the same-base gate
  runs, its Stage-1 eye readout should land near the validated Stage-1 numbers (mismatch/aux_loss
  ~0.078, bearing circ_corr ~0.54). If it does, the recipe reproduced whatever the original coef was.

## PLAN (Claude's rec, 2026-07-17; David corrected the seed logic — recorded below)
Sequence: declare recipe (coef 1.0) -> cheap eval power-up -> same-base FAST-FAIL gate -> fresh chain
-> 2-3 more fresh chains before anything builds on it.
- **SEED LOGIC (David's correction, the governing reading):** the same-base fresh-PPO-seed run is a
  FAST-FAIL GATE, not a green light. The eye is PART of the mechanism claim ("embodied vision produces
  prism adaptation"), so two Stage-2 seeds off ONE base are correlated through the shared eye — NOT
  independent N=2, and a reviewer would say so. Therefore: a PASS on the same-base gate tells you nothing
  about generality; you proceed to the fresh chain REGARDLESS. A FAIL kills the result cheaply in one run
  (adaptation isn't even robust on its own eye). Even the fresh chain is N=1 — budget 2-3 fresh chains
  eventually to turn "replicated" into a distribution; do NOT let one fresh success re-become the thing
  the LLM-grounding work builds on.
- **Cheap eval power-up (no training, existing checkpoints):** re-run the 60° battery at ~400 eps/cell
  + add a 45° dose-response point. Decides whether the CHOICE endpoint is genuinely flat or just
  underpowered (07-13 needed ~8x eps for choice) -> sets which metric the seed gate is scored on.

Reconstructed command (same-base fast-fail gate — NEW seed, identical recipe):
  python crawler/train_head_search.py --init-model results/stage1_ground_smoke_s0_best/best_model.zip \
    --xml crawler/mimo_crawler_pos_wide_prism.xml --prism-offset 30 --decoy \
    --mismatch-coef 1.0 --aux-lr 3e-4 --spawn-cone-deg 136 --spawn-radius 0.70 0.80 \
    --max-steps 1000 --ent-coef 0.01 --steps 300000 --seed 1 --run-tag stage2_adapt_s1
Then eval at 60° with `eval_prism_decoy.py` (base vs stage2_s1 sighted vs stage2_s1 ablated), render,
theory-monitor. Gate = ghost-fraction below chance under sighted + reverting to chance when ablated
(the 07-13 s0 signature). Per David: PASS -> proceed to fresh chain anyway; FAIL -> result is fragile,
stop and diagnose. STILL needs David's greenlight to spend compute.

## PRE-COMPUTE PANEL — RAN 2026-07-17 (strategist + lit-scout + local-book, in parallel). Verdicts:
- **experiment-strategist:** confirmed the gate command; THREE fixes — (1) pin `--n-envs 8` (default 8;
  a silent bump to 16 changes rollout/minibatch = reproduction confound); (2) watch first log line for
  `missing=0` (eye inherited) — `missing=6` means the eye did NOT load, kill the run; (3) coef=1.0 fine.
  Fresh chain = fresh Stage-1 from `decoy_v2_ext_s0_best` (seed 2, offset 0, 250K; expect missing=6 =
  correct fresh bearing head) -> Stage-1 gate (robust sign-concordance + vision-necessity, NOT circ_corr
  0.30) -> fresh Stage-2 (seed 2, 300K). HEADING (`displayed_red_frac`) is the confirmatory endpoint,
  CHOICE is exploratory/underpowered — PRE-REGISTER this. Ablated cell = internal chance ref (non-neg).
  BIGGEST HOLE: even the "fresh chain" shares the `decoy_v2_ext_s0` LOCOMOTION base -> not fully
  independent ("N=1 in disguise"); true independence needs a from-scratch base (`ext_s1`, big compute).
  ~9-12h wall-clock for gate+fresh-chain, sequential single-MPS.
- **literature-scout:** ML replication bar for this claim is ~8-20 independent seeds/arm with interval
  stats (rliable IQM + bootstrap CIs; AdaStop for early-stop), NOT 2-3 (Colas 2018; Henderson 2018;
  Agarwal 2021). Human-prism analogue is looser (~5-10 subjects) BUT only because within-subject design
  is strong + effect large vs spread. Full stack (pixels->mismatch-grounded encoder->RL->aftereffect)
  appears UNCLAIMED (novelty candidate; cite Feulner 2024 as emergence existence-proof; chase a
  "rotated-camera robot prism" lead before claiming first). KEY UPGRADE > raw seeds: lens-OFF AFTEREFFECT
  PROBE (field-standard positive endpoint) + ERROR-CLAMP (ghost offset independent of AB's aim, kills
  "just found a better policy"). Adopt implicit/explicit vocabulary (Taylor-Krakauer-Ivry 2014).
- **local-book-agent (Taylor):** tradition is N=1 WITHIN-subject dissociation, not between-subject
  averaging; core recalibration is a robust invariant across engaged subjects, route/rate vary by
  drive/training (= our winnability); the one failure case (71-day subject) was a READOUT artifact
  (adapted behaviorally, not in report) -> audit a null's readout before calling it failure. Gives NO
  quantitative seed number (that's the lit-scout's job).
- **DECISION FORK for David (post-eval):** WIDEN (more independent seed-chains toward 5-20 + interval
  stats) vs DEEPEN (build the lens-OFF aftereffect probe + error-clamp, run a couple independent chains
  through THAT). Claude's lean: deepen-then-widen (one clean aftereffect harder to dismiss than seeds
  3-through-20 of the lens-on test). DAVID CHOSE: "deepen then widen" (2026-07-17).

## DEEPEN ATTEMPT — lens-off aftereffect on stage2_adapt_s0 (2026-07-17, autonomous, eval-only). OUTCOME: NOT ESTABLISHED.
Ran the field-standard lens-off aftereffect probe two ways on the EXISTING adapted checkpoint (no training):
- **Representation route (bearing readout): UNINTERPRETABLE.** eval_bearing_readout instrument fails its
  own sanity check — validated base eye circ_corr +0.54 reads ~0 even after in-view filtering. Original
  readout settings never saved (reproducibility gap). Built eval_aftereffect_inview.py (in-view filter)
  but the base still doesn't reproduce -> instrument not trusted. Route abandoned.
- **Behavioral route (via trusted eval_prism_decoy @ offset 0): correctly-signed but NOT significant.**
  Full 2x2 {base,adapted}x{sighted,blind}, 300 eps, seed 0. Vision IS load-bearing (blinding swings net
  heading 31.8-38.5deg, z~15 -> refutes Story-B motor-habit). BUT the aftereffect (diff-in-diff, motor-drift
  controlled) = -6.7deg unpaired (z=-1.73, p=0.08) / -4.1deg PAIRED (z=-1.09, p=0.275). Paired = cleanest
  test (300/300 spawn-aligned), did NOT reach significance. ~half the naive raw -13.7deg shift was
  MOTOR-BIAS DRIFT between checkpoints, not vision.
- **Video (render rule):** ae_adapted_off0_sighted.mp4. Gemini="ragdoll/incidental" (known under-read);
  overridden by the z~15 ablation swing = vision-directed but crude belly-down gait.
- **METHODS LESSON (new, load-bearing):** a raw sighted heading shift between a base and a fine-tuned
  checkpoint CONFOUNDS visual recalibration with motor-bias drift from the extra training steps. The 2x2
  (add both blind cells) + difference-in-differences is MANDATORY to isolate the visual component. The
  raw number overstated the aftereffect ~2x.
- **FINDINGS entry:** 2026-07-17 "Lens-off aftereffect, full 2x2 control" + ADDENDUM (paired result +
  video + final verdict + cleaner-probe requirements). Theory-monitor verified 2x2 (no overclaim found).

## DEEPEN cont. (2026-07-17) — single-target no-feedback reach probe: INVALID INSTRUMENT.
Built crawler/eval_reach_aftereffect.py (single ball + early-window ballistic heading). Monitor caught +
code-confirmed: R²(true bearing, early heading) ~0 in ALL cells incl. SIGHTED -> the window=40 heading does
NOT measure vision-guided aim, it measures a target-independent initial crawl motion. My "+0.7deg = aims
true" headline was a SYMMETRY ARTIFACT -> RETRACTED. Root cause: this creature has NO ballistic-aim phase
(steers gradually over the approach; touch homes the contact) -> no early window to read an aftereffect from.
NET: the lens-off aftereffect is UNMEASURABLE with current tools (bearing-readout broken; early-window R²~0;
whole-episode homing-confounded), NOT shown absent. The decoy (-4.1, correctly-signed) and reach (+5.3,
wrong-signed) estimates don't even agree in sign -> effect at/below both instruments' noise floor.
FINDINGS entry: "2026-07-17 — Single-target no-feedback reach probe: INVALID INSTRUMENT".

## NEXT DEEPEN STEP = INSTRUMENT-BUILDING (not a seed, not widen):
Sweep the heading window; find a window where SIGHTED heading correlates with true bearing (R² high) AND
BLIND does not (validate the instrument measures vision-guided aim FIRST). Only then measure the aftereffect
(2x2 diff-in-diff, motor-drift controlled) in that VALIDATED window. Rule earned twice this session: never
trust an aim/aftereffect metric that hasn't passed the R²(sighted-high / blind-low) sanity check — the
bearing-readout and the early-window reach both failed exactly this.

## WIDEN — HELD (not launched). Gate driver staged at scratchpad/samebase_gate.sh (recipe pinned, --n-envs 8,
missing=0 guard). NOT launched: there is no VALID, significant aftereffect to replicate (both deepen
instruments were invalid/confounded). Resume widen only after a VALIDATED instrument shows a real effect.
All deepen work was eval-only; NO training compute spent this entire session.

## 2026-07-20 — SPATIAL-BEARING TEST (per CHECKPOINT_2026_07_20.md, agreed spec)
Question (plain English): can the walker steer to ONE ball by WHERE it is — i.e. does its vision
carry the ball's DIRECTION, not just its color? Phase V said direction was the thing vision failed at.
- [x] 1. Pre-experiment panel (process gate): experiment-strategist + literature-scout +
        local-book-agent in parallel; relay findings before training compute is spent.
- [x] 2. Env variant (minimal): `single_ball=True` flag on VisionSteerEnv — red ball at random
        bearing ±0.5 rad, 2.5–4 m, in view; blue decoy PARKED far ([50,-50], the gait-env fix);
        expose true gaze-relative bearing + turn cmd in `info` for the instrument.
- [x] 3. SMOKE-TEST THE ENVIRONMENT first (yesterday's lesson): N resets → red-in-view %,
        bearing spread, no fling/overlap; sanity render from an untrained model.
- [ ] 4. Train driver `vbear_s0`: transplanted frozen crawler eyes (--init-vision mildhead), frozen
        v10 gait, ~300K steps (same recipe as vsteer_v10).
- [ ] 5. New eval `eval_vsteer_bearing.py`: sighted vs FAIR blind (NOISE pixels, not zeros);
        contact%; INSTRUMENT CHECK R²(true bearing → turn/heading): sighted HIGH, blind LOW.
- [x] 6. Fix known caveat: make NOISE the standard blind in eval_vsteer_choice.py.
- [ ] 7. Render + WATCH episodes of the exact eval config (render-every-experiment rule).
- [ ] 8. theory-monitor independent verdict, then FINDINGS.md + THEORY_LOG.md entries (plain
        English, Taylor tie-in), glossary if new terms.
Expected per Phase V: this may FAIL (vision non-directional). If so, the identified fix is the
auxiliary "where's-the-ball" decode loss on the encoder — escalate to that, not to reward tweaks.

### Preflight results (2026-07-20) — the plan CHANGED before compute was spent
The agent panel + two measurements caught three things. Recorded because each would have made the
300K run uninterpretable:
1. **The frozen eyes DO carry direction — R²=0.996** (n=2000, held-out, ridge on the frozen conv
   trunk's 64-d latent -> sin of the true gaze bearing). This was the gate the strategist demanded,
   and the prior was pessimistic (a comparable crawler encoder measured lateral R²=0.010, chance).
   CONSEQUENCE: the direction information is already sitting in the latent. So if the trained driver
   fails to steer, that is a POLICY failure, not a representation failure — which is the exact fork
   the checkpoint said we needed to resolve, and it is now resolved IN ADVANCE.
2. **reach=0.75 m is UNWINNABLE — oracle scores 0%.** The strategist proposed tightening reach from
   1.0 to 0.75 to make the task harder. A scripted perfect-vision oracle scores 0% there: the ball
   (r=0.5, density=3 = light) gets punted, flooring closest approach at ~0.93 m. At reach=1.0 the
   oracle scores 100%. Adopting 0.75 would have been a winnability violation dressed as difficulty.
3. **cmd_turn=0 is NOT "straight ahead"** — the v10 gait drifts +77°..+112° left per episode. My
   first floor measurement (0%) was therefore an artifact, not evidence the task needs steering. The
   honest floor is the best FIXED turn: **30%** (at turn −0.2). Oracle ceiling 100%. 70-point window.
FINAL SPEC: single_ball, cone ±0.9 rad, reach 1.0 m, 2.5–4.0 m, ball in view 100% at reset (measured).
Panel notes carried forward: Taylor Ch.4 §4.6 predicts vision ALONE never determines the steering
action (it is joint with proprioception) — so a small blind-vs-sighted swing is ambiguous, and the
R² slope, not the ablation magnitude, is the disambiguator. Lit-scout: Wijmans et al. ICLR 2023 —
fully BLIND agents hit ~95% PointNav, so a sighted agent has no pressure to encode bearing; the
field's fix hierarchy is distil-then-RL over aux-loss.
- [x] 4. RUNNING: vbear_s0, 300K, transplanted+frozen eyes, frozen v10 gait.
- [x] 5. eval_vsteer_bearing.py written (first-step turn R², fair noise blind, stratified contact).
