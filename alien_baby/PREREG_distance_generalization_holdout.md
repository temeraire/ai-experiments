# Pre-registration — Distance generalization via a widened band with a held-out gap

**Status:** LAUNCHED 2026-07-18 (David chose distance-holdout first). Governing question: is AB's
movement-based reach a **learned law over distance**, or **memorization of the trained band**?

## LAUNCHED RUN (2026-07-18 ~18:59, PID 94849)
```
.venv/bin/python -u -m alien_baby.crawler.train_head_search \
  --init-model alien_baby/results/s1corr_plain_s0_best/best_model.zip \
  --xml alien_baby/crawler/mimo_crawler_pos_wide_prism.xml \
  --steps 200000 --run-tag dist_holdout_vis_s0 \
  --spawn-radius 0.35 0.85 --spawn-radius-hole 0.55 0.75 \
  --spawn-cone-deg 136 --gaze-spawn --decoy \
  --prism-offset 0 --seed 0 --n-envs 16 --device mps
```
Log: `alien_baby/results/dist_holdout_vis_s0.log`. Liveness gate PASSED at 10K (body_motion 2.28).
Deviations from the strategist's draft command, deliberate: `--xml …_prism.xml` (checkpoint is tied
to it — verified obs 6213 / act 26 match; the draft's default hs XML would have mismatched);
`--prism-offset 0` and `mismatch-coef` left at default 0.0 (pure distance manipulation, no prism
reward shaping). Pre-check: winnability rendered at r=0.35/0.65/0.85 (`scratch_render/`), balls
on-platform + in-view + reachable at all three (0.85 needs no pull-in); sampler unit-tested
(0/20000 draws in the hole on both spawn paths). Code: `--spawn-radius-hole` added to
`mimo_crawler_env.py` (`_draw_radius`), `train_crawler.make_env`, `train_head_search`; `--radius`
added to `render_prism_decoy.py`.

## POST-RUN plan (mandatory gates)
Three-cell eval on `dist_holdout_vis_s0_best` at offset 0, gaze-spawn, n≥60: trained-near
`--radius 0.35 0.55`, trained-far `--radius 0.75 0.85`, GAP `--radius 0.55 0.75`. Read contact rate
(primary) + directed-choice (co-primary) + steps-to-contact (corroborating). RENDER the GAP cell and
watch before any verdict. Then results-analyst + theory-monitor before anything is written as settled.

## PANEL REVISIONS (fold-in, 2026-07-18) — these override the draft below where they conflict
1. **Run on the VISION line, NOT proprio (load-bearing correction).** The distance law (R44, r=0.89)
   was measured on the CART, which carried AB's body to the ball. On the free crawler body, blind
   proprio locomotion is UNDIRECTED (mean Δdist-to-ball = +0.000 m) — it crawls one canned direction.
   So on this body, wide-cone far bins are unwinnable by construction, and "steps-to-contact ∝
   distance" presupposes homing the blind body lacks. The vision line (`s1corr`, 88% directed choice)
   is the ONLY config that homes across a wide cone → the only one that makes this test well-posed.
   The question upgrades to: *does the freshly-grounded eye's reach generalize across distance, or has
   it memorized the 0.70–0.80 shell?*
2. **Thresholds: contact rate is PRIMARY; steps-to-contact DEMOTED to corroborating.** This
   substrate's per-episode step variance is huge (documented ±600–1000); a ±20% band on a noisy
   steps-mean would flag noise as memorization. Require n≥60/bin; widen steps tolerance to ±35%.
3. **Add directed-choice-in-the-gap as a CO-PRIMARY.** It separates "can't SEE it at this distance"
   (choice→50%) from "can't CRAWL that far" (choice stays ~85%, contact dips) — which contact alone
   cannot. Pass = gap contact ≥0.8× trained-band mean AND gap choice stays near trained (~85%).
   Falsifier = gap contact ≤0.5× trained AND gap choice collapses toward 50%.
4. **Defer distance×size factored holdout to v2** (needs a training-time ball-size hook that doesn't
   exist; only an eval-time `geom_size` override does). Defer the gain-field arm too.
5. **Sequencing flag (for David):** this is a legitimate but SECONDARY line. The sharper
   North-Star-aligned next step already teed up is Stage-2, the +30° prism-lens recalibration test.
   Decide order before compute.

## Prior-art — this design is NOT novel; cite, don't claim (literature-scout)
- RL framing/vocab: **interpolation / combinatorial interpolation / extrapolation** over a *context
  set* — Kirk et al. 2022 (arXiv:2111.09794); train/held-out gap as the memorization metric — Procgen
  (Cobbe et al. 2020).
- The rule-vs-lookup LOGIC of "test the untrained values": **function learning** — DeLosh, McDaniel &
  Busemeyer 1997 (extrapolation is the diagnostic that separates a rule from memorized pairs; EXAM =
  a cheap null model to compare AB's gap behavior against).
- Motor control: distance/amplitude generalization is **broad but bounded by STATE OVERLAP** —
  Goodbody & Wolpert 1998; Mattar & Ostry 2010; workspace-distance recalibration — 't Hart/Henriques
  2014. (Fitts' law is a DIFFERENT phenomenon — do not cite it here.)
- Behavioral: **generalization gradients** — Guttman & Kalish 1956; Shepard's universal law 1987;
  **peak-shift caution** (Cheng & Spetch 2002) — the untrained middle can respond non-monotonically
  after discrimination training; pre-register this as a possible outcome, not noise.
- Factored holdout: **CLEVR-CoGenT** (Johnson 2017) / **SQOOP** (Bahdanau 2019); metric = the
  **"systematicity gap"** (held-out-cell minus trained-cell performance).
- Honest novel space: this exact interpolation/extrapolation + factored-holdout design applied to an
  **embodied, self-produced proprioceptive/visuomotor REACH law over metric distance×size** — the
  compositional lit is almost all vision/language/navigation. Method = reproduced; substrate = open.

## Local shelf — Taylor predicts the FORM of the result (local-book-agent)
- **§6.11 + §2.16–2.17 — the "automatic interpolation device."** You do NOT train every distance: a
  few trained magnitudes + overlapping generalization gradients fill untrained values as a **weighted
  average of the bracketing trained responses**. This is the theoretical warrant for train-ends /
  test-middle, and it predicts a *specific form*: gap response should sit BETWEEN the two bracketing
  trained distances, smooth, crude with few anchors and sharper with more → motivates an optional
  "vary number of trained distances" arm.
- **§6.4–6.8 — locomotion calibrated to distance** via retinal-image size, learned by reaching a ball
  *from many initial distances* (AB's task nearly verbatim); usable discrimination extends to ~2× the
  trained radius (expect graceful degradation, not a hard cliff, just outside range).
- **§4.10 / §6.6 — bounded extrapolation:** carries "a little beyond," then distinct far distances
  become indistinguishable — read a FLAT far response as "beyond the generalization horizon," not a
  broken policy.
- **§4.9 — reach-workspace geometry:** redundant-near / unique-far / unreachable(extinguished). Bin
  evals by these regimes; NEVER score the unreachable region as "AB failed" (winnability).
- **§6.14 — parameter-change breaks the mapping with a characteristic UNDERSHOOT** (change eye height
  → learned distance responses undershoot, must re-learn) — ties directly to the gaze-frame rule.
- **§9.19 — response-specificity:** a recalibration transfers ONLY over the region the behavior
  actually sampled — do not assume transfer to unsampled directions/bands; test per region.

## Origin (plain language)
David's proposal: AB's world is impoverished — one ball, a narrow slice of situations — versus the
many thousands of reach/locomotion instances a human infant gets. Idea: densely sample the ball's
position (a degree of azimuth at a time, a centimetre of distance at a time) to "compress years into
hours" and help generalization.

Refinement adopted here (why we do NOT just densify everything):
1. **Azimuth is already densely sampled.** Each episode already draws a random bearing in a wide
   cone (`--spawn-cone-deg`); random uniform coverage generalizes as well as or better than a fixed
   grid. Finer azimuth resolution buys ~nothing.
2. **Distance is the under-sampled axis.** Training uses `--spawn-radius 0.70 0.80` — a 10 cm shell.
   The Phase V distance law extrapolated impressively but from that thin band. Widening distance is
   the real, under-exploited lever.
3. **Density alone can mask memorization.** Give AB every position and it can build a dense lookup
   table and show zero underlying generalization — undetectable if you test where you trained. So the
   design MUST hold out a region and evaluate on it. Coverage + a deliberate gap = a real test.

## Design (minimal-change, existing flags only)
One factor, distance `d` = ball spawn radius. Widen the training band and cut a hole.

- **Widened range:** d ∈ [0.35, 0.85] m (vs the current 0.70–0.80 shell). Winnability: 0.35 m is
  inside prone-reach; 0.85 m is reachable by a short crawl. Verify both by render before compute.
- **Held-out gap (the hole):** train on d ∈ [0.35, 0.55] ∪ [0.75, 0.85], **never** on the middle
  band d ∈ (0.55, 0.75). Evaluate specifically inside the gap.
  - Implementation note: `--spawn-radius` is a single contiguous range. A two-interval "holed"
    sampler needs a ~5-line change in `mimo_crawler_env.py` reset (draw d, resample if it lands in
    the hole) OR a `--spawn-radius-hole LO HI` flag. This is the only code change required.
- **Everything else fixed** to the current plain-crawl recipe (no prism, no decoy unless the panel
  wants it): same cone, same body, same reward, same algorithm, same seeds. One variable: the
  distance distribution.

## Which policy
Default: the **plain / proprio-capable crawl policy** (the system whose distance law we are mapping),
seeded from the current `s1corr`/`s2corr` line. Rationale in the honesty note below. Panel may argue
for a vision arm; if so it is a SEPARATE arm, not a substitute.

## Metrics & falsifier (pre-registered thresholds)
Primary readout: **time-to-contact vs distance** (steps-to-contact ∝ range), plus hand-touch
fraction and net root displacement, scored on matched eval bins.

- **Marginal laws (trained region):** fit steps-to-contact = a·d + b on the two trained sub-bands.
- **Held-out prediction:** the fitted line predicts steps-to-contact at the gap distances.
- **GENERALIZATION (pass):** mean steps-to-contact in the gap is within **±20%** of the line's
  prediction AND contact rate in the gap ≥ 0.8 × the trained-band contact rate.
- **MEMORIZATION (fail / falsifier):** gap contact rate collapses (< 0.5 × trained) or steps-to-
  contact departs the predicted line by > 40% — i.e. AB can reach where it trained but not in the
  unseen middle it was never shown.
- **Null / inconclusive:** noisy gap with wide CIs → rerun with more eval episodes (≥30/bin) before
  any verdict. Informative-null rules apply.

## Winnability & liveness pre-checks (mandatory, before compute)
- Render one episode at d = 0.35, 0.65 (the gap), and 0.85 from the seed checkpoint via the eval's
  own `make_env`, watch via `describe_video.py` AND an experimenter's eye — confirm the ball is in
  view / bringable into view and reachable at every distance. If 0.85 is unreachable by crawl, pull
  the far edge in; do not score an impossible bin as "AB failed."
- Liveness gate ON for the training arm.

## Honesty note carried from the design discussion
This experiment sharpens and MAPS **proprio/movement** generalization (which already works). It is
**unlikely to fix vision inertness**, which the record attributes to a structural cause (no
directional action for vision to drive → no gradient to encode direction), not to sparse position
sampling. Coverage extends a capability that exists; it does not install a missing one. Keep the
claim scoped to the movement system unless a vision arm is added and independently shows a gap.

## Optional second factor (needs code, flag for panel)
Distance × size factored "holed product space" (the sharpest form, transcript line 1704): train
{small-near, large-far}, hold out {small-far, large-near}, eval the held-out corners. Requires a
training-time ball-size randomization hook (none exists today; eval-time `geom_size` override does).
Defer to a v2 unless the panel judges the code cost worth it now.

## Optional continuity arm
A slow within-episode ball drift (continuous tracking) as a richer, more developmentally faithful
signal than teleport-on-reset. Uses existing moving-ball machinery. Secondary; not part of the core
falsifier.
