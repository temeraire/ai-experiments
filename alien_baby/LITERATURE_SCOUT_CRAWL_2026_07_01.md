# Literature Scout — "Has anyone made a sim creature crawl?" (2026-07-01)

Triggered before committing more compute to the crawler track (after rnd_propulsion_400k
Verdict B + the hand_drive affordance diagnostic). Bottom line: **AB's locomotion failure is
a setup failure, not a learning failure**, and the setup is nonstandard on **four independent
axes, each individually known to break locomotion.** None of the fixes are research.

## Mission verdict
As of the 2026-06-16 scout we adopted the single most important solved ingredient (position-offset
PD control). The remaining failures come from three things the field would flag on sight, plus the
prone default pose:
1. **Clamps too tight + default pose is prone** → the reachable pose set physically excludes a crawl
   stride. (Our own hand_drive test already proved this = affordance failure, Verdict B.)
2. **No fall/early-termination signal.** Every from-scratch locomotion recipe uses early termination
   precisely because it removes the "lie-on-the-ground / do-nothing" local optimum. Its absence is why
   stillness and rocking are attractors.
3. **We reward |CoM velocity|.** Textbook gameable reward — rocking-in-place maximizes it with zero net
   displacement (exactly what rnd_propulsion_400k did).

## Per-question findings

### Q1 — Has anyone made MIMo crawl/locomote? VERDICT: OPEN (nobody has).
MIMo's published whole-body skill is **rolling only**; crawling is explicitly "ongoing/future work."
MIMo tasks to date: proprioceptive reaching, standup-from-crib, selfbody touch, catch, self-touch/
hand-regard (BabyBench 2025), and supine→prone **rolling** (2026). Repo cross-check: MIMo ships
`roll_over_scene.xml` and `standup_scene.xml`, no crawling scene.
- **Key source:** Philipp, López & Triesch, "Embodiment Shapes Rolling Behavior in a Multimodal
  Infant Model," ICDL 2026, arXiv:2606.17456. MIMo learns supine→prone rolls from scratch with
  **PPO + dense potential-based shaping + large sparse success reward (500) + metabolic-cost penalty
  (α=0.02), no imitation, supine start with mild limb jitter, spring-damper (46-DOF) actuators.**
  Concludes "physical strength is a critical component."
- "MIMo Grows!" (López et al., ICDL 2025, arXiv:2509.09805): crawling & self-touch are "ongoing work."
- **So what:** no crawler to download, but MIMo's own recipe for its one whole-body skill is
  PPO+potential-shaping+sparse-bonus+metabolic-cost+supine-start — differs from our sparse-contact
  SAC+RND on nearly every axis.

### Q2 — From-prone crawling as a distinct problem. VERDICT: PARTIALLY solved in principle, UNBUILT as reusable RL.
Belly-crawl-from-prone is a recognized distinct motor problem (clinically "crawling" = belly on floor,
vs "creeping" = hands-and-knees). No drop-in MuJoCo/Isaac RL crawler-from-prone exists.
- **Key source:** Yamada et al., "An Embodied Brain Model of the Human Foetus," Sci. Reports 2016
  (nature.com/articles/srep27893) — emergent rolling and **crawling-like** whole-body motion from a
  musculoskeletal body; existence proof, but bespoke spiking-muscle model, not reusable.
- Pattern in both fetus work and MIMo rolling: whole-body-from-prone emerges from **muscle-like
  actuation + strength + dense shaping**, not a stiff position servo held near a prone neutral.

### Q3 — Standard reliable recipe for from-scratch MuJoCo locomotion. VERDICT: SOLVED; we diverge on 3 load-bearing points.
Default recipe: **PD/position action space (target joint angles around a useful default pose)**;
**early termination on fall**; **reference-state / randomized initial-state distribution**; **simple
reward** (forward-velocity interval + alive bonus + small control cost), NOT a many-term or |velocity|
reward.
- **Key source:** Smith, Kostrikov & Levine, "A Walk in the Park," RSS 2023 / arXiv:2208.07860
  (code: github.com/ikostrikov/walk_in_the_park). Verbatim: agent "cannot make any progress in the
  unconstrained action space, while constraining the space leads to stable training"; PD offsets
  around a default pose o=[0.2,0.4,0.4] rad, Kd=10, DroQ, high UTD, **simple velocity-interval reward**
  (they explicitly reject multi-term rewards). ~20 min on real hardware.
- Peng & van de Panne, "Does the Choice of Action Space Matter?" SCA 2017, arXiv:1611.01055 —
  PD/position-target action spaces learn faster & better than torque.
- **How AB compares:** we correctly use DroQ + position-offset. We diverge on: (a) **no early
  termination** (implicated in the freeze); (b) **sparse-contact + |velocity| reward** instead of a
  clean signed velocity-interval reward (gameable, low-gradient); (c) offsets around a **prone** default
  pose instead of a locomotion-adjacent one.

### Q4 — Our exact failure modes. VERDICT: SOLVED — all three are named, textbook, with textbook fixes.
- **Freeze / do-nothing + rock-in-place:** DeepMimic (Peng et al., SIGGRAPH 2018, arXiv:1804.02717):
  early termination "eliminates local optima by penalizing the character when on the ground." Without
  it, lying prone doing nothing is an unpunished optimum — our exact attractor.
- **|velocity| gamed by oscillation = specification gaming.** Canonical DeepMind example is a
  locomotion one ("robot hooks its legs together and slides along the ground"). Fix: **signed,
  directional forward-progress reward (displacement toward goal), capped**, not |CoM velocity|.
  (Krakovna's specification-gaming list; DeepMind "Specification gaming" blog.)
- **Key source:** López, Ernst, Cruz, Hoffmann & Triesch, "Infant Spontaneous Movement Noise Improves
  Exploration in Deep RL," ICDL 2026, arXiv:2606.16590 — replacing white-noise exploration with
  **temporally-correlated ("colored") noise whose autocorrelation increases over training** produces
  structured coordinated exploration and better efficiency. A MIMo-native, more principled version of
  what our RND flag gropes toward.

### Q5 — Imitation / reference motion. VERDICT: SOLVED; minimal hand-authored reference works (no mocap needed).
- DeepMimic accepts **key-framed** reference clips, not just mocap. Peng et al., "Imitating Animals,"
  RSS 2020, arXiv:2004.00784: payload can be as little as **reference-state initialization (RSI) + a
  tracking reward from a single retargeted clip** (real quadruped).
- Even cheaper: Hämäläinen et al., "Self-Imitation via Termination Curriculum," MIG 2019,
  arXiv:1907.11842 — locomotion **without any reference clip**, via curriculum over the termination
  condition; RSI alone (seeding from a few good poses) often suffices.
- **Infant-crawl reference specifically:** CTU motion-retargeting (arXiv:2604.27583, code
  github.com/ctu-vras/motion-retargeting) retargets real infant-video motion onto **MIMo** with sub-cm
  accuracy — ready crawl reference trajectories (not yet an RL pipeline).
- AMP (Peng et al., SIGGRAPH 2021, arXiv:2104.02180) — adversarial motion prior, less brittle than
  phase-tracking.
- **Caveat:** reference motion only helps if the body can physically execute the poses → loops back to
  Q6. Reference inside our current clamps/pose would also fail.

### Q6 — Action-space clamps as a cause. VERDICT: over-restricting the reachable pose set is a known killer; exact width is body-specific.
Constraining the action space is necessary (Q3) but constraining it too tightly **around the wrong
default pose** makes the target behavior physically unreachable. "A Walk in the Park" width o=[0.2,0.4,
0.4] rad is **around a nominal standing pose** — offsets must be around a default pose already close to
the behavior. **Our default pose is flat prone**, so even generous offsets sweep a region that never
crosses into a propulsive stance. **Widening clamps alone (mimo_crawler_pos_wide.xml) is necessary but
not sufficient — change the default pose, not just the width.**

## Bottom line — highest-leverage next step (do NOT run another SAC/RND run in the current setup)
1. **Change the default pose off prone** → hands-and-knees / low-crouch "creeping" pose (or start supine
   and learn rolling→prop-up first, as ICDL 2026). Pair with the already-built widened clamps. Cheapest,
   an XML change; the single change our own affordance diagnostic points to.
2. **Fix the reward:** replace |CoM velocity| with **signed displacement toward the goal (capped)** +
   an **early-termination-equivalent** (terminate/penalize on torso-below-height / belly-collapsed).
   Kills both the freeze optimum (DeepMimic) and the rocking exploit (specification gaming).
3. **Adopt MIMo's own working whole-body recipe:** PPO + dense potential-based shaping + sparse success
   bonus + metabolic cost; consider spring-damper/muscle actuation over the stiff position servo.
4. **If pure RL still can't find the gait, seed it:** RSI + light tracking term from a few hand-authored
   crawl keyframes (no mocap), or the CTU infant-retargeted trajectories. Only after step 1.
5. **Swap white-noise exploration for colored/correlated noise** (arXiv:2606.16590) — MIMo-native
   replacement for the RND patch.

**Blunt flag:** running from a prone default pose with ±0.2–0.4 rad clamps, no fall termination, and a
|velocity| reward under sparse-contact SAC is nonstandard on four independent axes at once, each
individually known to break locomotion. Three failed runs are unsurprising; success would have been.

**Genuine novelty still available:** an infant-bodied agent that **belly-crawls from prone toward a
target it must perceive** is unbuilt (MIMo does rolling/reaching; the fetus model does task-free
crawl-like motion; quadruped RL does upright walking on non-infant bodies). Crawling-to-target on MIMo
would be a real contribution — but get it crawling *at all* first, with the solved recipe.

## Confidence & gaps
High confidence: Q1 (two Triesch-group papers + repo env list), Q3/Q6 (verbatim in local "Walk in the
Park" PDF + our own hand_drive diagnostic), Q4 (multiple sources + our logs). Less certain: exact clamp
width for MIMo kinematics (unpublished — empirical sweep on our body), and whether MIMo's rolling
reward/potential-shaping transfers to crawl (read abstract/HTML, not full methods). Follow-up: pull full
PDFs of arXiv:2606.17456 and 2606.16590 for exact reward terms & noise schedule; check MIMo GitHub for a
crawling branch.

## Sources
- arXiv:2606.17456 — MIMo rolling (ICDL 2026)
- arXiv:2606.16590 — Infant spontaneous movement noise / exploration (ICDL 2026)
- arXiv:2509.09805 — MIMo Grows! (ICDL 2025)
- arXiv:2208.07860 — A Walk in the Park + github.com/ikostrikov/walk_in_the_park
- arXiv:1611.01055 — Action space matters (Peng & van de Panne)
- arXiv:1804.02717 — DeepMimic (RSI + early termination)
- arXiv:2104.02180 — AMP · arXiv:2004.00784 — Imitating Animals (RSI on real robot)
- arXiv:1907.11842 — Self-imitation via termination curriculum
- Krakovna specification-gaming list · DeepMind "Specification gaming" blog
- nature.com/articles/srep27893 — Fetus embodied brain model (emergent crawl-like motion)
- github.com/ctu-vras/motion-retargeting — infant motion retargeting onto MIMo
</content>
</invoke>
