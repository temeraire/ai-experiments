# Literature Scout — Can AB Reuse a Crawling Body + Policy? (2026-06-16)

Scope: can the Alien Baby project ADOPT an existing locomoting creature (body, policy,
or proven recipe) so it can move toward and physically reach/touch a target, instead of
re-solving locomotion from scratch (which it tried and failed — FINDINGS.md:843, "the
policy did not learn to crawl ... it is not locomotion"). Every prior-work claim carries
a URL. Governing constraint applied throughout: **every episode must be WINNABLE** — the
target must be physically reachable and (for vision) bringable into view by an action the
agent can execute.

---

## PROJECT UNDERSTANDING

AB is a minimal MuJoCo creature trained with SAC/DroQ to (1) move, (2) reach/touch a
target by feel, and (3) eventually use a head camera. The project repeatedly failed to get
a free-bodied MIMo-style infant to locomote: under sparse contact reward the SAC actor
collapses to a no-move or one-move-then-freeze attractor (FINDINGS Phases I–VI). It worked
around this by mounting the body on an autonomously-moving **cart** (a substrate that
carries the body past the ball) — so AB's strong recent generalization results are NOT
self-propelled locomotion; the creature still cannot reliably crawl to a target it chooses.
The new governing rule is that training/eval episodes must be winnable: the target reachable
and in (or bringable into) view. The mission is to find whether a body+policy, or at least a
recipe, can be adopted to close the real perception->action->contact loop.

## LOCAL REFERENCES MINED

**`2509.09805v1.pdf` — "MIMo grows!" (MIMo v2), López, Lenz, Fedozzi, Aubret, Triesch
(2025, IEEE ICDL).** https://arxiv.org/abs/2509.09805 · code https://github.com/trieschlab/MIMo
- What it actually ships (verified in PDF): MuJoCo+Gymnasium infant body with
  proprio/vestibular/vision/touch; v2 adds growing body, foveated vision with developing
  acuity, sensorimotor delays, an IK/operational-space controller, and a random-environment
  generator (rooms + Toys4K objects). The one quantitative result is a **proprioceptive
  reaching** study (PPO) measuring how sensorimotor delay slows time-to-reach (Fig 4).
- DECISIVE for this mission (PDF, Conclusions, lines ~444): crawling is named as
  **"ongoing work"** — *"MIMo v2 is being used to model complex behaviors, including
  crawling and self-touch."* It is NOT a released, loadable crawling policy. The shipped
  task is target-aware reaching, not self-propelled go-to-target locomotion.
- Relevant references mined: [10] Mattern MIMo TCDS 2024 (arXiv:2312.04318); [11] Kim/
  Kanazawa/Kuniyoshi fetus-in-soft-uterus, ICDL 2022; [17] López/Shi/Triesch "Eye-hand
  coordination develops from active multimodal compression," ICDL 2023; [35] Toys4K
  (Stojanov/Thai/Rehg 2021); IK refs [32-34] (Nakamura task-priority redundancy; Khatib
  operational-space) — these are the controller AB could borrow to make reaching reliable.

**`documents/Learning_to_Walk_in_20_minutes.pdf` — "A Walk in the Park," Smith, Kostrikov,
Levine (2022, CoRL).** https://arxiv.org/abs/2208.07860 · code https://github.com/ikostrikov/walk_in_the_park
- The make-or-break ingredient, verified verbatim in the PDF (Sec V, lines ~470-526):
  the action space is **PD position targets defined as offsets around a default pose**,
  `[p - o, p + o]` with `o = [0.2, 0.4, 0.4]` (Fu et al. values), and *"We confirm that
  constraining the action space is crucial."* The ablation result, verbatim: **"the
  simulated agent cannot make any progress in the unconstrained action space, while
  constraining the space leads to stable training."** Also load-bearing: position-controller
  **damping** Kd=10 (Kd=1 unstable, Kd=20 too sluggish), DroQ (SAC+Dropout+LayerNorm), high
  UTD, synchronous per-step training, and a **simple** velocity-interval reward (DM-Control
  style) — they explicitly reject many-term reward functions.

---

## FINDINGS BY QUESTION

### Q1 — REUSABLE BODY + POLICY (downloadable policy, or only a body?)
**VERDICT: OPEN for a loadable infant crawling policy. PARTIALLY SOLVED for a reusable
infant BODY. PARTIALLY SOLVED for downloadable policies on the WRONG (adult/quadruped)
bodies.**

There is no open-source infant/small-humanoid body that crawls or locomotes-to-target WITH
a released, loadable checkpoint. You can get a body, or you can get a checkpoint for a
different body — not both.

- **MIMo / MIMo v2** — https://github.com/trieschlab/MIMo · https://arxiv.org/abs/2509.09805.
  Best-matched body. Ships proprio-only **reaching** as its benchmark. Crawling is explicitly
  *ongoing work*, NOT released. So: body yes, crawling policy no. (Verified in the PDF
  conclusions.)
- **BabyBench 2025** — https://babybench.github.io/2025/ · starter kit
  https://github.com/babybench/BabyBench2025_Starter_Kit. A MIMo-based ICDL-2025 competition.
  Ships the body + environments + evaluation metrics, and a baseline RL stack — but the two
  target behaviors are **self-touch and hand-regard only**, trained by **intrinsic motivation
  with no extrinsic reward**, and it ships **no pretrained policies** and **no locomotion /
  go-to-target task**. So it does not hand AB a crawler. (Verified by reading the README.)
- **Kuniyoshi fetus/neonate sims** — Yamada et al., "An Embodied Brain Model of the Human
  Foetus," Sci. Reports 2016, https://www.nature.com/articles/srep27893; Kim/Kanazawa/
  Kuniyoshi "Simulating a human fetus in soft uterus," ICDL 2022. These DO produce emergent
  rolling-over and crawling-LIKE motion from a musculoskeletal body with almost no innate
  circuitry — but it is a bespoke spiking/muscle model, not a MuJoCo+SAC drop-in, and there
  is **no released loadable policy or standard env**. Conceptual precedent, not reusable code.
- **CTU motion-retargeting (2026)** — "Simulating Infant First-Person Sensorimotor
  Experience via Motion Retargeting," https://arxiv.org/abs/2604.27583 · code
  https://github.com/ctu-vras/motion-retargeting/. Retargets real infant video motions onto
  iCub/pyCub/EMFANT/**MIMo** with sub-cm accuracy, code released (CC-BY-4.0). NOT an RL policy,
  but the retargeted trajectories could serve as **reference/demonstration motions** to seed
  or imitation-bootstrap an AB crawl (caveat: not advertised for RL; not yet peer-reviewed).
- **Standard MuJoCo crawlers as fallback** — DLR-RM rl-baselines3-zoo /
  https://github.com/DLR-RM/rl-trained-agents and HuggingFace https://huggingface.co/sb3 ship
  downloadable Ant / Humanoid / Walker SAC/TQC/PPO checkpoints (e.g.
  https://huggingface.co/sb3/tqc-Humanoid-v3). These ARE loadable policies — but they are the
  *standard* Ant/Humanoid bodies, the task is "maximize forward velocity in +x," NOT
  go-to-a-chosen-target, and the policy is tied to that body's exact observation/action
  layout, so it will not transfer to AB's morphology without retraining.

SO WHAT FOR AB: No shortcut exists where you download a crawling infant and a matching policy.
The realistic reuse is **body + recipe**, not body + policy. If you want a *loadable* policy
at all, it forces a body choice: either adopt a standard MuJoCo body (Ant is the path of
least resistance to "it actually moves to a target") or accept that the infant body must be
trained from the recipe in Q2.

### Q2 — REUSABLE RECIPE (and did AB skip the make-or-break ingredient?)
**VERDICT: SOLVED — the minimal recipe is well-established, and AB almost certainly skipped
the single most important ingredient: a constrained, position-based action space.**

- **Constrained action space is the make-or-break ingredient.** Smith/Kostrikov/Levine
  (2022), https://arxiv.org/abs/2208.07860, ablate it directly: **unconstrained action space
  makes ZERO progress; constraining it gives stable training.** The action is PD position
  *offsets around a default pose*, not raw torque. **AB's crawler uses raw `<motor>` torque
  actuators on its limbs** (`alien_baby/crawler/crawler.xml:239-273`: limb joints are
  `<motor>`; only the head is a `<position>` servo). This is exactly the configuration the
  paper shows cannot learn. AB's "SAC collapses to no-move / one-move-then-freeze" failure
  (FINDINGS Phases I–VI) is the textbook symptom of an unconstrained torque action space:
  most of the action volume tips/flails, so the safe expected-value action is ~zero.
- Supporting recipe pieces (all things AB partly has): DroQ — Hiraoka et al. 2022,
  https://arxiv.org/abs/2110.02034; REDQ high-UTD — Chen et al. 2021,
  https://arxiv.org/abs/2101.05982; minimal-effort learning — Ha et al. 2020,
  https://arxiv.org/abs/2002.08550. The "Walk in the Park" code (DroQ config, A1 env, sim
  + real dirs) is the canonical implementation to copy: https://github.com/ikostrikov/walk_in_the_park.
- A second skipped ingredient: **position-controller damping** (Kd=10) — the paper shows
  damping value alone separates "unstable" from "learns." Torque control gives AB none of this.

SO WHAT FOR AB: The cheapest high-leverage change is to rewrite the limb actuators from
`<motor>` (torque) to `<position>` servos with a default pose and clamp the policy output to
small offsets around that pose (`[p - o, p + o]`), with moderate damping. The field's verdict
is unambiguous: this is the difference between "cannot learn to locomote" and "learns in
minutes." It is very likely the actual reason AB's from-scratch locomotion failed, and it is
a small XML/wrapper change — not new research.

### Q3 — DIRECTED, VISION-IN-THE-LOOP LOCOMOTION / REACHING (move toward a target you can see)
**VERDICT: PARTIALLY SOLVED for manipulators with active vision; OPEN for an *infant body*
that crawls toward a *visible* target and keeps it in view as learned behavior.**

- Closest infant-model work, and a direct hit on AB's thesis: **López, Shi, Triesch,
  "Eye-hand coordination develops from active multimodal compression," ICDL 2023**,
  https://ieeexplore.ieee.org/document/10364414/. An embodied infant model learns to move its
  eyes to track an object held in its hand by compressing vision+proprio into a joint code.
  Key demonstrated result: the multimodal representation **only improves tracking if it
  emerges AFTER the single-modality systems are established** — a "less-is-more" effect. This
  is independent published support for AB's proprio-first-then-vision ordering, and it closes
  a gaze->hand loop, but it is gaze-control, not crawling-to-target.
- Active vision for reaching/grasping (manipulators, not bodies): Cheng et al., "RL of Active
  Vision for Manipulating Objects under Occlusions," https://arxiv.org/abs/1811.08067 — agent
  learns to MOVE its camera to locate the target before grasping. "Vision-Based Manipulators
  Need to Also See from Their Hands," Hsu et al. ICLR 2022, https://arxiv.org/abs/2203.12677 —
  on a left/right-randomized reach, a proprio-only policy **collapses to always reaching one
  side** because it cannot disambiguate direction without exteroception. This is precisely
  AB's "one canned paddle direction" problem, with an external citation that vision (here, a
  hand camera) is what fixes it.
- Whole-body move-toward-what-you-see (adult avatars, very recent): "Moving by Looking"
  (CLOPS), https://arxiv.org/abs/2509.19259 — an avatar navigates to and discovers a goal
  driven by egocentric vision; low-level motion is learned from data first, RL then maps
  pixels to motion commands. Demonstrates the closed perception->locomotion loop, but on a
  pre-trained motion prior, not an infant learning from scratch.

SO WHAT FOR AB: The exact thing AB wants — an infant body that crawls toward a target it
visually perceives and keeps in frame — is essentially unbuilt. The pieces exist separately
(gaze->hand in MIMo; camera-move-to-find in manipulation; vision->locomotion in avatars). The
strongest reusable idea is the López/Shi/Triesch "single modalities first, fuse later" result,
which says AB's developmental ordering is the *right* way to make the joint code informative —
and Hsu 2022 gives AB a citable reason the proprio-only policy is stuck on one direction.

### Q4 — ENFORCING WINNABILITY (keep the target reachable / in view during training)
**VERDICT: SOLVED — the field has standard, proven machinery for exactly this.**

- **Reverse / start-state curriculum:** Florensa et al., "Reverse Curriculum Generation,"
  CoRL 2017, https://arxiv.org/abs/1707.05300 — start near the goal (trivially winnable) and
  push start states outward only as the agent succeeds. Directly implements "every episode is
  winnable, difficulty rises with competence."
- **Automatic goal generation (keep goals in the feasible band):** Florensa et al.,
  "Automatic Goal Generation for RL Agents," ICML 2018, https://arxiv.org/abs/1705.06366 —
  a GAN proposes goals of *intermediate difficulty* (feasible-but-not-yet-mastered),
  explicitly filtering out impossible goals. This is the formal version of AB's "target must
  be reachable."
- **Goal-of-intermediate-difficulty / region growing:** "Region Growing Curriculum,"
  https://arxiv.org/abs/1807.01425; BaRC backward-reachability, https://arxiv.org/abs/1806.06161
  — same principle, reachability-aware.
- **Sparse->dense / toddler reward transition (winnable AND well-shaped early):**
  "Unveiling the Significance of Toddler-Inspired Reward Transition," AAAI 2024,
  https://arxiv.org/abs/2403.06880, and the extended "From Sparse to Dense,"
  https://arxiv.org/abs/2501.17842 — start with dense distance shaping (so early episodes
  give gradient), transition to sparse later; shown to improve sample efficiency and
  generalization on robotic-arm and **egocentric 3D navigation** tasks.
- **MIMo v2's own scaffolding:** its environment generator "encourages locations close to
  reachable areas" (PDF, line ~422) — the platform already biases toward winnable configs.
- **Foveation / gaze for keeping target in view:** MIMo v2 foveated vision + the saccade-
  vergence line (López et al. ICDL 2024; Raabe et al. ICDL 2023, cited in the MIMo v2 refs)
  give a principled way to keep a target in the high-acuity center as a learned act.

SO WHAT FOR AB: AB does not need to invent winnability enforcement. Adopt a reverse/start-state
curriculum (spawn the target within reach + in FOV first, expand outward as touch-rate rises)
and a sparse->dense reward transition. This is the standard answer to "episodes must be
winnable," and it directly replaces AB's hand-built cone/eccentricity scaffolds with a
principled, cited mechanism.

### Q5 — BOTTOM LINE
**VERDICT: AB can reuse a BODY and a RECIPE, but NOT a ready crawling policy; directed
crawl-toward-a-visible-target remains substantially OPEN and is where AB's real work is.**

The single most actionable finding: AB's locomotion likely failed for a **known, solved**
reason — raw-torque (unconstrained) action space — and the fix (position-offset action space
+ damping, the "Walk in the Park" recipe) is a small change, not research. Path that best
satisfies the winnability constraint, in order of leverage:
1. Switch limb actuators to **position-offset control** (Q2) so locomotion becomes learnable
   on the existing body.
2. Wrap training in a **reverse/start-state + sparse->dense curriculum** (Q4) so every
   episode is winnable and the target starts in-reach and in-FOV.
3. Borrow **MIMo v2's IK/operational-space reach controller** for the final reach so contact
   is reliable once the body is near.
4. Only then layer vision, using the **"single modalities first, fuse later"** result
   (López/Shi/Triesch) to make the visual code informative — the proprio-first ordering is
   externally supported, not idiosyncratic.

---

## WHAT TO STEAL (ranked)
1. **Constrained position-offset action space + damping** from "Walk in the Park"
   (https://arxiv.org/abs/2208.07860, code https://github.com/ikostrikov/walk_in_the_park).
   Highest leverage by far — likely the actual cause of AB's locomotion failure; small XML/
   wrapper change.
2. **Reverse/start-state curriculum + automatic goal generation** (Florensa CoRL 2017
   https://arxiv.org/abs/1707.05300; ICML 2018 https://arxiv.org/abs/1705.06366) to guarantee
   winnable episodes with rising difficulty.
3. **Sparse->dense (toddler) reward transition** (https://arxiv.org/abs/2403.06880) so early
   episodes have a learnable gradient and late ones are honest.
4. **MIMo v2 as the body + its IK/operational-space reach controller and reachable-area
   biased environment generator** (https://github.com/trieschlab/MIMo,
   https://arxiv.org/abs/2509.09805).
5. **DroQ + high UTD** (already in use; https://arxiv.org/abs/2110.02034,
   https://arxiv.org/abs/2101.05982) — keep, now correctly paired with #1.
6. **"Single modalities first, fuse later"** eye-hand result (https://ieeexplore.ieee.org/document/10364414/)
   as the principle for when to add vision.
7. **A downloadable Ant/Humanoid SAC checkpoint** (https://huggingface.co/sb3) only as a
   sanity baseline that "go to a target and touch it" is achievable in your harness — NOT as
   AB's body.
8. **CTU infant motion-retargeting trajectories** (https://github.com/ctu-vras/motion-retargeting/)
   as optional crawl reference motions for imitation bootstrapping.

## WHERE AB MIGHT BE NOVEL
- **An infant-bodied agent that crawls toward a target it visually perceives and keeps in
  view as learned behavior** — genuinely unbuilt. MIMo does proprio reaching; BabyBench does
  self-touch/hand-regard; CLOPS does adult avatars with a motion prior. AB's exact combination
  is open.
- **Proprio-first crawl-to-target, then vision layered with load-bearing measured as the
  outcome** — components exist (López/Shi/Triesch ordering result), the closed end-to-end
  locomoting version does not.
- STOP-REINVENTING flags: do NOT keep hand-tuning cone/eccentricity scaffolds (reverse
  curriculum + auto goal gen are the solved version); do NOT keep fighting torque-space SAC
  collapse (position-offset action space is the solved version); do NOT expect BabyBench or
  MIMo to hand over a crawling policy (neither ships one).

## CONFIDENCE & GAPS
- **Sure of:** "Walk in the Park" ablates unconstrained-action-space to total failure
  (read verbatim in the local PDF); AB's limbs use raw torque `<motor>` (read in crawler.xml)
  — so AB skipped the make-or-break ingredient. MIMo v2 crawling is "ongoing work," NOT
  released (read in the PDF). BabyBench ships body+envs but no policies and only
  self-touch/hand-regard (read the README). Winnability has standard solutions (Florensa,
  toddler-reward). Downloadable policies exist only for standard Ant/Humanoid bodies.
- **Less sure of:** whether the CTU 2026 retargeting and the "From Sparse to Dense" 2025
  extension are peer-reviewed yet (treat as preprints); whether MuJoCo Playground ships a
  go-to-target (not just joystick-velocity) humanoid task with checkpoints — its locomotion
  tasks are velocity/joystick-tracking, and I did not confirm a target-seeking task with a
  released checkpoint usable on a custom body.
- **Deeper follow-up should cover:** (1) the BabyBench baseline code to see exactly which
  intrinsic-motivation algorithm it ships (it may give AB a working MIMo control stack to
  build the position-offset action space onto); (2) whether the MIMo repo has a crawling
  branch or PR despite the paper calling it "ongoing"; (3) a direct read of "Moving by
  Looking"/CLOPS for whether its vision->locomotion mapping is transplantable to a learned-
  from-scratch infant; (4) the MIMo IK controller API, to wire the reach stage.
