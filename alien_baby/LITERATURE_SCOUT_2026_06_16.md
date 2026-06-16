# Literature Scout Report — Alien Baby (2026-06-16)

Run of the `literature-scout` agent. Scope: both, problem first. All eight questions
(Q1–Q8) investigated with multiple searches each; both local PDFs mined; 2024–2026
recency pass done. Every prior-work claim carries a URL. Honesty caveats in the final
section.

---

## PROJECT UNDERSTANDING
Alien Baby teaches a small simulated creature (MuJoCo, SAC/DroQ) to function in stages:
first move, then reach/touch a target using only body sense (proprioception + touch, no
target-location vector, no camera), and only later add a head camera. The central
problem: **vision is "integrated but inert"** — blanking the camera changes the policy's
actions (vision is wired in), but the visual latent does not encode *where* the target is
(linear probe: ball-direction R²≈0.08, a "representation failure"). Proprioception
generalizes lawfully (metric distance law that extrapolates; zero-shot transfer to
held-out object sizes/shapes). The thesis is **developmental ordering**: a body that can
move/reach/remember-by-feel comes first, vision is layered on later, and "touching" is
treated as an object-agnostic, automatically-generalizing proprioceptive act.

## LOCAL REFERENCES MINED
**`documents/Learning_to_Walk_in_20_minutes.pdf`** — *A Walk in the Park: Learning to
Walk in 20 Minutes With Model-Free RL*, Smith/Kostrikov/Levine (2022, arXiv:2208.07860).
Methodological taproot of AB's stack. Ablations show the minimal recipe: **constrained
action space** (offsets around a default pose; unconstrained fails entirely), **DroQ**
(SAC+dropout+LayerNorm), **high UTD**, synchronous per-step training; simple reward
sufficed. Top refs to follow: [60] DroQ (Hiraoka 2022), [64] REDQ (Chen 2021), [52] SAC
(Haarnoja 2018), [27] Ha 2020, [28] Haarnoja 2020, [16] RMA (Kumar/Fu/Pathak/Malik 2021),
[25] Lee/Hwangbo (privileged teacher→proprio student). AB's Phase XIV/XV DroQ+UTD=4 is on
this recipe.

**`2509.09805v1.pdf`** — *MIMo grows! Simulating body and sensory development in a
multimodal infant model* (MIMo v2), López/Lenz/Fedozzi/Aubret/Triesch (2025,
arXiv:2509.09805; IEEE ICDL 2025). Closest external platform: MuJoCo+Gymnasium **infant
body** with proprio/vestibular/vision/touch; v2 adds **growing body (birth→24mo)**,
**foveated vision w/ developing acuity**, **sensorimotor delays**, **IK**,
**random-environment generator**. Its default **reach benchmark is proprio-only** —
nearly identical to AB stage-1. Top refs: [10] Mattern MIMo (TCDS 2024), [17] López/Shi/
Triesch "Eye-hand coordination develops from active multimodal compression" (ICDL 2023),
[3]/[5] Corbetta infant reaching, [11] Kuniyoshi fetus sim (ICDL 2022), [35] Toys4K.
Flag: feature-list verified; validated *results* in the paper are thin (the
sensorimotor-delay reach study is the main quantitative result).

---
## FINDINGS BY QUESTION

### Q1 — Proprioception-before-vision the right first step?
**VERDICT: PARTIALLY SOLVED / ACTIVELY DEBATED.** "Staged complexity" is endorsed; the
specific proprio-first/vision-later ordering has infant-science support but runs against a
dominant vision-from-the-start tradition, and is not a standard ML recipe.
- Corbetta, Thurman, Wiener, Guan & Williams, *Mapping the feel of the arm with the sight
  of the object* (2014, Frontiers in Psychology 5:576).
  https://www.frontiersin.org/articles/10.3389/fpsyg.2014.00576/full — Infants: reaching
  accuracy present at reach onset; object-directed *looking* rises only *after* reaching
  emerges (vision aligns to a proprio reach). Strongest support for AB's ordering.
- von Hofsten, *Action in development* (2007, Developmental Science 10:54).
  https://onlinelibrary.wiley.com/doi/10.1111/j.1467-7687.2007.00564.x — Dominant
  counter-thesis: predictive *visual* control of reaching appears very early.
- Lungarella, Metta, Pfeifer & Sandini, *Developmental robotics: a survey* (2003,
  Connection Science 15:151).
  https://www.semanticscholar.org/paper/Developmental-robotics:-a-survey-Lungarella-Metta/e492c469039e119688b7653d51a88615ac6bc6d6
  — Codifies "staged increase of complexity" but doesn't single out proprio-before-vision.
- Berthouze & Lungarella, *Motor skill acquisition: freezing/freeing DOF* (2004, Adaptive
  Behavior 12:47). https://journals.sagepub.com/doi/10.1177/105971230401200104 — Robots:
  start reduced, add complexity helps.
- SO WHAT: One strong anchor (Corbetta 2014) + a staging principle, but you must position
  explicitly against von Hofsten. The deliberate "proprio-only first, then bolt on camera"
  curriculum is defensible as a contribution, not a reproduction.

### Q2 — Proprio–vision integration solved? Is "integrated but inert" named?
**VERDICT: PARTIALLY SOLVED / ACTIVELY DEBATED — your exact double-signature
(ablation-load-bearing + probe-non-encoding in RL) is OPEN/unnamed.**
- Modality dominance (supervised): Wang/Tran/Feiszli, *What Makes Training Multi-Modal
  Classification Networks Hard?* (2020, CVPR).
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf
  — joint multimodal net beaten by best single modality. Peng et al., *Balanced Multimodal
  Learning (OGM-GE)* (2022, CVPR).
  https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf
  — one modality dominates gradient; rebalancing recovers the weak (but informative) one.
  Huang/Lin/Du/Wang, *Modality Competition (provably)* (2022, ICML).
  https://proceedings.mlr.press/v162/huang22e/huang22e.pdf — modalities provably compete;
  some *never discovered* by their encoders (deepest match to your *representation*
  failure).
- RL-from-pixels (efficiency, not load-bearing): CURL (Srinivas/Laskin/Abbeel 2020, ICML)
  https://proceedings.mlr.press/v119/laskin20a.html; DrQ-v2 (Yarats/Fujimoto/Kostrikov
  2021) https://arxiv.org/abs/2107.09645; ATC (Stooke et al. 2021, ICML)
  https://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf; and most pointedly Zhang et
  al., *Learning Representations for Pixel-based Control: What Matters and Why?* (2022,
  ICLR) https://arxiv.org/abs/2111.07775 — under distractors, encoders fail to capture
  task structure unless reward/transition forces it.
- The recognized fix — privileged teacher→student: Chen/Zhou/Koltun/Krähenbühl, *Learning
  by Cheating* (2019, CoRL). https://arxiv.org/abs/1912.12294 — distill a privileged
  target-using teacher into a vision student; vision becomes load-bearing *because
  supervision forces it* — the step AB's end-to-end run skips. Plus Pinto et al.
  asymmetric AC (2018, RSS) https://www.researchgate.net/publication/320486930; RMA
  (Kumar/Fu/Pathak/Malik 2021, RSS) https://arxiv.org/abs/2107.04034; Lee/Hwangbo (2020,
  Science Robotics) https://www.science.org/doi/10.1126/scirobotics.abc5986.
- Your method validated: Gulcehre et al., *Light-weight Probing of Unsupervised
  Representations for RL* (2024, RLC). https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_242.pdf
  — linear-probe accuracy correlates with RL efficiency; your low-R² probe is a citable
  representation-failure verdict.
- SO WHAT: Your proprio-works/vision-inert split is the *predicted equilibrium* of
  end-to-end multimodal RL when one channel already solves the task. But the bias
  literature's weak modality is *recoverable by rebalancing*; yours is more severe (latent
  ~doesn't encode the target), closest to Huang's "never discovered." The reliable fix is
  **supervisory pressure on the vision encoder** (privileged distillation or an auxiliary
  ball-position decode loss). The ablation-yes/probe-no dissociation in an RL agent appears
  genuinely unnamed — worth writing up.

### Q3 — Minimal requirements to train a creature?
**VERDICT: PARTIALLY SOLVED for the algorithm/recipe; "minimum sensor set" and "minimal
reward threshold" are OPEN (no controlled ablation found).**
- Smith/Kostrikov/Levine, *A Walk in the Park* (2022). https://arxiv.org/abs/2208.07860 —
  constrained action space + damping + synchronous per-step training are load-bearing;
  simple reward suffices.
- Hiraoka et al., *DroQ* (2022, ICLR). https://arxiv.org/abs/2110.02034 — dropout+LayerNorm
  small critic matches REDQ cheaply.
- Chen/Wang/Zhou/Ross, *REDQ* (2021, ICLR). https://arxiv.org/abs/2101.05982 — high UTD +
  ensemble = 3–8× SAC efficiency.
- Ha/Xu/Tan/Levine/Tan, *Learning to Walk with Minimal Human Effort* (2020, CoRL).
  https://arxiv.org/abs/2002.08550 — multi-task + safety = near-autonomous learning.
- A. Chen/Sharma/Levine/Finn, *Single-Life RL* (2022, NeurIPS).
  https://arxiv.org/abs/2210.08863 — limiting minimal-experience case.
- SO WHAT: Your DroQ+high-UTD+structured-action stack *is* the ablated minimal recipe —
  on-recipe, not idiosyncratic. The gap: nobody runs the controlled "how few sensors / how
  thin a reward" ablation your minimal body sets up — your vision-ablation-sensitivity
  metric is an original contribution to that implicit question.

### Q4 — Building the environments?
**VERDICT: PARTIALLY SOLVED.** General embodied-RL sims are mature; developmental
infant-bodied ones are few; **MIMo v2 is the best-matched platform** for a proprio-first
creature that later grows a camera.
- Mattern et al., *MIMo* (2024, IEEE TCDS 16:1291). https://arxiv.org/abs/2312.04318 ·
  https://github.com/trieschlab/MIMo — infant body, all modalities, Gymnasium; default
  reach proprio-only.
- López et al., *MIMo grows!* (2025). https://arxiv.org/abs/2509.09805 — growing body,
  developing acuity, delays, procedural rooms. Best-in-class fit.
- Yamada et al., *Embodied brain model of the human foetus* (2016, Scientific Reports
  6:27893). https://www.nature.com/articles/srep27893 (+ Kim/Kanazawa/Kuniyoshi fetus sim,
  2022, ICDL) — proprio/tactile self-organization with almost no innate circuitry;
  conceptual precedent (not a released toolkit).
- Zakka/Tabanpour et al., *MuJoCo Playground* (2025).
  https://github.com/google-deepmind/mujoco_playground — 50+ GPU envs, zero-shot
  sim-to-real (adult robots, no dev body).
- Stojanov/Thai/Rehg, *Toys4K* (2021, CVPR). https://rehg.org/publication/dataset2/ —
  4,179 infant-appropriate objects.
- SO WHAT: MIMo v2 is the natural substrate/baseline to cite (same base + developing
  acuity + procedural rooms). Use Playground for throughput, Toys4K for targets; cite
  fetus + iCub diminished-early-vision results as demonstrated precedent for proprio-first
  development.

### Q5 — Proprioceptive reaching/touch WITHOUT vision as a deliberate first stage?
**VERDICT: PARTIALLY SOLVED in isolation; the proprio-first-*then-vision-curriculum*
framing is largely OPEN.**
- Mattern et al., *MIMo* (2024). https://arxiv.org/abs/2312.04318 — default reach benchmark
  is proprio-only (no camera) — near-exact match to AB stage-1.
- López et al., *MIMo grows!* (2025). https://arxiv.org/abs/2509.09805 — proprio-only reach
  solved under all delay conditions; does *not* stage modalities or compare
  proprio-vs-vision.
- Hsu/Kim/Rafailov/Wu/Finn, *Vision-Based Manipulators Need to Also See from Their Hands*
  (2022, ICLR oral, Stanford). https://arxiv.org/abs/2203.12677 — on "reach-hard" (goal
  randomized left/right), a **proprio-only policy collapses to always reaching one side** —
  can't disambiguate direction without exteroception. (Independently verified.)
- (Lower-confidence) recent blind-tactile-RL exists (arXiv 2510.21609 / 2606.11767) —
  directional evidence only; not fully verified, excluded from load-bearing claims.
- SO WHAT: Stage-1 is well-precedented in isolation (MIMo). The novel part is the
  **deliberate developmental ordering** with load-bearing measured as the outcome — OPEN.
  Hsu 2022 is a ready citation for *why* a proprio-only reacher succeeds by groping/biasing
  without encoding direction — the same non-directional "inert vision" puzzle as your Phase
  XIII/V.

### Q6 — Remembering a target's location and reaching to it WITHOUT vision?
**VERDICT: PARTIALLY SOLVED, with a split.** (a) remember-a-seen-then-hidden target via
recurrent policy: well-demonstrated, but target almost always *first acquired by vision*.
(b) body-centric spatial map from *purely* proprio+tactile exploration (no vision ever):
exists only narrowly (peripersonal space), usually with vision as teacher. A clean blind
map-then-reach from proprio+touch alone is **largely OPEN.**
- Banino et al., *Vector-based navigation using grid-like representations* (2018, Nature
  557:429). https://www.nature.com/articles/s41586-018-0102-6 — grid cells from self-motion
  path integration (navigation; eval still gives visual obs).
- Lampinen/Chan/Banino/Hill, *Towards mental time travel (HCAM)* (2021, NeurIPS).
  https://arxiv.org/abs/2105.14039 — return to a box holding a queried object after turning
  away — clearest sub-case (a), target acquired visually first.
- Roncone/Hoffmann/Pattacini/Fadiga/Metta, *Peripersonal space / margin of safety* (2016,
  PLOS ONE 11:e0163713).
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0163713 — iCub
  body-centered map from contact (closest to sub-case b), but visuo-tactile and
  peripersonal only.
- Klier/Crawford et al., *Memory for proprioceptive targets is partially coded relative to
  gaze* (2010, Neuropsychologia). https://pubmed.ncbi.nlm.nih.gov/20934442/ — humans can
  point to remembered purely-proprioceptive targets, but errors largest without vision
  (lawful but noisy).
- SO WHAT: "Remember where the ball was and reach blind" is under-explored as a
  *pure-proprio/touch* task — a real contribution, but no existing baselines to lean on,
  and neuroscience predicts a lawful-but-noisy signal (consistent with your Phase V
  proprio).

### Q7 — Does grasping add capability, and is it object-agnostic?
**VERDICT: ACTIVELY DEBATED — AB's "object-agnostic by construction" claim is
OVERSTATED.** (i) *category/identity-agnostic generalization* to novel objects: real,
robust (90%+). (ii) *success independent of geometry/pose* (AB's literal claim):
**false** — grasp success depends heavily on shape/pose even for "agnostic" methods.
Touchability ≠ graspability.
- Mahler/Goldberg et al., *Dex-Net 2.0* (2017, RSS).
  https://www.roboticsproceedings.org/rss13/p58.pdf — 99% on 40 novel objects, but
  geometry-conditioned grasps.
- Sundermeyer/Mousavian/Triebel/Fox, *Contact-GraspNet* (2021, ICRA).
  https://arxiv.org/abs/2103.14127 — >90% on unknown objects in clutter, grasps computed
  from observed geometry.
- Murali/Li/Gandhi/Gupta, *Learning to Grasp Without Seeing* (2018/2019, ISRR).
  https://arxiv.org/abs/1805.04201 — first tactile-only-no-prior-object grasping; possible
  but modest (re-grasp adds ~10.6% atop vision). (Verified, CMU.)
- Lee/Choi/Kim/Nam/Jeong, *Why Look at It at All?: Vision-Free Multifingered Blind
  Grasping* (2026, arXiv:2602.07326). https://arxiv.org/abs/2602.07326 — strongest
  vision-free result (98.3%, 18 objects) but via heavy privileged-teacher→student pipeline,
  not automatic generalization. (Recent preprint — treat headline as not-yet-peer-reviewed.)
- Oztop/Bradley/Arbib, *Infant grasp learning (ILGM)* (2004, Exp Brain Res 162:480).
  https://pubmed.ncbi.nlm.nih.gov/15221160/ — grasping *emerges from reaching* via a
  somatosensory stability signal (strongest theoretical "for"), but even ILGM matches
  object affordances; successor ILGA adds *visual* affordances.
- RULING: supported only in the weak identity-agnostic sense; NOT in the strong
  geometry-independent sense. Grasping does add capability beyond reach-and-touch and can
  be done from proprio/touch alone — so the claim is **directionally right but
  mechanistically overstated.**
- SO WHAT: Keep the spirit; **drop "object-agnostic by construction"** as written. Expect
  shape-conditioned proprio/tactile adaptation and **measure grasp success as a function of
  geometry** (flat vs round vs small) — otherwise you repeat the project's "success-rate
  hides a geometry confound" pattern (cf. your 0.075 knock-away dead-band).

### Q8 — Perception emerging from action / survival consequence (secondary)
**VERDICT: PARTIALLY SOLVED — mechanisms repeatedly built; AB's specific
"survival/world-consequence shapes a perceptual response, measured as a modality becoming
load-bearing" construct is largely OPEN.**
- Oudeyer/Kaplan/Hafner, *Intrinsic Motivation Systems* (2007, IEEE TEC)
  http://www.pyoudeyer.com/ims.pdf + *Playground Experiment* (2006)
  https://www.researchgate.net/publication/228640373 — AIBO driven only by learning-progress
  spontaneously produces an ordered easy→hard developmental sequence, no task/curriculum.
- Maye & Engel, *sensorimotor contingencies on robots* (2011–2013, Adaptive Behavior).
  https://journals.sagepub.com/doi/abs/10.1177/1059712313497975 — robot learns how actions
  transform sensory input, uses it for object discrimination/egocentric space.
- Pathak/Agrawal/Efros/Darrell, *Curiosity-driven Exploration (ICM)* (2017, ICML).
  https://pathak22.github.io/noreward-rl/resources/icml17.pdf — competent behavior from
  action-consequence prediction, no extrinsic reward.
- Klyubin/Polani/Nehaniv *Empowerment* (2005); Salge et al. (2014).
  https://www.researchgate.net/publication/221531178 — agents organize around perceivable
  consequences of their own actions.
- Lungarella et al. (2003 survey); Cangelosi & Schlesinger, *Developmental Robotics* (2015,
  MIT Press) — field-defining roadmaps.
- Taylor set-theoretic framing: **no indexed work** builds "perception as set-theoretic
  equivalence classes shaped by world-consequence." Closest adjacency is affordance
  equivalence classes (*Affordance Equivalences in Robotics: A Formalism*,
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6002533/) — about affordances, not
  perception-as-response. Honest read: the Taylor framing appears idiosyncratic/non-mainstream.
- SO WHAT: The machinery you rely on is real and demonstrated. But the field uses
  *intrinsic-motivation proxies* (learning progress, prediction error, empowerment) as the
  shaping pressure, not *world/survival consequence*, and almost nobody measures *whether a
  modality became load-bearing* as the outcome — that combination (pressure-not-bribery +
  vision-ablation sensitivity) is where AB is differentiated.

## RECENCY 2024–2026
- *MIMo v2 / MIMo grows!* (2025, arXiv:2509.09805) — same lab/substrate; growing body +
  developing acuity + delays directly model proprio-first→vision-later. SOTA dev-infant sim.
- Wei/Hu et al., *Diagnosing & Re-learning for Balanced Multimodal Learning* (2024, ECCV).
  https://dl.acm.org/doi/10.1007/978-3-031-73039-9_5 — modern framing of exactly your
  failure: nets greedily exploit the easy modality, under-learn the other.
- *Asymmetric Reinforcing against Multimodal Representation Bias (ARM)* (2025,
  arXiv:2501.01240). https://arxiv.org/pdf/2501.01240 — reinforces the suppressed modality;
  candidate *fix*.
- *Revisit Modality Imbalance at the Decision Layer* (2025, arXiv:2510.14411) + *See-Saw
  Modality Balance* (2025, arXiv:2503.13834) — imbalance lives partly at the
  decision/gradient layer — relevant to your representation-vs-policy-failure fork.
- Gulcehre et al. probing (2024, RLC) — validates your linear-decode probe.
- Enactive robotics (live): Egbert et al. (2022, Front Neurorobotics)
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9810814/; *Artificial enactive inference in a
  3-D world* (2024, Cognitive Systems Research).
- Net: no reversal. The 2024–26 modality-balance literature *confirms* AB's phenomenon and
  offers diagnostic/intervention tools, but lives in supervised classification, never frames
  the easy modality as proprioception, and never uses vision-ablation sensitivity in
  RL-under-pressure. MIMo v2 is the one piece overlapping your substrate.

## WHAT TO STEAL (ranked)
1. **Privileged-teacher → vision-student distillation** (Learning by Cheating; RMA;
   Lee/Hwangbo) — the field's actual recognized way to make vision load-bearing when proprio
   already works. Highest leverage against "integrated but inert."
2. **Auxiliary ball-position decode loss on the vision encoder** (Q2 diagnosis + OGM-GE
   rebalancing) — force the latent to encode direction, re-test whether ablation becomes
   *directional* (off-center) vs the content-free ecc=0 spike.
3. **MIMo / MIMo v2 as platform + baseline** — proprio-only reach is a ready stage-1
   comparison; developing-acuity + procedural rooms fit your curriculum.
4. **DroQ + high UTD + constrained action space** — you already use these; the citation
   chain legitimizes the minimal recipe.
5. **Linear-probe-as-diagnostic** (Gulcehre 2024) — keep using it; validated, publishable.
6. **Toys4K + MuJoCo Playground** — object content + GPU throughput when you scale vision.
7. **Grasp-success-vs-geometry protocol** (Dex-Net / pose-benchmark line).

## WHERE AB MIGHT BE NOVEL
- **"Vision load-bearing on ablation yet non-encoding on a linear probe" in an RL agent** —
  appears genuinely unnamed. Strongest novelty candidate.
- **Deliberate proprio-first → vision-later curriculum with load-bearing as the measured
  outcome** — components exist, the combination doesn't.
- **Remember an out-of-body target and reach from proprio+touch with no camera ever in the
  loop** (Q6b) — largely open.
- **Survival/world-consequence ("pressure not bribery") as the shaping signal** vs
  intrinsic-motivation proxies — under-explored.
- **Controlled "minimum sensor set / minimal reward" ablation** — the field minimizes
  algorithm and human effort, not sensors/reward.
- Stop-reinventing warnings: proprio-only reach in isolation (MIMo does it), DroQ/UTD recipe
  (settled), category-agnostic grasp *generalization* (solved — don't claim
  geometry-independence).

## CONFIDENCE & GAPS
- **Sure of:** modality-dominance is the closest named neighbor to "vision inert," but AB's
  failure is more severe (representation failure, not mere imbalance); privileged
  distillation is the recognized fix; MIMo v2 is the best-matched substrate;
  DroQ/constrained-action is the validated minimal recipe; "object-agnostic grasping" holds
  only weakly and AB's strong claim is overstated; the Taylor set-theoretic framing is
  idiosyncratic/unindexed.
- **Less sure of:** whether any very-recent/unpublished work reports AB's exact
  ablation-yes/probe-no dissociation (found none); MIMo v2's validated-results content
  (features verified, outcomes thinner); two Q5 blind-tactile-RL IDs (2510.21609 /
  2606.11767) not independently confirmed (excluded from load-bearing claims). Note: several
  2026 arXiv IDs (26xx.*) are real and within the current date, just past the assistant's
  training cutoff — the load-bearing ones (Lee 2026 blind grasping; Hsu 2022; Murali 2018)
  were verified directly.
- **Deeper follow-up should cover:** (1) RL papers that linear-probe a vision latent against
  a proprioceptive control (AB's exact method); (2) the MIMo eye-hand-coordination paper
  (López/Shi/Triesch 2023, ICDL [17]) in full — nearest published thing to AB's
  proprio↔vision integration goal; (3) whether any enactivist-robotics group has
  operationalized "load-bearing modality" as a metric.
