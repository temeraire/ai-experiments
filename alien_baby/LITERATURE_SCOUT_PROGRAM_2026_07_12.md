# Prior-art map — the whole AB *program* (2026-07-12)

Cross-disciplinary literature-scout pass answering the plain question: **"Has the
work I am doing already been done?"** — searched across developmental/epigenetic
robotics, motor-control & psychophysics, computational neuroscience, active
inference / predictive coding, enactivism / sensorimotor-contingency theory,
symbol grounding, and RL/robotics. Complements the earlier problem-level scout
(`LITERATURE_SCOUT_2026_06_16.md`); this one is at the program/thesis level.

**One-line answer:** the individual *ideas* in the program are each an established,
named research line — you should cite them, not claim them. The genuinely
under-explored part is a specific *combination* plus a *measurement discipline*.
Read this as good news: most of the machinery is known and borrowable.

---

## Verdict by component

### 1. A creature that learns to see by moving and touching — SOLVED (established field)
This is **developmental / epigenetic robotics** and **sensorimotor-contingency
theory**. AB is a new instance, not a new idea.
- Lungarella, Metta, Pfeifer & Sandini, *Developmental robotics: a survey* (2003,
  Connection Science).
- O'Regan & Noë, *A sensorimotor account of vision and visual consciousness*
  (2001, BBS); robot implementations reviewed in *Sensorimotor Contingencies as a
  Key Drive of Development: From Babies to Robots* (2019, Front. Neurorobotics).
- iCub eye–hand coordination from body babbling (Metta/Natale/Sandini lab).
- **Same infant-body substrate as us:** López, Shi & Triesch and the Frankfurt
  (FIAS) Active Efficient Coding line — see component 4.
- **Frame as:** "we instantiate developmental-robotics / sensorimotor-contingency
  learning on a MuJoCo infant body." Our differentiator is the *measurement*
  (vision-ablation sensitivity + linear decode probe as the pass/fail on "did
  vision become load-bearing"), which this field mostly does not do.

### 2. Prism / visuomotor-rotation adaptation in a *learned* embodied controller — PARTIALLY SOLVED (closer prior art than first thought — see correction)
Prism adaptation in humans is fully solved (not our claim). What we care about is a
*learned controller* that recalibrates vision under a displacement.
- Computational-neuroscience models of visuomotor-rotation adaptation reproduce
  aftereffects via population coding + gradient descent (e.g. *Adaptation to
  Visuomotor Rotation…* 2009; *De novo motor learning…* Nature Comms 2024) — but
  these are abstract reaching models, not embodied agents with a camera.
- Robot self-perception recalibrates under vision–proprioception conflict via
  **active inference** (Lanillos et al. 2021; rubber-hand-illusion robot, Patterns
  2023) — the sibling phenomenon, not framed as prism-with-aftereffect.
- **CORRECTION to the scout's "thin-to-absent" (found on close reading of
  Loquercio 2022 — see `papers/loquercio_2022_CMS_method_notes.md`):** that paper
  *does* run an explicit **prism-adaptation test on a real quadruped** (their §V
  "Visual Plasticity"). They rotate the camera ~30° in yaw (their stand-in for
  prism wedges), run the classic **pre-test → exposure → adaptation → post-test**
  protocol, and the policy re-learns to walk straight after ~80 s of data by
  fine-tuning the vision module on the cross-modal error. So "prism adaptation in a
  learned robot controller" **has been done** — the exact demonstration exists.
- **Where AB still has room (honest, narrowed):** Loquercio (a) uses a camera
  *rotation*, not a lateral field *displacement*; (b) recalibrates by *supervised
  fine-tuning of a terrain-height predictor*, not RL end-to-end; (c) does **not
  quantify a negative aftereffect** (only notes the post-test re-adjustment takes
  ~two trials); and crucially (d) does **not** distinguish genuine recalibration
  from strategy-substitution ("policy stopped using vision, fell back on a habit").
  AB's session result — that our creature did the *substitution*, not recalibration,
  and the machinery to *tell them apart* (heading-tracks-ball vs blind control,
  residual-slope across offsets) — is the part that is not in Loquercio.
- **Frame as:** "prism adaptation in a learned robot controller has been shown
  (Loquercio 2022); our contribution is the RL-from-pixels version and, above all,
  the recalibration-vs-habit-substitution measurement + aftereffect quantification."
  The aftereffect-as-proof-of-recalibration logic is textbook motor control
  (Redding & Wallace; Barrett 2012) — cite it, don't re-derive it.

### 3. Teach the eye from what the body felt on contact (the mismatch signal) — SOLVED (established mechanism)
This is the most important "stop reinventing" finding. The seen-versus-felt signal
we built this session is **cross-modal supervision**, already working label-free on
a real robot.
- **Loquercio, Kumar & Malik, *Learning Visual Locomotion with Cross-Modal
  Supervision* (2022)** — trains the vision module using **time-shifted
  proprioception** as the supervisory target: "I see point A ahead whose height is
  unknown; when my feet reach A, proprioception tells me its height — use that to
  teach what the eye should have predicted." That is exactly "use contact to teach
  the eye." Full method notes: `papers/loquercio_2022_CMS_method_notes.md`; PDF:
  `papers/loquercio_2022_cross_modal_visual_locomotion.pdf`.
- Body-schema cross-modal map learning: Yoshikawa, Hosoda & Asada.
- Prediction-error / active-inference recalibration: Lanillos & Cheng 2018; survey
  *Sensorimotor representation learning for an "active self" in robots* (2021).
- **Frame as:** "we apply cross-modal (proprioceptive/contact → visual)
  supervision, cf. Loquercio 2022." **Steal their time-shift trick.** Our open edge
  is doing it as a *developmental stage* and reading out load-bearing-ness, not the
  error signal itself.

### 4. Staged development / "interpenetration" (touch first, vision woven in) — PARTIALLY SOLVED (near-direct existence proof on our own substrate)
- **López, Shi & Triesch, *Eye-Hand Coordination Develops from Active Multimodal
  Compression* (ICDL 2023).** On the *same infant-model substrate*, the combined
  multimodal representation improves performance **only if it emerges after the
  single-modality systems are established** — an explicit **"less-is-more" staging
  effect.** This is almost exactly our "developmental order changes the internal
  representation," already demonstrated. Summary:
  `papers/lopez_triesch_2023_active_multimodal_compression_summary.md`.
- The **bare claim "staging helps" is therefore not new** — cite López 2023.
- Recency counter-voice worth citing for balance: *Position: Multimodal LLMs Should
  Learn from Children* (CVPR-W 2026) argues modalities unfold *together*, mildly
  against strict proprio-first ordering.
- **Genuinely unclaimed slice:** *measuring* interpenetration directly — using
  CKA / linear-probe / ablation to show vision entered proprioception's
  representation — rather than only showing staged training scores higher. Taylor's
  set-theoretic "interpenetration" framing (Ch. 5) is ours; nobody in ML/robotics
  operationalizes it as a measurement.

### 5. Grow language on an action-grounded substrate — PARTIALLY SOLVED (old idea; premise now empirically confirmed; our exact recipe untested)
- The idea is named and 35 years old: Harnad, *The Symbol Grounding Problem* (1990).
- Attempted on robots: Marocco, Cangelosi, Fischer & Belpaeme, *Grounding action
  words… simulated iCub* (2010, Front. Neurorobotics); Steels' language-game /
  physical-symbol-grounding program; Deb Roy's grounded language acquisition.
- **Our core premise is now empirically confirmed:** Xu et al., *Large language
  models without grounding recover non-sensorimotor but not sensorimotor features
  of human concepts* (2025, Nature Human Behaviour) — text-only LLMs match humans on
  abstract concepts but diverge most on **sensory/motor** ones. **Best single
  citation for "text embeddings lack the sensorimotor structure."** (The premise is
  therefore not ours to claim either.)
- Modern "bolt language onto perception" (PaLM-E 2023) is the *inverse* of our plan
  (pretrained vision + LLM, not grown-from-scratch).
- **Genuinely under-explored:** language grown on a **from-scratch,
  proprioception-first** substrate rather than bolted onto a pretrained encoder.
  An untested **bet**, reported with humility — not a result.

---

## Bottom line

**Stop framing as new / start citing:**
- "learning to see through action" → developmental robotics (Lungarella 2003).
- "use touch/proprio to teach vision" → cross-modal supervision (Loquercio 2022).
- "staging modalities helps" → López, Shi & Triesch 2023 (our substrate).
- "prism adaptation in a learned robot" → Loquercio 2022 §V.
- "ground language in sensorimotor experience" → Harnad 1990 / Steels / Roy /
  Cangelosi 2010; premise confirmed by Xu et al. 2025.

**The genuine open space (narrow but real):**
1. Prism **recalibration-vs-habit-substitution** distinguished in an end-to-end RL,
   from-pixels controller — including AB's negative result — with a **quantified
   aftereffect**. Loquercio did the prism test but not this distinction.
2. **Measuring interpenetration** directly (CKA/probe/ablation), not inferring it
   from scores.
3. **Load-bearing-ness as the outcome variable** — "did this sense become causally
   necessary?" — which developmental robotics builds systems but rarely tests.
4. Language grown from a **proprioception-first, from-scratch** substrate — a bet.

**Honesty caveats (from the scout):** absence of a clean citation is not proof of
novelty. The enactivist-robotics and cerebellar-internal-model literatures were not
exhaustively searched; a buried prism-aftereffect reproduction could exist there.

---

## What to steal, ranked
1. **Cross-modal (proprio→vision) supervision — Loquercio 2022.** Reframes our
   mismatch signal as established method; borrow the **time-shift** trick.
2. **López, Shi & Triesch 2023 "active multimodal compression."** Same substrate;
   "less-is-more" staging is both citation and template for measuring ordering.
3. **Xu et al. 2025 (Nature Human Behaviour).** Empirical anchor for the language
   premise.
4. **Active-inference robot self-perception (Lanillos).** Principled prediction-error
   alternative to reward-driven grounding; the natural language for our seen-vs-felt
   loop.
5. **Visuomotor-rotation computational models + Redding & Wallace prism framework.**
   Vocabulary and the aftereffect diagnostic.
6. **Marocco & Cangelosi 2010 iCub action-word grounding.** Closest prior attempt to
   grow language on a robot's own sensorimotor loop.

## Key sources
- Lungarella et al. 2003 — https://www.tandfonline.com/doi/abs/10.1080/09540090310001655110
- Sensorimotor contingencies in robots (2019) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6904889/
- O'Regan & Noë 2001 — https://www.researchgate.net/publication/11150760
- López, Shi & Triesch 2023 (ICDL, paywalled) — https://ieeexplore.ieee.org/document/10364414/
- MIMo grows! (2025, same lab, free, local `2509.09805v1.pdf`) — https://arxiv.org/abs/2509.09805
- Loquercio, Kumar & Malik 2022 — https://arxiv.org/abs/2211.03785 · project https://antonilo.github.io/vision_locomotion/
- Lanillos et al. 2021 — https://arxiv.org/abs/2105.04261
- Visuomotor-rotation model 2024 — https://www.nature.com/articles/s41467-024-48008-7
- Harnad 1990 — https://arxiv.org/abs/cs/9906002
- Marocco & Cangelosi 2010 — https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2010.00007/full
- Xu et al. 2025 (Nature Human Behaviour) — https://www.nature.com/articles/s41562-025-02203-8
