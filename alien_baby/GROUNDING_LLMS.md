# Grounding Language Models in Alien Baby's Perception

**A research program.** Draft 2026-07-09. Author: session with David.
Status: proposal + concrete first experiment. Not yet started.

---

## 1. The one-paragraph thesis

The Alien Baby (AB) project has, for eight months, been testing a theory of perception:
that the senses develop *sequentially* — touch and proprioception first, vision woven in
later — and that a later sense does not get its own module, it gets **interpenetrated** into
the earlier one. We now think the theory is largely right, and AB is the evidence. The next
question is not "is the theory true?" but "**what is it good for in artificial intelligence?**"
This document argues the answer: AB has built perceptual representations that are *grounded*
in a body's sensorimotor loop, and those representations are exactly what today's language
models lack. The program is to **use AB's grounded perception to ground an LLM** — and, in
doing so, to demonstrate a rival to the way AI currently bolts perception onto language.

---

## 2. Why this is a contribution to AI, not just to the theory

Current multimodal AI grounds language in vision by taking a **statically pretrained,
web-contrastive** image encoder (CLIP and its descendants) and attaching it to a language
model through a learned adapter (LLaVA, and essentially every open VLM). The encoder learned
its representation from hundreds of millions of image–caption pairs scraped from the web. It
never had a body, never acted, never had a sense that arrived *after* another and had to be
reconciled with it.

In AB's own vocabulary, **that is the "all-at-once fusion" architecture** — and AB spent its
first phase (v1) showing that this architecture is brittle. The all-at-once agent, given
vision and proprioception simultaneously, collapsed from 95% to 30% success under sensor
noise, while the *staged* agent (proprio first, vision interpenetrated after) degraded
gracefully. Later work (v5) made the alternative precise: with a protected proprio substrate
and a consistency loss, vision came to live *on proprio's manifold* (CKA = 0.998 with the
earlier sense) — additive, not displacing.

So the contribution is not "we grounded an LLM." VLMs already do a version of that. The
contribution is: **current grounding uses the architecture AB proved weak; AB embodies the
developmental, interpenetrated alternative; test whether that alternative produces grounding
that is more robust, more genuinely referential, and more like a human's.** That is a claim
with a control group, and half the control group's result is already in our own logs.

---

## 3. What AB has actually established (the assets we are bringing)

This program stands on results already in `FINDINGS.md`, not on hope. What AB has:

- **A grounded visual representation that reads direction from pixels.** The years-long
  representation failure (bearing-from-pixels decoded at R² ≈ 0.01–0.08) was fixed by the
  **distil-then-RL** recipe: Stage A distilled bearing into the encoder (R² = 0.84), Stage B
  let RL use it. Reward alone (Stage C) could not grow it. *We know how to build a grounded
  perceptual code, and we know it must be built before it can be used.*
- **Vision that is behaviorally load-bearing**, not decorative — the ±68° head-search task
  made orienting necessary; vision moved behavior (ablation gap +12.5 to +22.5, sign-robust
  across three seeds).
- **Genuine visual reference.** The two-ball decoy task produced **above-chance discrimination
  of which object to reach** (63–78%, blind floor 50% by construction, seed-robust 3/3). The
  prism-displacement test showed choice falling *below chance* as the visual field is rotated —
  proof the displaced picture, not mere arousal, steers the reach. AB's vision *refers to
  something in the world and acts on it*.
- **Genuine sensorimotor recalibration.** The prism-adaptation **negative aftereffect** — now
  **two-seed** (s0 near−far +71, s2 +67), with the aftereffect appearing exactly when
  recalibration occurs and absent when it doesn't (frozen encoder, over-gentled run). AB's
  perception–action map is *plastic and self-correcting*, the way a human's is under prism
  goggles.
- **Object-agnostic, learned invariance.** The color discrimination is **shape-invariant
  zero-shot** — trained only on spheres, it discriminates red-from-blue at full strength on
  boxes and capsules. Vision *could* see shape and chose to bind to color instead: invariance
  by learning, not by construction.

And, stated plainly, **what AB does not have**: it grounds a *handful* of invariants —
bearing, distance, toward/away, a color affordance ("approach red") — for a legless crawler
reaching for balls through 32×32 stereo vision. It does not ground "Tuesday," "justice," or a
noun vocabulary. Any honest version of this program has to cross that gap, not paper over it.

---

## 4. The honest gap, and the bridge that makes it tractable

The assumption behind this whole program — *our perceptions are the underpinnings of the
language we use* — is a real position in cognitive science (embodied / grounded cognition:
Lakoff & Johnson, Barsalou, and the Gibsonian lineage the source theory belongs to). But it
has a scope problem: AB grounds a few invariants; human language grounds thousands of concepts.

The bridge is the **image-schema / metaphor-extension** claim. In this literature, abstract
language is not grounded directly — it is *metaphorically extended* from a small set of
bodily-spatial schemas: source–path–goal, near–far, containment, balance, toward–away. "Grasp
an idea," "a close relationship," "where is this heading," "I see what you mean" — the abstract
vocabulary is scaffolded on the sensorimotor core. **If that is right, AB does not need to
ground the whole lexicon. It needs to ground the sensorimotor primitives that the rest is
built from** — and several of those (bearing, toward/away, near/far, reach) it already has as
measured, load-bearing quantities.

Two honesty flags that keep the claim from over-reaching:

1. **Mechanism, not content.** AB perceives like a crawler, not like a human. What could
   transfer is not the *content* of its percepts but the *structure* — the developmental,
   interpenetrated, response-conditioned way the code is organized. Keep the claim at the
   mechanism level.
2. **The bridge is a hypothesis, not a given.** Whether an LLM's spatial vocabulary actually
   aligns to AB's sensorimotor invariants is an empirical question — which is exactly what the
   first experiment (§7) is designed to answer, and a null result there is itself informative.

---

## 5. Why "weights as a governor" doesn't type-check — and the steelman

The tempting mental image is: take the weights AB learned and let them *govern* the LLM, like a
governor on an engine. Literally, this has no defined operation. AB's weights are a policy
network mapping (joint angles, touch bit, stereo pixels) → (torques, head yaw). An LLM's
weights are a transformer over token embeddings. Different input modalities, different
dimensionalities, no shared coordinate system. There is no multiplication or gate that couples
them.

What *does* transfer is not the weight tensors but the **grounded representational structure** —
the invariants AB learned (bearing, distance, toward/away, the recalibration map) and the
consistency constraints that shaped them. The real engineering question is therefore:
**how do you anchor an LLM's concepts to those grounded structures?** The next section gives
four answers, cheapest to most radical.

---

## 6. Four architectures

### (a) Concept-anchoring probe — *do this first; near-zero build*
Measure whether AB's grounded latent space and the LLM's word-activation space share
structure, without changing either model. Take AB's encoder latents for a set of scene states;
take an LLM's hidden activations for the words naming those states' invariants ("to the left,"
"far ahead," "moving toward"); test whether a linear/RSA alignment links them. If AB's "toward"
invariant predicts the LLM's "toward"-word geometry above a permutation baseline, we have first
evidence for the underpinning thesis. **A null is equally interesting**: it says text-trained
spatial words are ungrounded relative to sensorimotor structure — which is the whole motivation
for the bridge. Spec in §7.

### (b) Grounded soft-prompt bridge — *the direct A/B test of the thesis*
Mechanically the standard VLM trick (learn a small adapter projecting a perceptual latent into
the LLM's embedding space as a prefix / soft prompt), with one decisive change: the encoder is
**AB's developmentally-grounded sensorimotor encoder, not CLIP**. Then run the head-to-head no
one has run: **AB-grounded LLM vs. CLIP-grounded LLM** on spatial / affordance / embodied
reasoning. If sensorimotor grounding wins on the robustness and directional-consistency
dimensions AB is strong on, developmental grounding beats contrastive grounding — a real,
publishable claim.

### (c) Governor-as-consistency-critic — *the faithful realization of the intuition*
Make "governor" a *critic*, not a filter. When the LLM emits a claim with sensorimotor content,
AB's grounded model scores whether it is consistent with how an embodied perception–action loop
actually behaves — an RLHF-style reward model whose reward is *embodied-perceptual plausibility*
rather than human preference. Note this is **AB's own consistency loss scaled up**: AB already
enforces "vision must agree with proprio"; the governor enforces "language must agree with
grounded perception." Same primitive, one level higher.

### (d) Foundation — *the theory-faithful endpoint*
"Governor" is a post-hoc constraint on a finished LLM. The thesis — *perception underpins
language* — actually implies the opposite: the grounded system is the **base**, and language is
**interpenetrated onto it** the way vision was interpenetrated onto proprio. Don't connect a
finished LLM to AB; *grow* language on AB's grounded substrate, using the **distil-then-RL**
lesson (build the representation first; the higher process cannot grow it from scratch by its
own objective alone). This is a pointed critique of how LLMs are trained today — language
modeling with no grounded substrate — and it is the hardest, most research-heavy path.

### (e) World-model predictor — *grounded consequences, not just grounded reference*
(Added 2026-07-11, from David's `documents/reframed.docx` — the one genuinely additive idea in
that document.) Architectures (a)–(d) are all *representational*: they test or transfer the
structure of AB's percepts. This one adds a **predictive/causal** dimension: the language layer
must predict the *sensory consequences* of the actions it proposes, trained with a prediction
loss against an embodied forward model ("if I push this ball, it moves ~1 m and hits the wall").
A governor built this way scores claims against *predicted outcomes*, not just representational
plausibility — arguably a stronger constraint than the consistency-critic (c). It has a dormant
sibling already in the project: the deferred MICOA Phase II "anticipatory vision" (vision learns
to predict proprio's next-step distribution — *seeing contact before feeling it*, todo.md).
Cheap first step, analogous to probe (a): train a small forward model on existing AB rollouts
(state + action → next latent/proprio) and test whether AB's grounded latent supports
consequence-prediction at all. If it does, (e) becomes a real candidate; if not, that bounds
what the current substrate can govern.

**A cautionary note on external proposals (2026-07-11).** The other integration ideas in
`reframed.docx` — and, in our experience, the default shape of outside embodiment-LLM proposals
generally — converge on concatenating text + vision + proprio into a single state vector. That
is precisely the **all-at-once fusion** architecture AB's own v1 result refuted (95% → 30%
collapse under sensor noise, vs. graceful degradation for the staged/interpenetrated agent).
That the field's default is the architecture our findings argue against is part of this
program's contribution claim, and worth saying explicitly when writing any of this up.

---

## 7. The concept-anchoring probe, concretely (the recommended first step)

Zero GPU beyond one forward pass through an open LLM; uses AB checkpoints already on disk.

**AB side.** Load a grounded encoder (the Stage-A distilled `bearing_cnn`, or a decoy-policy
encoder). For a grid of scene states — ball at bearings across ±68°, at 2–3 distances, on
left/right — record: (i) the encoder latent (or the bearing slot), and (ii) the scalar
ground-truth invariants for that state: bearing θ, distance d, sign of toward/away, color.

**LLM side — a LADDER of embedding systems, not one model (see "Is it embedding-dependent?"
below).** For a small controlled vocabulary describing those same states — spatial prepositions
and motion verbs ("left," "right," "ahead," "near," "far," "toward," "away," "reach left")
embedded in minimal carrier sentences — extract a text representation from *several* systems and
compare the alignment across them:
- a sentence embedding (all-mpnet-base-v2, cached locally; and/or the OpenAI embeddings API);
- text-only LLM hidden activations at a mid layer, at ≥2 scales (Qwen2.5-7B; Llama-class);
- a **vision-grounded** text encoder as a deliberate contrast (Qwen2.5-VL-7B's text pathway) —
  its language has already been exposed to images, so it is the "contaminated positive" that
  tells us whether any alignment rides on prior visual grounding rather than language alone.

**Is it embedding-dependent? (yes — and that is designed in, not a flaw).** The alignment score
*will* vary by embedding system, because different systems carry different amounts and kinds of
spatial structure. So a single model gives an ambiguous answer (a null could mean "language isn't
sensorimotor-grounded" or just "this model is too weak to have the structure"). The scientifically
meaningful result is therefore the **pattern across the ladder**, not any one number:
- **RSA is the primary metric precisely because it is basis-independent** — it compares the
  *geometry* (dissimilarity structure) of the two spaces, not raw vectors, so it abstracts away
  each embedding's idiosyncratic dimensionality and scaling.
- **Does alignment scale with model capability?** A signal that grows with size/quality is
  evidence of real recovered structure; one that is flat or random across models is an artifact.
- **Text-only vs. vision-grounded contrast.** If alignment appears only in the vision-grounded
  encoder (Qwen-VL) and not in text-only models, then language-alone does *not* recover
  sensorimotor structure — which supports the whole premise and points straight at the bridge/
  foundation. If it appears in text-only models too, the metaphor-extension thesis gets direct
  support.
- **A highly embedding-dependent or generally weak result is itself a finding**, not a failure:
  it says grounding-to-sensorimotor-structure is not a stable property current LLMs have, which is
  exactly the deficiency this program exists to fix. We would only be *surprised* by a strong,
  consistent alignment in pure text models — that would undercut the premise that text needs
  grounding at all.
Nuisance parameters (layer, last-token vs mean-pool, carrier-sentence template) are swept and
reported as robustness bands, not fixed by fiat.

**Alignment test.** Two complementary measures, both cross-validated with permutation
baselines:
- **Representational Similarity Analysis (RSA):** build a dissimilarity matrix over AB states
  (by their invariants) and over the LLM word-activations; correlate the two matrices. A
  significant positive correlation means the *geometry* of AB's sensorimotor space matches the
  *geometry* of the LLM's spatial-language space.
- **Linear decode:** can a cross-validated linear map from AB's latent predict the LLM's
  activation for the matching word (and vice versa) above the permutation null?

**Read-outs.**
- *Positive & structured* (holds across multiple invariants, not just a single angle) → first
  evidence that the LLM's spatial vocabulary is anchored to sensorimotor structure AB has —
  green-light the soft-prompt bridge (b).
- *Null* → text-trained spatial words are ungrounded relative to AB's code → motivates (b)/(d)
  even more strongly (grounding must be *added*, not read off).

**Controls / honesty.** A trivial positive is possible if both spaces just encode a 1-D angle;
that is why the test spans several invariants and uses RSA, not a single regression. The
vocabulary is tiny — this is a proof-of-concept of the *bridge*, not a demonstration of full
grounding. Report exactly what was tested and what was not.

---

## 8. The decision that governs the whole program: governor vs. foundation

Everything above forks on one question. **Governor** (c) is a constraint applied to a
pretrained LLM — easy to prototype with today's tools, gives signal fast, but leaves the LLM's
ungrounded core intact. **Foundation** (d) makes grounded perception the base and grows
language on top — the theory-faithful reading of "perception underpins language," and the far
harder build.

Recommendation: **prototype the governor to get signal, but write the program's north star as
foundation.** The probe (a) and bridge (b) are the same regardless of which endpoint we commit
to, so they are the right first two moves either way.

**DECISION (2026-07-09, David).** Start with the **governor, then see**; **lean foundation** as
the eventual endpoint. The key refinement: these are *not* mutually exclusive, and both can work —
with different ceilings and failure modes. A governor is a correction applied at the surface of a
finished LLM; its ceiling is bounded by what the LLM already latently represents (it can reweight,
veto, amplify — it cannot install a representation the model never had). Foundation changes what
representations exist at all, so its ceiling is higher — **but only to the extent the grounded
substrate is broad.** There is a real regime where foundation is *worse*: grow language on too
narrow a substrate (only toward/away/red) and you get a stunted speaker starved of the breadth
that text-scale pretraining gives for free. So "foundation will be better" is a conditional bet:
it pays off exactly insofar as the perception program keeps widening what AB grounds — which is a
direct incentive to keep enriching AB, and ties the two halves of the project together. The most
likely real-world winner is a **hybrid**: text-pretrained breadth with a grounded core grown/
fine-tuned on the substrate — neither pure governor nor pure foundation. Governor-first is the
right de-risking order regardless of which ultimately wins, because it is the cheapest way to find
out whether grounded structure carries actionable signal at all before investing in growing it.

---

## 9. Recommended sequence

1. **Concept-anchoring probe (a)** — §7. Near-zero cost; falsifiable; tells us if the bridge is
   real before we invest. This is the embodied-cognition analogue of the free re-analysis
   diagnostics we run before committing compute.
2. **Grounded soft-prompt bridge (b)** — if (a) is positive. Build the AB-encoder adapter, run
   AB-grounded vs. CLIP-grounded head-to-head. This is the experiment that turns the thesis into
   a result.
3. **Governor-as-consistency-critic (c)** — once (b) shows the grounded channel carries signal
   an LLM can use.
4. **Foundation (d)** — the long-horizon endpoint; only after (b)/(c) establish that grounded
   structure transfers at all.

---

## 10. What would falsify the thesis

State it up front so the program can lose honestly:

- If the probe (a) finds **no** alignment between AB's invariants and any LLM's spatial
  vocabulary — and the bridge (b) then shows AB-grounding is **no better** than CLIP-grounding
  on embodied reasoning — the strong claim ("sensorimotor grounding is the better substrate for
  language") is dead, and we would have learned that web-contrastive grounding already captures
  whatever spatial structure language needs.
- If AB-grounding helps only on tasks whose answers are trivially readable from the invariants
  AB happens to expose (bearing/distance), and not on anything requiring composition or
  transfer, then it is a narrow sensor, not a grounding substrate.

Neither outcome is embarrassing; both are worth knowing.

---

## 11. Relation to the rest of the project

This program does **not** require pausing the perception work. It *consumes* it: every AB
result that makes vision more grounded (recalibration, object-agnosticism, richer invariants,
eventually locomotion) is a stronger encoder for the bridge. The clean division of labor:
`FINDINGS.md` keeps making the perception more genuinely grounded; this document is the plan for
what to do with that grounding once we have enough of it. The natural trigger to start is now —
the probe (a) is cheap and answers the load-bearing question before any large build.

---

## First probe results (v1, 2026-07-09) — the bridge is REAL but PHRASING-CONTINGENT

Ran the concept-anchoring probe (§7) end-to-end: AB Stage-A grounded latents (1039 in-view
states) vs. a 4-backend ladder × grid + both isolated-axis methods. Files:
`grounding/probe_ab_extract.py`, `grounding/probe_align.py`, `results/grounding/*`.

**What AB grounds:** its latent encodes **bearing (RSA +0.32) and essentially not distance
(+0.02)** — expected for a bearing CNN.

**Which axis each text model encodes varies enormously** (this is the embedding-dependence, made
concrete): sentence-embedding models (all-mpnet, OpenAI) and CLIP encode *distance* words
strongly (0.58–0.73) and bearing weakly; the contextual **text LLM (Qwen-1.5B hidden states)** is
the only one that encodes *bearing* (0.59). So which axis can even align depends on the model.

**Joint (grid) alignment is weak** — mpnet +0.07, OpenAI +0.11, **Qwen +0.22 (perm-p≈0.05, the
only ~significant one — because it is the one that represents bearing)**, CLIP +0.04. The
**vision-grounded CLIP did NOT align better** → the (weak) alignment is not riding on prior visual
exposure.

**The load-bearing finding — the probe is descriptor-dependent, not just model-dependent.** The
isolated-bearing alignment for the *identical* AB geometry swings from −0.37 to +0.58 purely with
phrasing: "far to my left…far to my right" (shared intensity word "far" folds the extremes) gives
**−0.37**; ordinal **"9 o'clock…3 o'clock" gives +0.58 (mpnet, p=0.016)**, Qwen +0.38; signed
degrees +0.19–0.24. So a genuine positive alignment between grounded bearing and language **exists,
but only when the linguistic encoding of direction is itself geometrically ordinal** (clock
positions); phrasings organized by intensity modifiers invert it.

**Verdict (v1, modest power):** the bridge is real but *conditional and fragile* — grounded
sensorimotor bearing structure aligns with language when language is geometrically faithful to it,
and is masked/inverted otherwise. That is neither "grounding is already in text" nor "no bridge";
it is "the bridge is phrasing-contingent," which is itself the interesting result and argues for
*building* grounding (bridge/foundation) rather than assuming it. Consistent with the program's
premise.

**Caveats / next:** left-skewed spawn (far-right bin empty → fix to symmetric coverage); single
AB encoder (add a decoy-policy encoder); sweep descriptor vocabularies systematically and report
the distribution rather than one phrasing; sweep Qwen layer/pooling; add a larger text LLM rung
for the scale-trend. The green-light criterion for building the bridge (b) is a *consistent*
positive alignment under faithful descriptors across the ladder — partially met (mpnet clock
p=0.016; Qwen positive), not yet decisive.

### Meaning-cloud test (2026-07-09) — averaging over phrasings does NOT recover meaning; the embeddings encode WORDS

Tested David's "meaning, not words" proposal directly (`grounding/probe_meaning.py`): represent
each bearing concept by a CLOUD of meaning-equivalent phrasings from 4 lexical families
(intensity / clock / degrees / casual) and use the centroid as the "meaning vector". Result, RSA
vs AB's grounded bearing geometry, all 4 backends:

| family | mpnet | openai | qwen | clip |
|---|---|---|---|---|
| intensity ("far to my left") | −0.37 | −0.32 | −0.26 | −0.21 |
| **clock ("9 o'clock")** | **+0.58** | +0.42 | +0.37 | **+0.77** |
| degrees ("−68°") | +0.04 | +0.09 | +0.59 | +0.42 |
| casual ("way over on my left") | −0.22 | −0.28 | −0.25 | −0.21 |
| **CENTROID of all** | **−0.30** | −0.28 | −0.06 | −0.11 |

**The centroid is negative/near-zero — averaging FAILED to recover meaning.** Reason: the centroid
of embedding *vectors* is not the centroid of *meanings*. The families impose qualitatively
different geometries — clock/degrees are ordinal (far-left ↔ right maximally distant, matching AB);
intensity/casual **fold** (they key on the modifier "far"/"slightly"/"a little", so left and right
extremes land *close*). Averaging mixes folded and ordinal geometries; it does not cancel to the
meaning, because the individual word-geometries are themselves not meaning-faithful.

**Interpretation — this supports David's premise, not the naive method.** "Meaning is general"
(invariant across phrasings) is exactly the property GROUNDING has and these language embeddings
LACK: AB puts far-left and far-right maximally apart because that is a spatial fact; the embeddings
put "far to my left" and "far to my right" *close* because they share the word "far". The models
represent surface-form geometry, not abstracted spatial meaning — except when the notation is
*accidentally* geometric (clock), where the words do the geometric work. So the invariant meaning
lives in AB's grounded representation, and off-the-shelf text embeddings do not supply it by
averaging. This is direct evidence for the "we may need to build our own embeddings" path
(= foundation): grounding is where the phrasing-invariant meaning geometry actually resides.
Caveat: naive mean is one aggregation; a smarter meaning-extraction that *selects/weights* faithful
phrasings could do better — but knowing which are faithful needs the grounded reference, which is
the point.

## Appendix — open questions for David

- **Which LLM** for the probe? A small local open model keeps it self-contained and cheap;
  worth fixing one before starting.
- **Which AB encoder** to probe first — the Stage-A distilled bearing CNN (cleanest, most
  purely "grounded direction") or a decoy-policy encoder (richer, includes the color
  affordance)? I lean Stage-A for the first pass, decoy for the second.
- **Governor vs. foundation** as the stated north star (§8) — a genuine strategic call, and the
  one place I most want your steer.

## Threads to pick up next session — building our own embeddings (David's ideas)

The meaning-cloud result pushed us toward "build our own embeddings" (the foundation path).
David has two concepts for HOW to build grounded embeddings, to be reintroduced/expanded next
session. Captured here so they aren't lost; the one-line gloss is my *tentative* read and should
be corrected by David, not treated as settled.

1. **Atomization of meaning** (David has raised this before). Tentative read: decompose meaning
   into primitive semantic *atoms* — grounded primitives — from which complex meanings are
   composed, rather than treating a word/phrase embedding as an unanalyzed whole. Natural fit with
   the image-schema core (§4) and with why the naive paraphrase-centroid failed: you don't average
   surface forms, you compose from grounded atoms. NEEDS David's actual definition.

2. **The physics of meaning** (raised 2026-07-09). Tentative read: meaning has law-like dynamics —
   positions, forces, invariants/symmetries — the way the AB grounded representation is literally
   shaped by sensorimotor *physics*. Possibly: an embedding space whose geometry obeys a "physics"
   inherited from the grounded world (so far-left/far-right are opposite by law, not by lexical
   accident). NEEDS David's actual definition.

Both connect to the same target: an embedding whose geometry carries the phrasing-invariant MEANING
that §"Meaning-cloud test" showed off-the-shelf text embeddings lack. Open question for the
discussion: do these two ideas compose (atoms = the elements, physics = the laws relating them)?
