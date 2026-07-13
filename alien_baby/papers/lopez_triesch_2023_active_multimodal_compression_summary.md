# López, Shi & Triesch (2023) — Eye-Hand Coordination Develops from Active Multimodal Compression

**Source status — READ THIS FIRST.** The full paper (2023 IEEE ICDL, pp. 437–442,
DOI at https://ieeexplore.ieee.org/document/10364414/) is **IEEE-paywalled with no
free PDF** as of 2026-07-12. This summary is assembled from the published **abstract**
(retrieved verbatim via web search) plus a description of the **Active Efficient
Coding** framework the paper extends. It is therefore a **faithful reconstruction, not
a full-text read** — treat method micro-details as indicative, and pull the PDF via
institutional access before quoting numbers. The freely-available companion from the
same lab (**"MIMo grows!"**, arXiv:2509.09805, stored locally as `2509.09805v1.pdf`)
covers the same body/substrate in depth and is the right thing to read in full.

Why it matters to AB: this is **the closest single paper to AB's staging thesis**,
run on the **same infant-model substrate**, and it already demonstrates that
developmental *order* changes what the combined representation is worth.

## What the paper does (from the abstract)
- Presents a **multimodal, hierarchical extension of the Active Efficient Coding
  (AEC) framework** to learn **eye–hand coordination** in an embodied infant model.
- The model **actively compresses visual and proprioceptive inputs into a combined
  multimodal representation**, and **learns eye movements to track an object held in
  its own hand** (self-generated target — the hand it controls).
- Headline result: the abstract multimodal representation **improves tracking
  accuracy only if it emerges *after* the single-modality (vision-only,
  proprioception-only) systems are already established** — a **"less-is-more"**
  effect for developing coordinated multimodal sensorimotor behavior.

## Active Efficient Coding (the framework it extends)
AEC is Triesch's line combining two ideas into one self-supervised loop:
- **Efficient coding** — the perceptual system learns a *compressed* (sparse /
  predictive) encoding of its sensory input, minimizing a reconstruction/coding cost.
- **Active perception** — the agent's *actions* (here, eye movements) are themselves
  learned, via reinforcement learning, to make the sensory input *easier to encode*
  (lower coding cost). Perception and behavior are optimized jointly: the eye moves so
  that what it sees is well-compressed, and the encoder adapts to what the eye brings.
No external labels or reward-for-task-success is needed — the "reward" is coding
efficiency itself. The multimodal extension adds a proprioceptive stream and a
**combined** code on top of the per-modality codes.

## The staging result, in AB's terms
Building the combined vision+proprioception representation **too early** (before each
single-modality system is settled) **hurts** — the joint code is worth having only
once the parts are in place. That is a direct, quantitative existence proof of AB's
own claim that **developmental order changes the internal representation**, on the
same MIMo-style body. Two consequences for us:
- The **bare claim "staging helps" is not novel** — cite this paper for it.
- It is also a **caution**: it predicts our own premature-stage failures (exactly the
  "don't be premature" concern), and gives an independent reason to gate each stage.

## Where AB is still differentiated (honest)
- **Mechanism of the staging metric.** López measures staging via **tracking accuracy**
  (a performance score). AB's proposed contribution is to measure the *representational*
  event directly — has vision **entered proprioception's representation** (CKA /
  linear probe / ablation sensitivity) — i.e. Taylor-style *interpenetration*, not
  just "staged scores higher." That measurement appears unclaimed.
- **Driver of learning.** AEC's driver is **coding efficiency** (compression); AB's is
  a **seen-vs-contacted mismatch / task-grounded** signal. Related in spirit
  (self-supervised, action-in-the-loop) but not the same objective.
- **Goal.** López stops at eye–hand coordination; AB aims the same substrate at prism
  realignment and, eventually, language grounding.

## To do
- [ ] Pull the full PDF via institutional/IEEE access and replace the reconstructed
      method details with the real ones (architecture, RL algorithm, exact staging
      protocol and numbers).
- [ ] Read `2509.09805v1.pdf` ("MIMo grows!") in full — same lab, freely available,
      the fuller substrate reference.
- [ ] Decide the interpenetration metric (CKA between modality manifolds vs. decode
      probe vs. ablation sensitivity) that would make AB's staging result *about the
      representation*, not just the score — the differentiator above.

## Sources
- Paper (paywalled) — https://ieeexplore.ieee.org/document/10364414/
- Abstract via search; AEC framework from the Frankfurt (FIAS) Triesch group line.
- Free companion, same lab/substrate — https://arxiv.org/abs/2509.09805 (local
  `2509.09805v1.pdf`).
