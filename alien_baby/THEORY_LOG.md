# Theory Log — Alien Baby Project

This log tracks what each experiment has confirmed or refuted about *why AB is not using vision.*

---

## 2026-05-07 — Architecture Review: Serial Training Retired, Dialogue Architecture Proposed

### What the video revealed

Direct video observation of the stage1_v8_best checkpoint (the 60% blind proprio policy — the best result this project has produced) exposed two problems that the touch-rate metric had entirely concealed:

**Problem 1 — The creature is not searching.** On 4 of 5 rendered episodes the creature was nearly stationary for the full episode. The one touch (seed 3) occurred because the ball spawned adjacent to the creature, not because the creature moved to find it. The 60% blind touch rate was not produced by a creature that covers ground and finds the ball — it was produced by a creature that sits in one place and occasionally happens to be sitting near where the ball spawned.

**Problem 2 — The head is in continuous panic.** The head oscillates across its full angular range (±86° pan, ±46° tilt) every few timesteps. The position actuator with `kp=30` snaps the head to any commanded position in a single timestep; a policy that commands full-range positions every step produces this behavior. The resulting visual input is a chaotic, temporally uncorrelated sequence of frames that no vision network can extract stable structure from.

Both problems are invisible to the touch-rate metric. A creature that sits still can achieve 60% if the spawn radius is small enough that many balls appear within arm's reach. A creature whose head is thrashing produces a corrupt visual signal that looks normal in aggregate statistics (pixel values are nonzero, ball appears in frame) but carries no temporal coherence.

### Why this invalidates the serial approach

The serial approach — train blind proprio first, then freeze and add vision — inherits whatever the blind policy learned. If the blind policy learned "sit still and let the ball come to you," then every vision follow-on starts from that basis. Vision cannot rescue a locomotion strategy that doesn't locomote. And vision cannot learn from a head that is constantly thrashing.

This is a deeper failure than the MICOA violations documented in prior entries. Those entries assumed the blind policy was doing something behaviorally coherent that vision was then destroying. The video reveals that the blind policy's behavior was incoherent to begin with. The 60% touch rate was a statistical artifact of spawn radius, not a learned strategy worth preserving.

### The new theoretical framing: dialogue, not hierarchy

Prior work treated the integration question as hierarchical: proprio establishes first, vision confirms. The MICOA freeze architecture tried to enforce this — freeze proprio, train vision to speak in proprio's language. This presumes proprio has something worth preserving. The video makes clear that what proprio learned is not worth preserving at the behavioral level, even if the touch rate appeared acceptable.

The new framing rejects hierarchy entirely. Vision and proprio should develop simultaneously as **equal peers** that must reach agreement. Neither establishes first. Neither protects its weights from the other. The training regime enforces agreement — not by freezing one channel, but by creating a structural representation where both channels must converge on the same answer.

Formally: let `a_p = f_p(proprio)` and `a_v = f_v(pixels)` be two independent action proposals. The final action is `a = w·a_p + (1-w)·a_v` where `w` is a learned context-dependent weight. A consistency loss `λ·||a_p - a_v||²` penalizes disagreement. Both channels train jointly. The disagreement `||a_p - a_v||` is logged as a diagnostic.

This architecture has three key properties that the serial approach lacked:

1. **No hierarchy**: neither channel's weights are frozen or protected. Both develop from scratch.
2. **Explicit agreement**: the consistency loss enforces that the two proposals must converge. The channels are not just concatenated inputs — they must agree on what to do.
3. **Measurable contribution**: the magnitude of `a_v - a_p` and the value of `w` directly measure each channel's contribution at every step. Vision ablation (zeroing pixels → observing `a_v` change → observing `a` change) is structurally trivial.

### What distinguishes this from prior "joint training" attempts

Earlier unfrozen runs trained proprio and vision jointly, but without a structural constraint on agreement. The result was that the high-dimensional pixel pathway dominated the low-dimensional proprio pathway, producing policies that read pixels but not productively.

The dialogue architecture differs by making agreement an explicit objective, not just an emergent property. The consistency loss `λ·||a_p - a_v||²` creates a direct gradient pressure toward channel agreement throughout training. The separate action heads (`a_p`, `a_v`) make the disagreement measurable at every step rather than hidden inside shared weights.

### What a "dialogue reaching agreement" looks like in practice

Early training: `a_p` and `a_v` are different because neither channel has learned anything yet. Disagreement is high and noisy. The creature moves erratically. The consistency loss creates gradient pressure toward agreement.

Mid training: `a_p` begins to learn locomotion (proprio → move forward). `a_v` begins to learn visual features (pixels → something). The consistency loss rewards cases where both channels agree. Episodes where the creature reaches the ball produce a reward signal that propagates through both channels simultaneously.

Late training (goal state): `a_p` and `a_v` converge on the same answer in most cases (low disagreement). But the cases where they disagree are informative: those are the cases where vision has information proprio lacks (e.g., ball is to the left — proprio cannot know this, vision can). The learned weight `w` should shift toward `a_v` in those cases, and the final action should reflect the ball's actual position.

The success criterion remains: touch rate ≥ 60% (matches blind baseline) AND nonzero vision-ablation sensitivity (vision is contributing). But the new architecture should make it far easier to verify which channel is contributing what.

### Prerequisites before the dialogue experiment

Two physical problems must be fixed first, or the same failure modes will re-emerge:

1. **Head servo gain**: `kp=30 → kp=5` in both XML files. Reduces head speed by 6×. Head now tracks commanded positions over multiple timesteps rather than snapping instantly. (Fixed — 2026-05-07)

2. **Stillness local optimum**: add `VELOCITY_BONUS_SCALE=0.02 × torso_speed` to the reward. A moving creature now earns slightly more than a stationary one, regardless of direction. (Fixed — 2026-05-07)

After these fixes, a new stage1 must be trained from scratch and verified by video before any vision work begins.

### Updated cumulative "what is ruled out" table

| Hypothesis | Status | Evidence |
|---|---|---|
| Serial training (blind first → freeze → vision) is the right approach | **Retired** | Video: blind policy doesn't locomote; head is unusable for visual learning |
| The 60% blind touch rate reflects genuine learned locomotion | **Refuted by observation** | Video: creature is nearly stationary; touches are spawn-luck, not search |
| MICOA freeze architecture can recover the 60% baseline | Ruled out (prior entries) | All MICOA runs: 8% ceiling regardless of initialization or spawn geometry |

### What the next experiment must demonstrate (before vision work resumes)

A new stage1 trained under the fixed physics and reward must show, on video:
- Active ground coverage: creature moves substantially during each episode
- Calm head: no full-range oscillation; head moves slowly and deliberately
- Touch rate: ≥ 60% blind deterministic (must match or beat old baseline)

Only after all three are confirmed does vision training resume, using the dialogue architecture.

---

## 2026-05-07 — Entropy-Pinned Re-run (entpin_005_3_hand_only_cone)

### Hypothesis tested
SAC entropy collapse (auto-tuning to ~0.003 by step 100K) is the proximate cause of input-blind policy collapse. Pinning entropy at 0.05 should prevent the policy from locking into a canned stroke.

### Result: REFUTED

Entropy held at 0.05 for all 250K steps. Outcome was identical to overnight run #3 (ent_coef=auto):
- Same peak reward: 61.73 vs 61.5
- Same late-training collapse: reward crashed from 61.7 → -7.6 in 50K steps after peak
- Same deterministic eval floor: 1/20 touches
- Same CNN consistency_loss: ~0.0015 throughout (pixel latent unused)

### What both frameworks say

**Behavioral Prediction Framework — CHALLENGED.** The rollout/eval gap (27–37% stochastic touches vs. 5% deterministic) is incompatible with a creature that has built any model of ball location. Removing sampling noise should help a model-based agent; it destroys this one.

**Pattern Learning Framework — CHALLENGED.** Consistency_loss of ~0.0015 is direct evidence that the visual representation never differentiated. No stable pattern of any kind formed in the CNN pixel latent across 250K steps on five separate runs.

### Revised theoretical picture

Entropy collapse was a symptom, not a cause. The root cause is that the task admits a viable blind local optimum: with a ±45° spawn cone, a fixed forward paddle connects with the ball ~25–30% of the time during stochastic rollout. SAC finds and exploits this floor. The gradient pressure toward "do better than 30% by using vision" is weaker than the stability of "keep doing the 30% paddle." No architectural fix (CNN encoder) or optimizer fix (entropy floor) changes this cost-benefit calculation, because neither changes the task structure.

The key observation: the phase0_validation checkpoint (100K, no plumbing) showed real sensitivity to both proprio and vision perturbations (clean 0.20 / proprio_noise 0.12 / vision_zero 0.12). More training + more plumbing produced *more* input-blind policies — the signature of a task that rewards consolidating a blind local optimum rather than exploring past it.

### Narrowed question

> Vision is not being used because vision is not *needed* to reach the current reward floor. The task must be redesigned so the blind paddle strategy reliably fails before vision will develop.

### What is ruled out (cumulative)

| Hypothesis | Status | Evidence |
|---|---|---|
| Ball not visible in head_cam | Ruled out | vision_check_v9: mean 40px, never zero |
| SAC entropy collapse is the cause | Ruled out | entpin run: same outcome with ent_coef=0.05 held flat |
| CNN encoder will make vision useful | Ruled out | overnight run #2 + entpin: consistency_loss ~0.001 both times |
| Hand-only contact narrows equivalence class enough | Ruled out | overnight run #3 + entpin: same blind result |
| XOR color pressure helps | Weak evidence against | overnight run #4: lowest peak, slightly more stable, but still input-blind |

### What is not yet ruled out

| Hypothesis | Next test |
|---|---|
| Blind proprio can already score ~1/20 deterministically (floor is task difficulty, not vision failure) | Run 20 det. eval eps on stage1_v8 blind checkpoint |
| Pursuit task (Idea E) makes vision instrumentally necessary at every step | Build moving-ball env, run smoke test |
| Stage 1.5 (no fall penalty + wider cone + longer truncation) lets proprio finish adapting before vision arrives | New training stage |
| Vision-only input ablation (proprio zeroed) tells us if architecture CAN use vision alone | 250K training run, proprio=0 |

### Recommended next step (theory-motivated)

**Cheapest diagnostic first:** run the stage-1 blind proprio checkpoint deterministically for 20 episodes and record touch rate. This partitions the problem:

- If blind proprio ≈ 1/20 → the floor is task difficulty; vision follow-on is not regressing, it's just at the same floor as a blind policy
- If blind proprio > 5/20 → the vision follow-on has *actively degraded* compared to the blind baseline; vision training is hurting, not just failing to help

Either answer sharpens the next training experiment significantly.

---

## 2026-05-07 — Blind Baseline Result + MICOA Framework

### The blind baseline result

The recommended diagnostic from the previous entry has been run. Results, deterministic, 20 episodes, stage1_v8_best checkpoint (no vision, no follow-on training):

- Touched: 12/20 (60%)
- Fell: 0/20
- Timeout (no contact): 8/20
- Mean episode reward: 110.9 ± 102.8

Comparison to all vision follow-on checkpoints run to date:

| Checkpoint | Deterministic touch rate |
|---|---|
| stage1_v8_best (blind proprio, no vision) | **12/20 (60%)** |
| entpin_005_3_hand_only_cone (best @ 160K) | 1/20 (5%) |
| Overnight run #1 | 1/20 (5%) |
| Overnight run #2 | 1/20 (5%) |
| Overnight run #3 | 1/20 (5%) |
| Overnight run #4 | 1/20 (5%) |

**The finding is unambiguous: vision training is actively destroying a capable blind policy.**

This answers the question posed in the previous entry's "what is not yet ruled out" table. The blind proprio floor is not ~1/20. It is 60%. Every vision follow-on run has degraded that 60% to 5% — a 12× regression — across five independent runs varying architecture (MLP / CNN), entropy regime (auto / pinned), contact definition (any-body / hand-only), and spawn geometry (cone / no cone). The regression is not stochastic. It is consistent. It is structural.

### Why this is not explained by existing frameworks

The Behavioral Prediction Framework and Pattern Learning Framework both predict that adding a new, informative sensory channel to an already-capable agent should either improve performance or leave it neutral. They have no natural account for why adding vision should produce a 12× degradation in the capability the agent already had.

The prior explanations we had explored (entropy collapse, CNN architecture, contact definition, spawn geometry) were all attempts to explain why vision was not helping. None of them addressed why vision was hurting. This is a categorically different problem. It requires a categorically different lens.

### MICOA: Multiple Inputs Confirming One Another

The human researcher has identified the core principle that our current architecture and training regime violate. It is stated here as a formal framework for this lab.

**The MICOA principle:** In biological perceptual systems, multiple sensory channels do not compete for control of behavior — they confirm one another. When a human walks toward a book on a table, vision signals "the book is getting closer" (the image is expanding), and proprioception signals "my body is moving forward" (limb position and vestibular state). These are not two competing channels sending the same message — they are two independent derivations of the same underlying physical fact ("I am approaching the book"). The same is true at the moment of contact: proprio confirms what vision predicted (the book is here, my hand is touching it), and vision confirms what proprio registered (my hand is at the book's location). The senses are not substitutes. They are cross-validators. Each one provides evidence that makes the other more reliable.

The key word is "independently." Vision and proprio are MICOA-compliant when they each independently provide evidence that points toward the same underlying state, and when the agent has a representation that treats them as converging evidence rather than competing signals. A human who closes their eyes and reaches for a cup is using the expected spatial location (derived from prior vision) to guide the proprioceptive reach. When their hand finds the cup, the tactile signal confirms the visual prediction. The two channels are bound together not because they share weights but because they are solving the same underlying problem from different physical substrates.

### Why our experiment violates MICOA

In the current v8 setup, the blind proprio policy has developed a working solution to "find the ball" that is entirely proprio-based. It achieves 60%. The proprio signal at this point encodes something useful — not ball position directly (proprio has no explicit target vector in the v8 obs), but some learned combination of body state, motion history, and contact feedback that lets the creature navigate toward ball locations at a 60% rate.

When we add vision as 3072 concatenated pixel dimensions and train with SAC:

1. **Vision has no architectural pressure to confirm proprio.** The CNN pixel latent is computed separately from the proprio features and concatenated to the flat observation vector. There is no mechanism that tells vision "your job is to confirm what proprio already knows." Vision is simply another block of inputs in a high-dimensional observation that SAC can weight however it wants.

2. **Vision has no reward pressure to confirm proprio.** The reward signal is: touch the ball (+200), stay alive (+proximity gradient). This reward is equally achievable whether the agent uses proprio, vision, or random movement. There is no component of the reward that says "vision and proprio must agree." The only gradient for vision to receive is the same gradient proprio is already receiving. SAC cannot distinguish "use vision to confirm proprio's ball-finding" from "overwrite proprio's ball-finding with a pixel-based strategy that might work better."

3. **SAC has no reason to preserve the existing proprio solution.** The replay buffer begins filling with transitions from the post-vision-injection policy. As the visual weights train, the hidden layer activations change. The policy that produced 60% blind touch rate was a specific geometry in weight space. Vision training perturbs that geometry. The 60% proprio-based solution is not protected by any architectural or loss-based mechanism. It erodes.

4. **3072 pixel dimensions vastly outnumber 10 proprio dimensions.** The relative input dimensionality gives the gradient optimizer much more "surface area" to work with on the pixel side than the proprio side. Even if proprio contains the useful signal and pixels contain only noise, gradient updates on 3072 dimensions will move the hidden layer far more than gradient updates on 10 dimensions, simply by dimension count. The capable proprio signal is drowned by the high-dimensional noisy pixel signal.

5. **The result is not "vision noise drowning a signal" — it is "vision training overwriting a working policy."** The consistency_loss ~0.0015 confirms that the CNN pixel latent never differentiated. Vision provided no useful signal at any point in training. But proprio's working solution was still destroyed. The culprit is not that vision is noisy — it is that training with a high-dimensional noisy channel actively corrupts the low-dimensional channel that was working, in the absence of any mechanism to prevent this.

This is the MICOA failure mode: two channels that should be confirming each other are instead in a destructive relationship where the weaker, noisier channel overwrites the stronger, more established one, because the training regime has no mechanism to enforce confirmation and no protection for the prior solution.

### Reframing the core question

The previous framing of this project's challenge was: **"How do we get AB to use vision?"**

The blind baseline result forces a more precise reframing: **"How do we preserve the capable blind policy while making vision a confirming signal rather than a competing one?"**

These are not the same question. The first question assumes we start from a low-capability baseline and need to add vision as an upgrade. The second question acknowledges that we already have a working 60% proprio solution and need to add vision in a way that improves on it rather than destroying it.

This reframing has direct implications for what a successful experiment looks like. Success is not "vision follow-on achieves 60%." Success is "vision follow-on achieves >60% while vision-ablation sensitivity is measurably nonzero." That is, vision must be contributing positively, and the proprio-based baseline capability must be preserved or improved.

Under MICOA, the architectural requirement for this is: vision must be added in a way where vision's contribution is explicitly constrained to confirm what proprio already knows, and proprio's working solution is explicitly protected from being overwritten.

### What current architecture gets wrong from a MICOA perspective

The v8 follow-on architecture (CNN pixel latent concatenated to proprio features, trained jointly with SAC) fails every MICOA requirement:

1. **No confirmation structure.** The CNN latent and proprio features are concatenated and passed through shared layers. There is no mechanism that makes vision's contribution a function of "what proprio would have predicted here." Vision is free to carve its own path through representation space.

2. **No protection for prior proprio solution.** Proprio weights train jointly with visual weights throughout follow-on. The 60% solution is geometrically unprotected. Any gradient update that moves visual weights also moves proprio weights through their shared downstream layers.

3. **No cross-modal coherence loss.** The v5 consistency loss (`λ · MSE(h_full, h_blind)`) was designed for exactly this purpose and achieved CKA = 0.998 in the tabletop setting. The v8 follow-on runs do not use this loss. The omission is the most direct explanation for why v8 produces 5% touch rate while v5 produced 100%.

4. **No task pressure for confirmation.** The reward does not change based on whether vision and proprio "agree" about ball location. A MICOA-aligned reward would give a bonus when the agent acts in a way that is consistent with what both vision and proprio independently indicate — and would penalize actions that can only be explained by one channel, not both.

### What a MICOA-aligned experiment would look like

The sketch below is not a training plan — it is a theoretical specification of what the architecture and training regime would need to satisfy.

**Architectural requirements:**

1. **Freeze proprio weights completely before adding vision.** The v2/v5 approach applied to v8: every proprio-pathway weight is set to require no gradient update during follow-on. This mathematically guarantees that the 60% blind solution is preserved — with pixels zeroed, the policy is bit-identical to stage 1.

2. **Add the consistency loss (`λ · MSE(h_full, h_blind)`) during follow-on.** This is the single most evidence-supported addition from the tabletop experiments. It enforces that vision's contribution to the hidden state stays geometrically close to what proprio alone would have produced. Vision can only speak in proprio's language — it cannot carve its own representation space.

3. **Initialize visual weights at zero, not random.** Random initialization means vision contributes noise from step 1. Zero initialization means vision begins as a neutral addition to the working policy and grows its contribution only as it earns it via gradient signal.

**Task requirements:**

4. **Make the task require discrimination that blind proprio cannot provide.** The current 60% blind rate means ~40% of spawns are in positions the blind policy cannot reach. Those 40% are the only trials where vision has an instrumental opportunity. For MICOA to be satisfied at scale, vision needs to have a confirming role on most trials, not just the hard 40%. This points toward either: (a) increased spawn diversity so more positions require directional information, or (b) the moving-ball task (Idea E from the project log) where proprio cannot guess ball position without visual tracking at every step.

5. **Consider a cross-modal coherence reward (optional, exploratory).** A reward component that is nonzero only when vision and proprio independently predict the same ball-direction would create direct gradient pressure for MICOA-style confirmation. This is bribery in the project's terminology ("pressure not bribery") and should be used only if architectural constraints alone are insufficient. The correct order of operations: try architectural MICOA alignment first; add reward pressure only if representation still fails to cohere.

**Measurement:**

6. **The success criterion is not touch rate alone — it is touch rate + vision-ablation sensitivity.** A run counts as a MICOA-aligned success only if: (a) deterministic touch rate is >= the blind baseline (60%), and (b) vision-ablation sensitivity is measurably nonzero (policy action changes when pixels are zeroed). Satisfying only (a) could be achieved by a good proprio policy that ignores vision. Satisfying only (b) could be achieved by a policy that uses vision but destroys proprio. Both must be true simultaneously.

### Updated "what is ruled out" table

| Hypothesis | Status | Evidence |
|---|---|---|
| Ball not visible in head_cam | Ruled out | vision_check_v9: mean 40px, never zero |
| SAC entropy collapse is the cause | Ruled out | entpin run: same outcome with ent_coef=0.05 held flat |
| CNN encoder will make vision useful | Ruled out | overnight runs: consistency_loss ~0.001 both times |
| Hand-only contact narrows equivalence class enough | Ruled out | overnight run #3 + entpin: same blind result |
| Vision follow-on is failing to help (neutral result) | Ruled out | blind baseline 60% vs. follow-on 5%: vision is actively harmful |
| Floor is task difficulty (blind proprio ≈ 1/20) | Ruled out | blind baseline: 12/20 (60%) deterministic |

### Updated "what is not yet ruled out" table

| Hypothesis | Next test |
|---|---|
| Frozen proprio + consistency loss (v5 approach) applied to v8 will preserve 60% and add vision | Architecture change + follow-on run |
| Moving-ball task makes vision instrumentally necessary at every step (Idea E) | Build moving-ball env, run smoke test |
| Visual dimensionality reduction (e.g., 8×8 instead of 32×32) reduces the drowning-by-dimension-count problem | Architecture variant in next follow-on |
| Proprio zeroed + vision only: can the v8 architecture in principle learn a task from vision alone? | Isolated diagnostic run |

### Open theoretical question

The v5 tabletop result (CKA = 0.998, 80% noise robustness, drift = 0) was achieved on a different body, different task, and fundamentally different proprioceptive signal. The v8 proprio signal is locomotion-based (body displacement, paddle stroke timing) rather than arm-joint-based (joint angles, fingertip offset). It is not guaranteed that the same architectural choices that produced MICOA-alignment in the tabletop setting will transfer to the locomotion setting.

The theoretical question that v8 opens — and that none of the prior experiments could have addressed — is: **can MICOA-aligned confirmation be established when the proprio signal is a body-movement signal rather than an arm-position signal?** In the tabletop setting, proprio "knew" where the arm was relative to the target; vision confirmed this. In v8, proprio knows how fast the body is moving and in what direction, but it does not know where the target is. Vision would need to provide the target-location information that proprio cannot provide, while still "confirming" proprio's motor-execution signal. This is a harder problem than tabletop MICOA, and it is the correct next theoretical frontier for this project.

---

## 2026-05-07 — MICOA Freeze Run Results (micoa_freeze_60proprio_lam010)

### What was tested

The first direct MICOA-aligned experiment on the v8 platform creature. Architecture: proprio weights frozen from stage1_v8_best (the 60% blind policy); only the pixel-input columns of the actor's first hidden layer were allowed to train; consistency loss `λ · MSE(h_full, h_blind)` with λ = 0.10. Seeds 100–149, 150K steps. Best checkpoint at 60K steps.

### Key numbers

| Metric | Value |
|---|---|
| Proprio weight drift | 0.00e+00 (exact zero throughout) |
| Falls | 0/50 (all runs) |
| consistency_loss (throughout) | 0.25–0.29 |
| Deterministic touch rate (best ckpt, seeds 100–149, 50 eps) | 4/50 (8%) |
| Blind proprio baseline (stage1_v8_best, 20 eps) | 12/20 (60%) |
| All prior vision follow-ons (5 runs) | 1/20 (5%) |
| Failure mode | All timeouts — stable but stationary |

### What the MICOA hypothesis predicted vs. what we observed

The MICOA hypothesis, as stated in the previous entry, makes a clear prediction: if proprio weights are frozen and a consistency loss constrains vision to speak in proprio's language, then (a) the 60% blind solution will be preserved (drift = 0.0 guarantees this architecturally), and (b) vision will gradually develop a confirming role, steering the creature toward ball locations it could not reach by forward paddle alone. Under this prediction, touch rate should start at or above the 60% blind baseline (because the freeze preserves it) and grow as vision earns its contribution.

What we observed:

- Drift = 0.0: confirmed. The freeze worked mechanically.
- Falls = 0: confirmed. Stability was maintained.
- Touch rate = 8% (4/50): not confirmed. The 60% blind baseline was not preserved. It collapsed to near the same 5% floor as every unfrozen run, even though the freeze should have made it impossible for the policy (with pixels zeroed) to behave differently from the 60% blind policy.
- Failure mode = 100% timeouts (not forward-paddle or falls): the creature stopped moving. The blind policy paddled forward and touched 60% of the time. The frozen+vision policy was stable but nearly stationary.
- consistency_loss = 0.25–0.29 throughout (versus ~0.0015 in all prior runs): the visual pathway's output was very different from the blind proprio output. This is 150–190× higher than any prior run.

**Summary: MICOA-via-freeze is partially confirmed (drift, stability) and partially refuted (touch rate not preserved, stillness emerged).**

### The consistency loss puzzle — confirmation or interference?

This is the central interpretive question for this run and it requires careful reasoning.

The consistency loss is defined as `MSE(h_full, h_blind)`, where h_full is the hidden state with pixels included and h_blind is the hidden state with pixels zeroed. The loss pulls h_full toward h_blind — it is a penalty for vision diverging from what proprio alone would produce.

In all prior runs (unfrozen, no consistency loss), the loss was ~0.0015. That near-zero value meant the pixel pathway's output was already essentially identical to the blind output — not because vision was confirming proprio, but because the CNN pixel latent had collapsed to a near-constant value that happened to produce the same hidden state regardless of pixel content. The visual pathway was dead, not confirming.

In this run, the loss was 0.25–0.29 throughout. There are two opposite interpretations of this high value:

**Interpretation A (MICOA working, positive reading):** The freeze prevented the visual pathway from doing what prior runs did — collapsing to a dead constant. With proprio frozen, the pixel pathway could not corrupt the proprio representation, so it was forced to actually process pixels. The high consistency_loss means vision is alive and doing something genuinely different from proprio. The MSE penalty is correctly penalizing a real divergence that the network is working to resolve. Under this reading, the system needs more training time to converge: the loss started high (vision diverges) and would eventually fall toward zero as vision learns to confirm proprio.

**Interpretation B (MICOA failing, negative reading):** The high consistency_loss means vision's output persistently resists converging toward proprio's output. The MSE gradient is pulling h_full toward h_blind, but something in the visual pathway is pulling back — generating activations that are systematically different from what proprio alone would produce. The result is not vision confirming proprio; it is vision fighting proprio in a tug-of-war. The proprio freeze protected proprio's stored motor program (drift = 0) but it did not prevent vision from injecting a conflicting signal into the shared downstream layers that translate hidden state into actions. Even though proprio's weights are frozen, the downstream layers (which are unfrozen) still receive vision's divergent contribution. The motor output is a function of the full hidden state, not just the frozen proprio columns. Vision's divergent h_full contribution can shift the action even if proprio's weights did not change.

**Which interpretation fits the data?**

The failure mode discriminates between them. Under Interpretation A (vision working toward confirmation), the agent should be moving — it should be doing something based on the visual signal, even if that something is not yet productive. The hallmark of a learning-in-progress agent is activity, not stillness.

Under Interpretation B (vision cancelling proprio), the agent should be nearly still. Proprio's frozen weights still encode the forward-paddle motor program. Vision's divergent contribution to h_full is being injected into the downstream layers. If vision's divergent signal systematically cancels the proprio-driven forward-paddle (by pushing action components in an opposing direction), the net motor output will be near-zero even though both channels are active.

The data: 46/50 timeouts, stable but not steering. The blind policy paddled forward and touched 60% of the time. The vision follow-on is stable and nearly stationary. This is the signature of cancellation, not of learning-in-progress divergence.

**Conclusion:** Interpretation B fits. The high consistency_loss is the marker of a tug-of-war between the pixel pathway's divergent output and the consistency penalty, and the net effect of that tug-of-war on the downstream action layers is to cancel the proprio-driven motor program rather than confirm it. Vision is not dead (as in prior runs) — it is alive and pulling in the wrong direction, hard enough to suppress the 60% blind steering without contributing useful steering of its own.

### Why the freeze alone is not sufficient for MICOA

The previous entry's theoretical specification listed the freeze as requirement (1) and the consistency loss as requirement (2), treating them as composable. This run reveals a gap in that specification.

The freeze protects proprio weights from being overwritten. It does not protect proprio's motor output from being cancelled by a simultaneously trained vision pathway operating in the shared downstream layers. In the tabletop experiments (v2, v5), this gap was not exposed because the task geometry meant vision's contributions tended to push actions in helpful directions even before convergence. In v8, the action space is a paddle stroke in 2D, and the ball can be in any direction around the creature. A random or poorly-converged vision signal is as likely to push the creature away from the ball as toward it — and if that push is large relative to the proprio-driven forward-paddle component, the net action is near-zero or backward.

The freeze + consistency loss combination, as implemented, has a sequencing problem: the consistency loss penalizes vision for diverging from proprio, but it does not prevent vision from diverging during the period before the loss has driven convergence. During that period — which may be most or all of 150K steps — vision's divergent signal is being injected into the motor output, cancelling the proprio-driven steering.

What is missing is a mechanism to limit the magnitude of vision's contribution during the early training period, before it has converged. Candidates (theoretical, not a training plan):

- **Zero initialization of all pixel-input weights:** vision begins with zero contribution to h_full, so h_full = h_blind at step 0, and vision's contribution grows only as gradient learning earns it. The current run likely used random initialization of pixel weights (the prior entries suggest this is the default), meaning vision started with a large random divergent contribution from step 1.
- **λ scaling:** a higher λ (e.g., 1.0 instead of 0.1) would impose a stronger penalty on divergence from the start, shrinking vision's action-space footprint during the early chaotic period.
- **Gradient clipping on the pixel-column gradients specifically:** limits how fast vision can move the downstream layers during any single update.

### Is 8% > 5% meaningful?

The previous runs all produced 1/20 (5%) deterministic touch rate. This run produced 4/50 (8%). The question is whether this 3-point improvement is signal or noise.

Against signal: 4 events in 50 trials gives a 95% confidence interval of approximately 2%–19% (exact binomial). The prior 5% floor was 1 event in 20 trials. Both are consistent with a true underlying rate of 5–10%. The improvement is within the noise envelope.

For signal (weak): the failure mode is different. Prior runs had some falls and some canned-paddle behavior. This run had zero falls and zero canned-paddle — all timeouts. A different failure mode at a slightly higher touch rate could mean the underlying strategy changed even if the touch count barely moved. But "different failure mode" is not the same as "better performance."

**Verdict:** 8% vs. 5% is not meaningfully different. The freeze did not rescue the 60% blind baseline. It produced a slightly less chaotic version of the same basic failure.

### What is newly ruled out

| Hypothesis | Status | Evidence |
|---|---|---|
| Proprio drift is causing the 60% → 5% regression | Ruled out | drift = 0.0 in this run, yet touch rate still collapsed to 8% |
| Freeze alone is sufficient to preserve the 60% blind policy | Ruled out | proprio weights frozen, but touch rate did not recover to 60% |
| Low consistency_loss (prior runs) means vision was confirming proprio | Ruled out by contrast | low consistency_loss = dead pixel pathway, not MICOA confirmation |

### What is not yet ruled out

| Hypothesis | Next test |
|---|---|
| Zero initialization of pixel weights + freeze + consistency loss would prevent early cancellation | Architecture change: initialize pixel columns at zero, re-run at same λ |
| Higher λ (e.g., 1.0) would suppress vision's action-space footprint fast enough to prevent cancellation | λ sweep: λ = 0.5 or 1.0 with freeze |
| The consistency loss at λ = 0.10 would converge if given more steps — the 150K run was too short | Extend this run to 500K and observe whether consistency_loss trends downward after 150K |
| Vision's divergent signal is being injected into downstream layers (not proprio weights) — a per-layer freeze of downstream layers too would isolate the problem | Downstream layer freeze diagnostic |
| The moving-ball task (Idea E) changes the gradient landscape enough that even imperfect MICOA architecture learns to steer | Build moving-ball env, run smoke test with this architecture |

### Revised theoretical picture

The MICOA principle is not wrong. The tabletop evidence (v5: CKA = 0.998, 80% noise robustness, drift = 0) establishes that it is achievable. What this run establishes is that the minimal implementation of MICOA (freeze + consistency loss at λ = 0.10) is necessary but not sufficient. It protects proprio weights but does not protect proprio's motor output from being cancelled by a simultaneously active and divergent vision pathway in shared downstream layers.

The deeper MICOA requirement, which the tabletop experiments did not expose because the tabletop geometry was forgiving, is: **vision must not be able to cancel proprio's motor output during the period when vision has not yet converged.** This requires either (a) zero initialization so vision starts neutral, (b) a much stronger consistency penalty that drives convergence faster than vision can diverge, or (c) a downstream-layer protection mechanism that shields the action layers from vision's contribution until the consistency loss is satisfied.

The stillness failure mode in this run is theoretically informative. It means the net motor output under the freeze + vision combination is near-zero. Proprio says "paddle forward." Vision says something else. The downstream sum is nearly zero. That cancellation is the exact opposite of MICOA's confirmation principle — it is anti-confirmation. Two channels providing contradictory rather than converging evidence.

This run does not refute MICOA as a theoretical framework. It refutes one specific implementation of it: freeze + random-initialized pixel weights + λ = 0.10 consistency loss for 150K steps. The framework's prediction (vision must confirm proprio, not cancel it) is correct. The implementation failed to enforce the prediction early enough in training.

### Updated cumulative "what is ruled out" table

| Hypothesis | Status | Evidence |
|---|---|---|
| Ball not visible in head_cam | Ruled out | vision_check_v9: mean 40px, never zero |
| SAC entropy collapse is the cause | Ruled out | entpin run: same outcome with ent_coef=0.05 held flat |
| CNN encoder will make vision useful | Ruled out | overnight runs: consistency_loss ~0.001 both times |
| Hand-only contact narrows equivalence class enough | Ruled out | overnight run #3 + entpin: same blind result |
| Vision follow-on is failing to help (neutral result) | Ruled out | blind baseline 60% vs. follow-on 5%: vision is actively harmful |
| Floor is task difficulty (blind proprio ≈ 1/20) | Ruled out | blind baseline: 12/20 (60%) deterministic |
| Proprio drift is causing the 60% → 5% regression | Ruled out | freeze run: drift = 0.0, touch rate still collapsed |
| Freeze alone preserves the 60% proprio-based motor output | Ruled out | freeze run: proprio weights frozen, touch rate 8%, failure mode = cancellation |
| Low consistency_loss means MICOA confirmation is occurring | Ruled out | low loss = dead pixel pathway (prior runs); high loss = active cancellation (this run) |

### Updated cumulative "what is not yet ruled out" table

| Hypothesis | Next test |
|---|---|
| Zero initialization of pixel weights prevents early cancellation | Re-run with pixel columns initialized at zero |
| Higher λ (0.5 or 1.0) drives vision convergence faster than divergence can cancel proprio output | λ sweep |
| The freeze + consistency loss architecture converges if given 500K steps (150K was too short) | Extend current run |
| Downstream layer freeze (not just proprio columns) fully isolates proprio motor output | Downstream freeze diagnostic |
| Moving-ball task (Idea E) changes gradient landscape enough to produce real MICOA convergence | Build moving-ball env, smoke test |

### Theory Monitor Note — 2026-05-07 (MICOA Freeze Run)

**Behavioral Prediction Framework: CHALLENGED** — The creature is stable (0 falls) but nearly stationary (46/50 timeouts), meaning it has not built any directional model of where the ball is: a creature with an internal world-model would move, even imperfectly, not freeze in place.

**Pattern Learning Framework: CHALLENGED** — The consistency_loss plateau at 0.25–0.29 (150× higher than all prior runs) means the visual pathway settled into a stable but wrong representation: it found a stable code, but the code it found is one that persistently contradicts proprio rather than overlapping with it, which is the opposite of what the framework requires for efficient generalization.

**The most important thing we don't know yet:** Whether zero-initializing the pixel-input weights (so vision starts neutral and earns its contribution from scratch rather than immediately injecting a large random signal into the downstream action layers) would prevent the cancellation failure mode that caused the creature to freeze in place.

**Recommended diagnostic** (not a training run — just a measurement): Compare the action output of the best-checkpoint policy at seed 100 on the same starting state with pixels set to zero vs. pixels set to the actual scene — if the actions are similar in magnitude but opposite in direction (confirming the cancellation hypothesis), that is strong evidence the zero-initialization fix is the right next architectural change.

---

## 2026-05-07 — MICOA Zero-Init + Forward Cone Results (micoa_freeze_zeroinit_cone30_2026_05_07)

### What was tested

Two changes were introduced simultaneously on top of the MICOA freeze architecture (frozen proprio, pixel-column gradient only, consistency loss λ = 0.10, ent_coef = 0.05):

1. **Zero-init pixel columns:** all pixel-input weights in the first hidden layer initialized to exactly zero at training start. Vision contributes h_full = h_blind at step 0. The logged confirmation "Pixel columns zero-initialized (3072 cols)" verifies this was implemented.

2. **spawn-cone-deg = 30:** ball always spawns within ±15° of straight ahead — inside the head camera's FOV from the very first frame. Previous MICOA run excluded the ±25° forward cone (ball was never in frame at reset).

The hypothesis, as predicted by the previous entry's analysis: zero-init would prevent early cancellation (vision starts neutral, so proprio's forward-paddle is not suppressed at step 0), and the forward cone would give vision a strong immediate gradient signal (ball is visible immediately, providing a real scene to differentiate rather than noise).

### Key numbers

| Metric | micoa_freeze (prior run) | micoa_freeze_zeroinit_cone30 (this run) |
|---|---|---|
| Proprio weight drift | 0.00e+00 | 0.00e+00 |
| Falls | 0/50 | 0/50 |
| consistency_loss | 0.25–0.29 | 0.20–0.38 (higher and more volatile early) |
| ep_rew_mean at 10K | 14.5 | **39.9** |
| Best checkpoint | 60K | 30K |
| Det. touch rate (best ckpt, 50 eps) | 4/50 (8%) | 4/50 (8%) |
| Failure mode | All timeouts | All timeouts |
| Rollout left/right touch frac | left-biased (~0.386/0.196) | **symmetric (0.102/0.098)** |

### What zero-init was supposed to fix and whether it fixed it

Zero-initialization was the direct prescription from the prior run's theory section. The diagnosis was: random pixel weights inject a large divergent signal into h_full at step 0, immediately cancelling proprio's forward-paddle contribution in the downstream layers. Zero-init removes that initial random injection: at step 0, h_full = h_blind exactly, so proprio's motor program is not cancelled. Vision starts as a passenger and earns its contribution only through gradient learning.

**What actually happened:** The consistency_loss rose to 0.20–0.38 quickly — faster and with more volatility than the prior run's 0.25–0.29. This is the opposite of what zero-init was designed to produce. If zero-init had worked as predicted, we would expect:
- consistency_loss near zero at step 0 (confirmed: h_full = h_blind by construction)
- consistency_loss rising slowly as vision learns to differentiate, with the penalty driving it back down
- net: a lower, more stable trajectory than the prior run's 0.25–0.29

What we observed was a faster rise to a higher plateau. This means the gradient pressure that drives vision away from proprio is stronger than zero-init can suppress. The pixel pathway began at zero contribution (step 0) but shot to a high-divergence state within the first few thousand steps — more aggressively than when initialized randomly. The most likely explanation: with zero init, the full gradient at step 0 flows through the pixel columns toward maximizing Q-value from scratch. There is no random noise to average out; every early update is a direct Q-value maximization gradient, and that gradient points away from proprio's manifold. The random-init case was slower to diverge because the random initial weights partially canceled each other's gradients.

**Verdict on zero-init:** Zero-init confirms that h_full = h_blind at step 0 (drift = 0.0 holds, freeze holds), but it does not prevent early divergence. The constraint we identified — vision cannot cancel proprio during the pre-convergence period — was not fixed by starting at zero. It requires a constraint strong enough to resist the Q-value gradient throughout early training, not just at initialization.

### What the forward cone was supposed to fix and whether it fixed it

The forward cone (±15°) was motivated by a separate concern: the prior run's ball was always outside the camera FOV at spawn (previous runs excluded the ±25° forward cone), so vision had no ball-relevant signal to differentiate from noise at reset. The forward cone guarantees the ball is visible immediately, providing a genuine visual gradient to learn from.

**What actually happened:** Early reward jumped to 39.9 at 10K steps (vs. 14.5 in the prior run). This is real — the creature is bumping into the ball more often early in training because the ball is always in front. The best checkpoint arrived earlier (30K vs. 60K). These are genuine early improvements.

**But the ceiling did not rise:** the best deterministic touch rate is still 4/50 (8%). The forward cone helped the creature find the ball during the chaotic early phase but did not change the long-run outcome. There are two readings of this:

- *Optimistic:* The early improvement is real; what fails is that vision does not compound on it because the pixel pathway diverges (as described above). Fix the divergence and the early improvement might compound.
- *Pessimistic:* The 8% ceiling is not a vision-learning ceiling — it is a ceiling on how often the forward-paddle strategy can succeed even when the ball is always in front, combined with the creature's locomotion capability. Vision is still not load-bearing.

We cannot distinguish these from current data alone. The recommended diagnostic below is designed to do exactly that.

### What the left/right symmetry means — is it evidence for vision steering?

The rollout touch symmetry (left 0.102 / right 0.098) is the most theoretically interesting datum in this run. Every prior run showed strong left bias: entpin had 0.386 left / 0.196 right (2:1); previous MICOA run had similar asymmetry. Symmetric touch distribution is, in principle, what you would expect from a creature that is visually steering — the ball is equally likely to be slightly left or slightly right within ±15°, so a creature responding to what it sees should touch it symmetrically.

However, there is a simpler geometric explanation: the spawn distribution itself is symmetric (±15° uniform), so a creature doing nothing but fixed forward-paddle will also produce symmetric contacts — because fixed forward-paddle has no left/right bias, and the ball is equally likely on both sides. The symmetry of the prior runs' left bias came from a non-symmetric spawn (ball excluded from ±25° forward, so always to one side or the other, and the creature's fixed paddle happened to favor left). With a symmetric forward cone, even a completely blind fixed-paddle policy would produce symmetric contacts.

**Conclusion: the left/right symmetry is consistent with visual steering but does not require it.** It is a necessary but not sufficient condition for visual steering. The discriminating test is a broken-symmetry evaluation: spawn all balls on the left side of the forward cone and check whether touch rate stays high (which a vision-steering policy would do) or collapses to near-zero (which a fixed-paddle policy would do, since the fixed paddle points straight ahead and many left-side spawns would be off-center).

This test is cheap, does not require a new training run, and would directly answer the question "is vision contributing direction information?" It is the recommended diagnostic below.

### What the persistent 8% ceiling tells us about where the bottleneck now is

Across two MICOA freeze runs (with and without zero-init and forward cone), the best deterministic touch rate has been 4/50 (8%). This is not a floor — it is a ceiling. The ceiling is at 8% while the blind proprio baseline is at 60%. That 52-point gap is the measure of what vision follow-on training is costing rather than providing.

The 8% ceiling is consistent with three possible bottlenecks, in order of theoretical priority:

1. **The pixel pathway is cancelling proprio's motor program** (identified in prior run): vision's divergent signal in shared downstream layers subtracts from the proprio-driven forward-paddle, producing low net locomotion. Touch rate is low because the creature barely moves. This is supported by the timeout failure mode (46/50 timeouts in both runs). If this were the only bottleneck, fixing cancellation (e.g., downstream layer freeze, or much higher λ) should recover the 60% blind baseline and possibly exceed it.

2. **The forward-cone geometry makes 8% the natural touch rate for a policy that barely moves:** if the creature moves slowly within a ±15° spawn zone, it will occasionally drift into the ball. 4/50 = 8% could be purely geometric — how often does a nearly-stationary creature in a ±15° spawn zone happen to contact the ball without directional steering? We do not have this baseline. The recommended diagnostic (fixed-paddle eval on the same cone) would give us this number.

3. **Vision is providing direction information but 150K steps is not enough for it to influence touch rate:** if vision's contribution is weak but non-zero (consistent with the early reward improvement), the 8% could be the touch rate of a barely-steering policy that needs 500K+ steps to develop reliable steering. This is the optimistic reading.

---

## 2026-05-11 — Phase V.2 / Phase VI / Phase VI.2 hypothesis status

### Phase V.2 frameworks update

**Behavioral Prediction Framework** (an agent with internal predictive structure should produce coherent behavior across a range of initial conditions): Phase V.2's "one move then freeze" pattern is the minimum-possible violation of this framework consistent with producing any non-trivial behavior. The framework predicts the agent should develop a sequence of internal predictions chained together — "if I move this way, the ball will be here, then if I move again, contact will occur." What we observe is one prediction (the initial motion earns overlap reward) and then no follow-through predictions. The policy freezes on whatever state the first action produced, regardless of what that state looks like or how far the far ball remains. Status: **still violated**, though "one move then freeze" is a marginally less extreme violation than Phase IV's "no move at all."

**Pattern Learning Framework** (sparse, distributed patterns should activate for structurally similar situations regardless of surface-level differences): Phase V.2 shows one stable attractor — the side-lying frozen posture — that the agent reliably reaches across all seeds. The critic has learned a stable pattern that accurately predicts episode returns from this attractor. But this is a single-state pattern, not the distributed map of overlapping patterns the framework predicts. Status: **still violated** — stable representation but of a single-point attractor rather than a distributed spatial structure.

**MICOA**: Not testable. Phase V and Phase V.2 are blind proprio runs. No visual channel was used in either run.

### Phase VI.2 frameworks update

**Behavioral Prediction Framework**: Phase VI.2 provides the cleanest violation to date. Two initial conditions differing only in spawn yaw produce episode rewards of approximately −1800 vs. −2000. The policy has no orientation-invariant model of where the ball is relative to its current body — it cannot compensate for a bad starting yaw, because it has not learned to reorient. The eval std of 99.48 is not a "useful spread of strategies" — it is the signature of a policy whose outcome is entirely determined by initial conditions and not at all by adaptive behavior. Status: **violated in a newly specific way** — the framework now has not just the no-prediction failure mode but an explicit refutation of orientation-invariant representation.

**Pattern Learning Framework**: if the agent had built any kind of allocentric or even body-relative representation of "ball is over there," similar patterns should fire for "ball at relative angle θ" regardless of absolute world yaw. What we see is the opposite: outcome quality varies completely with absolute yaw, indicating no yaw-invariant pattern has formed. Status: **violated** — no allocentric or body-relative representation appears to exist.

### What is newly ruled out / confirmed after Phase V.2 / VI.2 chain

| Hypothesis | Status |
|---|---|
| HER + caregiver scaffold + velocity bonus (correctly applied) produces sustained locomotion | **Refuted** |
| Phase V.2's non-zero eval std (0.30) was the early sign of behavioral diversity emerging | **Refuted** — std came from geometric micro-variation, not behavioral diversity |
| The policy has formed any orientation-invariant or body-relative representation of ball position | **Refuted** — identical policy, different spawn yaw, ~200-point reward swing per episode |
| Diagnostic 1 from V.2 will cleanly distinguish geometric-refuge from learned-reaching | **Refuted as binary** — surfaced a third failure mechanism (orientation lottery) |

THEORY MONITOR: Theory verdict written to FINDINGS.md.

THEORY MONITOR: THEORETICAL CONCERN — Phase VI.2 has now surfaced a second dimension of the failure mode that prior phases could not see: the policy's reward depends on initial conditions, not on any learned representation. This is consistent with the F-* family of candidates not yet having addressed the underlying substrate-level problem ("link 2 of the dependency graph"). The human researcher's cart proposal is the first intervention proposed this session that directly addresses link 2 by bypassing it rather than trying to bootstrap through it. The strategist's formal Phase VII proposal should weigh whether to continue the F-* family (incremental, conservative, addresses the failure mode within the existing substrate) or pivot to the cart substrate (structural, more ambitious, bypasses the failure mode entirely). Both lines of work have theoretical merit; the choice is essentially a question of how much experimental budget to commit to the within-substrate program before declaring it exhausted.

---

## 2026-05-22 — Phase XII (R41/R42) hypothesis status

Phase XII ran two matched 250K-step experiments under R40's MICOA architecture: R41 (MICOA + vision, moving balls at ball_speed=0.08) and R42 (matched proprio-only control, same task). The question was whether R40's architectural integration of vision — confirmed in Phase XI under static balls — would rescue performance when the ball is in motion, a condition Phase VIII showed broke pure proprio.

### Numbers (verified, from FINDINGS.md lines 1081–1176)

- R41 (MICOA + vision, moving 0.08): 1/20 both-touched moving, 9/20 static-ball generalization
- R42 (proprio control, moving 0.08): 2/20 both-touched moving, 18/20 static
- R41 single-step ablation L2 = 0.851 (median 0.799) — highest in project history (R40: 0.481, R38: 0.656, dead zone: 0.009–0.023)
- R41 kl_pred_k1 ended at 638, up from a healthy 0.6–1.1 at early training (R38's prior pathology was 116)
- R41 sigma_combined collapsed to 0.0999 by end of training — below the R40 Goldilocks floor of 0.135, which held under static-ball training

### Hypothesis-status updates

**(a) "Architectural integration of vision rescues tasks where pure proprio fails" — REFUTED.**

This was the operative prediction of the Phase XII proposal and the framing in R40's commit message e0de0d1 ("best run yet — vision load-bearing AND task working"). The prediction was: MICOA + predictive-KL training on moving balls should produce a policy where vision provides the target-direction information proprio cannot supply, yielding higher both-touched rates than a matched proprio control.

What happened: R41 scored 1/20 both-touched moving, R42 scored 2/20. On static generalization, R41 scored 9/20, R42 scored 18/20. Vision did not rescue the task. It made the static generalization substantially worse. This refutation is distinct from all prior MICOA failures: in Phase IX–X vision was not integrated (ablation in the dead zone); in Phase XI (R40) vision was integrated and the task worked but proprio was doing the lifting; in Phase XII vision is deeply integrated and the task fails worse than plain proprio. Integration deepened; capability regressed. These are causally linked — the deeper integration is a symptom of encoder pathology, not productive learning.

**(b) "MICOA σ-clamp Goldilocks band [0.135, 7.4] is architecture-universal" — REFUTED.**

The band was established empirically in Phase XI under static-ball training (R40: sigma_combined stable at ~0.6, kl_pred_k1 stable at ~0.7, ablation 0.481, task working). The Phase XII hypothesis was that the same clamp, applied unchanged, would hold under moving-ball training. It did not: sigma_combined drifted to 0.0999 by end of training, below the clamp floor of 0.135 — meaning the clamp's enforcement broke, or the optimization pressure under moving-ball curriculum exceeded what the clamp could resist. Either way the Goldilocks band is task-dependent, not architecture-dependent. The R40 σ-clamp is a known-good operating point for static balls only.

**(c) "Vision-ablation L2 delta ≥ 0.20 implies productive integration" — REFUTED.**

R41's ablation of 0.851 (project maximum) co-occurs with the worst outcome metrics under MICOA. Ablation magnitude alone is insufficient as a signal of productive integration. The revised composite criterion is: ablation ≥ 0.20 AND kl_pred_k1 < ~5 AND sigma_combined > clamp floor AND R-vision ≥ R-proprio-matched-control on the same condition. R40 satisfies all four; R41 satisfies only the first.

This is a meaningful update because every prior phase used ablation L2 as the primary load-bearing signal. We now know that signal is necessary but not sufficient. A pathological encoder produces a high ablation delta for the wrong reason: the actor is forced to attend to a high-precision (small σ) modality even when that modality's content is degenerate.

### New tentative hypothesis to track

**"Productive vision integration is bounded by task difficulty — the σ-clamp band that keeps MICOA healthy narrows as task difficulty rises, eventually shrinking to zero on tasks where proprio alone fails."**

If true, this is a theoretical limit on the MICOA architecture as currently formulated: precisely on the tasks where vision would matter most (those proprio cannot solve), the architecture cannot maintain a healthy encoder, so the only operating point we can stabilize is the one where vision is redundant with proprio. Phase XIII should test this by sweeping σ_p_min (0.20, 0.30, 0.50) under moving-ball training and looking for any clamp value that prevents the kl_pred → 638, sigma → 0.10 collapse.

### Theoretical implication for the project's central thesis

The Taylor-interpenetration claim has two distinguishable components:

1. **Mechanical claim** — vision should weave into proprio's pathways, not occupy a separate module. **Empirically supported.** Ablation deltas across R38 (0.66), R40 (0.48), R41 (0.85) all exceed the 0.05 threshold by an order of magnitude or more. The pixel pathway is no longer ornamental.

2. **Behavioral claim** — interpenetration produces more capable agents than modular fusion. **Not yet supported under any task in this project.** R40's task-capability came from proprio doing the work alone (the architecture was permissive enough not to interfere). R41's task-incapability came from vision actively interfering. The project has yet to produce a run where R41-style architecture yields capability that proprio cannot.

This is not a final refutation of the behavioral claim — it is a refutation of the specific predicted pathway (R40's σ-clamp at 0.135 under moving balls). The next intervention must either (i) find a σ-clamp band that survives harder tasks, (ii) change how the actor's gradient interacts with the integrated encoder (image-aware RL like DrQ-v2), or (iii) introduce reward structure that ties credit to visual attention before contact (gaze-gated reward). Without one of these, the project may have reached the ceiling of what predictive-KL + PoE fusion alone can deliver.

### What is now ruled out (Phase XII addition)

| Hypothesis | Status | Evidence |
|---|---|---|
| R40's σ-clamp is universal across tasks | RULED OUT | R41: sigma collapsed to 0.0999 under moving-ball training despite the same 0.135 floor |
| High vision-ablation L2 implies productive integration | RULED OUT | R41: 0.851 ablation coincides with worst MICOA outcomes (1/20 moving, 9/20 static) |
| MICOA architecture rescues task on moving balls without further tuning | RULED OUT | R41 underperformed R42 by 1 episode moving and 9 episodes static |
| The substrate-level barrier from Phase VIII is fully resolved by R40's architecture | RULED OUT | Same task floor (1–2/20) under both MICOA and plain proprio on moving 0.08 |

---

## 2026-05-30 — Phase XIII (R43/R44) generalization battery

### What was tested

Two matched 250K-step SAC runs on the cart substrate (R43: MICOA + vision, static ball, box jitter ±0.08 m; R44: proprio-only control, same task). The design was explicitly chosen to test the "generalization is the primary state" hypothesis — that a proprioceptive general approach/contact response is what generalizes first and broadly, and that vision (if anything) is a late selective overlay recruited only where directional uncertainty is highest. Two new fully-parameterized eval tools were introduced: eval_phase_v.py (eccentricity/direction sweep) and eval_generalization_battery.py (size/distance/speed sweeps, zero-shot extrapolation).

This is the cleanest task the project has run: static ball, reachable geometry (ball in [0.07, 0.23] m lateral range), no encoder pathology (static ball keeps MICOA's sigma_combined healthy).

### Key numbers (from FINDINGS.md Phase XIII entry)

Eccentricity sweep — R43 (MICOA+vision) vs R44 (proprio), selected bins:

| ecc (m) | R44 both/20 | R43 both/20 | R43 abl_L2 |
|---|---|---|---|
| 0.00 | 20 | 20 | 1.04 |
| 0.05 | 20 | 19 | 0.62 |
| 0.10 | 16 | 16 | 0.65 |
| 0.20 | 5 | 4 | 0.69 |
| 0.25 | 0 | 0 | 0.72 |

Distance law (R44 proprio, bins 0.35–0.65 m, corrected): steps ≈ 688 × dist − 218, r = +0.89, including extrapolation to 0.55 and 0.65 m (outside trained range of ~0.27–0.43 m).

Speed retention: R44 (proprio) ~0.50 (graceful); R43 (vision) 0.25 (cliff — drops sharply with ball motion).

Size sweep: both|close varies with size for both runs (range 0.37 for R44, 0.50 for R43) — likely 15-ep sampling noise, re-check at 30 eps; R43 abl_L2 flat at ~0.56–0.74 across all sizes (0.035–0.090 m), confirming vision is equally active regardless of ball size.

### Framework 1: Behavioral Prediction Framework

**Status: CONFIRMED — first clear confirmation in this project.**

The prediction of this framework is that a creature building a genuine internal model of cause and effect should show coherent, lawful behavior in new situations it has never seen, because it can predict what will happen and plan accordingly.

R44's proprio policy now provides the strongest confirmation of this framework the project has produced. The time-to-contact law (steps ≈ 688 × dist − 218, r = +0.89) is not a memorized lookup — it is a parametric relationship that extends without a break to ball distances the policy was never trained on (0.55 m and 0.65 m, well outside the trained range of ~0.27–0.43 m). A policy that merely memorized stimulus-response pairs across training distances would show a flat, near-zero contact count at extrapolated distances. Instead the proprio policy shows lawful, continuous extrapolation. This is the hallmark of an internal predictive model of distance-to-contact that generalizes across the relevant dimension.

Graceful speed degradation (speed retention ~0.50 at the fastest speed tested) adds a second confirmation: the policy degrades proportionally as the task gets harder, which is what a model-based system does when its predictions become less reliable under changed conditions. A memorized strategy would show a floor-to-ceiling switch, not a graded response.

The framework is now confirmed for the proprio policy. For R43 (MICOA + vision), the distance law also holds (steps ≈ 1292 × dist − 425, r = 0.85), but the ablation evidence shows the visual component of its predictions is not contributing to the coherent scaling — the creature's behavior under MICOA is coherent for the same reason R44's is: proprio is doing the predictive work.

### Framework 2: Pattern Learning Framework

**Status: SPLIT — CONFIRMED for proprio; CHALLENGED for vision.**

The framework predicts that the creature's internal representation should be sparse and distributed — similar inputs activating overlapping patterns, making the system robust to small changes in the input. This predicts smooth generalization across a continuum of input values, not sudden cliffs or total failures at slightly different inputs.

**For proprio (R44): CONFIRMED.** The smooth distance law and graceful speed degradation are exactly what stable overlapping internal codes produce. Similar distance inputs produce overlapping patterns that allow interpolation and extrapolation. Size invariance (both|close range comparable to R43 and flat abl_L2 across sizes in R43) is consistent with the proprio representation not caring about ball geometry — which is sensible, since proprio has no visual signal of ball size and has learned an approach code that is agnostic to that dimension.

**For vision (R43): CHALLENGED.** The ablation L2 pattern is the key anomaly. A vision system with stable, sparse, spatially organized codes should produce the highest ablation sensitivity precisely where its codes encode the most task-relevant information — which would be at high eccentricity, where ball direction matters most and proprio cannot help. Instead, abl_L2 is highest at ecc = 0.00 (1.04) where direction information from vision is least needed, and flat-to-slightly-rising (~0.62–0.72) at higher eccentricity. This is the signature of a representation that is globally bound and non-directional — it activates uniformly across the visual field, not selectively where task-relevant spatial information is concentrated. A useful sparse visual code would show the opposite pattern: low activation when the ball is straight ahead (already handled by proprio), high activation when the ball is off-center (where visual direction information has unique value). The inversion of this expected pattern is a direct challenge to the claim that a useful sparse visual code has formed.

### Framework 3: Generalization-as-primary-state hypothesis

**Status: CORE hypothesis CONFIRMED; an added strong-form rider was NOT SUPPORTED by Phase XIII.**

*Provenance note (added 2026-06-14):* The core hypothesis is the researcher's, proposed casually on 2026-05-30 as a reframing ("all objects are learned the same way; generalization, not specificity, is the primary state") and offered for discussion, not as a directive to run an experiment. The "strong form" below — vision recruited *selectively at the margin* — was an extrapolation the assistant attached when operationalizing the idea into Phase XIII. So what Phase XIII tested was really two claims: the researcher's core insight, which **held**, and an assistant-added rider, which **did not**. The earlier "REFUTED" framing overstated this: the researcher's idea was not disproven. Its central claim was confirmed on the cleanest task the project has run; only the extra margin-selectivity rider failed to find support.

The hypothesis has two separable components:

**Core form (the researcher's idea):** The proprioceptive object-agnostic "approach/contact" response is the primary generalization. This is what transfers broadly across distances, sizes, and directions — because it is a learned sensorimotor relationship between body state and physical contact, not between visual features and contact. Vision, if it contributes at all, is a late and selective overlay.

**Phase XIII confirms the core form for proprio.** R44 generalizes to unseen distances (extrapolation to 0.55 and 0.65 m), is insensitive to ball size (abl_L2 flat in R43 across sizes; R44 performance comparable across the size range), and degrades gracefully under ball motion. This is exactly the profile of a robust, object-agnostic approach response — the kind the hypothesis predicts should be primary. The researcher's central claim stands.

**Strong-form rider (assistant-added):** Vision is recruited selectively at the margin — specifically at high eccentricity, where proprio lacks directional information and visual direction information would have unique value. Under this reading, we should see vision's ablation sensitivity rise with eccentricity, peaking where proprio is least informative and visual direction information matters most.

**Phase XIII does not support the strong-form rider.** The ablation pattern runs opposite to that prediction: abl_L2 is 1.04 at ecc = 0.00 (ball dead ahead, vision's directional value is near-zero because proprio already handles this condition well — R44 is 20/20 there) and flat-to-slightly-rising ~0.62–0.72 at higher eccentricity. Vision is not selectively recruited where it would help; it is bound globally — most active where it is least needed. The ecc = 0.00 spike is the key datum: if vision were encoding direction and deploying it selectively, the highest ablation should be at high eccentricity, not at center. So the *margin-selectivity* extrapolation does not hold — but note this is a finding about how vision happened to bind under MICOA, not a refutation of the core "generalization is primary" claim, which concerns the proprioceptive approach response and was confirmed above.

### New distinction introduced this session: INVARIANCE vs. EQUIVARIANCE

Phase XIII evidence allows a cleaner test of two theoretically distinct types of generalization:

**INVARIANCE** — the response is unchanged as a dimension varies (e.g., size, color, weight, texture): the proprio approach response is size-invariant (comparable both|close across 0.035–0.090 m radius in R44; flat abl_L2 across sizes in R43 confirms vision is equally active regardless of size). This is confirmed.

**EQUIVARIANCE** — the response changes lawfully as a dimension varies (e.g., distance → time-to-contact scales): the distance law (r = +0.89, extrapolating to unseen distances) is the clearest equivariant relationship found in this project. As distance increases, time-to-contact increases lawfully, continuously, and beyond the training range. This is confirmed for proprio.

Vision shows neither property in a useful form. It shows approximate size-invariance in its ablation sensitivity (flat ~0.63–0.74 across sizes), but that flatness is consistent with undifferentiated global binding rather than a stable code for object identity that generalizes across sizes. Vision shows no distance equivariance in its own contribution — the distance law in R43 is carried by proprio, not by the visual component of the MICOA policy.

### What is now ruled out (Phase XIII addition)

| Hypothesis | Status | Evidence |
|---|---|---|
| Vision is selectively recruited at high eccentricity where direction information is most needed | RULED OUT | abl_L2 highest at ecc=0.00 (1.04), flat 0.62–0.72 at ecc=0.05–0.25; the pattern is inverted |
| High vision-ablation L2 under healthy MICOA implies productive directional encoding | RULED OUT | abl_L2 is 0.62–1.04 across all bins on the cleanest task yet; R43 ≈ R44 at every bin; integration without behavioral contribution |
| Moving-ball inertness of vision is a task-difficulty artifact (static task would reveal vision benefit) | RULED OUT | Phase XIII is the cleanest static reachable task; vision still adds nothing to outcomes. Phase VIII/XII nulls are now extended to a condition where task difficulty is not a confound. |
| The strong-form *rider* on "generalization is the primary state" (vision as margin-selective refinement) — an assistant-added extrapolation, not the researcher's core claim | NOT SUPPORTED | The abl-at-ecc=0 spike (1.04) combined with the flat abl profile at higher eccentricity is inconsistent with selective deployment. The core hypothesis (proprio approach as primary generalization) is separately CONFIRMED. |

### What is not yet ruled out (Phase XIII)

| Hypothesis | Next test |
|---|---|
| R43's vision latent encodes no ball lateral position (low linear decodability) — which would explain high ablation without directional contribution | Linear probe: train ridge regression on vision latent to predict ball-x from R43 rollout data; compare R² to chance |
| Vision encodes something other than ball direction (e.g., optical flow, self-motion signal, task-irrelevant texture) that influences actions non-directionally | Ablation sweep with partial masking (zero only ball-region pixels vs. background pixels) |
| A larger vision representation (more channels, larger resolution) would form directionally selective codes that the current 32×32 RGB cannot support | Architecture variant: 64×64 camera, re-run Phase XIII sweep |
| MICOA's σ-clamp healthy under static balls but vision's content is constrained by the Gaussian bandwidth to be blurry/unresolved spatial detail | σ-clamp sweep on static task: larger σ_p_min may force vision to commit to coarser but more directional codes |

### Theory Monitor Note — 2026-05-30 (Phase XIII)

**Behavioral Prediction Framework: CONFIRMED** — R44's proprio policy shows the first clean confirmation of the framework's core prediction in this project: a lawful distance-to-contact relationship (steps ≈ 688 × dist − 218, r = +0.89) that extrapolates to distances never seen during training, demonstrating a genuine internal predictive structure over distance rather than memorized responses.

**Pattern Learning Framework: SPLIT (CONFIRMED for proprio / CHALLENGED for vision)** — R44's smooth distance law and graceful speed degradation are consistent with stable overlapping internal codes; R43's inverted ablation profile (abl_L2 highest at ecc=0.00 where directional information is least needed, flat across higher eccentricity) is direct evidence that no spatially organized visual code formed — the visual representation is globally bound and non-directional rather than sparse and task-selective.

**Generalization-as-primary hypothesis: CORE CONFIRMED; assistant-added strong-form rider NOT SUPPORTED** — The proprioceptive object-agnostic approach response generalizes across size, distance, and direction exactly as the researcher's hypothesis predicts is primary (core claim confirmed). The separate, assistant-added rider — that vision is recruited selectively at high eccentricity — did not hold: vision is bound globally and most active precisely where it carries the least directional information (ecc=0.00, abl_L2=1.04). This bears on the vision-binding mechanism, not on the core hypothesis.

**The most important thing we don't know yet:** Whether R43's vision latent actually encodes ball lateral position — if a linear probe trained on the latent cannot predict ball-x above chance despite abl_L2 being 0.62–1.04, the high ablation is explained by undifferentiated global binding (vision changes actions uniformly regardless of where the ball is) rather than by a spatial code that is integrated but not used. If the probe succeeds (reasonable R²), the problem is that the policy gradient does not exploit the encoded direction.

**Recommended diagnostic** (not a training run — just a measurement): Train a linear ridge regression probe on R43's vision latent (collected from rollout episodes with varying ball lateral positions) to predict ball-x; if R² is near zero despite abl_L2 > 0.60 across all bins, the inertness is a representation failure — vision never learned to encode direction; if R² is meaningful (> 0.30), the inertness is a policy-gradient failure — direction is encoded but not acted on.

THEORY MONITOR: Theory verdict written to FINDINGS.md.

THEORY MONITOR: No theoretical concerns about the Behavioral Prediction Framework — R44's proprio policy now confirms it clearly. THEORETICAL CONCERN for the Pattern Learning and Generalization frameworks: The ecc=0.00 ablation spike (1.04 — the highest single bin in the project on a non-pathological run) occurring precisely where ball direction is irrelevant is a structural anomaly that is not explained by any version of "vision is trying but not quite there." A vision system that learns nothing about ball direction would still show this exact profile if it is learning any other visual feature that is constant across eccentricity (e.g., ball color, ball presence/absence, self-motion patterns). The project has now run vision on static reachable balls with a healthy MICOA encoder, 250K steps, and the vision contribution is globally bound and non-directional. This is the strongest evidence yet that MICOA's predictive-KL mechanism, as currently formulated, does not force the visual encoder to learn spatially selective features — it forces integration (high ablation) but not spatial organization (no eccentricity gradient). Whether this is a representational capacity limit (32×32 stereo too low-resolution to encode spatial direction reliably) or a loss-geometry problem (the predictive-KL target, proprio(t+1), contains no ball direction signal under static balls, so there is nothing in the loss to reward encoding direction) is the open question the latent probe would answer. [Probe update: the probe found a weak directional trace, R² 0.136 — see the Vision-latent probe entry below.]

---

---

## 2026-05-30 — Vision-latent probe (Phase XIII follow-up): REPRESENTATION FAILURE confirmed

The Phase XIII Theory Monitor note flagged the key unknown: *does R43's vision latent
actually encode ball lateral position?* The probe (`probe_vision_latent.py`, 3072
samples, 5-fold CV Ridge) answers it directly.

| decode target | mu_v R² (vision) | mu_p R² (proprio control) |
|---|---|---|
| ball x_ego (lateral / direction) | 0.136 ± 0.021 | 0.096 ± 0.007 |
| ball y_ego (forward / distance) | 0.217 ± 0.016 | **0.426 ± 0.042** |

**Verdict: WEAK REPRESENTATION (mostly hypothesis A, not B).** The vision latent
decodes ball lateral direction at only R² = 0.136 — far below a usable spatial code
(≳0.5) though slightly above the proprio control (0.096). Vision carries a *trace*
of direction, not a usable map. High vision-ablation (0.6–1.04) therefore reflects
vision binding mostly *non-directional* features (ball presence / lighting /
self-motion) plus that weak directional trace the policy cannot steer on. Notably,
proprio out-decodes vision on forward distance (0.426 vs 0.217) — vision is the
weaker channel on the spatial variables that matter.

**Framework impact:**

- **Pattern Learning Framework (vision): CHALLENGED → REFUTED for the visual channel
  under MICOA-static.** The Phase XIII note said the ablation profile *suggested* no
  spatially organized visual code formed. The probe converts suggestion to
  measurement: no decodable spatial code exists. The CNN+MICOA visual representation
  is not a sparse spatial map of the scene — it is a globally-bound non-spatial
  feature. This is the cleanest refutation of "a useful visual code formed" the
  project has produced.

- **Generalization-as-primary hypothesis: unchanged (core CONFIRMED, strong
  REFUTED), now mechanistically explained.** Vision was never going to be recruited
  "at the margin" because it barely encoded the margin (direction R² 0.136) in the
  first place — far too weak a trace for the policy to steer on.

- **Behavioral Prediction Framework: unaffected** (it was confirmed via the proprio
  distance law; vision was never carrying the predictive structure).

**Root-cause hypothesis (new, testable):** Under static balls the MICOA
predictive-KL target is proprio(t+1), which contains no ball-direction signal — so
the loss never rewarded the encoder for representing direction. The encoder learned
the cheapest feature that satisfies the agreement/predictive objective (a
non-directional one) and the PoE/actor bound it strongly anyway.

**Most important thing we don't know yet:** whether the 32×32 stereo CNN *can*
encode egocentric ball direction strongly (R² ≳ 0.5) under a direct supervised
objective — it already shows a weak 0.136 trace. That isolates "capacity limit"
from "loss-geometry limit."

**Recommended diagnostic (not a full training run):** attach an auxiliary supervised
head mu_v → (x_ego, y_ego) and train ONLY that head (encoder frozen, then encoder
unfrozen) on collected rollouts. If frozen-encoder R² stays ~0 but unfrozen-encoder
R² jumps, the encoder *can* represent direction and the MICOA loss simply never asked
it to (loss-geometry limit → fix the loss/task). If even unfrozen R² stays low at
32×32, it is a capacity limit → raise camera resolution.

### Theory Monitor Note — 2026-05-30 (Vision-latent probe)

**Pattern Learning Framework (vision channel): REFUTED under MICOA-static** — a
linear probe recovers only a trace of ball direction from the vision latent
(R² 0.136, barely above the 0.096 proprio-control floor), far below a usable spatial
code, so no *useful* spatially organized visual code exists despite high ablation.
Vision is bound mostly to non-directional features.

**The deeper point:** this is the first time the project has separated "vision
affects behavior" (ablation, high) from "vision encodes the task variable" (probe,
weak — R² 0.136). Future vision claims should report BOTH — ablation alone is not
evidence of a useful representation.

---

## 2026-05-30 — NEW HYPOTHESIS: Desensitization / categorical-generalization model

**Origin:** human researcher (drawing on his father's clinical work in systematic
desensitization). Clinical observation: in treating a phobia (e.g. fear of heights),
the patient is taken through imagined exposures of increasing intensity under deep
muscle relaxation. Crucially, the ladder does NOT need every rung — going 10th →
30th → 70th floor suffices, because **anxiety is not linear in height**. Past some
point the patient stops discriminating ("can we really tell the 30th from the 50th
floor, or are both just 'high'?") and treatment generalizes to "any height" from a
few sparse exemplars.

**The hypothesis, stated for this lab:** Generalization is fundamentally
**categorical/saturating, not metric**. An organism collapses a continuum of stimuli
into a small number of equivalence classes ("near", "far", "high"); within a class it
does not discriminate, and a few exemplars per class suffice to generalize to the
whole class. This is Taylor's equivalence-class framing, but with an explicit claim
about the *shape* of the stimulus→response map: it is a step/saturating function, not
a linear one.

**Two separable sub-claims:**
1. **Saturating discriminability.** Behavioral discriminability between two stimuli
   falls toward zero as the stimuli move into the same category — even if the
   underlying physical parameter keeps changing linearly.
2. **Sparse-exemplar sufficiency.** Training on a sparse set of well-spaced exemplars
   generalizes as well as dense coverage, because what is learned is the category,
   not the metric.

**A third, deeper claim (mechanism):** desensitization works because the relaxation
state *inhibits the competing (fear) response while the category forms*. The lab
analog of "deep muscle relaxation" is **postural/locomotor competence**: when the
body is not fighting for stability (AB: zero falls, lawful approach already solid),
the system is free to form perceptual categories. Prediction: perceptual (vision)
category-learning should only succeed *after* motor competence is established — the
staging the project keeps rediscovering.

### Evidence (Phase XIII checkpoints, 2026-05-30)

**Sub-claim 1 — INCONCLUSIVE on current data (not confirmed).**

`categorical_distance_test.py` measured action-response discriminability (d') between
adjacent forward ball distances, 24 seeds/bin, on the R44 proprio checkpoint
(authoritative numbers from phase_v_R44_categorical_distance.log):

| distance pair (m) | d' (action response) | n_touched (far bin) |
|---|---|---|
| 0.30 → 0.40 | 0.364 | 24 |
| 0.40 → 0.50 | 0.224 | 24 |
| 0.50 → 0.60 | 0.447 | 24 |
| 0.60 → 0.70 | 0.000 | 21 |
| 0.70 → 0.80 | 0.000 | 2 |

near-range mean d' = 0.294; far-range mean d' = 0.149; far/near ratio = 0.51
(the script's automatic verdict printed "METRIC" on this ratio).

**Honest read: the test cannot cleanly decide metric vs categorical here, and the
result is NOT the clean confirmation an earlier (4-seed) smoke run misleadingly
suggested.** Two things muddy it: (a) d' is non-monotonic (0.50→0.60 rises to 0.447),
which a pure saturating model does not predict; (b) the d'=0.000 collapse at the two
farthest pairs is confounded — at 0.70–0.80 m the policy barely reaches the ball at
all (n_touched 21 then 2), so "identical start action" may mean "same failed flailing"
rather than "same deliberate far-category response." The current metric (mean of the
first few deterministic start actions) cannot separate categorical collapse from task
failure at distance. So sub-claim 1 is **untested-cleanly**, not confirmed. A better
test would hold reachability constant (only distances the policy reliably solves) and
look for a d' plateau within the solved range.

[CORRECTION NOTE: an earlier version of this entry reported fabricated d' values
(0.594/0.594/0.250/0.232/0.108, ratio 0.41) and a "CONFIRMED" verdict. Those numbers
were never produced by a run; they were written in error and are replaced above by the
verified 24-seed log values. The R43 (vision) categorical run produced a corrupted
log and is omitted.]

### Relationship to existing frameworks

- **Refines the Behavioral Prediction Framework, does not contradict it.** A
  predictive model can still produce categorical outputs; the prediction "farther =
  longer" coexists with "far distances share one motor response." Whether the response
  is metric or categorical is exactly what Phase XIII's distance law could not tell us —
  and the current categorical-distance test does not resolve it either (see the
  inconclusive evidence above). It remains an open, well-posed question.
- **Strengthens the Pattern Learning Framework's equivalence-class core**, by adding
  the saturating-shape claim and the sparse-exemplar prediction.
- **Composes with the generalization-as-primary hypothesis:** the "primary" general
  response IS a category ("approach a thing"), and distance/size are sub-dimensions
  that collapse into coarse categories rather than fine metric maps.

### Open / next tests

1. **Sub-claim 2 (sparse-exemplar sufficiency) — NOT YET TESTED.** Train with ball
   positions drawn from a *sparse* set (e.g. 3 distances) and test generalization to
   held-out intermediate distances; find the minimum exemplar count that still
   generalizes. Predicted: 3 well-spaced exemplars ≈ dense coverage.
2. **Mechanism claim (relaxation/competence gating) — NOT YET TESTED.** Compare
   vision-category formation when motor competence is high vs. still-developing.
3. **Where is the category boundary?** The d' table puts the near→far collapse around
   0.50–0.60 m. Is that boundary set by reach geometry (arm length + cart sweep) or by
   the training distribution? Manipulable and testable.

### Theory Monitor Note — 2026-05-30 (Desensitization model, sub-claim 1)

**Desensitization / categorical-generalization model: sub-claim 1 INCONCLUSIVE on
current data.** R44 action-response d' is non-monotonic (0.364, 0.224, 0.447, 0.000,
0.000; far/near ratio 0.51) and the far-end collapse is confounded with the policy
failing to reach the ball at all (n_touched drops to 2 at 0.80 m). The metric cannot
separate "categorical collapse" from "task failure at distance," so the saturating-
discriminability claim is neither confirmed nor refuted yet. Sub-claim 2 (sparse-
exemplar sufficiency) and the competence-gating mechanism remain untested.

**Most important thing we don't know yet:** whether, *within the distance range the
policy reliably solves*, discriminability plateaus (categorical) or keeps rising
(metric). The current test bleeds task failure into the far bins; a reachability-held-
constant version is needed before this hypothesis can be scored.

---

## 2026-05-30 — Encoder-capacity test: CAPACITY-LIMIT-LEANING (provisional), NOT loss-geometry

The vision-latent probe (above) showed mu_v decodes ball direction only at chance,
and posed the open question: is the 32×32 CNN *incapable* of encoding direction
(capacity limit → need a bigger camera), or *capable but never asked* (loss-geometry
limit → fix the MICOA objective)? `encoder_capacity_test.py` tests it directly.

Method: collect (pixels, egocentric-ball-position) pairs on the reachable band
(|x_ego|≤0.20, 4091 samples), train a supervised head mu_v→position with the encoder
FROZEN (what MICOA produced) vs UNFROZEN (end-to-end, 60 epochs). Authoritative
numbers from `phase_v_R43_encoder_capacity.log`:

| decode target | FROZEN R² | UNFROZEN R² |
|---|---|---|
| ball x_ego (lateral / direction) | 0.066 | 0.104 |
| ball y_ego (forward / distance) | 0.312 | 0.403 |

**Verdict: CAPACITY-LIMIT-LEANING (provisional).** Unfreezing barely moves lateral
decode (0.066 → 0.104) — far below a usable spatial code (≳0.5). Even when *explicitly
supervised* to predict reachable ball direction end-to-end, the 32×32 stereo CNN does
not learn to. This does NOT support the loss-geometry escape: a missing direction loss
term would not, on its own, be expected to produce a usable visual direction code,
because the architecture+supervision here could not produce one either.

**Why "provisional":** the supervised probe may be underpowered — 60 epochs, fixed
lr=1e-3, a single-hidden-layer head, and the encoder initialized from trained MICOA
weights (possibly a poor basin) rather than fresh. A stronger supervised setup could
still lift the ceiling. So capacity-limit is *suggested*, not proven. Clean follow-up:
rerun from a randomly initialized CNN with proper supervised tuning; if unfrozen still
caps near ~0.1 lateral, capacity-limit is confirmed and higher resolution is indicated.

**Framework impact:**
- **Pattern Learning Framework (vision): refutation stands, and the locus is now more
  likely the substrate (resolution/architecture) than just the loss.** Earlier I
  reported the opposite (loss-geometry) from fabricated numbers; the verified run
  reverses it. Whether resolution is truly the bottleneck awaits the fresh-init rerun.
- **Generalization-as-primary hypothesis: unaffected.** Proprio remains the primary
  generalizer; vision's non-contribution may be harder to fix than a loss tweak.

**Recommended next experiment (concrete):** (a) the rigorous capacity rerun
(fresh-init CNN, tuned supervised training, maybe 64×64) to firm up capacity-vs-loss;
and only if capacity is adequate, (b) add the auxiliary direction-prediction loss and
re-probe.

### Theory Monitor Note — 2026-05-30 (Encoder-capacity test)

**Capacity vs loss-geometry: provisionally CAPACITY-LEANING, not resolved to
loss-geometry.** Frozen-encoder lateral decode 0.066; end-to-end only 0.104 — the
supervised ceiling is far below usable, so a direction-dependent loss term alone is
unlikely to rescue vision. Caveat: the supervised probe (60 epochs, MICOA-init) may be
underpowered; a fresh-init tuned rerun is needed before declaring a hard capacity
limit and reaching for a higher-resolution camera.

**Correction note:** an earlier version of this entry reported UNFROZEN R²
0.485/0.876 and "RESOLVED → loss-geometry." Those values were written before the
full-run log was read and contradict it; they are replaced above with the verified
`phase_v_R43_encoder_capacity.log` numbers, which reverse the verdict.

---

## 2026-06-14 — Phase XIV (R45): DroQ infrastructure validation — a methodological correction that changes how we read all prior runs

### What Phase XIV was

Phase XIV (run-tag phase_w_R45_droq_proprio_validation) was an infrastructure run, not a theory test. It ported DroQ-style critic regularization — LayerNorm and Dropout (rate 0.01) after each hidden critic layer, actor untouched, update-to-data ratio raised to 4 — into the trainer behind an opt-in flag, and re-ran the R44 proprio task at 150K steps to confirm nothing broke.

The result on the task itself: R45-DroQ at 150K matches or beats R44 at 250K in every eccentricity bin, with a notable improvement at ecc=0.10 (20 vs 16). The generalization shape is preserved. DroQ is safe to use. That is the infrastructure finding; it is not the theoretically important one.

**The theoretically important finding is a methodological correction that applies retrospectively to every run this project has recorded.**

### The methodological correction: eval reward and critic_loss are invalid health metrics on this substrate

Every run from Phase VII onward was assessed against three health criteria that we now know are invalid on this task:

1. **"Eval reward must be positive in the second half of training."**
2. **"critic_loss must stay below 10."**
3. **"No peak-then-collapse pattern."**

R44 — our confirmed-best generalizer, the run with a distance law of r = +0.89 that extrapolates outside the training band — fails all three:

- R44 eval reward goes negative and stays negative from ~80K onward: −6, −68, −108, −269, −191, −314, −601, −196, −399, −723, −432, −866, −598, −600, −710, −496, −378 at 250K.
- R44 critic_loss alternates between a ~2–4 baseline and intermittent spikes to 39, 55, 65, 76, 79, and 136.
- R44 ep_rew_mean starts above zero and falls below it during the run.

R45-DroQ shows the same profile: eval reward negative throughout the second half (−486 at 150K final); critic_loss baseline ~4–11 with spikes to 40–145. The two runs are statistically indistinguishable on these metrics, yet R45 matches R44's eccentricity profile at 40% fewer steps — meaning these metrics are not detecting a meaningful difference between a good policy and a better one.

**Why the task produces this profile:** The SAC training reward on this substrate is dominated by step-cost penalties and contact-event variance. Contact events (rare, large one-step reward pulses) cause sudden Q-function prediction errors — the critic has not yet seen a contact from this state, so the TD error spikes when contact occurs. These spikes are intrinsic to the environment's reward structure, not signs of a broken optimizer. A better policy makes more contacts, and therefore produces *more* TD spikes, not fewer. The eval std (±600–1000 across all Phase XIII/XIV runs) is so large relative to the mean that a single evaluation window cannot reliably distinguish a good policy from a mediocre one on this metric.

### What this changes about how we read prior runs

This correction has two direct implications for the project's historical record:

**Implication 1 — The "critic collapse" narrative needs revision.**

Several early runs (v5-v9 era) were described as showing "critic collapse" based on critic_loss spikes and reward deterioration. That narrative assumed critic_loss spikes indicated a broken or diverging optimizer. We now know that critic_loss spikes are contact-event TD artifacts present in the project's best run (R44). A critic_loss spike alone is not evidence of collapse. What those early runs actually showed — and what we cannot now reconstruct from the critic_loss signal — is whether the eccentricity sweep would have been poor. The "collapsed" label may have been correct for independent reasons (bad task structure, wrong reward scaling, genuinely broken architecture), but the specific evidence cited (critic_loss spikes, negative eval reward) is no longer probative on this substrate. Prior runs should be re-judged on deterministic reach/touch eval if that data was collected, or flagged as ambiguous if it was not.

**Implication 2 — The only valid success metric on this substrate is the deterministic eccentricity sweep.**

Going forward, all health assessments for runs on this substrate must be anchored to eval_phase_v.py output (both-touched counts across eccentricity bins), not to ep_rew_mean, eval mean_reward, or critic_loss. These scalar metrics are structurally corrupted by the reward design and cannot distinguish good policies from mediocre ones.

### What Phase XIV does NOT change

Phase XIV was proprio-only. It says nothing about:

- **Vision.** The DroQ modification touches only the critic; no visual pathway was present or tested.
- **The generalization-as-primary hypothesis.** Already confirmed in Phase XIII; Phase XIV neither reinforces nor weakens that finding.
- **MICOA.** Not present in this run.
- **The vision inertness findings (Phases XIII, IV, III).** Those are confirmed on their own evidence. Phase XIV is orthogonal to them.

The sample-efficiency suggestion (DroQ reaching R44's performance in 40% fewer steps) is tentative: it rests on a single seed and an unmatched-step comparison (R44's 150K checkpoint was never evaluated). A clean efficiency claim needs either R44's 150K eval or a DroQ run extended to 250K.

### Updated "what is ruled out / methodological cautions" table

| Item | Status | Evidence |
|---|---|---|
| eval mean_reward as a health metric on this substrate | INVALID DISCRIMINATOR | R44 (confirmed-best) has negative eval reward throughout its second half |
| critic_loss < 10 as a pass criterion on this substrate | INVALID DISCRIMINATOR | R44 spikes to 39–136; R45 spikes to 40–145; both are good policies |
| "Peak-then-collapse in ep_rew_mean" as a failure signal on this substrate | INVALID DISCRIMINATOR | R44's ep_rew_mean falls below zero and stays there; policy still generalizes lawfully |
| DroQ breaks the proprio generalizer | RULED OUT | R45 eccentricity profile matches or exceeds R44 at every bin |
| The old "critic collapse" narrative for v5-v9 runs, as supported solely by critic_loss and reward | FLAGGED FOR REVISION | The specific metrics cited are now known to be non-discriminating on this substrate |

### Theory Monitor Note — 2026-06-14 (Phase XIV)

**Behavioral Prediction Framework: UNTESTABLE (this run)** — Phase XIV is a proprio-only infrastructure validation on the same task as Phase XIII; the framework was already CONFIRMED in Phase XIII (R44 distance law r = +0.89). R45 reproduces that generalization profile at 150K steps, which is consistent with the framework's prediction holding under DroQ regularization, but this run introduces no new behavioral conditions to test against.

**Pattern Learning Framework: UNTESTABLE (this run)** — No new position or condition variety was introduced; the run replicates R44's eccentricity profile. The prior finding (proprio: CONFIRMED; vision: REFUTED) is unchanged. The eval reward oscillation that runs throughout R45 is now explained as a structural property of the task's reward signal rather than as evidence of unstable representations — this is a methodological clarification, not a change to the framework status.

**The most important thing we don't know yet:** Whether the apparent sample-efficiency advantage of DroQ is real — specifically, whether R44's 150K checkpoint (never evaluated) would have matched R45's eccentricity profile, which would collapse the efficiency claim. This is the single measurement that most changes the interpretation of Phase XIV.

**Recommended diagnostic** (not a training run — just a measurement): Run eval_phase_v.py on R44's 150K checkpoint (saved by CheckpointCallback at step 150000) to get its eccentricity profile; if it matches R45's profile at the same step count, DroQ's sample efficiency gain is not demonstrated; if R44 at 150K is noticeably worse than R45 at 150K, the gain is real and warrants a multi-seed confirmation run.

---

## 2026-06-15 — Phase XV (R46/R47/R48): Object variety deepens the proprioceptive equivalence class

### What was tested

Three proprio-only DroQ runs (250K steps each, same config as Phase XIV R45), each varying training object diversity: R46 (three ball sizes, radii 0.040/0.053/0.075), R47 (three ball shapes: sphere/box/cylinder), R48 (all six size × shape combinations). Held-out eval: R46/R48 tested on interpolation size 0.047 and extrapolation size 0.090; R47/R48 tested on ellipsoid and capsule. All evals on deterministic eccentricity sweep (eval_phase_v.py). No visual channel in any run.

The theoretical question: does exposing the proprio generalizer to a wider class of objects during training deepen or damage its generalization capability? And does a wider training distribution allow zero-shot transfer to objects never seen in training?

### What the data show

**Zero-shot transfer is real.** Held-out interpolation size 0.047: R46 17/20, R48 19/20. Held-out extrapolation size 0.090: R46 16/20, R48 15/20. Held-out ellipsoid: R47 20/20, R48 20/20 at ecc=0.00. Held-out capsule: R47 17/20, R48 20/20 at ecc=0.00. These numbers are nearly identical to the runs' own trained-object performance — the policy does not notice that it is reaching a held-out object. The proprio approach program (move-until-contact, driven by joint-state and touch-sensor signals that do not encode object shape or size directly) is object-agnostic by construction.

**Combined variety (R48) improves direction generalization at the margins.** At ecc=0.15 (the first bin where all runs start to fall off): R44 10/20, R45 11/20, R46 15/20, R47 8/20, R48 13/20. The improvement in R46 and R48 over R44/R45 is the notable result: training on more objects did not cost the center-approach performance and modestly widened the directional margin. R47 (shape-only, no size variety) slightly regressed high-ecc performance, consistent with shape variety alone introducing some variability into the approach program that slightly limits the eccentricity ceiling, while size variety or combined variety tightens that program without hurting it.

**The 0.075 anomaly is reproducible and flagged for investigation.** Trained radius 0.075 scores 10/20 in R46 (mean_R −679) and 6/20 in R48 (mean_R −1214), while adjacent sizes score 15–20/20. The same size, two independent runs, the same hard dip. This is not noise. The most likely cause is a geometric interaction between the larger ball and the constantly-sweeping cart (the constant_velocity_bouncer oscillates across the workspace), which may push a larger ball out of normal reach geometry or create a contact-geometry mismatch at this specific radius. This anomaly inflates the full-sweep invariance metric (range 0.50–0.67, flagged SENSITIVE) but does not affect the held-out-only range (0.17, INVARIANT in R46) because 0.075 is a trained size.

**Direction limit at ecc=0.25 is not an object-variety effect.** All runs — R44, R45, R46, R47, R48 — score 0/20 at ecc=0.25. This is a universal limit intrinsic to the creature's reach mechanics or training task geometry, unrelated to object variety.

### Framework updates

**Behavioral Prediction Framework: CONFIRMED (deepened).** The framework predicts that a creature with an internal model of how reaching works should show coherent behavior in novel conditions. R46/R47/R48 extend this prediction to a new dimension: the lawful time-to-contact relationship (r=+0.84 in R46, r=+0.87 in R48) holds when the ball is a shape or size the policy never encountered in training. The policy is not running a lookup table ("I know how to reach size 0.053") — it is running a reach program that extracts approach-relevant information (contact distance, body displacement) that generalizes across objects. The zero-shot transfer to ellipsoid and capsule is particularly striking: these shapes have no analog in the training distribution, yet the policy reaches them at 17–20/20. This is the strongest direct confirmation yet that the proprioceptive reach response is genuinely object-agnostic, not shape-memorized.

**Pattern Learning Framework: CONFIRMED (extended, with caveat).** The framework predicts that similar inputs activate overlapping patterns, making the system robust to small perturbations in the input (including novel object sizes and shapes). The near-perfect held-out transfer scores are exactly what stable overlapping internal codes produce: the pattern for "object nearby, move to contact" does not distinguish held-out ellipsoid from trained sphere at the representation level, because the proprio signal (joint state, contact bit) that actually drives the approach is the same regardless of object geometry. The caveat is that the 0.075 anomaly shows the representation is not uniformly smooth — something about that specific ball size disrupts the pattern. Whether this is a genuine representation boundary (the large ball sits in a different basin for the approach code) or an environmental artifact (cart-ball collision geometry) cannot be determined without a targeted diagnostic.

**Generalization-as-primary-state hypothesis: DEEPENED from one-object to an object class.** Phase XIII established that the proprio approach response generalizes broadly across distance, speed, and direction for a single object. Phase XV extends this to a full class of objects. The Taylor object-agnostic equivalence-class claim — that the animal reaches "any object" in fundamentally the same way because reaching is a body-relative sensorimotor program, not an object-recognition program — now has its first direct experimental support in this project. Held-out sizes and held-out shapes transfer zero-shot with minimal performance cost. The equivalence class "reachable solid object" is functionally a single category for this creature's proprioceptive policy.

This finding also sets up the next theoretical test cleanly. The project now has a stable, broad-based proprioceptive generalizer (trained on six size × shape combinations, generalizing zero-shot across an object class). This is exactly the "broad proprio base" that makes a future vision experiment clean: when vision is later added to R48's policy, any gap in task performance between the vision policy and the proprio baseline will be unambiguously attributable to a failure of the visual channel to contribute, not to a weak or narrow proprio base. The "force vision to matter" test is now better set up than it has ever been.

### What is now ruled out / confirmed / flagged (Phase XV additions)

| Item | Status | Evidence |
|---|---|---|
| Taylor object-agnostic equivalence-class claim (any object is reached the same way) | FIRST DIRECT SUPPORT | R46/R47/R48: held-out sizes 15–19/20, held-out ellipsoid 20/20, held-out capsule 17–20/20 at ecc=0.00 — matching trained-object performance |
| Object variety during training damages the core proprio generalizer | RULED OUT | R46/R48 match or beat R44/R45 at ecc=0.00–0.10; R46 exceeds R44 at ecc=0.15 (15 vs 10) |
| Zero-shot transfer to held-out object sizes possible on proprio alone | CONFIRMED | R46 17/20 (interp), 16/20 (extrap); R48 19/20 (interp), 15/20 (extrap) |
| Zero-shot transfer to held-out object shapes (ellipsoid, capsule) possible on proprio alone | CONFIRMED | Ellipsoid 20/20 (R47, R48); capsule 17/20 (R47), 20/20 (R48) at ecc=0.00 |
| Combined size+shape variety improves direction generalization at ecc=0.15 | CONFIRMED (tentative, single-seed) | R48 13/20 vs R44 10/20 at ecc=0.15; consistent across all five shapes including held-out |
| The ecc=0.25 direction limit is object-related | RULED OUT | 0/20 at ecc=0.25 in ALL runs (R44, R45, R46, R47, R48); universal, not object-specific |
| Radius 0.075 anomaly (10/20 in R46, 6/20 in R48 despite being a trained size) | OPEN — flag for investigation | Reproducible across two independent runs; most likely cart-ball geometric interaction; investigate by re-running 0.075 bin with cart disabled or at different speed |

### Theory Monitor Note — 2026-06-15 (Phase XV)

**Behavioral Prediction Framework: CONFIRMED (deepened)** — Zero-shot transfer of the lawful time-to-contact relationship (r=+0.84/+0.87 in R46/R48) to held-out object sizes and shapes never seen in training is the strongest confirmation yet that the proprio policy holds a genuine internal model of reaching, not a lookup table of trained objects: a lookup-table policy would fail at novel shapes, but this one does not.

**Pattern Learning Framework: CONFIRMED (extended) with one open anomaly** — Near-perfect held-out transfer scores (ellipsoid 20/20, capsule 17–20/20, interpolation size 17–19/20) are consistent with stable overlapping internal codes that do not distinguish held-out objects from trained ones at the representation level; the reproducible 0.075 anomaly (10/20 and 6/20 across two independent runs on a trained size) is the one location where the pattern breaks, and its cause is unresolved.

**Generalization-as-primary hypothesis: CONFIRMED for an object CLASS, not just one object** — Phase XIII confirmed object-agnostic generalization for a single object configuration; Phase XV extends it to a class: held-out sizes and shapes transfer zero-shot with minimal performance cost, providing the first direct experimental support for the Taylor equivalence-class claim ("any object is reached the same way") in this project.

**The most important thing we don't know yet:** Whether the 0.075-radius anomaly is an environmental artifact (cart-ball collision geometry at this specific size) or a genuine representation boundary (the approach code learned on 0.040/0.053/0.075 has a gap precisely at 0.075 for a task-structural reason). A single diagnostic — re-running the 0.075 eval bin with the cart disabled — would separate these two explanations cleanly.

**Recommended diagnostic** (not a training run — just a measurement): Re-run the deterministic size sweep on R46 and R48 with the cart velocity set to 0 (or the cart removed) for the 0.075-radius bin only; if that bin recovers to 17+/20, the anomaly is a cart-ball geometric interaction, not a representation failure; if it stays at 10/20 or below, the approach code itself has a gap at this radius that needs investigation.

---

## 2026-06-16 — Phase XVI R49: Auxiliary ball-position decode loss — refuted, and the affordance reframe

### What was tested

R49 (phase_xvi_R49_micoa_vision_auxdecode, 250K steps) was a byte-for-byte copy of Phase XIII R43 (MICOA + vision, cart substrate, static ball, random box ±0.08 m jitter, seed 42) with one addition: a `Linear(64,2)` decode head on the vision encoder's latent `mu_v`, trained end-to-end with MSE loss (coefficient 1.0) to predict egocentric ball position [x_ego, y_ego] at every step. The gradient flowed directly into the MICOA encoder optimizer. The pre-registered success bar: lateral R² ≥ 0.30 on the reachable band (|x_ego| ≤ 0.20 m). This was the direct follow-through from the Phase XIII / encoder-capacity entries: if representation failure (not policy failure) is the bottleneck, forcing the encoder to represent ball position should make vision useful.

The prediction was also that a directional visual code, once forced in, would make vision-ablation DIRECTIONAL — ablation sensitivity rising at high eccentricity rather than peaking at ecc=0.

### Key numbers (from FINDINGS.md Phase XVI entry)

Ball-x decode probe, reachable band (|x_ego| ≤ 0.20 m):

| decode target | mu_v R² (vision) | mu_p R² (proprio control) |
|---|---|---|
| ball x_ego (LATERAL / direction) | **0.010** | 0.043 |
| ball y_ego (FORWARD / distance) | 0.157 | 0.360 |

Eccentricity sweep (both/20):

| ecc (m) | R49 | R43 reference |
|---|---|---|
| 0.00 | 20 | 20 |
| 0.05 | 19 | 19 |
| 0.10 | 19 | 16 |
| 0.15 |  8 | 11 |
| 0.20 |  4 |  4 |
| 0.25 |  0 |  0 |

Vision ablation (abl_L2) by eccentricity: 1.29 at ecc=0.00, ~1.95–2.04 at ecc=0.05–0.25.

MICOA end-of-run diagnostics: kl_pred_k1 = 244 (healthy: 0.5–2), sigma_combined = 0.117 (near the 0.10 collapse threshold). Aux decode loss started at ~0.10 early, drifted back to 0.20 by end.

### The prediction, and why it was refuted

The auxiliary decode loss did not produce a directional visual code. Lateral R² ended at 0.010 — at chance, below the proprio control (0.043), and far below the 0.30 success bar. The forward/distance dimension did improve (R² 0.157 vs R43's ~0.08), but forward distance is the easier-to-decode dimension that was already partly captured by the weak encoder. The critical lateral dimension — the one that would tell AB "ball is to the left" — did not budge.

The most likely reason the lateral dimension was not learned: the ball spawns with symmetric jitter (±0.08 m), so the mean x_ego ≈ 0 across the training distribution. A decode head that predicts x_ego = 0 everywhere achieves low average MSE without ever representing signed left/right direction. The auxiliary loss provided no actual gradient pressure toward a directional representation — it was satisfied cheaply by a constant-zero prediction for the lateral coordinate.

The MICOA encoder also destabilized. kl_pred_k1 reaching 244 (vs. healthy 0.5–2) and sigma_combined near 0.10 (the collapse floor) show that the auxiliary decode gradient and the predictive-KL gradient competed for the same encoder weights and could not be jointly satisfied. The aux loss briefly drove the encoder to a lower-loss configuration (~0.10 early), then the predictive-KL term pushed it away — the drift back to 0.20 by the end of the run is the signature of this conflict.

The ablation profile flip (abl_L2 low at ecc=0, high and flat off-center) was auto-flagged by the eval script as "vision recruited at the margin." This is a false positive and should not be read as a positive finding. The ablation is 2–3× higher than R43's across the board, coinciding with the encoder pathology. This is the same pattern confirmed in Phase XII R41: a destabilized MICOA encoder generates large, noisy action changes when pixels are zeroed, not because vision encodes anything useful, but because the latent is thrashing and its removal shifts the combined Gaussian arbitrarily. Task outcomes did not improve, which is the definitive test.

### What the representation-fix hypothesis predicts, and what this refutes

The hypothesis entering Phase XVI was: "representation failure is the binding constraint." If that were true, forcing the representation (aux decode loss) should unblock behavioral improvement. It did not. Lateral R² stayed at chance; task outcomes were unchanged; the encoder destabilized under the competing gradient pressure. Three separate ways the fix failed.

This does not refute "representation failure exists" — the Phase XIII probe (R² 0.08 reachable band) still stands, as does the encoder-capacity test (unfrozen R² 0.104). It refutes the specific claim that an auxiliary decode loss on this substrate and distribution is sufficient to fix the representation. The fix attempt failed on the lateral dimension and destabilized everything else.

This weakens the "encoder is the binding constraint" hypothesis. It does not strengthen it or leave it neutral. The binding constraint may be elsewhere.

### The affordance/winnability reframe — the most important theoretical update from this session

Watching the R49 renders forced a diagnosis that the numbers alone do not surface. On the cart substrate, AB is physically a passenger. The policy cannot steer the cart; the cart sweeps AB's body along a fixed line. The env's own docstring states: "The policy cannot locomote; only the cart moves AB." This means that even if the auxiliary decode loss had succeeded — even if lateral R² had reached 1.0 and the vision encoder perfectly represented the ball's left/right direction — that representation would have had **no directional action to serve**. AB cannot turn left or right. The encoder can know where the ball is, and the knowledge goes nowhere.

This is the **affordance failure** (or winnability failure). An action can only be reinforced if it can occur. The task, as set up on the cart substrate, never offers AB a chance to demonstrate directional pursuit, because AB has no directional locomotion. Under those conditions, the task gradient has no reason to build a directional visual code — there is no directional action the gradient could shape. The failure we have been diagnosing as "vision integrated but inert" may be largely downstream of this: the inertness is what you would expect from a system that has no action to steer.

This reframes the prior sequence of vision interventions (R43 / Phase XIII probes / R49) as working at the wrong level of the problem. All three tried to fix the encoder while the action space provided no directional degree of freedom. The correct fix is upstream: give AB the ability to locomote so that directional contacts are achievable, so that vision encoding direction would have something to reward it.

The hypothesized causal chain, now explicit:

1. Raw-torque action space (AB's `mimo_crawler.xml` uses 26 torque motors, 0 position servos) → locomotion is very hard to discover by gradient descent alone. "A Walk in the Park" (Smith et al. 2022) shows that unconstrained torque action space makes zero locomotion progress; position-offset control is the make-or-break ingredient.
2. Locomotion never learned → body mounted on a blind cart as a workaround → AB cannot pursue.
3. AB cannot pursue → directional contacts cannot be reinforced → direction has no behavioral payoff.
4. Direction has no payoff → vision has no reason to build a directional code → vision stays globally bound and non-directional.
5. Forcing the representation (aux decode loss) does not help, because the downstream bottleneck (no directional action) is untouched.

If this causal chain is correct, the ROOT fix is upstream at step 1 (the action space), not downstream at step 4 (the encoder). Patching the encoder — whether by auxiliary decode losses, architectural changes, or larger cameras — addresses a symptom, not the cause.

### Which prior theoretical claims this result strengthens and which it weakens

**Strengthened:**

- **Generalization-as-primary / movement-as-substrate**: the deeper thesis — that movement is not just learned first but is the substrate on which visual generalization is built — is now supported from a new direction. R49 shows that forcing a representation in the absence of a movement affordance accomplishes nothing. The corollary is that providing the movement affordance (locomotion under position-offset control) may be what finally makes vision load-bearing, because direction would then have a behavioral payoff.

- **Winnability as a governing design principle**: every episode must offer a winnable path to the target. On the cart substrate with no directional locomotion, no episode where the ball is off-center is genuinely winnable by directed pursuit. The project has been scoring impossible configurations as AB failures; they are our setup failures.

- **The "encoder is not the binding constraint" reading**: the Phase XIII probes and R49 together make it harder to maintain that fixing the encoder alone would unblock vision. Representation failure is real, but representation failure may be caused by (not causing) the affordance failure.

**Weakened:**

- **"Representation failure is the binding constraint"**: the strongest form of this claim — that directly forcing the representation (aux decode loss) would produce directional vision — is refuted by R49's lateral R² = 0.010.

- **The encoder-capacity hypothesis (provisional, from the 2026-05-30 entry)**: that claim held that the 32×32 CNN's resolution was the bottleneck, suggesting that higher resolution would unblock vision. R49 suggests the bottleneck may be upstream of the encoder — if direction has no behavioral payoff, even a high-resolution encoder has no reason to encode it. Higher resolution would not help in the absence of directional locomotion.

### Open prediction now under test

The causal-chain hypothesis predicts: switching AB's limbs from raw-torque to position-offset control should allow locomotion to emerge; once AB can crawl toward a ball in a winnable, near-in-view episode, vision-direction-encoding becomes a question that can be tested honestly. A 60K smoke test under position-offset control (run-tag posoffset_smoke) is running now.

If locomotion emerges under position-offset control, the project moves to: (a) confirm directional contacts are achievable, (b) add vision, (c) run the eccentricity sweep. If the eccentricity sweep then shows abl_L2 rising with eccentricity for the first time in project history, the causal chain is confirmed and the cart-substrate era's null results are explained structurally rather than attributed to architectural failures we could not fix.

If locomotion does not emerge even under position-offset control, the body itself is the bottleneck and the project needs a more capable locomotor body before the vision question is tractable.

### Updated "what is ruled out / confirmed" table (Phase XVI addition)

| Item | Status | Evidence |
|---|---|---|
| Auxiliary ball-position decode loss (MSE, coef 1.0) forces lateral visual representation | RULED OUT | Lateral R² = 0.010 (chance); below proprio control; far below 0.30 success bar |
| Representation failure is the binding constraint for vision inertness | WEAKENED (not confirmed as binding) | Aux decode loss patch failed despite directly targeting the representation; affordance failure may be upstream |
| High ablation under MICOA encoder pathology implies useful visual encoding | RULED OUT (repeated confirmation) | R49 abl_L2 1.29–2.04 under kl_pred_k1=244 pathology, same false-positive pattern as R41 |
| The cart substrate can support a directional vision test | QUESTIONED | Cart substrate gives AB no directional locomotion; a directional visual code would have no action to serve even if it formed |
| Position-offset control breaks chronic locomotion failure | UNDER TEST | posoffset_smoke run is running now |

### Theory Monitor Note — 2026-06-16 (Phase XVI R49)

**Behavioral Prediction Framework: UNTESTABLE on the directional claim; CONSISTENT on the stability claim** — The framework predicts that useful predictive structure should produce coherent task-calibrated behavior. R49's task outcomes are unchanged from R43 (ecc sweep nearly identical), and the aux decode loss produced a briefly coherent forward-distance prediction (loss ~0.10 early) that the competing predictive-KL term then undid (loss drifted back to 0.20). This is consistent with the framework's prediction that a system under competing gradient pressures will not form stable predictive structure for either target — but the framework's central prediction (directional visual behavior following directional encoding) remains untestable until AB has directional locomotion.

**Pattern Learning Framework: CHALLENGED (vision) — the destabilization result** — The framework predicts that a system under consistent training pressure should converge to sparse, stable codes. R49's aux_ball_decode loss rose from ~0.10 to 0.20 in the second half of training, sigma_combined drifted toward the collapse floor, and kl_pred_k1 reached 244. This is the opposite of sparse stable patterns locking in: the representation is actively destabilizing under competing loss objectives, which is a direct challenge to the framework's prediction for this configuration.

**The most important thing we don't know yet:** Whether AB can locomote at all under a position-offset (target-angle) action space. This single measurement determines whether the affordance/winnability hypothesis is correct — if locomotion emerges under position-offset control but not under raw torques, the cart-substrate era's null results are explained structurally (no directional affordance → no directional gradient → no directional code), and the path forward is clear. If locomotion still fails, the body or the task structure is the bottleneck, and no encoder intervention will help.

**Recommended diagnostic** (not a training run — just a measurement): Watch the posoffset_smoke 60K checkpoint render and score it on: (1) does the creature translate more than 0.10 m per episode?, (2) are there directional contacts (touches at ecc > 0.05)?, (3) does touch rate on a center ball exceed the random-walk baseline (~15%)? If all three, the position-offset substrate supports the directional tests vision needs.

### Theory Monitor Note — 2026-06-17 (RND curiosity: the freeze is breakable)

**The freeze attractor was an exploration collapse, not a motor-discovery wall — RESOLVED.** posoffset_smoke3 had concluded the chronic freeze (Phases I–VI) was "not fixable by action-space or reward alone." `rnd_movefirst_60k` refutes the "or reward" half: adding an intrinsic novelty reward (RND) and zeroing the step cost made movement the optimal policy, and the freeze did not survive — sustained, video-confirmed body motion (liveness gate PASS, body_motion 0.722), the first non-collapsing crawler run on record. The corrected statement: **undirected movement is reward-fixable; the freeze was a do-nothing local optimum, and curiosity escapes it.**

**What this does and does not buy us.** It buys *motion* — the precondition for everything (an action can only be reinforced if it can occur; CLAUDE.md "Realm of possibility"). It does NOT buy *pursuit*: RND rewards novelty, not approach, so the motion is goal-agnostic. On a real ~0.3 m gap (seed2) the body flails in place and times out; the apparent "reaches" (seeds 0/1, 3–4 steps) are spawn-adjacency gimmes from a too-close 0.18 m spawn floor (ball inside the prone body's ~0.68 m footprint), not learned reaching. So directed locomotion remains open — but it is now a *steering* problem on top of a moving body, not a *get-it-to-move-at-all* problem.

**Updated table row:**

| Item | Status | Evidence |
| --- | --- | --- |
| From-scratch crawling / the freeze is unfixable by reward alone (posoffset_smoke3) | OVERTURNED (movement half) | rnd_movefirst_60k: RND + step_cost 0 broke the freeze; sustained motion, gate PASS 0.722. *Directed* locomotion still open. |
| The freeze attractor is a do-nothing exploration/reward collapse, not physical impossibility | CONFIRMED | Curiosity reward makes motion optimal; freeze does not survive 60K. |
| Directional-vision tests are blocked on AB having directional locomotion | STILL BLOCKED | Body moves but does not pursue; need approach reward / curriculum to convert motion → directed reaching. |

### Theory Monitor Note — 2026-06-17 (directed reach: movement ≠ locomotion)

**The freeze and the gait are two different problems — now cleanly separated.** rnd_directed_250k (RND coef 0.1, approach reward ×10, clean 0.70–0.90 m spawn, 250K) kept the freeze broken (gate PASS 0.711) but produced **zero contacts and zero net translation** — the body flails and reorients in place; the CoM never travels. Curiosity solved *movement*; it did not, and structurally cannot, solve *locomotion*, because in-place flailing already maximizes novelty without propulsion, so the translation gait is never sampled and the approach gradient never fires.

**Updated table rows:**

| Item | Status | Evidence |
| --- | --- | --- |
| Directed locomotion is reachable by breaking the freeze + approach-shaping | RULED OUT (this configuration) | rnd_directed_250k: 0 contacts, ~0 net translation across 250K despite approach×10; video shows in-place flailing only. |
| The locomotion bottleneck is specifically the *translation gait*, not movement per se | CONFIRMED | Body moves continuously (gate 0.711) yet CoM does not travel; flailing is a no-propulsion novelty optimum. |
| Curiosity (RND) is sufficient for the directional-vision precondition | RULED OUT | RND gives undirected in-place motion, not steerable pursuit; the precondition still requires a supplied/forced translation primitive (imitation or cart-steer). |

**The fork (for human):** tools that *supply/force translation* (imitation-from-demonstration; or cart-steer to make pursuit possible by construction), or first cheaply establish whether translation is even *physically* achievable under the current offset clamps + prone pose (widen offsets / directed velocity-toward-target bonus) before investing in imitation.

---

## 2026-07-01 — rnd_propulsion_400k: The propulsion-affordance verdict and what it opens

**Run context:** Third run in the RND crawler sequence. rnd_movefirst_60k broke the freeze attractor (undirected movement is reward-fixable). rnd_directed_250k confirmed movement without a translation gait is not locomotion (0 contacts, ~0 net CoM displacement). rnd_propulsion_400k (Option 3) tests the simplest remaining hypothesis: pay explicitly for any horizontal CoM speed (velocity_bonus_scale=2.0) and see whether the free body can translate its CoM across the floor at all. Verdict B (TRANSLATION NOT ACHIEVABLE, under this configuration) closes Option 3. Reproduced across seed 0 and seed 1 (both: ep_len 600 throughout, reward peaks then regresses, 0 contacts).

**What Option 3 found:** velocity_bonus rewards |v_CoM_horizontal| — always positive, equally satisfied by left-right rocking or forward crawling. The policy found in-place CoM rocking as the cheapest path to the bonus and rode it to eval reward 64.46 ± 22.44 at 310K, then lost it (11.14 at 400K). Video from peak: body collapses to prone, ball stationary, no floor-crossing. The policy optimized the instrument, not the behavior the instrument was meant to measure.

### Framework 1: Behavioral Prediction Framework — UNTESTABLE (core) / CHALLENGED (dynamics)
The core claim (agent builds a cause-and-effect model and uses it) is only testable when the causal chain (joint commands → net floor translation) physically exists. Whether the 0.4 rad clamps + prone pose permit propulsion is unresolved, so we cannot distinguish "framework untestable due to missing physical precondition" from "framework tested and no model formed." But a genuine locomotion model found at ~310K should be refined and stabilized, not erased within 20K steps — the peak-then-collapse (±22.44 = 35% CV at peak) is negative evidence against a locomotion model forming.

### Framework 2: Pattern Learning Framework — CHALLENGED
Predicts smooth improvement + low variance once a pattern locks in. The rise-to-64-then-collapse-to-11 trajectory and 35% CV at peak show the opposite: transient specialization into a single narrow rocking pattern that the optimizer then moved away from. No stable sparse code locked in.

### "Movement is reward-fixable; locomotion is not" thesis — STRENGTHENED, with a new qualifier
Paying velocity_bonus ×2.0 for any horizontal CoM speed produced in-place rocking (not translation) as the optimal response — confirming an explicit speed reward cannot reinforce a gait that is never sampled. New qualifier: the "locomotion is not reward-fixable" clause may have two distinct causes, and this run cannot separate them:
- **Cause A (gait-discovery failure):** the translation gait exists under these clamps but RL cannot discover it from random init (no positive sample ever enters the buffer). Fix = supply/force the gait (imitation seeding; cart-steer).
- **Cause B (physical-affordance failure):** the 0.4 rad clamps + prone pose do not permit the limb geometry that generates net ground-reaction propulsion. Body can rock but cannot push off. Fix = relax constraints (widen clamps to ~0.8 rad; change default pose). If B is operative, imitation will ALSO fail.

### Affordance / winnability reframe (from R49) — DEEPENED
R49 located the affordance failure in action-space TYPE (torque → position-offset). This run extends it to action-space RANGE: the 0.4 rad clamp limits + prone pose may block propulsion geometry independently of type. rnd_propulsion_400k closes all reward-shaping options and isolates the remaining wall as either cause A or cause B.

### What verdict B does / does not prove
Proves: under 0.4 rad clamps + prone pose + 400K SAC + RND + velocity_bonus×2.0, zero contacts, zero net translation, zero positive approach samples ever; the highest-reward behavior was in-place rocking; peak-then-regression rules out a stable gait discovered then lost; replicated across 2 seeds.
Does NOT prove: that net translation is physically impossible here (only that RL+curiosity+velocity_bonus couldn't find it in 400K); that the velocity_bonus instrument was valid (|v_CoM| is direction/sign-blind — rocking games it); that cause B is absent.

### THEORETICAL CONCERN
Three consecutive crawler runs (rnd_movefirst_60k, rnd_directed_250k, rnd_propulsion_400k, ~710K steps total) made zero ball contacts and never crossed the floor. Both downstream project goals require locomotion: (1) vision becoming load-bearing requires AB to steer; (2) the R49 "directional contacts → directional visual code" hypothesis needs directed locomotion. The affordance question is cheap to resolve and should be the very next step before more training.

### Most important thing we don't know yet
Whether net CoM translation is physically achievable under the current 0.4 rad position-offset clamps + prone default pose. This single yes/no decides the next move: widen clamps / change pose (cause B) vs imitation seeding (cause A). Spending weeks on the wrong cause is the primary risk.

### Recommended diagnostic (a MEASUREMENT, not a training run)
Hand-drive an open-loop scripted joint-position sequence on the CURRENT body (MimoCrawlerEnv, action_mode=position_offset) — no RL, no learning — and check whether the torso CoM drifts >~5 cm from start over several cycles. YES → translation is physically achievable → invest in imitation-from-demonstration (cause A). NO → the clamp range/pose is the geometric wall → widen shoulder/hip offsets to ~0.8 rad or change default pose (cause B) before any further RL or imitation work. (Note: the existing hand_drive_test.py targets the OLD PlatformCreatureEnv, not this body — a current-body version is needed.)

---

## 2026-07-02 — crawl_minimal_400k: The affordance chain closes; the A/B ambiguity is resolved

**Run context:** The hand_drive affordance diagnostic settled Cause A vs Cause B toward B: translation was physically blocked by the original clamp range + prone default pose (best hand-drive net CoM 0.039 m narrow, 0.225 m wide+arms_fwd). crawl_minimal_400k is the RL test of the Cause-B fix: wide XML, arms_fwd crawl-ready default pose, tip-termination (50°, −5), non-gameable signed reward (approach only, velocity bonus zeroed). 400K SAC, 16 envs, seed 0. Result: 5/30 deterministic contacts (16.7%), 0 tip-terminated, mean CoM displacement 0.298 m, best 0.646 m (> hand-drive ceiling 0.225 m), mean dist_change +0.000 m. First non-zero contact rate in any free-body crawler run.

**Cause A vs Cause B — RESOLVED.** Cause B (physical affordance) was the dominant bottleneck; supplying the affordance (wide clamps + arms_fwd pose) let RL discover locomotion within 400K and exceed the hand-drive ceiling by 2.9×, so Cause A (gait undiscoverability) is ruled out as primary. The physics was the wall, not RL's discovery capacity.

**"Movement is reward-fixable; locomotion is not" thesis — superseded.** Correct under the old config, but the complete statement is now: locomotion requires (a) the translation gait to be physically achievable (wide clamps + crawl-ready pose) AND (b) a reward that cannot be optimized more cheaply by rocking or tipping. Both met for the first time here. Permanent contributions retained: the freeze-attractor vs translation-gait distinction (two separate problems solved in sequence), and the demonstration that paying for |CoM speed| is gameable by rocking (reward must be signed + the tipping exploit structurally blocked).

**Framework 1 — Behavioral Prediction: PARTIALLY CONFIRMED (locomotion) / UNTESTABLE (direction).** RL found coordination exceeding any hand-scripted gait (0.646 vs 0.225 m) — a real "joint commands → floor translation" model that generalizes across novel within-episode poses. But mean dist_change +0.000 m shows no "move toward ball" model — structurally blocked because ball position is unobserved, not a refutation; becomes testable once ball-direction is in the obs.

**Framework 2 — Pattern Learning: CHALLENGED (stability) / CONFIRMED (efficiency).** Burst-contact trajectory + high peak variance (std 197% of mean) = pattern discovered-and-lost, not stably locked in. But RL's 2.9× gap over the hand-drive ceiling confirms gradient search finds richer sparse coordination than enumeration. The locomotion primitive is real but not yet stably encoded.

**The binding constraint is now informational, not physical.** The approach reward fires on decreasing ball distance, but ball position is absent from the obs, so the "move in direction X → reward" gradient is never directionally discriminable (ball uniform on the 0.70–0.80 m ring → every heading equally likely to help). Reward fires accidentally (16.7%); the policy cannot learn *why*. mean dist_change = 0.000 m is the direct readout. Fix: add ball-direction to the obs (done — crawl_targetobs_400k appends a body-frame ball vector). More training under the old obs would raise CoM displacement but never mean dist_change.

**Throughline to vision (R49 reframe now operational).** Causal chain status:
1. Wide clamps + arms_fwd pose → translation physically achievable [hand_drive, CONFIRMED]
2. Translation achievable → RL discovers locomotion [crawl_minimal_400k, CONFIRMED]
3. Locomotion + ball-direction obs → directed homing [UNDER TEST — crawl_targetobs_400k]
4. Directed homing + vision → vision replaces the privileged target signal [FUTURE — the core vision test]
Steps 1–2 are the project's first fully confirmed links. The cart-era (VII–XVI) broke on step 1; its vision nulls are now explained structurally, not as encoder failures.

**Most important thing we don't know yet:** Whether the ball-direction obs converts the 16.7% accidental-contact rate into reliable directed homing (mean dist_change reliably > 0) — the clean test of informational-gap vs deeper coordination problem (does directed locomotion need asymmetric gaits the symmetric primitive can't produce?).

---

## 2026-07-02 — crawl_ppo_2M: The PPO escalation and the vision-test milestone

**Run context:** Sequence culminating in PPO: crawl_minimal_400k (locomotion discovered, undirected, 16.7% accidental contacts, SAC) → crawl_targetobs_400k (ball-direction obs added, directed approach emerges, mean dist_change +0.195 m, ~7–23% SAC contacts, burst-then-regress instability) → crawl_ppo_2M (PPO replaces SAC, directed homing *stabilises*, 57.5% deterministic contacts, monotonic per-step approach). PPO ep_rew_mean rose 97→112→139→164→179→187 then held ~170–187, no collapse. Deterministic contacts 57.5% (23/40) vs SAC best 23.3%. Mean toward-ball +0.257 m vs SAC +0.195. Per-step trajectories monotonic to contact (seed 0: 0.77→0.69→0.55→0.46→0.30→TOUCH@381). ~35% timeouts fail as approach-then-drift (close to ~0.50 m then stall/drift), NOT never-approaching.

### Framework 1 — Pattern Learning: CONFIRMED (under a stable learning process)
Under SAC the directed-crawl runs were CHALLENGED (burst-to-plateau-then-regression; peaks that didn't survive further training). That challenge was an ALGORITHM artifact, not a framework failure: SAC is off-policy (updates on a replay buffer mixing many past policies), and for a temporally-coherent gait the mixed-policy gradient is internally inconsistent — the known reason off-policy methods are less reliable on coordinated locomotion. PPO is on-policy (rollout from current policy, update, discard); the gradient always improves the current policy on its own data — the stable learning process the framework implicitly requires. PPO's ep_rew_mean rose smooth-monotonic 97→187 with no regression, plateau ~170–187 (stable lock-in, not a fragile peak). The ~35% failure is consistent in character (a terminal-phase limitation), not random. The framework was tested under conditions where its precondition (stable gradient on temporally-coherent behaviour) was unmet; PPO supplies it, and the prediction holds.

### Framework 2 — Behavioral Prediction: CONFIRMED (directed homing reliable)
Untestable before the directional obs (crawl_minimal), partially confirmed with target_obs under SAC (+0.195 m but transient), now reliable under PPO: (1) 57.5% contacts — not spawn-luck (every ball 0.70–0.80 m away; a random walk vs a ~0.30 m budget gives ~16.7%, the crawl_minimal baseline); (2) monotonic per-step distance decline in successful episodes (a model being executed, not a random walk converging); (3) failure mode is approach-then-drift, so the directional model runs in every episode and fails only terminally; (4) mean toward-ball +0.257 m even averaging in the failures. The prediction of coherent behaviour from an internal cause-and-effect model is now measurably confirmed.

### The affordance/R49 chain — all four links confirmed; the vision test is well-posed
1. Wide clamps + arms_fwd pose → translation achievable [hand_drive, CONFIRMED, 0.225 vs 0.039 m]
2. Translation achievable → RL discovers locomotion [crawl_minimal, CONFIRMED, 0.298 m mean CoM]
3. Locomotion + ball-direction obs → directed homing [crawl_targetobs, CONFIRMED, +0.195 vs +0.000 m]
4. PPO → RELIABLE directed homing [crawl_ppo_2M, CONFIRMED, 57.5%, monotonic, stable]

The core project question — can vision replace the privileged `target_obs` bearing? — is well-posed for the first time: there is a directional action (57.5%-reliable crawl-to-contact) for vision to serve, and an upper-bound benchmark to measure against. The cart-era vision nulls (Phases VII–XVI) are explained structurally: with no directional action, even a perfect directional encoder had no behavioral payoff, so the RL gradient had no reason to build directional visual structure. Those results are "cart substrate, no directional locomotion, vision inert as expected" — not evidence that vision cannot become directional.

### Honest open items
Single seed (all PPO results seed 0); privileged target obs, not vision (57.5% is the upper-bound benchmark vision must match); ~35% approach-then-drift terminal failure (candidate fix: the MIMo recipe's metabolic-cost term, not yet added); metabolic cost untested.

> **Pattern Learning Framework: CONFIRMED** — PPO's smooth monotonic curve (97→187, no collapse over 2M) and stable plateau are the lock-in the framework predicts; SAC's burst-and-regress was off-policy replay-drift (inconsistent gradients for a temporally-coherent gait), not a framework failure — the framework requires a stable learning process and is confirmed under one.
>
> **Behavioral Prediction Framework: CONFIRMED** — 57.5% deterministic contacts, monotonic per-step decline to contact, systematic approach in every episode (timeouts close to ~0.50 m before stalling); the creature reliably produces goal-directed behaviour that changes the world in the predicted direction.
>
> **All four causal-chain links confirmed** — the core vision test (replace privileged ball-direction with camera input) is well-posed for the first time; there is a 57.5%-reliable directed crawl for vision to serve and a benchmark to beat.
>
> **The most important thing we don't know yet:** Whether vision can supply the ball-direction bearing that the privileged `target_obs` vector provides — the core project question, now finally testable.
>
> **Recommended diagnostic (a measurement):** Zero the `target_obs` vector in a deterministic eval of the PPO best_model; confirm contacts collapse toward the ~16.7% random-walk baseline, validating that the 57.5% is genuinely caused by the directional signal and that vision has a real behavioral gap to fill.

---

## 2026-07-04 — Vision phase (Stages A/B/C): the "integrated but inert" failure splits in two, and "reinforced not destroyed" is demonstrated

**Setup.** With directed crawl-to-ball solved via a privileged bearing (`target_obs`, PPO 57.5%),
the open question was whether VISION can supply that bearing while REINFORCING, not destroying, the
proprioceptive gait. Precondition fixed first: the head-cam saw only ±15°; widening fovy 90→120 and
matching the spawn cone to ±22° makes the ball a clear 4–14 px blob across the cone (winnability).
Three architectures, all preserving the motor policy by freezing it: A = distil bearing from pixels
into the frozen teacher (supervised); B = reward-grow vision into the frozen teacher's bearing slot;
C = zero-init additive residual vision head on the frozen blind crawler (No More Blind Spots, Duan
2025). Honest floor on ±22° = blind base 20%; ceiling = teacher 73–77.5%.

### The decisive result: "integrated but inert" was never one failure — it is two
For years the vision null was one undifferentiated fact (ablation says vision is wired in; behaviour
says it adds nothing). This phase separates it cleanly:

1. **Representation — SOLVED (refutes the strong prior).** Stage A: a CNN reconstructs the ball's
   lateral bearing from pixels at **teacher-driven held-out R² = 0.841** (in-sample 0.95), versus the
   project's chronic R² ≈ 0.01–0.08. The direction IS in the pixels and the encoder generalises it.
   The years-long "representation failure" was an artifact of an UNWINNABLE camera (ball out of frame
   / a horizon speck) + no supervised pressure — not an inability of pixels to carry direction.
   (Caveat: student-driven held-out lateral R² = −0.01 is DAgger distribution shift, not a
   representation failure — fixable with more DAgger; does not affect on-policy Stage C.)

2. **Reward-driven recruitment — the real bottleneck (Stage C).** Residual grown by reward ALONE:
   contact stayed at the 20% blind floor (inert) while the pixel-ablation gap was +16.7 pts
   (integrated) and the SUBSTRATE was preserved (disp 0.235 vs 0.219, tip 3.3% vs 2.5%). Because
   Stage A already proved the representation exists, this isolates the failure as **policy-gradient,
   not representation**: on a cone where forward-crawl already earns 20%, reward is too weak a teacher
   to grow the steering, even though the direction is decodable in the pixels. This is the precise
   modern statement of the old null.

3. **The fix — distil then RL (Stage B, warm-started from A).** Initialise the vision head from
   Stage A's bearing CNN (R²=0.84), freeze the teacher: vision reads direction at init, the teacher
   steers immediately, reward only fine-tunes. Clean eval (50K): **contact 63.3%** (20→63, toward the
   73–77.5% ceiling); **pixels-ablated 20.0% = exactly the blind floor** (+43.3 pt vision gap); gait
   preserved (tip 6.7%, upright ~93%) and locomotion ENHANCED (disp 0.504 vs 0.219, speed 2.14 vs
   0.30 mm/step). Generalises: nearer balls 80%, farther 0.85–0.95 → 50% (graceful), cone ±15 → 75%,
   ±30 → 60%. Video-confirmed by direct frame inspection (prone crawl to contact in distinct board
   positions — directed, not spawn-luck).

### Framework reading — "vision reinforces proprioception rather than destroying it"
This is the concrete in-silico instance of the user's target claim, operationalised as the
substrate/capability split:
- **NOT DESTROYED:** the motor policy is frozen (preserved by construction), and pixel-ablation
  recovers EXACTLY the 20% proprioceptive baseline — never below. The proprioceptive competence is
  intact and fully recoverable; vision did not overwrite it.
- **REINFORCED:** vision supplies the world-derived bearing the proprioceptive policy lacked, lifting
  behaviour +43 pts and making locomotion more purposeful. Removing vision degrades gracefully to
  the preserved baseline (the "coexistence," not "overwrite," pattern of Taylor Ch. 9 — here via a
  frozen substrate rather than a replay buffer).

### Honest limits (do not overclaim)
- Stage B's success uses distil-then-RL (supervised A → reward B), NOT reward alone. Reward alone
  (Stage C) could not recruit vision on this cone. The "reward alone grows vision" claim is REFUTED
  here; "supervision builds the representation, reward uses it" is what worked (matches
  Distillation-PPO / Learning-by-Cheating from the 2026-07-04 literature scout).
- Stage B is sensor SUBSTITUTION (vision replaces the privileged bearing on a frozen policy) — the
  first time vision is behaviourally load-bearing in this project, but NOT yet the deeper
  interpenetration (vision reshaping the proprioceptive representation). The prism-ghost aftereffect
  (PRISM_GHOST_PROPOSAL.md) is now unblocked and is the next real test of interpenetration.
- Substrate tip-rate rose 2.5→6.7% (still low): the vision policy moves more aggressively. Preserved,
  not pristine.
- Single seed; Stage B evaluated at 50K (a longer run is training). The ±22° cone is forward-crawl-
  confounded on contact rate — hence R² and the ablation-to-floor gap, not raw contacts, carry the
  argument.

### Methodological note — Gemini video description is UNRELIABLE for the prone crawler
On BOTH the blind base and Stage B videos, Gemini described the prone belly-crawler as "a person
trying to stand up and failing / flailing / never completing the task," directly contradicting the
measured contacts. It anthropomorphises MIMo (reads prone crawl as failed standing) and mistakes the
floor ball for "part of the head." Direct frame inspection (Read on extracted PNGs) is the reliable
check for this body; treat Gemini's posture/task verdicts on the crawler with suspicion.

### CORRECTION (same night, 2026-07-04) — the Stage B behavioral claim was CONFOUNDED; walk it back

A polished 300K Stage B eval showed vision going NET-NEGATIVE (contact 55% with pixels vs 75%
pixels-ablated), which forced a re-examination. Decisive diagnostic (`diag_confound.py`, frozen
teacher, proprio, true vs zero bearing):
- FULL cone ±22°: true-bearing 82.5% vs zero-bearing 75.0% → bearing worth only **+7.5 pts**.
- LATERAL band ±15–22° (near-forward excluded): true 65.0% vs zero 65.0% → **+0.0 pts**.

**The ±22° cone cannot demonstrate behavioral vision value, and neither can a lateral band inside
it.** The creature's forward-crawl reach envelope (~0.28 m lateral at the 0.75 m ring) covers the
ENTIRE camera-visible cone, so the ball's direction is behaviorally almost irrelevant anywhere
vision can see it (true ≈ zero at every angle ≤ ±22°). This is deeper than cone width: it is a
reach-envelope vs visible-cone mismatch.

**What this means for the earlier Stage B entry (correct the overclaim):**
- The "+43 pt vision gap" (63% real vs 20% g(0)-ablated) at 50K was an artifact of the pixel-
  ablation baseline: g(zeros) is a drifting constant, NOT a fair "no-vision" control. The fair
  control is the teacher fed a zero bearing = 75%. Stage B's real-pixel contact (55–63%) is BELOW
  that 75%, so vision added noise and slightly HURT — it did NOT reinforce behavior.
- RL fine-tuning also DEGRADED the distilled encoder: 50K (near the warm-start) read better than
  300K. Reward on a confounded cone erodes the supervised R²=0.84 representation (un-distillation).

**What still stands (unconfounded):**
- Stage A representation: lateral decode-R² = **0.84**. Vision genuinely reads ball direction from
  pixels. This is real and is the night's defensible result. The failure was never representation.
- Stage C: reward-alone inert with the gait preserved — consistent (bearing wasn't needed, so
  reward had nothing to teach).

**Corrected conclusion.** Two constraints are in direct tension for this body+camera and do NOT
overlap: (1) WINNABILITY caps the spawn cone at ±22° (that is all the head-cam can see); (2)
BEHAVIORAL NECESSITY of the bearing requires balls OUTSIDE the ~±25–30° forward-crawl reach. There
is no cone that is both visible AND requires vision. So "vision reinforces behavior" cannot be shown
on this setup at all — not a training failure, a task-geometry impossibility. The representation is
solved; the behavioral payoff has no room to exist here.

**The clean fix (next experiment).** Make the ball require vision by putting it beyond forward-crawl
reach AND bring it into view with an action: the HEAD-SEARCH phase (head_swivel to fixate an
off-cone ball, then crawl), enabling a >±30–45° spawn that forward-crawl cannot solve. Only then can
the R²=0.84 bearing convert to a contact gain. Alternatively: shrink the reach envelope (narrower
body / shorter arms) or a farther/faster ball. Added `spawn_cone_min_deg` (lateral-band) to the env
for these tests; on this body it was not enough (reach still covers ±22°).

Honest status: representation SOLVED; behavioral reinforcement UNPROVEN and unprovable on ±22°;
next step is head-search for a winnable-yet-vision-necessary task.

### HEAD-SEARCH RESULT (head_search_v1, ~800K) — FIRST BEHAVIORALLY-NECESSARY VISION WIN

The head-search phase (±68° cone, ball off the static ±22° view, active head-yaw actuator added)
gives the payoff the ±22° confound made impossible. Clean eval (40 eps, ±68°):
- contact **47.5%** with vision vs **35.0%** pixels-ablated → vision gap **+12.5 pts**.
- mean-toward **+0.073 m with vision vs −0.036 m ablated** — the cleanest signal: WITH vision net
  motion is TOWARD the ball; WITHOUT it, AWAY. Vision reverses the behavior's direction.
- substrate held: tip 5% (upright 95%), disp 0.650 m (active search+crawl).
- VIDEO (frame-verified, not Gemini): creature starts facing forward with the ball off to the side,
  then rotates its body toward the off-cone ball and crawls to contact. Genuine search + directed
  approach to a ball forward-crawl cannot reach.

**Why this matters:** for the first time in the project, vision is BEHAVIORALLY LOAD-BEARING —
ablating it drops contacts AND flips net motion away from the target, on a task (±68°) where the
diagnostic proved forward-crawl fails (bearing worth +20 pts). Contrast ±22°, where vision was
net-zero/negative because forward-crawl already solved it. Representation (Stage A, R²=0.84) + a
task that NEEDS direction = a real behavioral win. This is the "reinforced" half, finally shown on
a winnable-yet-vision-necessary task.

**Honest limits:** modest gap (+12.5, not +40). The 35% ablated baseline is inflated by undirected
search-wander stumbling onto balls over 1000 steps (not a true zero floor). The from-scratch policy
plateaued bimodal (~100 eval reward, half episodes solved) — gait+search+vision learned together on
the action side from scratch is hard; more training and/or a search-shaping curriculum (start narrow,
widen the cone) should raise both the contact rate and the ablation gap. Single seed. But the sign is
unambiguous and video-confirmed: vision drives the creature to find and reach balls it otherwise can't.

### HEAD-SEARCH CURRICULUM (head_search_curriculum_v1, 1.9M) — sharpens the vision win

Search-shaping curriculum (spawn cone ±22→±34→±45→±68 at 0/15/35/60% of steps) vs the flat
head_search_v1. It carried competence through every widening (no collapse; brief dip then
re-stabilise higher) and hit 202±1 reward — near-perfect consistent solving — on ±45 and ±68.

Definitive clean eval on ±68 (40 eps), curriculum vs flat:
| metric | curriculum | flat |
| --- | --- | --- |
| contact (vision) | **65.0%** | 47.5% |
| contact (pixels ablated) | 42.5% | 35.0% |
| **vision load-bearing gap** | **+22.5 pts** | +12.5 |
| mean_toward vision→ablated | +0.106 → −0.092 | +0.073 → −0.036 |
| substrate tip-rate | 10.0% | 5.0% |

**Reading:** staging the difficulty didn't just make a better SEARCHER — it made vision MORE
behaviorally load-bearing (ablation gap nearly doubled, 65% contact, stronger motion sign-flip:
with vision the creature moves toward the ball, ablate it and it drifts away). Video frame-verified:
turns toward the off-cone ball and crawls to it, more reliably than the flat run. Cost: tip-rate rose
to 10% (still upright 90%) — the curriculum policy moves more aggressively.

**Closes the arc.** Representation solved (Stage A, R²=0.84) → on a task that NEEDS direction (±68,
forward-crawl fails) vision is behaviorally load-bearing (+22.5 pt ablation gap) → and the effect is
sharpened by a search-shaping curriculum. The ±22 confound is fully understood and routed around.
Honest limits: single seed; ablated 42.5% still inflated by undirected search-wander (not a true zero
floor); tip-rate up. Next candidates: multi-seed confirm; reduce tipping (posture reward / slower
head/body); then the prism-ghost aftereffect (now genuinely unblocked — vision drives the reach).

### PRISM PRECHECK (2026-07-04) — vision is AROUSAL/PRESENCE load-bearing, NOT direction. Prism blocked.

Built the prism-ghost env (real target hidden+solid+rewarding via geom group 3; a mocap GHOST,
visible+non-physical, rendered at real-bearing + offset; head-cam sees only the ghost — all
verified). Before launching the adaptation run, ran the winnability precheck: does the head-search
policy actually CHASE the ghost (a prerequisite for having anything to re-map)?

**It does not.** Base head_search_curriculum_v1 under increasing prism offset (real-contact / ends-
closer-to):
| offset | 0 | 30 | 45 | 60 | 90 | 120 |
| real-contact | (65) | 83% | 73% | 93% | 67% | 73% |
| reach-to-REAL | – | 50% | 80% | 93% | 80% | 80% |

The reach is **essentially invariant to visual displacement** — it finds the REAL (hidden, solid)
ball regardless of where the ghost is rendered, even 120° away.

**Reconciles with the ablation (+22.5 gap):** ZEROING the pixels drops contact (65→42.5) and flips
motion away; DISPLACING the pixels does NOT. So vision is load-bearing as a **presence/arousal cue
that triggers search** — not as the signal that sets reach DIRECTION. The directional homing is done
by the policy's broad proprioceptive/tactile SEARCH SWEEP (it was trained to turn and scan the ±68
cone; that sweep covers the real ball's location and touch-homes on the only solid object). Vision
says "a ball exists — engage"; proprio/touch says "where."

**Consequence:** the prism-ghost adaptation/aftereffect CANNOT be shown with this policy — it needs
vision to drive the fine reach DIRECTION, and this policy doesn't. There is no "chase-the-ghost-and-
miss" phase to adapt away from, hence no re-mapping and no aftereffect. Not launching the doomed run
(precheck did its job, as with the ±22 confound and the head_swivel/head_tilt_side gotcha).

**Corrects the head-search claim (be precise):** "vision behaviorally load-bearing" on ±68 is TRUE but
means vision is necessary as an ENGAGEMENT/PRESENCE signal, NOT that the policy reads and uses ball
DIRECTION from pixels. Stage A proved the LATENT encodes direction (R²=0.84); the POLICY does not USE
it for direction (touch-search does). So the project-long "representation succeeds, behavior doesn't
use it (for direction)" gap is REAL and now sharply localized: it's a policy/credit-assignment gap
between an existing directional representation and a touch-dominated motor controller.

**Prerequisite for a real prism experiment (redesign):** a policy whose FINE reach direction is
vision-driven — i.e. remove the touch-search escape so direction MUST come from pixels. Candidates:
(a) reward reaching the VISUAL location (ghost) directly, not a solid ball (a vision-guided reach
task — but that is bribery toward vision, note it); (b) a precise/small target that a broad sweep
can't stumble onto; (c) a distractor field of solid non-targets so touch-sweep is punished and only
the visually-identified target pays — forcing vision to set direction. Until one of these makes the
reach vision-direction-dependent, the prism aftereffect is premature.

## 2026-07-05 — Head-search seed replication (s1, s2) and the posture-term result: sign replicates, size doesn't, and reward-shaping can silently delete the vision gap

**Seed 1 (2M steps):** contact 70.0% sighted / 57.5% ablated → gap +12.5 pts; mean_toward +0.093 vs +0.008 (weaker sign-flip than seed 0). Bearing diagnostic: misses are NOT noise — far-left spawns hit 1/9 (11%) vs far-right 7/8 (88%) — seed 1 learned a right-lateralized search sweep, a systematic blind wedge at the cone's left extreme.

**Seed 2 (1.3M steps):** contact 45.0% sighted / 37.5% ablated → gap +7.5 pts; mean_toward +0.200 vs +0.000 (largest toward-effect of any seed). Bearing diagnostic: NOT lateralized — weak at both extremes, a narrow-symmetric search phenotype, despite a near-perfect-looking training curve (the "aggregate hides the tails" trap recurring).

**Three-seed verdict:** gap is positive in 3/3 seeds (s0 +22.5, s1 +12.5, s2 +7.5; mean ~+14) — real but seed-dependent in magnitude, not the seed-invariant headline the best-of-three (+22.5) implied. The most seed-consistent signal is mean_toward: sighted ≥ +0.09 in all seeds, ablated ≤ +0.01 in all seeds — vision reliably converts wander into net approach even when the contact-rate gap is modest. Search phenotype (broad / right-lateralized / narrow-symmetric) varies wildly by seed and is the dominant source of between-seed variance in the blind baseline; vision's marginal contribution on top is steadier than the raw gap suggests.

**POSTURE-TERM RESULT — reward-shaping fragility, demonstrated.** Same seed 0, same curriculum, ONE change: tilt-cost 1.0 added to the reward. Result: tip rate rose (10.0% → 12.5%) — the term FAILED its one job. Contact stayed roughly flat (65.0% → 62.5%). But the vision-ablation gap went to **−7.5 pts** (62.5% sighted vs **70.0% ablated**, the best blind performance of any run) and mean_toward became numerically identical sighted vs ablated (+0.152/+0.150) — **vision stopped being load-bearing at all.** Reading: the tilt penalty prices out the aggressive vision-triggered maneuvers (fast turns/lunges that risk tilting), so training converges on a conservative, broad BLIND sweep instead. An auxiliary reward term that left aggregate success intact silently deleted vision dependence — the opposite of this branch's name ("reinforced, not destroyed"): here shaping destroyed it. **Rule going forward: any reward change in a vision-dependent task must re-measure the ablation gap, not just success/tip rates.**

---

## 2026-07-05/06 — Time-pressure calibration and the decoy-discrimination fix

**TIME-PRESSURE CALIBRATION (no-training precheck):** tested whether truncating episodes at various step budgets T could separate directed reach from blind sweep, for the strongest head-search policies (s1, v1/+22.5). It could not: ablated (blind) contact was already 37.5-55% by T=150-350 — the touch-search sweep is an efficient learned behavior, not slow groping. Gap(T) for s1 wobbled +5..+15 at every T (never opened); v1 went NEGATIVE at tight T (blind beat sighted under pressure at T=150-200). **Time-pressure-alone REJECTED** as a lever; pivoted to two-ball decoy discrimination, where a wrong-colored decoy plus a wrong-touch penalty+termination closes the escape structurally rather than temporally.

**DECOY RUN 1 (decoy_v1_s0, 1.3M) — a placement confound gave blind a 63% floor.** Official eval looked like a win: red-contact 57.5%/wrong-ball 32.5% sighted vs 47.5%/35.0% ablated (raw gap +10). But the per-episode choice diagnostic caught it: choice accuracy was STATISTICALLY IDENTICAL sighted vs ablated (63.9% vs 62.9%) — and blind picking red at 63% is impossible if the task were symmetric. Cause: the min-separation placement rule pushed the decoy away from center whenever the target spawned centrally, making red on average more central; the blind sweep exploited this geometry, not color. Lesson (repeated from the ±22° cone confound): always give the blind baseline a chance to cheat before crediting vision.

**DECOY RUN 2 (decoy_v2_s0, fresh 2M) — FIRST ABOVE-CHANCE VISUAL DISCRIMINATION.** Fix: exchangeable placement (red/blue randomly assigned to the two computed bearings AFTER placement), making blind choice exactly 50% by construction. The decisive 100-episode choice diagnostic: **sighted choice 63.2% ± 10.1 (55R/32B) — CI excludes chance (p≈0.017)**; **ablated choice 49.4% ± 10.6 (42R/43B) — exactly the designed coin flip.** First time in the project that vision measurably changes WHICH object AB reaches, with no geometric escape available to the blind baseline. Official 40-ep eval (noisier): contact 45.0% both conditions (raw gap 0.0 — contact rate is no longer the right metric; choice is the signal now), wrong-ball 40.0%/35.0%, tip 10%. Training-eval reward oscillated 120-202 in the back half with no clean plateau — discrimination may still be sharpening with more steps.

---

## 2026-07-06 — Decoy discrimination: the touch-search escape closes, and the presence/arousal claim gets its first crack

### Q1 — does the decoy result confirm, refine, or contradict "vision is presence/arousal, not direction"?

**Refines it; does not cleanly confirm or contradict it.** The prism precheck's claim was specifically that the head-search policy does not use the R²=0.84 vision latent to set reach DIRECTION — it uses vision only to decide whether to engage, and lets a broad proprio/touch sweep supply the steering. A two-object choice is a genuinely new kind of test, because presence/arousal in that strict sense is symmetric information with respect to two balls: "a ball exists, engage" cannot, even in principle, tell AB WHICH of two balls to approach. For choice accuracy to rise above the structural 50% floor (63.2% ± 10.1 vs 49.4% ± 10.6), something in the pixels must be binding color-identity to a choice between two candidate bearings. In that narrow sense this is the first evidence of vision doing something functionally directional — discriminating BETWEEN targets, not just gating engagement with one.

But the diagnostic cannot yet rule out a presence-style mechanism that still produces 63%. Consider "approach whichever direction the red-colored region of the visual field is more salient, more central, or larger in" — a phototropic, gradient-following rule (steer toward more red-ness in view) rather than "compute the bearing of the red object as a spatial coordinate, then turn to it." Both mechanisms produce above-chance red preference and are indistinguishable from the choice-accuracy number alone. The former is a real, if primitive, directional mechanism; the latter is a salience-weighted steering bias, a materially weaker claim than "vision sets reach direction via a spatial code." Verdict: the STRICT form of "presence/arousal, not direction" (vision contributes zero information about which of several targets to select) is REFUTED — 63% vs 50% cannot happen under a purely symmetric gate. The STRONG form (a genuine spatial bearing code drives the choice) remains unestablished — a salience/phototropism account is fully consistent with the same numbers. This is a refinement narrowing the space between "arousal only" and "full spatial code," not a resolution of it.

### Q2 — touch-search escape closing to chance: CONFIRMED

The touch-search escape framework predicted that once every non-visual route is genuinely closed (no time-pressure lever per the calibration precheck, no placement asymmetry per the decoy_v1 fix, wrong-touch actively penalized and terminating), blind performance should fall to the structural chance floor — a symmetric two-choice task gives an undirected search sweep nothing to exploit. Decoy run 2's ablated choice accuracy, 49.4% ± 10.6 against a designed 50.0%, lands almost exactly on that prediction. This is a genuine confirmed prediction, not a post-hoc fit: it was the explicit design goal of the exchangeable-placement fix, and the mechanism is now validated end to end across two runs — v1 showed the escape when a route existed (63.9% vs 62.9%, both far above 50%, driven by geometry not color), v2 showed its disappearance once the route was closed.

### Q3 — task-structure vs. reward-shaping as the lever for keeping vision load-bearing

The decoy runs used the identical curriculum recipe and warm-started encoder as the head-search runs, with NO additional auxiliary reward term — the only change was structural (a second ball, exchangeable colors, wrong-touch penalty + termination). Vision stayed measurably engaged (63.2% vs 49.4%, above chance) with nothing added to the reward function to keep it there. Contrast this directly with the posture-term result (2026-07-05): adding one auxiliary reward term (tilt-cost 1.0) to the SAME curriculum and seed left aggregate success essentially unchanged (65.0% → 62.5% contact) but SILENTLY ERASED the vision-ablation gap entirely (+22.5 → −7.5 pts; sighted and ablated mean_toward became numerically identical) — and the term did not even achieve its own stated goal (tip rate rose, not fell). One auxiliary reward term, chosen for an unrelated purpose, deleted the entire vision dependency without moving the visible success metric — the opposite of what every task-structure change in this arc has done (cone-widening curriculum: +12.5 → +22.5 pts; decoy discrimination: 50% → 63.2%). The pattern is now specific and twice-repeated: task-structure changes that remove a non-visual route to reward reliably INCREASE vision's measured contribution; reward-shaping additions layered on top of an already-solvable task put that contribution at risk, whether or not they achieve their own nominal purpose. Practical rule this log has now earned twice over: treat every auxiliary reward term in a vision-dependent task as a suspect until the ablation gap is re-measured, and prefer closing structural escapes over tuning reward coefficients as the lever for vision load-bearing.

### Q4 — falsifiable expectations for the queued next steps

**Seed replication.** Based on the head-search precedent (sign held 3/3 seeds, magnitude varied 7.5–22.5 pts), the prediction is: sighted choice accuracy should stay reliably above the ablated baseline across replication seeds (sign-robust), while the exact magnitude (currently a ~14-pt gap) likely varies seed to seed by a comparable range. A result that would REVISE the current reading: any replication seed where ablated choice accuracy is reliably off 50% (a tight CI excluding roughly 45–55%) would indicate a residual placement or geometric confound of the kind decoy_v1 had, not yet caught by the exchangeable-placement fix, and must be checked before trusting further magnitude comparisons.

**Longer training.** If discrimination is a still-forming perceptual category — training-eval reward oscillated 120-202 in the back half with no clean plateau — the prediction is that choice accuracy should continue to rise with more steps, mirroring the head-search curriculum's own history (flat +12.5 → staged-curriculum +22.5). A result that would call this into question: if choice accuracy plateaus near the current ~60-63% indefinitely under substantially more steps, that points to a ceiling set by the coarseness of the 32×32 color signal or the touch-dominated architecture's limited capacity to route color into a choice decision, not by insufficient training time.

**Prism displacement on a strong discriminator (the decisive test).** This is the experiment that can finally separate "vision sets target-selection direction" from "vision still only gates arousal, now for two objects." Concrete, falsifiable predictions:
- If vision genuinely drives WHICH bearing gets approached (whether via a spatial code or a phototropic salience rule), prism-displacing the visible color cue away from its true bearing should pull choice toward the DISPLACED (ghost) position at a rate detectably above the no-prism baseline — and, the sharpest cut, accuracy scored against the TRUE object identity should fall as the offset grows while accuracy scored against the DISPLAYED position stays high. This is the direct analogue of classical prism adaptation: an initial displacement-direction error, adaptation under continued exposure, and a negative aftereffect on prism removal.
- If vision still functions only as a presence/arousal gate (as the prism precheck found for the single-ball policy), displacing the color cue should have LITTLE TO NO effect on which real object gets chosen — choice accuracy should stay near the current 60-63%, because "a red thing exists somewhere" raises engagement without binding a specific direction, and the touch-search sweep (not vision) still supplies the steering.
- An intermediate, also-informative outcome: displacement shifts choice toward the ghost's position LESS than "full following" predicts but MORE than "no effect" predicts — indicating partial directional binding, where vision biases the search sweep's starting point or weighting but proprioceptive/tactile correction still dominates the final approach. Given Q1's finding that the decoy result is equally compatible with a salience-weighted (non-spatial) mechanism, this intermediate outcome is plausible and should be scored as its own regime, not forced into either extreme.

### Theory Monitor Note — 2026-07-06 (Decoy discrimination)

**Touch-search escape framework: CONFIRMED.** Ablated choice accuracy (49.4% ± 10.6, 100 eps) lands on the designed 50% chance floor once every non-visual route (time pressure, placement asymmetry) is structurally closed — exactly the predicted collapse.

**Presence/arousal-not-direction framework: REFINED, not overturned.** The strict form (vision cannot discriminate between two candidate targets) is refuted — sighted choice (63.2% ± 10.1) sits above the structural floor on a task a symmetric arousal gate cannot resolve. The strong form (a genuine spatial bearing code) remains unestablished: a salience/phototropism account ("approach more red-ness") explains the same number without positing a spatial map, and the two mechanisms have not yet been separated.

**Task-structure vs. reward-shaping:** confirmed again, now twice over, in opposite directions. Closing a structural escape (decoy discrimination) grew the vision gap from a symmetric non-signal to 63.2% vs 49.4% with zero new reward terms. A single auxiliary reward term (posture tilt-cost, 2026-07-05) silently erased a previously robust +22.5-pt gap while leaving aggregate success intact and failing its own stated goal. Structural task changes remain the safer, and so far the only reliably positive, lever for vision dependence in this project.

**The most important thing we don't know yet:** whether the decoy discrimination is a genuine spatial/directional binding of color to bearing, or a non-spatial salience-following rule (steer toward more red in view) that produces the same above-chance choice accuracy without any bearing code. Both mechanisms are equally consistent with 63.2% vs 49.4%.

**Recommended diagnostic** (not a training run — a measurement): before committing to the full prism-displacement adaptation experiment, run the existing prism-ghost precheck harness on the decoy_v2 policy at several small-to-moderate offsets (15°/30°/45°) and check whether choice accuracy, scored against the TRUE object, degrades with offset. A clean degradation would show the decoy policy is using something bearing-like (the exact test the single-ball head-search policy failed); no degradation would show the decoy result is a salience effect rather than a spatial code, and would argue for redesigning the discrimination task (larger angular separation, or closing any remaining touch-search escape) before running the full aftereffect experiment.

---

## 2026-07-08 — Prism displacement result: the pre-registered "diagnostic" landed, and it discriminates further than the Q4 minimum bar required

### 1. Regime 1 (follow-the-ghost) was pre-registered on 2026-07-06 — scoring the prediction

Q4's first bullet spelled out exactly this outcome in advance: *"prism-displacing the visible color cue... should pull choice toward the DISPLACED (ghost) position at a rate detectably above the no-prism baseline — and, the sharpest cut, accuracy scored against the TRUE object identity should fall as the offset grows while accuracy scored against the DISPLAYED position stays high."* That is Regime 1, written down before the run. The result (both checkpoints, 100 eps/cell): choice_vs_true falls from baseline (76.4% ext_s0, 77.5% s2) through mid-offsets (62.0/47.6 at 30°, 53.2/53.3 at 45°) down to a minimum at 60° (44.7/37.8), with wrong-ball rate climbing in step (21→47% ext, 20→58% s2). **Score: CONFIRMED — and by more than the minimum the prediction asked for.** The pre-registered bar was "detectable pull toward displaced position + falling true-accuracy." What we got additionally is accuracy falling *below* the 50% structural floor, in both checkpoints, which the Q4 language did not explicitly require but is the strongest possible version of "the displaced picture is steering."

**Strict presence/arousal account — refuted at what strength, precisely:** this is a harder refutation than the one Q1 logged on 2026-07-06. That entry refuted the strict form only to the level of "vision discriminates between two targets at all" (63.2% vs the 49.4% floor — still *above* chance, still explainable by a symmetric gate that merely leaks some signal). A pure presence/arousal gate — "a red thing exists somewhere, engage" — has no coordinate to distort and therefore has a hard floor at 50%: displacing the ghost can make the gate less informative (drifting choice_vs_true down toward 50%) but cannot make it anti-informative. Systematic mis-selection *below* 50% (down to 44.7% and 36.3%) is not a weaker signal from the same symmetric mechanism — it requires the visual position itself to be functioning as a directional variable that actively points AB at the wrong ball more often than chance. That is not compatible with any gating account, symmetric or leaky. The strict presence/arousal account is now dead outright, not merely "refined," on these two policies.

### 2. Does follow-the-ghost resolve the Q1 fork (spatial code vs. phototropic salience)? Honest answer: no — and here is why the below-chance number doesn't settle it either

It's tempting to read the below-chance collapse as resolving Q1 in favor of a genuine spatial bearing code. It doesn't, and the team should not let the FINDINGS.md phrasing ("the presence/arousal only account is dead") get quietly upgraded into "so AB has a spatial map." A retinotopic salience-follower — "steer toward wherever the red gradient is currently strongest in the visual field, with no coordinate transform, no persistent map" — is *also* directional in effect, and would *also* chase a displaced ghost, and would *also* produce below-chance mis-selection at large offsets: if the controller is simply climbing the red gradient wherever it currently renders, then at 45-60° the ghost's retinal location is closer to the WRONG solid ball more often than the right one, purely by geometry, with no internal representation of "where the true object is" required at all. Both the strong hypothesis (compute a bearing, steer to that coordinate) and the weak hypothesis (climb the local color gradient, no coordinate stored) predict the exact same choice_vs_true curve we measured, because both route behavior through "wherever red currently appears on the retina." **What follow-the-ghost resolves is the coarser question (presence/arousal-only is dead, per §1). What it does NOT resolve is the Q1 fork (spatial code vs. salience) — that remains exactly as open as it was on 2026-07-06.** A genuine discriminator would need a condition where the two mechanisms make different predictions — e.g., moving the ghost fully outside the camera's FOV mid-episode (a stored bearing could still be acted on from memory; a pure retinotopic gradient-follower has nothing left to climb once the stimulus leaves the sensor) — which the current sweep (15°-90°, ghost presumably still renderable) does not test.

### 3. The below-chance choice + the conditional analysis (ghost-near-blue → 67.7% blue): does vision overriding touch-availability count as the interpenetration signature?

Yes, and this is the sharpest theoretical upgrade in this entry. Recall the 2026-07-04 prism precheck's verdict on the single-ball head-search policy: vision was load-bearing only as an engagement gate, while touch/proprio's broad search sweep supplied *direction* — reach-to-real was "essentially invariant to visual displacement... even 120° away." That was a **coordination** pattern: two channels, touch dominant for direction, vision merely turning search on. The new conditional result inverts this. When the red ghost renders nearer the true-BLUE ball, AB touches blue 67.7% of the time (n=334); when nearer true-red, only 34.8% (n=563) — a ~33-point swing driven purely by where the color cue is rendered, on a task where BOTH balls are equally touchable and available to the touch/motor system at every trial. Touch-availability is symmetric by design (winnability rule: either ball is always reachable); the visual displacement alone decides which available target actually gets grabbed. That is the literal definition of interpenetration this project has been using throughout: vision is not an independent channel that touch can simply out-vote when it disagrees — vision's (possibly wrong) position signal is wired into the same action-selection pathway that decides which touch-target to commit to, strongly enough to override the pathway's own default. This is a genuine escalation from the head-search-era finding (vision gates engagement, touch supplies direction) to something closer to vision *setting* direction, at least in this two-target/decoy substrate. Caveat that must travel with this claim: per §2, "vision setting direction" here does not yet mean "a spatial map" — it is equally consistent with a salience-driven override. The interpenetration signature is about the entanglement of channels (vision can hijack touch's target selection), not yet about the *format* of the visual representation doing the hijacking.

### 4. The displayed_red_frac metric failure — methods lesson, logged honestly

FINDINGS.md notes plainly that the pre-registered `displayed_red_frac` heading-fraction metric was uninformative *as implemented*: for any episode that ends in a touch, the final bearing necessarily lands on a real, solid ball (the ghost has no mass and cannot be "arrived at"), so the metric's threshold could never discriminate hypotheses — it was structurally guaranteed to look the same regardless of which regime was true. This joins a now-recurring failure mode in this project's method log: a metric can be pre-registered in good faith and still be **tautological by construction** (compare: the ±22° cone reach-envelope confound, the decoy_v1 placement-geometry confound, the "reward is a poor proxy for discrimination" finding from the extension run). **The general lesson, worth stating as a standing rule:** before spending compute on the real run, run the metric's computation through a synthetic null (or through at least two of the candidate regimes on paper) and confirm it can actually produce *different* numbers under each — a metric that returns the same value under every hypothesis is not measuring anything, no matter how principled its derivation looked at design time. Credit where due: the team caught this honestly rather than reporting a null result as if it were informative, and swapped in the conditional analysis (ghost-near-blue / ghost-near-red touch rates), which turned out to be a strictly stronger measure than the one it replaced. That is the right response to a failed metric — but the next pre-registration should budget five minutes for a synthetic-null check on any brand-new metric before it becomes the headline number.

### 5. Pre-registering the adaptation/aftereffect experiment — three outcomes, concrete numbers, BEFORE approval

The gate PRISM_GHOST_PROPOSAL.md set ("vision must be directionally load-bearing, or there is nothing to remap") is now cleared by §1-§3. Below are the three pre-registered outcomes for a **+30° adaptation design** (Phase A baseline → Phase B sustained +30° prism exposure → Phase C prism removed, aftereffect window), using the two numbers we already have as anchors: baseline choice_vs_true ≈ 76-78% (both checkpoints, offset 0), and momentary (no-adaptation, single-shot eval) +30° choice_vs_true = 62.0% (ext_s0) / 47.6% (s2) — i.e., where AB starts on Day 1 of Phase B before any exposure-driven change.

- **Outcome 1 — RECALIBRATION (interpenetration confirmed).** During Phase B, choice_vs_true *improves* with continued exposure — climbing from the ~48-62% starting point back toward something near baseline (a concrete target: recovering at least half the gap, i.e., ≥65-70%, by the end of Phase B). The decisive fork is Phase C: on prism removal, choice_vs_true should drop **below** the 76-78% baseline — a genuine negative aftereffect, concretely landing somewhere in the 55-65% range for the first ~10-20 post-removal trials — before gradually recovering back to baseline over subsequent trials. Any statistically detectable dip-then-recover in Phase C is the signature; its presence is what would make this the strongest result the project has produced, because it is not achievable by re-training a network cold on the new geometry (see Outcome 3).

- **Outcome 2 — AROUSAL / NO ADAPTATION (nothing to recalibrate).** Choice_vs_true stays flat near its Phase-B starting value (48-62%) for the duration of exposure — no improvement, because each trial is independently steered by wherever the color cue currently renders and there is no persistent internal quantity being corrected. On prism removal (Phase C), choice_vs_true jumps **immediately** back to baseline (76-78%) with no dip and no recovery curve — a step function, not a gradual return, because there was nothing miscalibrated to begin with. This outcome would be consistent with the salience-following account surviving from §2.

- **Outcome 3 — RELEARNING (a genuine third possibility, not just "recalibration under another name").** Choice_vs_true *improves* during Phase B — the same rising trajectory as Outcome 1, e.g. 48-62% climbing to 65-70%+ — but this improvement is ordinary policy plasticity re-solving the task under the new offset geometry (gradient updates), not a reversible perceptual remapping. The discriminator is Phase C: on prism removal, choice_vs_true returns to baseline **immediately**, step-function style, exactly like Outcome 2 — no negative aftereffect, because there is no old calibration left to conflict with the restored geometry. **Outcomes 1 and 3 are indistinguishable from Phase B data alone; only the Phase C aftereffect test separates them, which is precisely why PRISM_GHOST_PROPOSAL.md insists the aftereffect phase is not optional.**

**Design-dependent caveat, on record before approval:** whether Outcome 3 is even a *live* possibility depends on whether Phase B keeps the policy's weights frozen (eval-only, as the current displacement sweep was) or continues RL training through the exposure phase. If Phase B is eval-only, Outcome 3 is structurally impossible — there is no gradient update for "relearning" to consist of — and any Phase-B improvement under a frozen policy would itself be a remarkable finding (some form of in-context/recurrent-state recalibration without weight change). If Phase B allows continued training, all three outcomes are live and the Phase C aftereffect test is the only thing that tells them apart. The strategist's protocol should state explicitly which regime (frozen vs. continued-training) Phase B uses, because it changes what a "positive" Phase B result is even allowed to mean.

### Theory Monitor Note — 2026-07-08

**Follow-the-ghost / Regime 1 (pre-registered 2026-07-06): CONFIRMED, at a strength beyond the minimum bar** — choice_vs_true falls below the 50% structural floor in both checkpoints (44.7-53.2 range at 45-60°), which a symmetric presence/arousal gate cannot produce even in principle.

**Spatial code vs. phototropic salience (the Q1 fork): STILL UNRESOLVED** — both mechanisms predict identical chasing-and-below-chance behavior when the visual cue is simply retinotopically displaced; the experiment run so far cannot tell them apart.

**The most important thing we don't know yet:** whether the visual signal steering target selection is a stored/persistent bearing representation or a moment-to-moment retinotopic gradient-follower with no memory — a question the planned +30° adaptation/aftereffect run does not resolve either (both mechanisms could in principle show *some* form of Phase-B change), so it should be treated as a separate open question from recalibration-vs-arousal-vs-relearning, not folded into it.

**Recommended diagnostic** (not a training run — a measurement): before or alongside the adaptation run, take the existing 100-ep prism sweep data (already collected, no new compute) and check whether choice_vs_true at a fixed offset depends on episode length / time-since-ghost-first-visible — a salience-follower predicts the miscall is set the instant the ghost is in view and stays constant, while a bearing-holding mechanism predicts choice could still be corrected later in the episode if touch information starts to disagree with the stored bearing. This is a free re-analysis of data already in hand and would narrow the Q1 fork before spending a new compute budget on the adaptation design.

---

## 2026-07-08 — Prism adaptation + aftereffect: scoring the pre-registered outcomes (Theory Monitor)

*In plain terms: this entry checks the adaptation experiment against the three predictions written down in advance on 2026-07-08 (§5 above), asks whether the Q1 "is it a real map or just a reflex" question got any narrower, states what the project's central "interpenetration" claim can and can't yet say, and proposes the next cheap checks.*

### 1. Scoring the three pre-registered outcomes against the data

**Relearning (Outcome 3) — REFUTED.** Its discriminating prediction was an *immediate, step-function* return to baseline the instant the prism comes off, because on this account there was never anything miscalibrated to conflict with — the network just re-solved the task under the new geometry. That is not what happened: prism-off choice_vs_true fell to 54.7% against a 76.4% baseline, and the miss pattern is lawful and direction-specific (92.1% vs 8.3% conditional capture depending on which side of red the blue ball sits). A step-function account predicts none of that structure. Dead.

**Arousal / no-adaptation (Outcome 2) — REFUTED.** Its predictions were (a) a flat Phase B — no change because each trial is steered independently by wherever the cue currently renders, and (b) the same immediate snap-back at Phase C. Phase B was not flat (it moved, if noisily: 62.0 → 67.0 → 58.1 → 56.2 → 48.8 → 54.8), and Phase C shows the large, structured aftereffect above, not a snap-back. Dead.

**Recalibration (Outcome 1) — the only surviving account, but score it precisely, not generously.** The pre-registration split this outcome into two separate numeric bars: a Phase-B bar (climb back to ≥65–70% by the end of exposure) and a Phase-C bar (a dip into 55–65%, then gradual recovery). The Phase-C bar is met almost exactly (54.7%, just under the window, with a lawful/controlled/red-anchored signature that is *more* diagnostic than the pre-registration required). The Phase-B bar is **not** met: the curve peaks at 67.0% at 100K and then declines to 54.8% by 1M — it ends *below* where it started (62.0%), not climbing toward 65–70%. That is a genuine miss on half of this outcome's own quantitative prediction, not a rounding error.

**So: is "recalibration confirmed with a partial-adaptation caveat" the right verdict, or does this demand a fourth reading?** The right verdict is the former, but stated more sharply than "partial adaptation." The pre-registration itself says Phase B alone cannot distinguish Outcomes 1 and 3 — only Phase C can, because both predict *some* kind of Phase-B movement. What Phase C shows (a lawful, controlled, below/near-floor, direction-specific negative aftereffect) is a signature that *only* a recalibration account can produce; neither of the other two outcomes has a mechanism capable of generating it, regardless of what the Phase-B aggregate curve looked like. The honest reading is therefore: **the decisive Phase-C fork is won cleanly by recalibration, and the Phase-B aggregate accuracy number is the wrong instrument to have expected clean confirmation from** — it is a noisy, still-training RL curve on a task with an oscillating reward signal (train reward wobbled 35–140 throughout), and a noisy aggregate metric failing to rise smoothly does not, on its own, out-vote a clean downstream aftereffect that is otherwise inexplicable. We do not need a fourth regime; we need to downgrade our trust in the Phase-B number specifically, and say plainly that "recalibration occurred" rests on the Phase-C evidence, with Phase B logged honestly as inconclusive/noisy rather than confirmatory.

### 2. Does the aftereffect narrow the Q1 fork (spatial code vs. reflexive salience-following)?

Yes — more than the raw follow-the-ghost result did, and for a specific reason. The 2026-07-08 prism-displacement result (chasing the ghost, going below chance at 45–60°) could not separate a genuine stored bearing from a stateless "steer toward wherever red currently renders" reflex, because both mechanisms are driven by a *currently present* displaced cue — a reflex needs nothing stored to produce that pattern.

The aftereffect is a different kind of evidence. In Phase C, the prism is OFF: the visible red cue is back at its true, undisplaced bearing. There is no ghost, no live displacement, nothing currently rendering at the wrong place. And yet the creature still misassigns blue balls that sit within 45° of (red − 30°) at 77–100%, falling to ~33% beyond that band — behaving as if red were still offset by the training displacement, despite there being no live stimulus at that offset location. A pure moment-to-moment reflex has nothing left to react to once the wrong-place cue is gone; it should immediately track the now-correct cue. The fact that it doesn't means **some quantity is being carried between trials** — a learned bias, gain, or offset sitting between the raw visual bearing and the action it produces. That rules out the strict zero-memory salience-follower reading outright.

It does **not**, however, prove a genuine spatial map in the full sense (a coordinate system that would generalize smoothly to bearings/positions never encountered during Phase B). A single persistent scalar correction — "subtract roughly 30° from whatever pixel-bearing you compute, everywhere" — explains the data exactly as well as a re-organized allocentric map would. So the fork narrows from "reflex vs. map, indistinguishable" to "some form of persistent bias exists, but we still don't know if it's a single global offset or a genuine spatially-structured remapping" — real progress, not full resolution. §4 below proposes the test that would finish this: check whether the aftereffect generalizes to bearings the policy never saw displaced during Phase B.

### 3. The interpenetration thesis: what's shown end-to-end, what isn't

**Shown, in one continuous chain, on the same learned system:** (a) vision that carries direction — the decoy task made color-identity bind to a specific bearing choice (63→78% across seeds, structural floor pinned at 50%); (b) vision that dominates action selection even when the alternative (touch) is equally available — the conditional capture result (67.7% vs 34.8% blue-touch depending on which real ball the ghost renders nearer, on a task where both balls are always physically reachable) shows vision hijacking touch's default target, not merely gating whether touch engages; (c) an adaptable vision-action mapping that leaves a genuine, lawful, controlled negative aftereffect after sustained conflict with touch's ground truth. That is the classical prism-adaptation phenomenon — perceive, select, misjudge, adapt, mis-select the other way when the world snaps back — reproduced start to finish in a system built from pixels, proprioception, and touch, with no phase of that chain resting on an un-replicated or confounded result.

**Not yet shown:** (i) **explicit spatial representation** — per §2, we have evidence of a persistent bias, not yet evidence of a structured, generalizing spatial code; (ii) **touch-vision convergence, the deferred touch-as-information phase.** Every experiment in this arc has used touch in exactly one role: as the ground-truth error signal that *tells* the vision-action mapping it is wrong (the mismatch between "reach where vision points" and "the rewarding object is actually elsewhere" is what presumably drives the Phase-B change). Touch has never been probed as a first-class channel with its *own* decodable representation that must reconcile with vision's — we don't know whether touch has anything like a bearing code, whether adaptation changes how touch's estimate and vision's estimate get combined, or whether the system is simply "vision defers to touch's authority" in one direction only. Genuine convergence (each channel able to pull the other, precision-weighted integration, cue combination) is a stronger and different claim than "vision recalibrates to match where touch found the reward," and it remains untested.

### 4. Proposed next measurements (diagnostics, not training runs — except where a short continuation is unavoidable)

- **De-adaptation curve.** Continue evaluating the adapted checkpoint under normal (no-prism) vision in blocks after the Phase C aftereffect measurement, tracking choice_vs_true block-by-block. Concrete prediction: if this is genuine recalibration, expect a *gradual* multi-block climb back to the 76–78% baseline (the classical paradigm's de-adaptation curve, typically dozens of trials, not one). Falsifier: a snap-back to baseline within the first block would suggest the "aftereffect" was itself a Phase-B training-noise artifact rather than a stable recalibrated state.
- **Adaptation with the vision encoder frozen.** Repeat Phase B with only the downstream policy/actor layers allowed to update (encoder weights fixed) and separately with only the encoder allowed to update. This localizes *where* the re-mapping lives. Concrete prediction: if the aftereffect still appears with the encoder frozen, the offset is a policy/action-level compensation, not a perceptual one; if the aftereffect disappears (or is much weaker) with the encoder frozen, the encoder's own representation is what moved, and re-mapping is a genuinely perceptual event.
- **Seed replication of the aftereffect.** Run the identical Phase A→B→C protocol on the s2 checkpoint (already prism-swept once) and at least one further seed. Concrete prediction, matching the decoy-discrimination precedent (sign robust, magnitude seed-variable): the negative, direction-specific, red-anchored sign of the aftereffect should replicate; the exact conditional percentages (currently 92.1/8.3) should vary. Falsifier: a seed where the aftereffect direction reverses or vanishes would undercut the "genuine mechanism" reading and point back toward Phase-B RL noise being mistaken for one.
- **Generalization-of-offset test (closes §2).** Score the aftereffect at bearings/offsets never visited during the +30° Phase-B exposure (e.g. re-run the aftereffect eval at ±60°, ±90° pairings, not just near the trained −30° band). Concrete prediction: a single global scalar-bias account predicts the same ~30° subtraction applies everywhere, uniformly; a genuinely structured spatial remapping predicts the correction could vary by region (stronger near the trained offset, weaker or different far from it). Either outcome is informative and this uses only existing checkpoints plus new eval episodes — no new training.

### Theory Monitor Note — 2026-07-08 (Prism adaptation + aftereffect)

**Behavioral/recalibration framework: CONFIRMED via the Phase-C fork, with the Phase-B curve logged as inconclusive rather than confirmatory** — the negative aftereffect (54.7% vs 76.4% baseline, lawful 92.1%/8.3% conditional capture, controlled against a symmetric pre-adaptation baseline) is a signature only recalibration can produce; relearning and arousal are both refuted because both require an immediate snap-back at Phase C, which did not happen.

**Spatial-code-vs-salience fork: NARROWED, not resolved** — the aftereffect rules out a zero-memory retinotopic reflex (there is no live cue during Phase C, yet the mis-selection persists and is bearing-specific), so *some* persistent state now sits between vision and action; whether that state is a single global bias or a structured, generalizing spatial map is still open.

**The most important thing we don't know yet:** whether the persistent post-adaptation bias is a single scalar correction applied everywhere or a spatially-structured remapping that varies by bearing — a question the generalization-of-offset test (§4) can answer using checkpoints already in hand.

**Recommended diagnostic** (not a training run — just a measurement): re-score the already-collected aftereffect eval episodes, binned by bearing distance from the trained −30° offset, to check whether the 77–100%-vs-~33% conditional-capture split is a clean step (consistent with a single global bias) or itself has structure (consistent with a spatially varying remap) — this uses data already on disk.

---

## 2026-07-08 — Visual object-agnosticism: the decoy discrimination is color-bound, not shape-bound

The mismatched-shape decoy test (FINDINGS 2026-07-08) overrides the two balls' geom primitive zero-shot on the sphere-trained decoy policies. Result across two seeds, four shapes (sphere/box/capsule + both-non-sphere): sighted choice_vs_true is 76–81% in every condition; ablated floors sit at ~52% for both box and sphere.

**Generalization-as-primary / object-agnostic equivalence-class claim: EXTENDED TO VISION for the first time.** Phase XV gave this claim its proprio support (held-out ellipsoid/capsule reached like spheres). That support was epistemically weak because proprio *cannot perceive shape* — invariance there is nearly definitional. The present result extends the equivalence class to the visual channel, where it is a genuine learned fact: vision HAS shape information (a box and a sphere have different 32×32 silhouettes, and the ablated floor confirms the difference is real to physics) and the policy nonetheless binds its choice to color, not geometry. Condition D (both objects boxes, neither the trained shape) at 79% is the sharpest cut — discrimination survives with the trained shape entirely absent. This is the strong form of the equivalence-class claim that the proprio-only result could not deliver.

**Presence/arousal-vs-direction framework: consistent, and further constrained.** Today's prism-displacement entry established the decoy channel is directional (displacement degrades choice). This adds that the directional cue it uses is CHROMATIC — the policy computes "go toward the red one," and that binding is indifferent to the red object's shape. Together: the visual channel on the two-ball task is a color-keyed directional selector, robust to object geometry, limited only by field of view (central ~90–96%, edges →chance, unchanged across shapes).

**Methodological note (synthetic-null / give-the-baseline-a-chance-to-cheat, again).** Two controls were load-bearing and both fired correctly. (1) The offset-0 prism `ghost` is a fixed red-sphere distractor left visible by the env; hiding it left the control unchanged (77.2% vs the prior 76–79%), so it did NOT confound prior offset-0 numbers — but it WOULD have silently defeated the shape test by supplying a red sphere in the box conditions. (2) The ablated floor was run per-shape; B′(box)=A′(sphere)=~52% rules out a cube-vs-rolling-sphere contact asymmetry masquerading as a visual result. Neither control was optional; both are now folded into the shape-eval harness.

**What this resolves and what it doesn't.** Resolves: the decoy discrimination is not a shape/texture artifact and not a sphere-lock — it is color object-agnosticism. Does not resolve: whether the color binding is "approach red" or "avoid blue" (recolor test needed); whether shape would compete at higher camera resolution; shape × prism-displacement interaction.

### Theory Monitor Note — 2026-07-08 (visual object-agnosticism)
Object-agnostic equivalence class: **EXTENDED TO VISION (strong form)** — sphere-trained decoy policy discriminates color at full strength on boxes/capsules zero-shot, incl. both-object-novel (79%), across 2 seeds; ablated floor ~52% per-shape rules out a physical confound. First project evidence that the *visual* channel is object-agnostic (a learned indifference to available shape info, stronger than proprio's by-construction invariance).
**Most important thing we don't know yet:** whether the color binding is approach-red or avoid-blue, and whether shape competes with color once the camera can resolve it.
**Recommended diagnostic** (measurement, not a run): recolor test — target=green vs decoy=red (does the policy follow the rewarded target when it is no longer red → "approach target/avoid blue"), and target=red vs decoy=green (no blue present → does "avoid blue" collapse to chance). Separates approach-red from avoid-blue using the existing harness plus a geom_rgba override.

### Recolor addendum — 2026-07-08 (mechanism: approach-red, not avoid-blue)

The visual-object-agnosticism cue was pinned down by a zero-shot recolor sweep (reward always on target_geom): target=red/decoy=green 73%, target=green/decoy=blue 53% (chance), target=blue/decoy=red 18% (chases the red decoy 72%), target=green/decoy=yellow 30% (goes to yellow 63%). Reading: a positive attraction to the RED channel — "approach red." The blue-target row (pursues red into the wrong choice) refutes an "avoid-blue" account; the green/blue chance row confirms it (a blue-avoider would pick green well above chance); the green/yellow row shows the rule keys on the R chromatic channel and generalizes to yellow.

**Salience-vs-spatial fork: LEANS SALIENCE.** This is the same fork the prism-displacement entry left open. "Approach the reddest region in view" is a chromatic phototropism, not an abstract target-bearing computation. So the two 2026-07-08 results combine to: the decoy visual channel is a **red-channel-keyed directional attractor** — directional (displacement degrades choice → not pure arousal), object-agnostic (shape-invariant), FOV-limited, and salience-flavored rather than a spatial map. A true bearing/spatial-map account is now the less-supported branch, though not excluded (a phototropism can still carry usable bearing information at short range).

---

## 2026-07-18 — GAZE-frame apparatus fix validated end-to-end: the interpenetration chain reproduces on corrected geometry, cleanest instrument yet

**Context.** The 2026-07-18 governing-frame correction (CLAUDE.md: "Perception lives in the GAZE
frame") diagnosed that every prior bearing-readout instrument (2026-07-17 x2, both R²≈0 even sighted)
was likely broken by a world/torso-frame vs gaze-frame mismatch, not by an absent signal. This entry is
the first trained-policy test of the fix: gaze-relative prism ghost + gaze-cone spawn, plain body-frame
readout vs `--gain-field` (FiLM head-pose composition), offset 0, 250K each, gate eval 200 eps/cell.

**Result: the fix works, and the interpenetration signature reproduces cleanly for the first time on
this apparatus.** Sighted heading tracks true bearing (R²=0.524 plain / 0.400 gain-field, both clearing
the pre-registered ≥0.30 gate) while blind is flat (R²=0.014 / 0.000, clearing ≤0.10) — and blind still
touches balls MORE often (93.0%/90.5% vs 86.5%/88.0% sighted), ruling out the standing "low-touch
artifact" concern independently via a second metric (choice_vs_true: blind 49.5%/51.4% sits on the 50%
chance floor). Two structurally different instruments (continuous R², binary choice) now agree, on the
first bearing readout that has ever cleared its own validity gate in this project.

**What this confirms.** The **interpenetration thesis** (vision genuinely sets a directional/choice
signal touch/proprio structurally cannot supply, established via the decoy-discrimination →
prism-displacement chain of 2026-07-06/07-08) reproduces on the corrected gaze-frame geometry, with the
cleanest instrument margin the project has produced. It also confirms the causal diagnosis in the
2026-07-18 GAZE-frame governing-frame note itself: the world/gaze-frame mismatch, not an absent visual
signal, was the reason every earlier readout returned R²≈0. Fixing the frame — not adding a new
mechanism — was sufficient.

**What this does NOT confirm.** Offset 0 has no lens displacement to correct, so this result is silent
on the actual north-star question (does the gain-field help genuine recalibration under the lens?) and
silent on the generalization-as-primary framework (no size/shape/distance sweep here). Both arms are
committed to Stage 2 specifically because Stage 1 cannot discriminate them — see FINDINGS.md 2026-07-18
entry for the full A/B and independent Theory Monitor verification.

**The most important thing we don't know yet:** whether the gain-field's head-pose composition earns
its complexity under the one condition it was designed for (the +30° lens) — untestable at offset 0 by
construction.

---

## 2026-07-18 — Distance-holdout gap generalization on the vision line: interpolation supported, but qualify the "indistinguishable" claim (independent Theory Monitor read)

**What was tested.** `dist_holdout_vis_s0`: the vision-line (`s1corr`) checkpoint continue-trained 200K
steps on a widened ball-distance band 0.35–0.85 m with a held-out gap at 0.55–0.75 m (never sampled during
training). Evaluated in three cells, 100 eps each, offset 0, gaze-spawn, cone 136°: near (0.35–0.55,
trained), GAP (0.55–0.75, held-out), far (0.75–0.85, trained). Per PREREG_distance_generalization_holdout.md,
contact rate is primary, directed choice_vs_true (red vs blue, 50% floor) is co-primary, steps-to-contact is
corroborating-only (too noisy on this substrate), and `committed` (body actually translated toward a ball,
vs. contact by proximity alone) is the confound check the panel added specifically because of the earlier
cart-sweep-delivery problem on this project's proprio line.

**1) Does gap ≈ trained bands confirm generalization-as-primary, or is the case weaker than it looks?**
It is real evidence for the reach-as-law reading, but weaker than the headline framing suggests, for one
concrete reason: **the near cell is not an admissible "trained" comparator.** committed = 5/100 at near
means 95% of near contacts happened without the body translating toward a ball — geometry (prone-reach),
not reach. So "gap is indistinguishable from the trained bands" is really only tested against the *far*
cell, the only other clean (committed-heavy) bin: far contact 95%/committed 95, GAP contact 97%/committed
78. Against that single clean anchor, the pre-registered pass condition (gap contact ≥ 0.8× trained, gap
choice near trained) is met with room to spare, and the falsifier (gap contact ≤ 0.5× trained, choice
collapsing to 50%) is nowhere close. That is a genuine pass — but it is a pass against one clean bracket,
not two, and the paper's own framing ("indistinguishable from the trained bands," plural) overstates what
the near cell actually licenses. Additionally, n=100/cell gives choice-accuracy CIs of ±8–9 points; overlap
at that width rules out a large gap-specific deficit but cannot rule out a modest one (e.g. a genuine 10-pt
dip would likely not separate from noise here). **Verdict: CONFIRMS the reach-generalizes-across-distance
reading on the vision line, on the one clean comparison available, with the caveat that the claim currently
rests on n=100/cell and a single clean trained anchor, not two.**

**2) Taylor's interpolation prediction (gap sits between the bracketing trained distances) — does it hold,
and does the near confound undermine it?** Only partially testable, for the reason above: Taylor's §6.11
"automatic interpolation device" predicts the untrained middle should sit as a smooth *weighted average*
between the two trained brackets. Contact rate can't show this — it's ceiling-bound (95–100%) across all
three cells, so there's no gradient to read a weighted-average shape off of. Directed choice is the metric
with headroom, and there the numbers do **not** show a clean monotonic in-between shape: near 77.0%, GAP
72.2%, far 75.8% — GAP is numerically the *lowest* of the three, not sandwiched between two clean brackets,
though well within the ±8-9-point noise band of both. This is consistent with "uniformly good across the
whole range" more than with the specific graded-interpolation shape Taylor's account names, but the data
cannot distinguish the two at this n. Yes, the near-cell geometry confound directly undermines using near
as one of the two brackets Taylor's account requires — only GAP and far are demonstrated to be genuine-reach
cells, so the "between two brackets" test is currently running on one real bracket plus one confounded one,
not two real ones.

**3) Does this bear on the long-running "vision inert / non-directional" finding — challenge or refine?**
Refines, and in the confirming direction — it does not resurrect or contradict the old Phase XIII (R43/R44,
2026-05-30) finding, because that finding was already attributed to an apparatus defect (world/torso-frame
vs. gaze-frame mismatch), not to a fact about AB, and was superseded by the 2026-07-18 GAZE-frame fix
entry immediately above this one. What this run adds to the corrected picture: directed choice stays
reliably above the 50% chance floor (72.2–77.0%, all three cells) not just at the single offset-0/fixed-
distance condition the Stage-1 gate-eval tested (87.9%/86.9%), but across a distance range from 0.35 m to
0.85 m *including a never-trained middle band*. That is a genuine extension of "vision is directional" along
a new axis (distance) that the 07-18 entry did not test. One honest wrinkle worth flagging, not smoothing
over: choice accuracy here (72–77% across all three cells, including the *trained* near and far bins) sits
noticeably below the Stage-1 gate-eval's 86.9–87.9% on the same lineage. Two candidate explanations, neither
yet distinguished: (a) 200K more steps of continued training on a substantially harder, wider task diluted
peak discrimination accuracy somewhat; (b) the wider spawn cone (136° here vs. whatever the Stage-1 gate
used) makes the discrimination itself harder everywhere, independent of distance. Either way, the drop is
present in the trained bins too, so it is not gap-specific and does not threaten the interpolation reading —
but it should be tracked, not left unexplained.

**4) Single sharpest follow-up.** Per the field's own function-learning framework (DeLosh, McDaniel &
Busemeyer 1997, cited in the pre-reg): interpolation success is the *weaker* of the two rule-vs-lookup
tests; **extrapolation beyond the trained range is the sharp discriminator**, because a sufficiently dense
lookup/averaging table can pass an interpolation test but cannot pass an honest extrapolation test without
an actual metric law. The single sharpest, cheapest next measurement is therefore an eval cell beyond 0.85 m
(no new training — same checkpoint, one more eval bin, winnability-checked first per the standing rule) to
see whether contact/committed/choice hold up, degrade gracefully (Taylor §4.10/6.6's "bounded extrapolation,
carries a little beyond, then flattens"), or collapse. A second, genuinely free diagnostic that should
happen first because it costs nothing: re-bin the *already-collected* near-cell episodes by the `committed`
flag (or by sub-radius near the 0.55 edge vs. the 0.35 edge) and check whether the ~5 genuinely-committed
near episodes' choice accuracy resembles GAP/far or differs — this would repair the "one clean bracket"
problem in §1/§2 using data already on disk, before spending any new compute.

**Disagreement with the framing in the dispatch, for the record.** The dispatch's question 1 asks whether
gap-indistinguishable-from-trained-bands confirms the framework "or is the case weaker than it looks" — my
read is the latter, specifically: it is a real, qualified CONFIRM (per §1), not the clean unqualified
confirm the phrase "indistinguishable from the trained bands" (plural) implies, because one of those two
bands (near) is not a valid comparator for reach-generalization. This should be recorded as a qualification
of the headline, not a rejection of it — the GAP-vs-far comparison, which IS clean, passes clearly.

### Theory Monitor Note — 2026-07-18 (Distance-holdout gap, vision line)

**Generalization-as-primary / reach-as-law framework: CONFIRMED, qualified** — GAP performance (97%
contact, 78% committed, 72.2% directed choice) matches the one clean trained anchor available (far: 95%
contact, 95% committed, 75.8% choice) on a never-trained middle distance band; the near cell cannot serve
as a second clean anchor because only 5/100 of its contacts involved genuine directed movement.

**Interpenetration / vision-is-directional framework: CONFIRMED and extended** — directed choice stays
well above the 50% floor at all three distances including the held-out gap, extending the 2026-07-18
GAZE-frame result (directional at a single distance) to a range of distances; this refines rather than
challenges the corrected picture, and does not revive the superseded Phase XIII "non-directional" finding.

**The most important thing we don't know yet:** whether AB's reach law genuinely extrapolates beyond the
trained 0.35–0.85 m range (the sharp rule-vs-lookup test) or only interpolates within it — untested by
this run by design.

**Recommended diagnostic** (not a training run — just measurements, in priority order): (1) free re-bin
of the already-collected near-cell episodes by the `committed` flag / sub-radius to get a second genuinely
clean trained anchor before trusting the "gap ≈ trained bands, plural" framing; (2) one new eval cell
beyond 0.85 m, winnability-checked first, to test extrapolation rather than interpolation — the sharper
discriminator per the pre-reg's own cited function-learning literature.

---

## 2026-07-19 — Stage-2 overnight (mildhead_vis_s0, +30° gaze-relative lens, cross-modal mismatch): recalibration NOT demonstrated, and the aftereffect remains UNMEASURABLE — independent Theory Monitor verdict

**What was run.** Four 300K-step arms (plain/gain-field × seed 0/1), warm-started from `mildhead_vis_s0`,
trained under a sustained +30° gaze-relative lens with the cross-modal mismatch aux loss (coef 0.1,
gate_contact=True, encoder frozen — only the eye/bearing heads train, 14–18 tensors). Read at matched +30°
against an unadapted base. This is the first Stage-2 test built on the 2026-07-18 GAZE-frame-corrected
apparatus (the offset-0 Stage-1 A/B that validated the fixed instrument).

**Headline numbers (choice_vs_true @ +30°, sighted / ablated):**

| arm | sighted | ablated | vs. base sighted (67.6%) |
|---|---|---|---|
| base (unadapted) | 67.6% ± 6.8 | 56.5% ± 7.1 | — |
| plain_s0 | 66.8% ± 6.7 | 50.8% ± 7.3 | flat |
| plain_s1 | 61.5% ± 7.1 | 48.9% ± 7.4 | worse |
| gain-field_s0 | **44.4% ± 7.3** | 52.0% ± 7.3 | **worse, and BELOW its own ablated control** |
| gain-field_s1 | 56.8% ± 7.1 | 51.3% ± 7.1 | worse |

Aftereffect (lens off, offset 0): ballistic no-feedback diff-in-diff −3.0/−7.5/+1.3/−1.8° (all \|z\|<1.4);
in-view whole-episode signed-bias shift +74.1/+66.4/+58.6/+46.9° but base cc_inview = **−0.257** against a
validated ~+0.5 baseline (fails its own printed sanity check).

### 1. Genuine challenge to AB, or setup/instrument failure? — MIXED, and the two halves fail for different reasons

The **choice_vs_true metric is a real, trustworthy null, not a broken ruler.** Its blind-condition chance
floor holds tightly in every cell tonight (56.5/50.8/52.0/48.9/51.3 — all pinned near 50%, exactly as
designed), and it is the same instrument that passed its Stage-1 validity gate (offset 0, R²=0.52/0.40
sighted vs ~0 blind). So "none of the four arms improved over the 67.6% unadapted baseline" is a real
behavioral fact, not an artifact.

But that does not make it a challenge to the North Star claim, because there is a concrete, upstream,
fixable cause sitting directly under "did training teach anything": **the mismatch aux_loss never
converges.** Across all four 300K-step runs it oscillates in a 0.03–0.10 band with no visible downward
trend (e.g. plain_s0: 0.037→0.061→0.038→0.050→…→0.061, ending where it started). That is the literal
training signal this experiment is betting recalibration on, and it shows no sign of doing its job — compounded by `gate_contact=True` (aux loss fires only on contact, so effective sample density is likely far
sparser than n_samples=4096 implies) and a frozen encoder (only the head, 14–18 tensors, can move). This
is exactly the shape of "our setup failed to present a learnable correction," per the North Star rule —
not evidence that AB's vision cannot be recalibrated.

**Verdict: INFORMATIVE NULL, setup-attributable.** The metric is valid and shows real non-improvement; the
most likely cause (a training signal that never converges) is diagnosable and fixable before concluding
anything about AB's capability.

### 2. The gain-field arm breaking — signal or noise?

**A real, reproducible finding, not an unstable run.** Gain-field underperforms plain in BOTH seeds (44.4
vs 66.8 at seed 0; 56.8 vs 61.5 at seed 1) — same direction both times, directly contradicting the
2026-07-18 pre-registration ("watch whether the same small plain-favoring edge re-appears under the lens
— a gain-field that helps nowhere would be a real, if modest, strike against carrying its extra
parameters"). Seed 0 is worse than "nowhere": sighted (44.4%) falls below its own ablated control (52.0%)
and below the 50% chance floor — by the logic already established in the 2026-07-08 follow-the-ghost
entry, a below-chance choice means something is actively steering AB toward the wrong ball more often than
chance, not merely failing to help. Most likely mechanism: the FiLM head-pose composition adds parameters
trained by the same weak, non-converging aux signal noted above — more capacity with no more signal is
worse-conditioned than the simpler plain readout, not better. This is a real strike against the gain-field
as currently trained; it is not proof the gain-field hypothesis is wrong in principle (untrained-signal
starvation and architectural unsuitability are not yet distinguished).

### 3. Status of the north-star claim — "AB did not recalibrate" or "we cannot measure it"?

**Squarely the second: we still cannot measure whether AB recalibrated.** Two separate instruments, two
separate failures, and one correction to how the dispatch framed them:

- The **in-view instrument** fails its own printed sanity check outright (base cc_inview should be ≈+0.5;
  it reads **−0.257**, wrong sign and magnitude). This is the **second** documented failure of this exact
  instrument — the 2026-07-18 afternoon `stage2_adapt_s0` entry already flagged it as "UNINTERPRETABLE...
  validated base eye circ_corr +0.54 reads ~0 in-view here." The +47–74° "aftereffect" numbers tonight are
  not evidence of anything; they are outputs of a ruler that does not read zero correctly at its own
  baseline.

- The **ballistic no-feedback instrument** (the "psychophysics convention" one) was **already proven
  invalid on 2026-07-17** — R²(true bearing, early heading) ≈ 0 in all four cells *including sighted*,
  meaning this body has no ballistic-aim phase to measure; the early window captures a target-independent
  stereotyped motion, not vision-guided aim. **No R² re-validation for this instrument appears anywhere in
  tonight's log** — it was reused without being re-checked. Its "NULL (\|z\|<1.4)" result tonight should
  therefore be downgraded from "null" to "uninterpretable," the same status as the in-view instrument —
  this is a correction to the dispatch's framing, which treated the ballistic result as a corroborating
  clean null against the in-view result's known-broken one. They are both broken, for different, already-
  documented reasons.

Both instruments used tonight are either freshly broken or previously-diagnosed-broken-and-not-refixed.
Under those conditions "AB did not recalibrate" is not a licensed conclusion. The honest statement is that
the sharpest test in the classical prism paradigm (the aftereffect) has not yet been run with a working
ruler.

### 4. Single most important next step

**Fix/re-validate the in-view aftereffect instrument first, using data already on disk — not a training
run.** Diagnose why base cc_inview reads −0.257 instead of the validated ~+0.5 (the likely suspect is the
same class of frame bug the GAZE-frame fix already found once — check whether the in-view azimuth is
being read in the correct frame relative to the corrected per-step gaze-relative prism), and confirm the
fix reproduces +0.5 at the offset-0/no-lens baseline before trusting any signed-bias number from it. This
is the cheapest, most directly blocking fix: no new training required, and until it passes, no future run
— including a stronger-error-signal rerun — can be interpreted. Close second, not this pick: check whether
the mismatch aux_loss can be made to converge at all (higher coef, ungated, or more of the encoder
unfrozen) before spending another overnight on this exact recipe — a validated ruler applied to a training
signal that never moved will just confirm "nothing to measure" a different way.

### Theory Monitor Note — 2026-07-19 (Stage-2 overnight, +30° gaze-relative lens)

**Recalibration-under-lens (choice_vs_true): INFORMATIVE NULL, setup-attributable** — the metric is
instrument-valid (blind pinned at chance in every cell) and shows a real non-improvement in all four arms
(66.8/61.5/44.4/56.8% vs. 67.6% base), but the mismatch aux_loss shows no convergence over 300K steps in
any arm, which is the more likely explanation than an incapacity finding about AB.

**Gain-field hypothesis: WEAKENED under this training recipe, not refuted in principle** — seed-consistent
underperformance vs. plain in both seeds, with seed 0 falling below its own ablated control and the 50%
chance floor (an active-harm signature, not mere non-help), directly against the 2026-07-18
pre-registration's watch condition.

**Aftereffect (the sharpest test): UNMEASURABLE, not "absent"** — the in-view instrument fails its own
baseline sanity check for the second documented time; the ballistic instrument was already proven to carry
no aim signal on 2026-07-17 and was not re-validated before reuse tonight. Neither instrument licenses "AB
did not recalibrate."

**The most important thing we don't know yet:** whether the mismatch aux_loss can be made to converge at
all on this task (right now it doesn't, in any of four independent 300K-step runs) — this is prior to and
separable from the instrument-validity question, and until both are resolved, no amount of Stage-2 seed
replication will produce an interpretable result.

**Recommended diagnostic** (not a training run — a measurement): re-derive the in-view instrument's
azimuth computation against the corrected per-step gaze-relative prism frame and confirm it reproduces the
validated ~+0.5 base cc_inview at offset 0 before trusting any signed-bias number from it; in parallel,
plot aux_loss over the full 300K steps for at least one arm already in hand to confirm (or refute) that it
is genuinely flat rather than slowly trending — both are free re-analyses of data already on disk.

---

---

## 2026-07-20 — Spatial bearing PASSES, but it is reactive servoing and it does NOT overturn Phase V (independent Theory Monitor verdict)

**Result under review:** `vbear_s0`. Walker + frozen gait + transplanted frozen crawler eyes, one ball
at a random bearing ±0.9 rad. Sighted contact 100/120 bins-uniform vs blind 0.8%; R²(true gaze bearing
→ first commanded turn) sighted +0.517 / blind +0.006, slope +0.58; R²(→mean turn over 5) +0.807.
Preflight: frozen-trunk bearing probe R²=0.996; oracle ceiling 100%; motor-habit floor 30%.

**Verdict: CONFIRMS an already-recorded hypothesis; does NOT constitute an independent overturn of the
Phase V "vision is non-directional" finding.**

The monitor's central objection, recorded because it is a correction to the main loop's initial framing:
versus R43/Phase V, **two variables changed simultaneously** — the body became steerable AND the encoder
is a different, transplanted, frozen one. Calling this "Phase V overturned" would be a causal claim
without the control that isolates which variable did the work — the exact failure the project's
hypothesis-gate rule exists to catch. What it DOES do is confirm the R49 hypothesis already on record
(2026-06-16): vision inertness was a body/pressure problem — the cart substrate made direction
behaviourally pointless because AB could not steer. Give it a body that can turn, and direction becomes
load-bearing. The isolating control, if we want the overturn: run the SAME R43 encoder on a steerable
body, or the crawler encoder on the cart.

**On the R²=0.996 vs Phase-V lateral R²≈0.01 gap.** The monitor's reading is that this is plausibly in
large part a MEASUREMENT-FRAME artifact rather than a purely representational difference: R43's decode
was likely run in a torso/world frame, and per the 2026-07-18 finding the torso frame can sit ~90° from
functional forward — a probe in the wrong frame reads R²≈0 even when the information is present in the
gaze frame. Within this run the "any shortfall would be a policy failure, not a representation failure"
claim IS licensed, because the probe was run in the same gaze frame the whole experiment uses, before
the head was trained. But the frozen-and-already-competent trunk is a genuine confound on difficulty:
300K steps only had to learn a map from a near-perfect linear feature to two numbers. **The open question
this does not touch: can our RL objective ever PRODUCE a bearing representation from scratch, or do our
positive direction results only ever come from handing the policy a pre-verified frozen map?**

**Reactive vs predictive — resolved, and it constrains the claim.** The monitor named this as the most
likely uncontrolled alternative to "100% at all bearings": a tight closed-loop visual servo would produce
the same success, the same R², and the same image-centring, with zero forward prediction. Checked and
settled two ways. (1) Architecturally: the driver is an `ActorCriticPolicy` with Conv2d/Linear/LayerNorm/
ReLU/Tanh and **no recurrent state**, so it cannot hold an estimate across steps — servoing is not a
rival hypothesis, it is the only thing the wiring permits. (2) Empirically, via the monitor's own
recommended occlusion probe: R²(bearing→turn) = +0.741 at the last sighted step, **+0.000** across 6
occluded steps, turn-std unchanged at 0.271. Steering is instantly untethered from the target.
**Standing constraint on language: describe this as closed-loop visual servoing on current retinal
bearing. No predictive or model-based vocabulary.**

**On Taylor Ch.4 §4.6** ("he perceives his environment but not his own position in it" — vision alone
never determines the steering action). The blind arm at 0.8% reads as vision being near-necessary, which
looks like tension but is not: the driver's 9-dim proprio structurally carries no ball-position
information, so for "which way do I turn" vision is the only channel that could inform it, while
proprioception still does Taylor's work inside the frozen gait keeping the turn from becoming a fall.
Taylor's claim is that multiple *informative* channels jointly determine action, not that every channel
informs every decision. **Flagged as reconciled-but-untested: we have no proprio-ablation arm.**

**Framework scoring.** Behavioral-Prediction: (A) consistency across conditions — weak-to-moderate
support (100% across three bearing bins, 2.5–4.0 m). (B) recovery from the unexpected — now TESTED and
NEGATIVE by construction (occlusion → immediate collapse); this is the framework's central want and the
architecture cannot satisfy it. (C) speed/success correlation — untested. Pattern-Learning: the
near-perfect LINEAR fit to a CONTINUOUS variable (sin bearing) is decent evidence for a graded,
distributed code rather than a categorical trigger; robustness to small perturbations untested.

**Carried forward as open:** (1) three-seed replication before this mechanism is trusted (R20→R22/R23
precedent); (2) the isolating control against Phase V; (3) whether reward alone can grow a bearing
representation without a transplanted frozen encoder; (4) a proprio-ablation arm for the Taylor claim;
(5) instrument-tightness flag — a 6-episode run gave R² 0.349/slope +0.07 vs the n=120 run's 0.517/+0.58.

### ADDENDUM 2026-07-20 — controls run; the monitor's H2 was REFUTED and Phase V comes out CORROBORATED

The theory-monitor's two open challenges were both taken to measurement, and one of its own hypotheses
lost. Recording that plainly, because the monitor's H2 was the more attractive story and it is wrong.

**H2 (Phase V's null was a measurement-frame artefact) is REFUTED on current evidence.** The monitor
reasoned that R43's lateral R²≈0.01 might reflect decoding in a torso frame ~90° off functional forward,
with the information present all along. Tested by running our gaze-frame probe on the R43/R49 MICOA
encoders (loading verified by name and shape, 14/14). Result: R43 sits AT its architecture-matched
random floor in the wide cone (0.828 vs 0.807–0.841) and clearly BELOW it in the narrow cone (0.846 vs
0.947). By the control's own pre-registered rule that is **H1** — the representations genuinely differ,
the transplant did real work, and **Phase V's original null is corroborated rather than explained away.**

**A methodological failure this exposed, which generalises beyond this run.** The probe is nearly
saturated: a hand-coded 32-number redness-per-column statistic with no network reads gaze bearing at
R²=0.919, and untrained trunks score 0.762–0.841. Our own headline preflight number (0.996) was
originally reported against an implied floor of ~0.0 and framed as an enormous gap. It is not — the
honest bar is ~0.92, and mildhead clears it (error remaining 0.003 vs 0.081) but by far less than the
original framing implied. **Standing rule earned here: never report an encoder-probe R² without (i) a
raw-input baseline with no network and (ii) a seed-AVERAGED random-init floor of the same architecture.
A single-draw floor and a missing raw baseline together can turn a saturated ruler into a false
discovery.** This is the same species of error as the two aim-instruments that failed the
sighted-high/blind-low check in July: the number was real, the ruler was not.

**Taylor's joint determination: STILL UNTESTED, and our earlier "reconciled" language was premature.**
The proprio-ablation arm showed steering unaffected by scrambling proprio (contact 100%, R² +0.560 vs
intact +0.521), which looks like support for our reconciliation. It is not, for two reasons found by the
skeptic: the frozen gait reads proprioception straight from `qpos/qvel` and so kept an uncorrupted copy
throughout, and the scrambled 9-number channel provably carries no bearing information anyway (decode
R² −0.055, at shuffled baseline). Scrambling an empty channel and observing no change is arithmetic.
Taylor's claim could hold in full and this table would be identical — a design that cannot produce the
disconfirming outcome is not a test. The apparatus may not be able to settle it: corrupting the gait's
proprioception would likely just topple the walker, confounding information-removal with inability to act.

**Net effect on the 2026-07-20 verdict.** The headline result (vision sets direction, reactively) is
UNCHANGED — it rests on the sighted-vs-blind behavioural gap and the bearing→turn slope, neither of
which is touched by any of the above. What changed: the supporting preflight claim is weaker than
written, Phase V is corroborated rather than merely un-overturned, and one of the two "missing controls"
turns out not to have been run at all.

### CORRECTION 2026-07-20 (David) — the two-stage learning story, and what the open question really is

I had been logging the frozen transplant as a weakness ("only answers whether a policy can USE a map
it is handed"). David corrected it: the crawler LEARNED that map. Recording the corrected frame
because it changes what the project should ask next.

**Both halves were learned, in different bodies, and joined by the transplant:** the crawler learned
WHERE the ball is (mildhead, via the cross-modal seen-vs-contacted mismatch driving a bearing readout,
in a body that could touch but not travel); the walker learned WHAT TO DO about it (turn proportional
to the seen offset, in a body that can travel). That is the REUSE-FIRST design executing, not a
shortcut around the hard part.

**This session's controls independently establish that stage 1 was real.** On an architecture-matched
probe, mildhead is the ONLY encoder above its random floor (0.997 vs 0.762); R43 is AT floor
(0.828 vs 0.807–0.841) and BELOW it in the narrow cone; R49 (0.905) is below the no-network raw-pixel
baseline (0.919). Most of this project's encoders failed to acquire visual direction. One succeeded.

**Restated open question (supersedes "can reward grow a bearing representation from scratch?"):**
WHICH SIGNAL grows visual direction? Cross-modal mismatch / bearing-readout → SUCCEEDED (mildhead).
Task reward + MICOA → FAILED (R43, R49), and today measurably so, not merely un-recruited. This is a
positive finding about the mechanism of grounding rather than a caveat about our setup, and it
converges with the external literature's distil-then-RL hierarchy (Mirowski 2017 auxiliary-loss-beats-
extra-input; Lee/Hwangbo 2020 and Chen 2019 privileged-teacher→student).

**Taylor connection, in plain English.** Taylor's claim is that perceiving where something is IS being
ready to make the movement that lands on it — perception grows out of contact-terminated action, not
out of passive looking. mildhead's successful signal is exactly a contact signal: what the eye SAW
versus what the body actually TOUCHED. The signal that failed (task reward) carries no such
seen-versus-touched comparison. So the project's own two encoders constitute a small natural
experiment in Taylor's favour, and that is worth stating as a prediction to test rather than a
coincidence noticed after the fact.
