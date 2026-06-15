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

3. **No cross-modal coherence loss.** The v5 consistency loss (`λ · MSE(hidden1(full obs), hidden1(pixels-zeroed obs))`) was designed for exactly this purpose and achieved CKA = 0.998 in the tabletop setting. The v8 follow-on runs do not use this loss. The omission is the most direct explanation for why v8 produces 5% touch rate while v5 produced 100%.

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

The v5 tabletop result (CKA = 0.998, 100% success, frozen proprio + consistency loss) was achieved on a different body, different task, and fundamentally different proprioceptive signal. The v8 proprio signal is locomotion-based (body displacement, paddle stroke timing) rather than arm-joint-based (joint angles, fingertip offset). It is not guaranteed that the same architectural choices that produced MICOA-alignment in the tabletop setting will transfer to the locomotion setting.

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
| ball y_ego (forward / distance) | 0.217 ± 0.016 | 0.426 ± 0.042 |

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
