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

The previous theoretical picture assumed that vision follow-ons were failing to help. The blind baseline reveals the more precise and more alarming diagnosis: they are actively overwriting a working solution.

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
| CNN encoder will make vision useful | Ruled out | overnight run #2 + entpin: consistency_loss ~0.001 both times |
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

There is currently no way to distinguish these three bottlenecks from the numbers alone. The left/right asymmetry ablation and a vision-on vs. vision-zeroed comparison at the best checkpoint are both needed.

### Theory update: what constraint has not yet been relaxed?

The previous MICOA theory section listed three MICOA requirements:
1. Freeze proprio weights (done — drift = 0.0)
2. Consistency loss pulling vision toward proprio (done — λ = 0.10)
3. Zero-initialize pixel weights (done — confirmed in this run)

All three have now been implemented and the result is still 8%. The theoretical question is: what constraint remains unrelaxed?

The answer suggested by this run's data is: **the downstream layers are unfrozen and are accessible to vision's gradient throughout training.** Vision cannot overwrite proprio's stored weights (freeze holds). But vision can, and apparently does, inject a divergent signal into the unfrozen downstream layers (hidden2 and the output layer) that changes the motor output even when proprio's hidden1 contribution is unchanged. The consistency loss fights this injection at the h_full level, but it is fighting an ongoing gradient that is always pointing away from proprio's manifold.

The constraint that has not yet been relaxed is: **downstream layer gradient access for the vision pathway during the pre-convergence period.** Two candidate relaxations:

- **Freeze all downstream layers (hidden2, output) until consistency_loss falls below a threshold** — vision can only modify pixel-column weights in hidden1, not anything downstream, until the hidden1 representation has converged. This is the most direct way to enforce "vision cannot cancel proprio's motor program during pre-convergence."
- **Much higher λ (e.g., 1.0 or 5.0)** — if the consistency penalty is large enough, vision's contribution to h_full is strongly suppressed during the divergent period, limiting its action-space footprint even without a hard downstream freeze.

Neither of these has been tested. The theory predicts that relaxing the downstream constraint should recover something closer to the 60% blind baseline while still allowing vision to eventually contribute steering information.

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
| Low consistency_loss means MICOA confirmation is occurring | Ruled out | low loss = dead pixel pathway; high loss = active cancellation |
| Zero-init prevents early vision divergence | Ruled out | consistency_loss rose faster and higher than prior run despite zero start |
| Forward cone + zero-init is sufficient to raise touch rate above 8% | Ruled out | same 4/50 ceiling as prior MICOA freeze run |

### Updated cumulative "what is not yet ruled out" table

| Hypothesis | Next test |
|---|---|
| Left/right touch symmetry reflects genuine visual steering (not geometric spawn artifact) | Left-only spawn eval: 30 det. eps with ball always 10–15° left |
| Downstream layer freeze prevents cancellation and recovers the 60% proprio motor output | Architecture change: freeze hidden2 + output layer until consistency_loss < threshold |
| Higher λ (1.0 or 5.0) suppresses vision's action footprint enough to prevent cancellation | λ sweep on frozen-proprio + zero-init architecture |
| Moving-ball task (Idea E) changes gradient landscape so MICOA convergence is achievable | Build moving-ball env, smoke test with current architecture |
| Vision is contributing some direction information at 8% — 500K steps would compound it | Extended run (500K) from current best checkpoint |
| Best-checkpoint policy (30K) scores differently with pixels on vs. zeroed — first direct vision-load-bearing test | Vision ablation eval: same 50 seeds, pixels zeroed, compare touch rate |

### Theoretical verdict

The zero-init + forward cone run tells us something definitive: the bottleneck is not in the initialization or in whether the ball is visible from frame one. Both of those variables have now been optimized in the best-case direction and the outcome is unchanged. The bottleneck is structural. It is in how the downstream layers of the frozen-proprio architecture respond to the pixel pathway's gradient signal during the pre-convergence period. Until that structural bottleneck is addressed — either by a downstream layer freeze, a much stronger consistency penalty, or a task redesign that makes proprio's blind strategy fail — the 8% ceiling will persist regardless of what we do to the pixel-column initialization or the spawn geometry.

The theory predicts the next constraint to relax is downstream layer protection. The experiment that tests this prediction is a follow-on with hidden2 and the output layer frozen until consistency_loss drops below 0.05, then unfrozen to allow the now-converged vision representation to improve motor output. If touch rate recovers toward 60% under this regime, the structural bottleneck hypothesis is confirmed. If it does not, we must reconsider whether the frozen downstream geometry established by blind proprio is simply incompatible with any vision contribution — in which case the moving-ball task (where proprio's strategy structurally fails) becomes the correct next step.

---

## 2026-05-07 — BREAKTHROUGH: Vision Confirmed Load-Bearing (First Time)

### What was confirmed

The vision-ablation sensitivity test on the micoa_freeze_zeroinit_cone30 best checkpoint (30K steps) returned:

- Mean |action_full - action_blind|: **0.456**
- Max difference: **0.886**
- Fraction of steps with non-trivial difference (> 0.01): **100%**
- Fraction of steps with meaningful difference (> 0.05): **100%**
- Steps measured: 192 (40 episodes, up to 5 steps each)

This is the first time in this project's history that vision has been confirmed load-bearing. Every prior run — five independent runs across different architectures (MLP, CNN), entropy regimes (auto, pinned), contact definitions, and spawn geometries — returned near-zero sensitivity. The camera was, in all of them, effectively ignored by the policy. This checkpoint is different. On every single measured step, the policy produces a meaningfully different action when the camera sees the actual scene versus when the camera sees a blank screen. The difference is not marginal (mean 0.456, max 0.886 — on an action space that spans roughly ±1). Vision is being read.

### Why this matters

This is a milestone because it answers the question the project has been asking since the overnight sweep: **has the creature ever, at any point in training, actually looked at the world and let what it saw change what it did?**

The answer was "no" for five runs and hundreds of thousands of training steps. It is now "yes."

This matters for the theory. The project was motivated by the idea that a creature with a proper developmental sequence — proprio established first, vision added as a confirming signal on top — should develop vision that is genuinely integrated into behavior, not merely appended to it. The prior runs were failing at the most basic prerequisite: the camera wasn't even contributing to the action. We couldn't test any claim about *how* vision was integrated because vision wasn't integrated at all. Now it is. We can now ask richer questions.

### The exact boundary of the claim

"Vision load-bearing" means one thing precisely: the policy's action changes when pixels are zeroed. That is the full content of the claim. It does not mean:

- **Vision is steering correctly.** The directional spawn test (ball fixed at +15° right: 0/30 touches; ball fixed at -15° left: 0/30 touches) shows that vision's influence on action has not yet translated into reliable directional guidance. The creature looks at the scene and acts differently because of what it sees — but the different actions it produces do not yet consistently move it toward the ball.

- **Vision is helping more than it hurts.** The 8% deterministic touch rate is still far below the 60% blind baseline. Vision is load-bearing in the sense that it influences every step, but the influence is not yet net-positive.

- **Vision will remain load-bearing with more training.** The late-training collapse pattern (peak at 30K, then degradation) means we do not know whether this sensitivity is stable or whether longer training would cause the pixel pathway to drift back toward irrelevance.

The correct precise statement is: **at the 30K checkpoint of the micoa_freeze_zeroinit_cone30 run, vision is influencing every action, but has not yet learned which visual patterns should produce which directional motor responses.**

### Distinguishing "vision affects behavior" from "vision affects behavior correctly"

This distinction is the central theoretical question the project now faces. It replaces the previous central question ("why isn't vision used at all?"), which is now answered.

To understand the distinction, consider what a vision-guided creature would need to do. Suppose the ball is 15 degrees to the creature's left. The camera image has a colored ball visible in the left portion of the frame. A vision-guided creature would need to have learned a mapping from "red blob in left half of image" to "execute leftward turn: increase left-arm push, decrease right-arm push." That mapping is the pixel-to-direction coupling.

What we observe is: the camera image is influencing the action — the action is different when pixels are present versus zeroed. What we do not yet observe is: the influence is specifically "red blob left → turn left, red blob right → turn right." The directional spawn test tells us this mapping has not formed. Both conditions (ball precisely left and ball precisely right) produce 0/30 touches. If the mapping existed, ball-precisely-left would produce a higher touch rate than ball-precisely-right, or vice versa.

The visual influence is real. The visual influence is not yet directionally differentiated.

### Framework assessment

**Behavioral Prediction Framework — PARTIALLY CONFIRMED.**

The framework predicts: a creature that builds an internal model of cause and effect will let sensory inputs shape its decisions on every trial, not just occasionally. The 100% sensitivity rate (every step, every episode) is exactly this. The camera is being consulted on every decision. That is the "affect behavior" half of the prediction — and it is confirmed.

The framework also predicts: a creature with a genuine model of "where the ball is" will move toward the ball when it sees the ball. This half is not confirmed. The directional test returns 0/30 in both conditions. The creature is using the camera, but the camera's contribution to behavior does not yet encode "ball location → move toward ball."

The previous state of this framework in this project was "CHALLENGED" on every prior run. It can now be upgraded to "PARTIALLY CONFIRMED" — the use-vision half is confirmed, the use-it-correctly half is not yet demonstrated.

**Pattern Learning Framework — PARTIALLY CONFIRMED.**

The framework predicts: the creature's internal representation should be stable and respond consistently to similar visual inputs. The 100% sensitivity rate across 192 measured steps is exactly this — a stable representation that responds to visual input on every trial without dropping out. This is the consistency the framework predicts.

The framework also predicts: similar visual inputs should activate overlapping internal patterns that produce similar (but not identical) motor outputs. The directional test — ball left vs. ball right — is the first test of this prediction: do slightly different visual scenes (ball offset by 30 degrees) produce appropriately different motor outputs? The 0/30 result in both conditions tells us the pattern for "ball slightly left" and the pattern for "ball slightly right" have not yet differentiated in a way that steers the creature differently. The representation is stable and active; it is not yet spatially differentiated.

The previous state of this framework was "CHALLENGED." Like the Behavioral Prediction Framework, it can now be upgraded to "PARTIALLY CONFIRMED" — stability and consistency of visual influence are confirmed; directional differentiation is not yet demonstrated.

### Updated "ruled out" table

Items that can now be crossed off or updated in light of this result:

| Hypothesis | Previous status | Updated status | Update reason |
|---|---|---|---|
| Vision is never load-bearing under any MICOA architecture | Open (five failed runs) | **RULED OUT** | 0.456 mean ablation sensitivity, 100% of steps |
| The MICOA freeze + zero-init + forward-cone architecture produces an input-blind policy | Open | **RULED OUT** | Vision sensitivity confirmed at 30K checkpoint |
| Consistency_loss 0.20–0.38 means vision is alive but wrong (not dead) | Interpretation | **CONFIRMED** | Ablation confirms pixel pathway is active and influencing actions |
| Left/right touch symmetry (0.102/0.098) reflects genuine visual contribution | Ambiguous | **PARTIALLY SUPPORTED** | Vision is load-bearing, but directional test shows the influence is not yet directional |
| Vision load-bearing is achievable without making blind proprio fail | Open | **CONFIRMED** | Achieved at 30K without changing task structure |

### The new central question

The previous central question was: "Why isn't vision being used at all?"

That question is answered. Vision is being used.

The new central question is: **"Why is vision changing actions on every step, but the changed actions are not yet moving the creature toward the ball in a direction-specific way?"**

There are two distinct possible answers, and distinguishing them is the next theoretical priority:

**Answer A — The visual influence is directionally undifferentiated:** The camera is changing the action, but the change is not specifically "ball on left → turn left." The pixel pathway has learned that *some* visual input is present and should influence behavior, but has not yet learned *which* visual features (ball position in frame) should produce *which* directional motor outputs (left vs. right arm). The visual contribution is more like a general "alertness signal" (something is visible, adjust behavior) than a directional guidance signal (the ball is at bearing X, move that way). Under this answer, the fix is more training — the pixel-to-direction mapping needs more gradient signal to converge.

**Answer B — The visual influence encodes direction, but locomotion cannot execute the turn:** The creature correctly reads "ball on left" vs. "ball on right" at the pixel level — the internal representation does differentiate left from right — but the motor system is not capable of reliably converting a "turn left" signal into an actual leftward turn. The paddle-arm locomotion in v8 requires asymmetric pushing (left arm harder to turn right, right arm harder to turn left), and this asymmetric motor capability may not have been learned well enough at the 30K checkpoint for directional visual signals to produce reliable contact. Under this answer, the fix is motor competence improvement, not more visual training.

The two answers have different implications for what to build next. Answer A implies: extend training, or add more explicit gradient pressure on the pixel-to-direction mapping. Answer B implies: improve locomotion first (more stage-1 training, or a simpler motor task), then re-test visual steering.

**The diagnostic that distinguishes them:** Measure the mean action vector separately when the ball is always left versus always right. If the action vectors are systematically different between conditions (even though neither produces a touch), Answer B is more likely — the creature is detecting direction but failing to execute. If the action vectors are indistinguishable, Answer A is more likely — the direction hasn't been encoded yet. This does not require a new training run.

### What the theory predicts about how to fix the directional gap

Under the MICOA framework, the directional gap is explained as follows. Proprio in v8 is a locomotion signal — body speed, arm positions, contact with the ground. It does not encode ball position. Vision is being used (confirmed), but vision's contribution to the hidden state has not yet been shaped by enough trials where "ball on left, action taken, ball contacted" versus "ball on right, action taken, ball contacted" to learn the directional mapping. The freeze architecture means vision can only influence the downstream layers through the pixel-column contributions to h_full; those contributions are being made, but the 30K checkpoint is early — only 30,000 gradient steps of pixel-to-direction learning have occurred.

The theory predicts: **the directional mapping should emerge with more training, provided the architecture does not collapse before it can form.** The late-training collapse pattern is the threat. If the 30K checkpoint is the peak and the policy degrades after this, the directional mapping may never get enough gradient signal to form. Protecting the 30K checkpoint state and continuing training — or using the 30K checkpoint as a starting point for a longer, lower-learning-rate run — is the theory-motivated next step.

There is also a MICOA-specific prediction: the directional mapping will be easier to learn if proprio can provide *any* directional signal, even a weak one. In the current setup, proprio has zero information about ball position (no target vector in the observation by design). The pixel pathway must independently discover the pixel-to-direction mapping with zero proprio confirmation of direction. In the tabletop setting, proprio always knew the target direction — vision confirmed it. In v8, proprio cannot confirm direction because it does not know direction. Vision must earn the directional mapping from scratch, using only contact reward as the feedback signal. This is harder, and it may require more steps than a setting where proprio provides even a weak directional cue.

### Open theoretical questions going forward

The first-order question was: "Does the MICOA architecture produce a vision-load-bearing policy?" That is answered: yes.

The second-order questions, now live:

1. **Is the visual influence directionally differentiated?** Measurable now with the action-vector diagnostic described above. Does not require training.

2. **Does the directional mapping emerge with more training, or does the late-training collapse prevent it?** Requires either extending the 30K checkpoint or understanding what causes the collapse at 150K.

3. **Does the v8 architecture support the full MICOA confirmation loop?** In tabletop, proprio confirmed vision's prediction at contact. In v8, proprio cannot confirm direction. Is the one-way confirmation (vision influences action, pero proprio cannot validate vision's direction estimate) sufficient for the directional mapping to stabilize? Or does the absence of proprio-directional confirmation mean the mapping will remain noisy?

4. **What is the correct MICOA architecture for a locomotion problem where proprio is directionally blind?** The tabletop setup gave proprio a 3D fingertip-to-target vector. v8 deliberately omits this. The theoretical question is: what is the minimum proprio structure that allows MICOA-style confirmation when proprio cannot directly confirm ball direction? Candidates: body velocity (already in proprio — can vision confirm "I am moving in the direction I see the ball"?), or contact feedback (vision predicts contact will occur soon, proprio confirms it).

---

### Summary: what changed on 2026-05-07

Before the ablation test: the project had five failed runs, a working 60% blind baseline, and a clear diagnosis of why vision wasn't being used (MICOA violation). The question "is vision load-bearing?" had never been answered yes.

After the ablation test: the project has its first positive vision result. The MICOA architecture (freeze + zero-init + forward cone) produced — at the 30K checkpoint — a policy that uses its camera on every step. The challenge is now a second-order one: the camera is being used, but not yet in a directionally correct way.

This is the right kind of progress. It does not close the project. It opens a richer set of questions that were previously inaccessible because the most basic condition (vision being used at all) had not been met.

---

## 2026-05-07 — Stage1 Headfix: Locomotion Baseline Reset

### Why stage1_v8_best was retired

The old stage1_v8_best checkpoint scored 60% deterministic touch rate across 20 episodes and zero falls. Those numbers looked solid on paper. Video review shattered that reading. In 5 rendered episodes the creature was nearly stationary on 4 of them — seed 3 touched because the ball spawned adjacent to the creature, not because the creature moved toward it. Additionally, the head oscillated across its full angular range every few timesteps, a behavior produced by the kp=30 servo gain snapping to any commanded position in a single step. The visual signal from a head moving like that is a chaotically uncorrelated sequence of frames that no downstream network could extract stable structure from.

The theoretical implication is blunt: every MICOA vision experiment that followed stage1_v8_best was built on a foundation that didn't locomote and produced an unusable visual signal. The 60% touch rate was a statistical artifact of spawn radius geometry. The serial approach — train blind proprio first, then freeze and add vision — inherits the blind policy's behavior. If the blind policy learned "stay still and let the ball come to you," no follow-on architecture can rescue that.

### What the new run's numbers say about locomotion quality

The stage1_headfix_velbonus_2026_05_07 run shows three improvements over the old baseline that go beyond the raw touch rate.

First, the training trajectory was stable. The previous stage1 runs showed early peaks followed by catastrophic collapse. This run sustained rewards in the 90–135 range for most of 200K–500K steps with a peak of 176.9, suggesting the creature is not oscillating between good and bad local optima in the way that plagued prior runs.

Second, mean_dist_mean of 0.417 and ep_len_mean of 151 steps indicate the creature is covering ground. The old stage1_v8_best did not report mean_dist_mean — this is the first run in the project where we have positive evidence of ground coverage.

Third, the fast contact at seed 3 (step 39) would be nearly impossible for a stationary creature. The ball would need to spawn within a very small radius for step-39 contact to occur without movement. This is the strongest single datum suggesting real locomotion has emerged.

The 45% deterministic touch rate falls short of the 60% old baseline by raw number, but the old baseline is now understood to be invalid. Whether 45% from a moving creature is sufficient to resume vision work requires video confirmation.

### Is 45% "real" locomotion or still luck?

This is the open question the project currently sits on. There are two distinct scenarios that are both consistent with 45% touches and the numbers reported.

Scenario one: the velocity bonus broke the stillness optimum and the creature now actively paddles toward likely ball positions. Under this scenario, the 45% reflects a learned spatial strategy — the creature knows, through proprio feedback and reward shaping, that forward movement leads to contact, and it moves forward reliably. Seed 3's step-39 contact is the strongest evidence for this scenario.

Scenario two: the velocity bonus created movement but not steering. The creature now moves in one direction consistently (possibly forward, possibly randomly varying), and 45% of ball spawn positions happen to lie within its path. Under this scenario, 45% is still spawn-luck, just from a moving rather than stationary creature. The right-side rollout touch fraction of 1.0 versus 0.569 left-side fraction is a flag for this scenario — a truly locomotion-competent creature should not have this asymmetry unless there is a systematic structural reason for it.

Video review is the discriminating test. A creature executing scenario one will visibly move toward the ball. A creature executing scenario two will move in a direction and either contact the ball if it is in the way or time out if it is not, regardless of where the ball is.

### What this means for the MICOA roadmap

The MICOA roadmap, as established in earlier entries, specifies that vision follow-on cannot resume until the blind proprio policy shows, on video: active ground coverage, calm head motion, and touch rate that reflects movement rather than spawning.

This run is a necessary reset, not a sufficient one. The physics fixes (kp=5, velocity bonus) address the root causes of the two failures identified in video review. The training numbers are encouraging. But the decision gate is the video, not the numbers.

Until a human reviews the 5 rendered episodes at `alien_baby/results/videos/v8_sanity_stage1_blind_trained_headfix_velbonus_best_seed{0-4}.mp4` and confirms calm head motion and active locomotion, the MICOA roadmap is on hold. If the videos confirm the behavioral prerequisites, the next step is a joint dialogue-architecture follow-on from this checkpoint — the two-stream (proprio + vision, equal peers, explicit agreement loss) design proposed after the video-review direction change earlier today. If the videos reveal persistent stillness or head thrashing, the physics parameters need further adjustment before any vision work begins.

### The theoretical bottleneck this run addresses

All previous MICOA vision experiments failed because the substrate they were building on was broken. The freeze held, the consistency loss fired, the ablation showed sensitivity — but the creature the vision system was trying to steer couldn't reliably move. The theoretical claim "vision must confirm proprio" requires that proprio has something worth confirming. A stationary creature's proprio signal ("I am not moving") is technically valid, but it is not the kind of directional locomotion signal that vision can usefully extend. For MICOA confirmation to work in the locomotion setting, proprio must encode movement-toward-target, and vision must learn to extend that movement toward targets proprio cannot identify directionally. The headfix and velocity bonus are the prerequisites for that encoding to be possible.

The outstanding theoretical question remains the one posed after the MICOA blind baseline result: can MICOA-aligned confirmation be established when proprio is a body-movement signal rather than an arm-position signal? The tabletop experiments had proprio directly encoding fingertip-to-target distance. v8 proprio cannot know ball position. Vision must provide directional information that proprio cannot confirm. This asymmetry — vision informing, proprio executing, but proprio unable to validate vision's directional estimate — is the distinctive challenge of the locomotion setting that the tabletop setting could not expose.

A locomotion-competent stage1 baseline is the first prerequisite for testing whether that challenge can be solved. This run is the attempt to establish that baseline. Video review will determine whether it succeeded.

---

## 2026-05-07 — Stage1 v2: New Locomotion Baseline Established (stage1_headfix_velbonus2_cone180_600steps_2026_05_07)

### Why 85% is qualitatively different from the old 60%

The old stage1_v8_best benchmark of 60% was achieved with a narrow spawn cone, short episodes, and — as video revealed — a nearly stationary creature whose success depended on the ball spawning within arm's reach. Touching 12 out of 20 balls under those conditions required almost no locomotion. That baseline was declared invalid.

This run's 85% was achieved under three more demanding conditions simultaneously. The spawn cone was 180 degrees, meaning the ball appeared anywhere in the creature's front hemisphere — it was not possible to succeed by sitting still and waiting for the ball to appear nearby. The episode budget was doubled to 600 steps, which means the creature had more time but also faced more demanding ball positions that required sustained travel rather than a brief nudge. The velocity bonus was 2.5 times stronger than in the v1 run, creating a persistent gradient away from the stillness local optimum that plagued every prior stage-1 attempt.

Achieving 85% against that broader challenge requires real locomotion. A creature that paddles in one fixed direction and relies on spawn luck would not achieve 85% across a full 180-degree front hemisphere. The mean episode length of 198 steps — well short of the 600-step ceiling — shows the creature is finding the ball actively rather than timing out. The three timeouts in 20 episodes represent the hard tail of the spawn distribution, not a systematic locomotion failure.

The most important single datum: 5 consecutive new best-checkpoints appeared early in training. This is the signature of a policy that is genuinely learning and compounding its improvement, not oscillating around a lucky initial state. The 85% best-checkpoint result is the peak of a learning curve, not a noise spike.

### What this means for the MICOA roadmap

The MICOA roadmap was put on hold pending video confirmation that the v1 headfix run (45% deterministic) showed active locomotion and a calm head. That decision was correct — the project cannot build a vision architecture on a locomotion substrate it hasn't verified. The v2 run strengthens the case for video confirmation: if the videos show the creature actively searching across the 180-degree cone (including seeds where the ball is far away), the MICOA roadmap prerequisites are met and vision follow-on can resume.

More specifically, the v2 result changes what the MICOA roadmap is protecting. Every prior MICOA experiment — five runs from the old stage1_v8_best — was trying to add vision to a creature that sat still. The theoretical diagnosis was that proprio didn't have anything worth confirming: a creature that doesn't move provides a locomotion signal ("I am not moving") that vision cannot usefully extend into directional steering. This run changes that diagnosis. If the creature is genuinely covering ground and finding balls across the 180-degree front hemisphere, then proprio encodes something valuable: active locomotion, varying speeds, asymmetric paddle strokes that correlate with eventual ball contact. That is a locomotion signal vision can potentially extend. "I am moving at speed X in direction Y, and contact occurs when I'm oriented toward the ball" is a predictable relationship that vision, by adding ball-direction information, could enhance. The old stage1_v8_best offered no such relationship.

The implication is concrete: a follow-on run seeded from this v2 checkpoint, using the MICOA architecture (frozen proprio, zero-init pixel columns, consistency loss), is testing a different and more favorable hypothesis than all prior MICOA runs. The substrate is locomotion-competent. The question is now whether vision can contribute directional steering on top of an already-functional locomotion engine, rather than whether vision can rescue a creature that doesn't move.

### What the prerequisites for vision follow-on now are

Two prerequisites remain before vision follow-on resumes, and neither has been waived.

The first is human video review. A human must watch the rendered episodes and confirm three things: the creature's head moves slowly and deliberately (not oscillating across full range), the creature covers ground during each episode (not stationary), and contacts are the result of movement toward the ball (not spawn adjacency). This is the same decision gate as for the v1 run. The v2 numbers are more encouraging, but numbers have fooled this project before — the 60% stage1_v8_best numbers looked solid until video revealed they were meaningless.

The second is the directional asymmetry question. The v2 rollout shows touched_left = 0.549 and touched_right = 0.347 — the creature makes about 58% more left contacts than right contacts during stochastic training. This asymmetry could mean the creature has a preferred direction (paddles more easily to the left), or it could be a spawn artifact from the 180-degree cone distribution if the ball spawning is not perfectly symmetric. It does not invalidate the 85% result, but it is worth noting for video review: if the creature is strongly left-biased in its movement, the 85% touch rate under cone180 may be partly explained by the ball frequently spawning on the left side. A video-confirmed right-side contact would be reassuring.

These prerequisites exist not to delay the project but to protect it from repeating the mistake of the MICOA experiments built on stage1_v8_best. One bad foundation cost the project the entire MICOA experimental series. Verifying the foundation before building is cheap insurance.

### What is ruled in and ruled out by this result

This run rules in one thing clearly: the combined parameter set (kp=5 head servo, VELOCITY_BONUS_SCALE=0.05, cone180, 600 steps) produces a locomotion policy that achieves 85% deterministic touch rate with zero falls across 20 episodes. That combination works for stage-1 locomotion training.

It rules out the hypothesis that the velocity bonus at 0.02 was sufficient to fully break the stillness local optimum at scale. The v1 run achieved 45%, which was an improvement but not a breakthrough. The 2.5x increase to 0.05 moved the needle from 45% to 85% — a nonlinear improvement that suggests the 0.02 bonus was pushing in the right direction but not strongly enough to consistently dominate the stillness incentive across all spawn positions.

It does not rule out or confirm anything about vision. This is a blind proprio run. The vision question is entirely deferred to the follow-on.

### Framework assessment for this run

**Behavioral Prediction Framework:** The 85% rate under cone180 is consistent with the framework's prediction that a creature develops internal models of where contact is likely to occur and moves to produce that contact. A 180-degree spawn distribution forces the creature to either have a generalizable search strategy or fail on the half of spawns that land outside its default direction. The fact that only 3/20 timed out suggests the creature is doing something more than pointing straight ahead and hoping. However, confirming this interpretation requires seeing the creature turn toward off-center balls — which only video can show.

**Pattern Learning Framework:** The 5 consecutive new bests in early training are consistent with the framework's prediction that the agent learns stable distributed patterns that generalize: each improvement represents a more robust locomotion pattern that succeeds across more ball positions, not a one-off win. The consistent 0 falls across all 20 deterministic episodes reinforces this — a fragile or noisy policy would produce occasional tipping, but none occurred. The pattern for "locomote stably" appears to have solidified fully.

### The outstanding theoretical question this baseline opens

The project's central unresolved question — whether MICOA-aligned confirmation can work when proprio is a body-movement signal rather than an arm-position signal — has not been answered by this run. It has been made testable. For the first time, the substrate has the locomotion competence that the question requires. Whether proprio's movement encoding is rich enough for vision to extend into directional steering is the question the follow-on will test.

The specific form of that question, given this v2 result: the creature already achieves 85% blind. Vision follow-on must achieve more than 85% — or achieve 85% with confirmed nonzero vision-ablation sensitivity — to represent a genuine improvement. A follow-on that achieves 85% through proprio alone (vision ignored) is indistinguishable from the blind baseline by touch rate alone. The ablation test remains the decisive measurement.

---

## 2026-05-11 — Overnight session: No-bribery substrate, dialogue arch, fixed-ball test

### What changed since the last theory entry

Three new experiments tonight, all on the MIMo crawler, all designed to test where the previous failure modes were located in the dependency graph from body → motion → contact → memory → spatial structure → vision. The summary of what was tested and what we learned, in MICOA / pressure-not-bribery terms:

**Phase A** (mimo_substrate_A, 250K): no-bribery substrate (no velocity bonus, no approach reward) + 360° spawn + 2000-step episodes + guardrails + 15° camera tilt + existing flat-concatenation StereoCrawlerCNN architecture. The question being tested: does removing bribery and broadening the spawn distribution make vision become load-bearing without changing the architecture?

**Phase B** (mimo_dialogue_v1, killed 112K): same substrate as Phase A but with the dialogue architecture from the May 7 retirement memo — two-stream actor (proprio + vision), learned scalar gate w, consistency loss λ · ||μ_p − μ_v||² with λ=0.05. The question: does the structural change to a symmetric peer architecture with explicit agreement pressure produce MICOA-aligned integration?

**Phase C / C2** (mimo_phase_c, killed 56K each): no bribery, but two fixed ball positions at (0.7, 0.0) and (0.0, 0.7), random starting orientation per episode, blind proprio. C2 also added memory_obs: two binary flags (touched_ball1, touched_ball2) appended to proprio so the stateless SAC can condition on its own past contacts. The question (proposed by the human researcher mid-session): does stable spatial structure — a world the agent can build a model of, rather than a fresh random problem each episode — produce learnability that random spawn doesn't?

### Numerical reality

All three experiments produced essentially the same diagnostic signature: the deterministic eval mean reward equaled −100.00 with standard deviation 0.00 (or very close), meaning every one of 20 evaluation episodes returned exactly the no-action baseline (2000 steps × −0.05 step cost). Stochastic rollouts showed some contacts (ep_rew_mean around −20 to −70, corresponding to ~10–30% one-ball touch rate), but those contacts disappeared the moment exploration noise was removed. The deterministic policy mean was approximately zero, the creature did not move in eval, and no policy variant produced a learnable improvement.

Phase B's gate trajectory is the most informative new datum: w collapsed from random init (~0.5) to ≈ 0.005 within 40K steps, then slowly recovered to 0.07 by the 112K kill. Disagreement between the streams grew from 0.09 to 0.42 over the same window. Consistency_loss grew from 0.01 to 0.31. This is direct refutation of MICOA convergence under the dialogue-architecture-as-implemented: the streams diverged rather than agreeing, and the gate excluded one stream rather than blending them. The architecture is structurally correct (two streams, explicit gate, agreement objective) but the implementation has an asymmetric gradient flow — with gate ≈ 0, the SAC actor loss flows almost entirely to μ_v, while μ_p receives only the (smaller) consistency-loss gradient. μ_v drifts toward whatever Q-maximizes; μ_p is pulled along by consistency but cannot keep up; the gate sees the resulting disagreement and further suppresses the lagging stream. A self-reinforcing collapse.

### The pressure-not-bribery line, sharpened

This session forced a more precise articulation of the pressure-not-bribery principle than the project has had before. The human researcher articulated the distinction explicitly: enabling a capacity (the body being able to move, the camera being aimed correctly, memory architecture for storing past events) is substrate, not bribery. Paying for a specific behavior (move forward gets +reward, look at ball gets +reward, approach ball gets +reward) is bribery. The two are categorically different even though both involve the experimenter adjusting things.

Under this articulation, the runs we did tonight tested whether the substrate alone (without any bribery) is sufficient. The answer is unambiguously no for SAC + MIMo crawler with sparse contact reward: the agent never reaches a state where the substrate features could pay off, because SAC's deterministic policy collapses to zero-action when the only reward signal is sparse contact that the policy hasn't yet figured out how to produce reliably. The contact reward isn't dense enough to bootstrap locomotion, and without locomotion, the agent never produces the contacts that would teach it locomotion. The chicken-and-egg.

The earlier 85% blind baseline result avoided this trap by including a velocity_bonus_scale=0.05. That bonus, in the new framing, is on the bribery side of the line — it pays for a specific behavior (any motion). It is what kept the policy out of the zero-action attractor. Without it, the no-bribery setup loses the locomotor base entirely.

The cleaner re-articulation, then: a small velocity bonus is not bribery in the same sense an approach reward is. An approach reward pays for movement *toward a specific target* — it encodes the answer to the question we are asking the agent to learn. A velocity bonus pays only for *not being still* — it tells the agent that the zero-action attractor is suboptimal but does not say where to go. Under this reading, velocity_bonus belongs to the "enabling" category (closer to providing energy to a body that needs to move) rather than the "bribery" category (closer to giving the agent the answer). This is the line of reasoning that the next experiment (Phase D, not yet run) is designed to follow up.

### The two-balls-define-a-line hypothesis

The human researcher's most important contribution to the theoretical framing this session was the observation that random ball spawn destroys learnability in a deeper sense than just making the task hard. With random spawn, the spatial structure of the world is *re-randomized* every episode, so nothing the agent encounters about the world's structure can transfer between episodes. The agent gets contact rewards but those rewards do not aggregate into a model of *the world* — they aggregate into a stimulus-response habit ("twitch and sometimes you bump into something"). With fixed ball positions, by contrast, there is a stable spatial structure that the agent could in principle build a representation of. After many episodes the agent could learn that contact is achievable at specific world-coordinate locations, and the policy can be conditioned on its own position estimate to navigate to them.

The two-ball variant adds the further insight that two stable landmarks define a coordinate system. Once the agent has touched both balls in the same episode, it has implicitly identified a line segment in the world, with its own (path-integrated) position registered against that line. Future episodes can then use this learned reference frame to navigate efficiently: touch one ball to register, then beeline to the other.

This is a real cognitive proposal. It maps onto place cells (hippocampal cells that fire at specific world positions), grid cells (cells that fire on a hexagonal grid relative to learned landmarks), and the Numenta thousand-brains framework's emphasis on reference-frame learning. The proposal is that the right experimental substrate for testing reference-frame development is not "random task, see if it generalizes" but "stable world, see if a map forms."

The empirical result (Phase C, Phase C2) is that the proposal is not refuted but is also not yet testable in the current setup. The agent never reaches the state of reliably touching even one ball under no-bribery conditions, so it never gets the data points that would seed a map. Fixed positions and memory flags are downstream-of-locomotion features. They cannot pay off in an agent that does not move.

### Updated dependency graph

The session has clarified the project's effective dependency graph in a way that earlier failures had only hinted at:

```
body works (physics, mass, joints)
        ↓
can move (a policy whose mean produces locomotion)
        ↓
can encounter things (stochastic motion produces contact)
        ↓
can remember encounters (within-episode memory, weight memory)
        ↓
can build a spatial map (stable world structure + memory + locomotion)
        ↓
can use vision distally (CNN learns object/location associations)
        ↓
can integrate vision with proprio (MICOA confirmation)
```

The new finding tonight is that **the chain breaks at link 2 under sparse-contact-only reward**. Every previous failure mode the project has documented (entropy collapse, gate collapse, consistency loss diverging, vision silent) is downstream of this break. You cannot test gate behavior in a policy that does not move. You cannot test memory in a policy that does not encounter. You cannot test vision integration in a creature that does not produce visual evidence.

This is not a methodological criticism of the prior runs — those runs included velocity_bonus and so kept the chain intact at link 2. The criticism is that the prior runs *also* included a behavior-specific component (approach reward, FOV reward, velocity bonus tuned to 0.05) and we could not tell which components were necessary for the locomotor base and which were unnecessary bribery for the downstream question. Tonight's runs separate those concerns: with all of them removed, the chain breaks at link 2. The implication is that *some* enabling signal at link 2 is necessary, and the design question becomes how to provide it without contaminating the test of links 5–7.

### Updated "what is ruled out" / "what is not yet ruled out"

| Hypothesis | Status before tonight | Status after tonight |
|---|---|---|
| No-bribery substrate alone makes vision load-bearing under flat-concat architecture | Open | **Refuted** (Phase A: 0% deterministic, ablation 0.0226) |
| Dialogue architecture (gate + consistency loss) escapes flat-concat's failure modes on the same substrate | Open | **Refuted under the as-implemented λ=0.05 with no gate floor** (Phase B gate collapse) |
| Random ball spawn is the load-bearing failure (rather than the architecture) | Live | **Not testable in current configuration**; the upstream locomotion failure preempts the test (Phase C, C2) |
| Memory of past contacts solves the two-ball task | Live | **Not testable**; same upstream issue (Phase C2) |
| Some non-zero velocity bonus is required to maintain the locomotor base under SAC + sparse contact reward | Implicit | **Confirmed by absence** (every run with velocity_bonus_scale=0.0 collapsed at the policy mean) |

| Hypothesis | Next test |
|---|---|
| velocity_bonus_scale=0.02 (small, enabling) + fixed balls + memory + random orientation produces learnable behavior | Phase D — same env as C2 with velocity_bonus_scale=0.02 |
| Once locomotion is stable, the dialogue architecture with a gate-floor of 0.2 (prevents collapse) produces MICOA-aligned integration | Phase E — dialogue arch + gate floor + locomotor-enabling substrate |
| A recurrent (LSTM) policy on the Phase D substrate develops the two-ball spatial map | Phase F — LSTM-augmented SAC or PPO + LSTM on Phase D's setup |

### Frameworks: what each one says after tonight

**Behavioral Prediction Framework** (the creature should build internal models that let it act in advance of contact): the deterministic-zero-action collapse across A, C, C2 is the strongest violation the project has seen. The framework predicts the agent should at least produce *some* motion in its mean policy, even if that motion is poorly directed. The observed result is that the mean policy produces *no* motion. This is a stronger failure than the framework anticipates, and it tells us the framework's prediction is conditional on a locomotor base existing. With no base, there is no policy mean to be predictive — there is only exploration noise. The framework needs to be re-stated to account for the prerequisite: an agent with a working motor system that produces consequences will build internal models of those consequences. If the motor system is not working, no models will form.

**Pattern Learning Framework** (the agent's internal representations should respond to recurring features of the environment with stable distributed patterns): tonight's runs included environments with extreme regularity (fixed ball positions, fixed platform, fixed guardrails) and the agent still did not develop stable patterns that could be used for control. This is informative because it means the environmental regularity is not sufficient — the agent must also be in a state where regularity *produces consequences* that can be learned from. With no locomotion, the agent does not produce the consequences that would let the pattern learning occur. Same prerequisite issue as the behavioral prediction framework.

**MICOA** (multiple inputs confirming one another, not competing): Phase B's gate-collapse result is direct evidence that the implementation of MICOA-style architectures must contend with gradient-flow asymmetry. The clean theoretical statement ("two channels that must agree") becomes, in practice, "two channels with a gating mechanism that creates a feedback loop where the channel with smaller actions gets weighted more, which starves the other channel, which then diverges, which the gate then suppresses further." The next implementation should include either a gate floor or a different weighting scheme entirely (e.g., simple averaging with stop-gradient on the gate during early training) to break this feedback loop.

### The central question, restated

Before tonight, the central question was: "Can MICOA-aligned vision integration be achieved on a locomotion body where proprio cannot directly encode target direction?" This question presumed a working locomotor base.

After tonight, the more precise central question is: **"What is the minimum enabling signal that maintains the locomotor base without becoming behavior-specific bribery, so that the substrate-level tests (fixed structure, memory, vision integration) can be conducted on top of it?"** The principled answer (a small velocity bonus, justified as energy-cost rather than direction-specific shaping) is the design hypothesis for the next experimental cycle. If that hypothesis is right, the project regains its ability to test the substrate questions on a foundation that doesn't collapse. If it is wrong — if even a small velocity bonus still corrupts the test — then the project needs to consider whether SAC + MIMo + sparse-contact-reward is fundamentally the wrong combination for this experimental program, and whether a different algorithm (PPO with HER, or a model-based learner like Dreamer) is required.

The session does not close the project. It clarifies the dependency graph and re-locates the open question.

