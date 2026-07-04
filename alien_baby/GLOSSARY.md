# xGlossary

Terms that show up in FINDINGS.md, the training scripts, and our conversations. Kept plain-English; each entry aims for "what is it, why does it matter here."

---

## Reinforcement learning (the algorithm side)

**SAC (Soft Actor-Critic).** The learning algorithm we use. It's a modern reinforcement-learning method for problems with continuous actions (like joint torques). "Soft" because it rewards the agent for keeping some randomness in its behavior — it discourages premature lock-in to a single strategy. It's the standard workhorse for robotic-arm tasks in simulation.

**Actor.** The neural network that chooses actions. Input: the observation (proprio + pixels). Output: a distribution over actions. In our case it's a small MLP: obs → 256 → 256 → action.

**Critic.** A second neural network that estimates "how good was that action in that state?" — a scalar value. SAC actually uses two critics and takes the minimum (a trick to stop the agent from over-trusting its own value estimates). The actor trains itself to pick actions the critic rates highly.

**Policy.** The rule that maps observations to actions — i.e., the creature's moment-to-moment behavior strategy. In our code the policy *is* the actor network: feed in the observation vector (proprio + pixels), out comes a distribution over the action vector (joint torques, head commands). "Running the policy" = executing that mapping step-by-step through an episode. SAC's whole job is to shape the policy over training so that actions it picks lead to high reward. When we talk about "vision-ablation sensitivity" (how much the action changes when pixels are zeroed), we are asking: how much does the policy depend on vision?

**Episode.** One trial: reset the environment, let the agent act for up to 200 timesteps or until it touches the target (whichever comes first). Our task gives +10 reward for touching the target.

**Timestep / step.** One full decision cycle of the RL loop: (1) agent sees the observation, (2) picks an action, (3) env advances the physics, (4) returns the new observation + reward + done flag. In our env each step packs 5 internal MuJoCo physics substeps of 0.01 sim-seconds, so one step = **50 ms of simulated time**. The agent only *decides* every 50 ms, but physics is computed every 10 ms.

So a 400-step stage-0 episode = 20 simulated seconds of creature-time. 200,000 steps of training ≈ 2.8 sim-hours of creature experience (~8 min wall-clock at ~400 FPS).

Worth distinguishing:
- **Step** ≠ **sim-timestep** — a sim-timestep is 0.01 s of physics; a step bundles 5 of those.
- **Step** ≠ **episode** — an episode is a full reset-to-termination run (up to 400 steps now in stage 0).
- **Step** ≠ **video frame** — videos render one frame per step, so timelines happen to align, but they're different things.

**Rollout.** A sequence of timesteps, usually a full episode or a batch of episodes. "Roll out the policy" = run it in the environment and record what happens.

**Replay buffer.** A rolling cache of the last 50,000 transitions `(state, action, reward, next_state)` the agent has experienced. SAC learns *off-policy*: instead of only learning from the latest episode, it keeps past experience around and re-learns from random samples. This is why the agent can keep improving long after its behavior has changed.

**SAC update batch (or "gradient step").** Once per environment timestep, SAC pulls 256 random transitions from the replay buffer and does one gradient update on the actor and critic using that batch. "Batch" here means "the 256 samples used for one update step" — not a batch of episodes.

**Gradient step.** One application of backpropagation + optimizer step. Our runs do one gradient step per environment step, so 200k environment steps ≈ 200k gradient steps.

**Checkpoint.** A saved copy of the trained model (weights + optimizer state). We can load it back later and keep training or just evaluate it. Our checkpoints live under `alien_baby/results/` (e.g., `followon_v5_checkpoint.zip`).

**Deterministic vs stochastic evaluation.** During training the actor samples actions from a distribution (stochastic) to explore. During evaluation we use `deterministic=True`, which just takes the distribution's mean — a stable, repeatable behavior.

**Entropy coefficient (ent_coef, sometimes α or "temperature").** SAC's knob for how much randomness it encourages in the actor. SAC tunes it automatically during training. You'll see it logged as `ent_coef` and `ent_coef_loss`.

**Polyak update / target network.** A stability trick. The critic has a "shadow" copy (the target) whose weights slowly drift toward the main critic's (`τ = 0.005` per update). The main critic trains against the slowly-updating target, preventing runaway feedback loops. Not something you need to interact with — it's just internal machinery.

---

## Neural-network plumbing

**MLP (Multi-Layer Perceptron).** A plain feed-forward neural network: a stack of linear layers with non-linearities between them. Our actor is `Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, action_dim)`. No attention, no recurrence — just matrix multiplies and ReLUs. Contrast with Transformer/LLM architectures, which are much larger and use attention.

**Linear layer.** `y = W·x + b` — a matrix multiply plus a bias vector. Our first layer's weight `W` has shape (256, obs_dim); it's the part we talk about when we say "proprio columns" (the first 7) vs "pixel columns" (the remaining 12,288).

**ReLU.** `max(0, x)` applied element-wise. The most common non-linearity. Without it, stacking linear layers would collapse to a single linear map.

**Activation / hidden activation.** The intermediate vector produced by a layer. `hidden1` in our code = the output of the first Linear+ReLU of the actor. That 256-dim vector is what we probe for CKA and neighbor-consistency — it's the network's internal "picture" of the state.

**Feature extractor.** For our observations it's the identity (just flattens the obs into a vector). More complex setups use CNNs for images; we skipped that for simplicity.

**Encoder (in our context).** The part of the network that turns raw sensory input into the compact internal latent the policy actually reads. We have two, one per modality (see MICOAExtractor): the *proprio encoder* — a 3-layer MLP that maps the 7 proprio numbers to a 64-dim Gaussian `(μ, σ)` over the shared latent `Z` — and the *vision encoder* — a DrQ-v2 4-conv CNN that maps the stereo camera image to its own 64-dim `(μ, σ)`. "The visual latent" we keep probing *is* the output of the vision encoder. So when we say "representation failure" or "the encoder never encoded ball direction," we mean: the vision encoder's output doesn't carry that information for any downstream reader to use. The encoder is the fix target whenever the problem is representational rather than a policy-gradient problem.

**Decoder (in our context).** A readout that runs the other direction — from the internal latent back out to some interpretable quantity. We don't keep a decoder as a permanent part of the agent; we attach them in two roles. (1) As a *probe* — a read-only diagnostic: the ball-x linear probe is a decoder (vision latent → ball lateral position), and its R² ≈ 0.08 is how we measured representation failure. (2) As an *auxiliary loss* — a trainable decoder whose error is pushed *back into* the encoder, forcing the encoder to put the target into the latent in the first place (see *auxiliary decode loss*). Read-only decoder = measures the failure; auxiliary-loss decoder = tries to fix it.

**Weight / parameter.** A single learnable number inside the network. The actor has ~3.2M weights total; we freeze most of them and only let a small subset (the pixel-column part of the first layer) train.

**Freezing weights.** Setting `requires_grad = False` so the optimizer won't update them. In v2/v3/v5 we freeze all of the actor's weights except the first-layer pixel columns — proprio's behavior is mathematically preserved.

**Gradient hook / gradient mask.** A PyTorch feature that intercepts gradients as they flow back through a tensor. We use it to zero out the gradient on the proprio columns of the first-layer weight — so even though that tensor is marked trainable, the proprio slice effectively isn't. Belt-and-suspenders on top of freezing.

**Backprop / gradient descent.** The training loop: run the network forward, compute a loss (how wrong the output was), compute the gradient of the loss w.r.t. each weight, nudge each weight a tiny bit in the direction that reduces the loss. Repeat.

**Learning rate.** The size of the nudge. We use 1e-4 (0.0001) for v2–v5.

**Loss function.** A single scalar that measures how badly the model is doing on a batch. The optimizer minimizes it. SAC's actor loss has two parts (task performance and entropy); v5 adds a third (consistency). MSE — mean squared error — is the simplest loss: average of `(prediction − target)²`.

**KL divergence (Kullback-Leibler).** A number measuring how different two probability distributions P and Q are, written `KL(P || Q)`. Zero when P = Q; grows as P diverges from Q. For two Gaussians it has a clean closed-form formula in terms of their means and variances, which makes it cheap to compute and differentiable — you can use it directly as a loss term.

Crucially, KL is **asymmetric**: `KL(P || Q) ≠ KL(Q || P)` in general. The asymmetry matters. `KL(P || Q)` heavily penalizes P putting probability mass anywhere Q assigns near-zero probability — informally, "P should be careful never to claim things Q rules out." The direction you write the KL in expresses *which* distribution is the reference. When neither side should be treated as ground truth, you average both directions and call it **symmetric KL**.

In Phase IX MICOA we use symmetric KL between the proprio and vision encoder distributions (neither is ground truth, both are penalized for disagreement). In Phase X we use the asymmetric form `KL(vision(t) || proprio(t+1))` with proprio detached — vision is pulled toward proprio's future, but proprio is unaffected by the loss (proprio is the ground truth for what vision should be predicting).

---

## Representation-analysis tools

**R² ("R-squared").** The coefficient of determination. A goodness-of-fit number for regression, in range [−∞, 1]. R² = 1 means the regressor predicts targets perfectly; R² = 0 means it does no better than guessing the mean; negative R² means it's worse than guessing the mean. When we said "vision's target-localization accuracy capped at R² = 0.35," we meant: a linear readout trained to decode target position from the network's hidden activations explained only 35% of the variance in target position.

**Linear probe.** A simple trick for asking "does this internal representation contain information X?" You freeze the main model, train a *linear* readout (one matrix multiply) to predict X from the model's hidden activations, and measure how well it works. If a linear probe gets high R², the info is there and easily extractable. Low R² could mean info is absent — or just tangled up in a non-linear way.

**Ridge regression.** The specific linear-probe regressor we use (`sklearn.linear_model.Ridge`). It's linear regression plus a small penalty on large weights, which prevents overfitting. Nothing exotic.

**Representation failure vs. policy failure.** The two ways a vision system can "be active but useless," and the fork the ball-x decode probe was built to resolve. *Representation failure:* the visual latent never encoded the task-relevant variable (e.g. ball direction) in the first place — a linear probe reading the latent can't decode it above chance. If so, no amount of fixing the RL update helps, because there's nothing in the latent for the policy to use; the fix target is the encoder / the representation-learning objective. *Policy failure:* the latent *does* encode the variable (probe R² is meaningful) but the policy gradient never learned to act on it — the fix target is the RL update / actor, not the encoder. Phase XIII's probe came back **representation failure**: R43's vision latent decodes ball lateral direction at R² ≈ 0.08, below even the proprio negative control, so vision's high ablation reflects binding to non-directional features (ball presence, lighting, self-motion), not a usable spatial code. Why it matters: this is the first time the project separated "vision *affects behavior*" (ablation, high) from "vision *encodes the task variable*" (probe, ~chance) — ablation alone is not evidence of a useful representation.

**CKA (Centered Kernel Alignment).** A similarity measure between two sets of activations from two different networks on the same inputs. CKA = 1 means "these two networks represent the input in essentially the same way (up to a linear reshuffling)." CKA = 0 means "totally unrelated representations." We use linear CKA; the math is `‖Xᵀ Y‖² / (‖Xᵀ X‖ · ‖Yᵀ Y‖)`. Why it matters here: if v5's hidden1 has high CKA with stage1_v3's hidden1, vision is living on proprio's manifold — *additive* interpenetration. Low CKA = vision has carved out its own manifold and ignored proprio's.

**k-NN (k-nearest neighbors).** For each point, find the k closest other points (we used k=5, closeness measured in hidden1 space by Euclidean distance). Used here as a probe: if two observations are neighbors in hidden1 space, are their *actions* also similar? If yes, we have an equivalence class (different inputs → same action → same perception, in the theory's sense).

**Neighbor-consistency ratio.** Our equivalence-class metric: `mean(action-distance among hidden1 neighbors) / mean(action-distance among random pairs)`. Lower = tighter equivalence classes. Stage1_v3 came out at 0.07 (neighbors' actions are 14× more similar than random); v1 agents were ~0.45 (weak class structure).

**k-means.** An unsupervised clustering algorithm. Give it N points and a k; it finds k "center" points such that each observation ends up assigned to its nearest center. We use it to carve hidden1 activations into discrete Π-classes we can inspect.

**Silhouette score.** A measure of how clean a clustering is: high = points are much closer to their own cluster's center than to other centers; low = clusters bleed into each other. Range [−1, 1].

**Spearman correlation (distance correlation).** Correlation between the *ranks* of two variables (not the raw values). We use it to ask "when two observations are close in hidden1 space, are their actions also close?" Spearman is robust to outliers and non-linear monotonic relationships.

**MSE (mean squared error).** Average of squared differences. Our v5 consistency loss is `MSE(h_full, h_blind)` — push the two hidden1 vectors to be numerically close. Same quantity R² implicitly measures.

---

## MICOA (Multiple Inputs Confirming One Another)

The architecture we built in Phase IX and Phase X to address the problem that vision was never load-bearing across 7 prior experiments (Phase VII/VIII). The diagnosis was structural: concatenating proprio and pixels into one flat vector for SAC is "welding two tubes end-to-end" — it makes a longer rod, not a box. SAC just follows whichever gradient is easier (always proprio), and the pixel encoder gets no useful training signal. To make a box, the architecture needs a component whose job is to *detect* when two independent streams are constraining the same world-state simultaneously. That detector is what MICOA adds.

**Product of Experts (PoE).** The parameter-free math at the heart of MICOA. Each modality's encoder outputs a Gaussian distribution `(μ, σ)` over a shared 64-dim latent space `Z` instead of a single point. The PoE fuses them with Bayesian precision-weighted averaging:
```
μ_combined  = (μ_p/σ_p² + μ_v/σ_v²) / (1/σ_p² + 1/σ_v²)
σ_combined² = 1 / (1/σ_p² + 1/σ_v²)
```
When both encoders point at the same region of Z, the precisions add and σ_combined shrinks — that tightening is the "corner" forming. When they disagree, σ_combined stays large and the system represents its own uncertainty. The PoE has no learned weights; the confirmation emerges from the mathematics of multiplying independent Gaussians.

**MICOAExtractor.** Our SB3-compatible feature extractor that implements PoE. Each modality has its own encoder (a 3-layer MLP for proprio, a DrQ-v2 4-conv CNN for stereo pixels), each outputs `(μ, σ)`. PoE fuses them. The downstream policy sees `[z_combined | μ_p | μ_v]` (3 × 64 = 192 dims) so the actor can detect "all three columns are similar" → high confidence vs. "columns diverge" → caution.

**MICOASAC.** A SAC subclass that adds the MICOA agreement loss(es) on top of SAC's normal actor/critic gradients. Each gradient step, after SAC's own backward pass, MICOASAC samples a fresh batch from the replay buffer and runs an extra forward + backward over the encoder's parameters with `_micoa_opt` (a separate Adam optimizer over the extractor only). The SAC training loop itself is untouched — the auxiliary loss only shapes the encoders.

**Symmetric KL agreement loss (Phase IX).** The first MICOA loss term, weighted by `--micoa-beta`:
```
loss_sym = β_sym * 0.5 * (KL(p || v) + KL(v || p))
```
where `p = N(μ_p, σ_p)` and `v = N(μ_v, σ_v)` come from the same observation at the same timestep. Symmetric so neither encoder is treated as ground truth — both are equally penalized for outputting different distributions. This is "confirmation" — the two encoders agree about *now*. **Result across R36/R37: vision learned to mirror proprio's encoding (cheapest way to satisfy the loss) and remained inert in behavior.** This is the "confirmation without anticipation" failure mode.

**Temporal predictive KL loss (Phase X).** The second MICOA loss term, weighted by `--micoa-pred-beta`:
```
loss_pred = β_pred * KL(N(μ_v(t), σ_v(t)) || N(μ_p(t+1), σ_p(t+1)).detach())
```
Vision at time `t` is pulled toward the distribution proprio will encode at `t+1`. Asymmetric and one-sided: the `.detach()` on proprio's future means only the vision encoder is updated by this loss. Vision is the predictor; proprio is the ground truth. This is "anticipation" — vision must learn to see *what proprio is about to feel*. Mirroring proprio(t) won't satisfy this loss because in general proprio(t) ≠ proprio(t+1); vision has to actually predict the change. Whether this gets vision over the load-bearing threshold is what R38 tests.

**The "corner" vs "tube" framing.** A flat MLP over `[proprio | pixels]` builds a longer tube. Two encoders + PoE builds a corner: σ_combined shrinks precisely when both encoders are constraining the same Z, which is what makes the corner rigid. The empirical question is whether SAC's task gradient is strong enough — combined with MICOA's KL pressure — to make the corner load-bearing for behavior, not just present as a representation. Phase IX result: corner forms, behavior doesn't change. Phase X is the test of whether temporal asymmetry breaks that pattern.

**σ_combined and kl_agreement.** The two diagnostic scalars MICOA logs every 500 steps (look for `[MICOA]` lines in the training log). σ_combined falling = corner is forming. kl_agreement near zero = encoder distributions overlap. We learned to read these *together*: σ_combined falling while kl_agreement stays moderate-and-falling = healthy corner formation. σ_combined falling while kl_agreement crashes to zero in the first 4K steps = forced collapse (β too high; what R36 did).

**Ablation delta (the real success metric).** How much the agent's action changes when we zero all pixel inputs at inference time. Measured as the L2 norm of `action(full_obs) - action(blind_obs)`. The pre-approved success threshold for Phase IX/X is `> 0.05`. Across 9 prior runs (Phase VII/VIII, R36, R37) the number sits around 0.001–0.002 — vision has been silently inert. Whether MICOA + temporal prediction can push this above 0.05 is the question that breaks the pattern.

---

## The simulation stack

**MuJoCo.** A fast, accurate rigid-body physics simulator. Handles the arm, the objects, contact forces, gravity. Bought by DeepMind in 2021 and open-sourced.

**Center of mass (CoM).** The average position of all mass in a body, weighted by mass. A tall narrow object with mass up top (a standing pencil) has a *high* CoM and tips easily: small disturbances tilt it far enough that gravity acts *outside* the base of support and pulls it over. A squat object with mass near the ground (a bowling pin butt) has a *low* CoM and stays put. Relevant here because AB's chassis stability is essentially a CoM-vs-wheelbase geometry problem: tipping is stable iff the CoM ends up outside the wheel contact polygon.

**Gymnasium.** The standard Python interface for RL environments (`env.reset()`, `env.step(action)`, etc.). Fork of OpenAI's old `gym`. Our `TabletopReachEnv` inherits from `gym.Env`.

**Stable-Baselines3 (SB3).** A Python library of clean, tested RL algorithm implementations (including SAC). We use it instead of writing SAC from scratch.

**FPS (in training logs).** Environment steps per second — how many simulate-one-timestep-and-get-a-result cycles the computer completes per second of real wall-clock time. Each "step" is: take an action → run the physics forward → get the new observation and reward. So 200 FPS means the creature experiences 200 moments of simulated time per real-world second. This is *not* video frames — nobody is watching. It's the speed of the training loop. Higher FPS = faster training. Vision slows it down (rendering pixels costs CPU time): Stage 1 (proprio only) ran at ~5000 FPS; Stage 2 (with vision) runs at ~200 FPS.

**Proprioception (proprio).** The sense of your own body's position — where your joints are, how fast they're moving, whether something is touching you. In our env: 3 joint positions + 3 joint velocities + 1 touch bit (+ 3 rel-target-pos in the non-blind variants). Distinct from *vision* (the camera image) and *exteroception* more broadly.

**hide_target_offset.** A flag on our env. When True, proprio drops the 3-dim fingertip-to-target vector, so the agent can't cheat by reading target location from proprio — it has to see it or feel it via contact.

**FlattenVisionWrapper.** A tiny wrapper that turns a dict observation `{proprio: [7], vision: [64,64,3]}` into a flat vector of length 7 + 12,288 = 12,295. Lets us use SB3's standard MlpPolicy instead of writing a custom policy.

**Follow-on (v2/v3/v5 architecture).** Our preferred term instead of "staged." Stage 1 is trained first on proprio only, then frozen. Stage 2 ("the follow-on") layers vision on top — only the new pixel-input columns train. Proprio behavior is mathematically preserved (drift = 0.0). Contrast with v1 "staged," which let Stage 2 rewrite every weight — we now read that as obliteration, not interpenetration.

**Consistency loss (v5).** An auxiliary term on the actor loss: `λ · MSE(hidden1(full obs), hidden1(pixels-zeroed obs))`. Forces whatever vision contributes to hidden1 to stay close to what proprio alone would have produced. "Vision must confirm, not reshape."

---

## Source-theory terms (for reference)

**Equivalence class.** the theory's central construct. Many different sensory states (instances) that map to the same conditioned response. "The class *is* the perception." Our neighbor-consistency ratio is a proxy for class structure.

**Π (Pi).** the theory's notation for a *readiness state*: the set of conditioned responses an organism is currently prepared to execute. Perception in the source theory is just the simultaneous pattern of Π.

**Interpenetration (Ch. 5).** "Properties of the perceptual field determined by one set are incorporated in the perceptual field determined by the other set." Our operational reading: vision's representation should inherit and extend proprio's, not overwrite it. v5's consistency loss is an explicit test of this reading.

**Unconditioned vs conditioned response.** Unconditioned = reflexive, built-in (e.g., contact → grasp). Conditioned = learned to fire from a previously-neutral cue (e.g., the *sight* of an object → grasp, after learning that seeing precedes touching).

**Coordination vs interpenetration (Numenta vs Taylor/MICOA).** The precise axis on which AB diverges from Numenta's Thousand Brains Theory. *Coordination* (Numenta's voting): each cortical column / Learning Module builds its own object model independently; modules exchange messages and reach consensus on the answer, but no module's internal representation is restructured by another's learning history — experts vote, and the vote doesn't rewire any expert. *Interpenetration* (Taylor, Ch. 5; the thing MICOA is built to produce): two modules' representations become mutually constitutive — vision comes to encode the world *the way proprio does* because proprio's learning history reshapes vision's representation; the learning is entangled, not just the outputs. The disagreement is NOT about whether a domain-general *mechanism* exists — Numenta grants one (reference frames are used everywhere), which is its own form of "generalization is general." It is specifically about *content-merger*. The wedge that separates them is **prism adaptation** (see next): pure coordination predicts permanent vision-vs-proprio disagreement under reversing prisms; the fact that vision instead *recalibrates* is interpenetration, and coordination-only cannot produce it. Caveat we hold honestly: a general mechanism does not guarantee automatic success — AB's own vision stayed inert (R²≈0.08) despite seeing all the variation, so "general" describes the mechanism's reach, not a promise every channel generalizes.

**Prism adaptation (movement-primacy evidence).** Classic perceptual-rearrangement evidence (Stratton 1897 inverting lens; Held & Hein 1963 kitten carousel) that movement/proprioception is the developmental and authoritative anchor and vision is the plastic party — the empirical backbone of AB's proprio-first thesis. Don left-right reversing or displacing prisms and the visual world is wrong; over days of *active, self-produced movement* the sensorimotor system recalibrates until visually-guided action is accurate again. Two load-bearing facts: (1) **adaptation requires active movement** — Held & Hein's passively-carried kitten gets identical visual input and does *not* adapt; exposure alone is insufficient, action is necessary (the "passive kitten" is what a pure confirmation/agreement loss reduces to). (2) **The motor/world frame is the anchor**: when vision and action conflict the system re-anchors vision toward action, not the reverse — "reality and movement don't switch; vision does" (honest nuance: total adaptation splits between a visual and a proprioceptive shift, but the held-fixed reference is consistency with self-produced action). Directly justifies AB's asymmetry — the `.detach()` on proprio in the Phase X temporal-predictive loss (proprio = ground truth, vision = predictor that must adapt) — and points at what was missing when vision stayed inert: prism adaptation is driven by a *prediction error between vision and proprio under active movement*, which a confirmation-only signal lacks but the temporal-predictive and R49 decode losses supply.

---

## Generalization, robustness & recent methods (Phase XII–XV)

**DroQ (Dropout Q-functions).** A tweak to SAC's critics: add dropout + layer-norm inside the critic networks and raise the update-to-data ratio. The regularization lets you do many gradient updates per environment step without the critic overfitting/diverging, so the agent learns more per unit of experience. Introduced in Phase XIV (R45) as critic regularization; we run it with `--droq --dropout-rate 0.01 --utd 4`. Validated as a no-regression infrastructure change before being used as the Phase XV workhorse.

**UTD (update-to-data ratio).** How many gradient updates the agent does per environment step. Vanilla SAC is UTD=1 (one update per step); DroQ lets us push it higher (we use UTD=4) for more learning per sample. Higher UTD = more compute per step but faster learning in samples — only stable with regularization like DroQ.

**Critic dropout.** Randomly zeroing a fraction of critic-network activations during training (`--dropout-rate 0.01`). A standard regularizer that, in DroQ, is what keeps the high-UTD critic from over-trusting its own estimates.

**Eccentricity (ecc).** How far off-center the target is from AB straight ahead, in meters of lateral offset. ecc=0.00 = ball dead ahead (easiest); larger ecc = ball further to the side (harder to reach). Our standard bins run 0.00, 0.05, 0.10, 0.15, 0.20, 0.25.

**Eccentricity sweep.** The deterministic eval that scores both-touched success at each eccentricity bin (e.g. "20,20,19,13,5,0" across the six bins). Per the Phase XIV methodological finding this is the *only* valid health metric on this substrate — eval mean_reward and critic_loss are invalid here.

**Vision-ablation sensitivity (ablation L2 / abl_L2).** Our primary measure of "does vision matter to behavior?": zero out the pixel columns of the observation and measure how much the policy's action vector changes (L2 distance). High = the policy depends on vision; near-zero = vision is inert ("dead zone" ≈ 0.002–0.024). Preferred over task-success because AB can solve the task by proprio-grope alone — ablation directly asks whether vision is load-bearing. Caveat learned across phases: high action-level ablation does NOT imply vision is *useful* (R41 had the highest ablation ever, 0.85, yet performed worse than the proprio control).

**Load-bearing vs productive.** Two different bars for vision. *Load-bearing* (action-level): zeroing pixels changes the action (ablation > threshold). *Productive* (outcome-level): having vision actually makes AB succeed more often than a matched proprio-only control. The whole project's recurring finding is that vision keeps clearing the first bar without clearing the second.

**Generalization-as-primary.** Our reframing of the project's thesis after Phase XIII: the interesting claim is not "vision helps" but that AB builds a representation that *generalizes* — lawful behavior on object/target configurations it was never trained on. The acid test is zero-shot transfer to held-out objects, not raw task success. Confirmed (core) for proprio; the strong form (vision recruited at the margin) was refuted.

**Winnability (every-episode-must-be-winnable).** The project's governing design rule, articulated 2026-06-16 after watching R49 renders where AB, on a blindly-drifting cart with no locomotion, simply could not get to the ball: *we must provide AB the tools it needs to succeed; if AB never gets near the target, that is our setup failure, not AB's learning failure, and it teaches nothing.* Operationally: every training/eval episode must be **winnable** — the action we want reinforced (touching the ball) must be physically possible for AB's body (target inside its real reach/locomotion envelope), and for any vision run the target must be in the field of view or bringable into view by an action AB can execute (a head turn). Never present a target that cannot be seen and cannot be found. Two consequences: (1) curricula and eval bins must stay inside the winnable set, and we must not score impossible configs as "AB failed" (that measures our setup's impossibility, not AB's ability — separate "AB couldn't" from "we made it impossible"); (2) body fidelity never outranks winnability — if the MIMo infant body makes the target unreachable, change the setup, not the standard. Rationale: an action can only be reinforced if it can occur. This rule is what reframed the project away from the inert-vision decode-loss patch (R49) toward giving AB the affordance to pursue (cart-steer / locomotion).

**Movement-as-substrate (of generalization).** The project's deeper thesis (a step beyond generalization-as-primary): movement isn't just learned first, it is the *substrate* the generalizing representation is built ON. Evidence: proprio's lawful, extrapolating distance law (movement-substrate working for proprio); prism adaptation showing the motor frame is the anchor vision recalibrates onto. The open, not-yet-confirmed bet: that building in *richer* movement makes *vision* generalize too (R49 and DOF-curriculum runs are the cheap probes of this causal link, before any expensive simulator port). Paired strategic rule — **"faithful to the mechanism, free on the implementation"**: AB reconstructs human perception but is not bound to the human developmental path. Decide each human fact by whether it is *load-bearing for the mechanism* (active self-produced movement; motor competence established before vision is trusted; graded freezing/freeing of DOF — keep these) or an *incidental detail of being a mammal* (gestation, helpless emergence, the literal year of crawling — drop or compress these into a curriculum). Diverge freely where it *serves* the mechanism (privileged training signals, near-perfect proprioception, repeatable resets, compressed lifetimes); stay faithful only to movement-first, interpenetration, and the equivalence class. Two concrete build-in levers: (1) graded DOF freezing/freeing (Berthouze & Lungarella — cheap, a curriculum change in MuJoCo today); (2) migrating to a richer/growing body (MIMo v2 — expensive, justified only once the substrate→vision-generalization link is confirmed on the current body).

**Zero-shot transfer.** Performance on objects/conditions AB was never trained on, with no extra training. Phase XV's central test: train on a subset of an equivalence class (e.g. ball radii {0.040, 0.053, 0.075}) and measure both-touched on held-out members ({0.047, 0.090}).

**Held-out / interpolation vs extrapolation.** A *held-out* value is deliberately excluded from training so it can test transfer. *Interpolation* = the held-out value sits inside the trained range (e.g. 0.047 between 0.040 and 0.075); *extrapolation* = outside it (e.g. 0.090 beyond 0.075). Extrapolation is the harder, more telling test.

**Distance law / time-to-contact.** The empirical finding that steps-to-reach scales lawfully with target distance (Phase XIII: R44 proprio ≈ 688·d − 218, r≈0.89), and crucially *extrapolates* to untrained distances — evidence the representation is metric, not memorized. Confound: the autonomously-sweeping cart logs spurious ~1-step contacts that add scatter, so the exact slope is noisier than a clean reach task would give.

**Equivalence-class test (object variety).** The Phase XV paradigm: treat "ball" as a class with members varying along a dimension (size, shape, …), train on some members, hold others out, and ask whether AB treats the held-out members as the same class (still reaches them). Directly operationalizes the source theory's equivalence-class construct.

**Domain randomization (variation-makes-generalization).** The robotics-RL name for the mechanism that builds an equivalence class: train across many variations of a situation (object size, position, lighting, dynamics) that all earn reward for the *same* outcome (touch/grasp), and the breadth of variation you trained on becomes the breadth you generalize to. It's the engineering version of the source theory's "many instances → one conditioned response → the class *is* the perception." Two caveats we learned the hard way: (1) in RL the "same outcome" is defined by the *reward function* (reward declares which varied states count as equivalent), standing in for what a developing body gets for free from its unconditioned reflex; (2) variation is *necessary but not sufficient* — proprio saw the variations and built a lawful, extrapolating class (the distance law), but vision saw the *same* variations and still failed to generalize (R²≈0.08) because nothing forced its latent to encode what varied. Variation only generalizes when the learner is *compelled to represent* it — which is exactly what R49's auxiliary decode loss manufactures by hand.

**The 0.075 anomaly.** A reproducible dip in both-touched success at the *trained* ball radius 0.075 (≈6/20) seen in both R46 and R48, even though smaller and larger sizes do better. Anomalous because a trained size should be easy; flagged for a focused diagnostic rather than explained.

**constant_velocity_bouncer (cart mode).** The substrate AB sits on: a cart that drifts at constant speed and bounces off the arena walls (`--cart-speed 0.15`), giving gentle ongoing motion. Distinct from a static base or a ball that itself moves (`--ball-speed`, set to 0 in Phase XIV/XV).

**Freeze attractor.** The recurring failure mode where RL training drives AB's policy onto "do nothing" (or "make one move, then go inert") — motionless body, reward dead-flat at pure step cost (e.g. −18.7), zero contacts, every episode times out. Called an *attractor* in the dynamical-systems sense: it's the basin training keeps sliding into regardless of seed or starting pose (the behavior is conditionally uniform — always freeze — even when the reward varies by lucky spawn orientation). Two causes, neither sufficient alone: (1) a raw-torque action space (*A Walk in the Park* shows torque control "cannot make any progress" from scratch), and (2) reward/exploration collapse — standing still is a genuine local optimum because moving costs energy and rarely hits a sparse contact reward. Switching to position-offset control fixed (1) but left the attractor *unbroken in 60K* steps, proving (2) is real. Seen across Phases IV–VI and again in R50 steer-mode video.

**RND (Random Network Distillation).** An intrinsic-motivation / curiosity method (Burda et al. 2018) for getting an agent to explore. Keep one *fixed, randomly-initialized* net (the target) and train a second net (the predictor) to match its output on visited states. The predictor's error becomes a bonus reward: familiar states are predicted well (low error → boring), novel states aren't (high error → curiosity bonus). Directly attacks the freeze attractor — a motionless body sees the same state forever, error → 0, so stillness stops paying and movement does. Simplest robust curiosity method (no learned dynamics model, just two nets and a subtraction), which is why it's the natural first lever to bolt onto the crawler.

**Movement vs. locomotion.** A distinction the project's whole crawler track now turns on. **Movement** is any change in the body's configuration — joints rotating, limbs splaying, head turning, torso twisting — the body *doing something* rather than sitting inert. **Locomotion** is the strict subset of movement that *translates the whole body through space*: the center of mass actually travels from A to B across the floor. All locomotion is movement; most movement is not locomotion — you can flail every limb continuously while your CoM stays put (net ground-reaction forces cancel or only rock the body). The bridge between the two is a **gait** (see next entry). This is why `rnd_directed_250k` failed: RND + zero step-cost reliably produced *movement* (novelty is cheap to get by fidgeting), but the body never *translated*, so the approach reward — which can only fire on distance-closed — never received a positive sample and no pursuit gradient formed. The corrected slogan: **movement is reward-fixable; locomotion is not** (it needs the gait discovered or supplied). Not word-play — locomotion demands temporal *coordination* that mere movement doesn't.

**Gait.** The coordinated, repeating push-off pattern that turns movement into locomotion. To translate the CoM, the limbs must push against the ground in the *right sequence and phase* so ground-reaction forces sum to a net thrust in one direction instead of cancelling — that phased, cyclic pattern is a gait (crawling, walking, swimming, slithering are all gaits). The hard part isn't moving each joint; it's discovering the *sequence* that accumulates propulsion. A gait is a narrow region of action-space that novelty-seeking (RND) is unlikely to stumble into, and it can't be reward-shaped into existence if it's never once sampled — which is why the indicated tools for AB are the ones that *supply* a gait (imitation from a crawling demonstration, seeding the replay buffer) or *force translation directly* (a velocity-toward-target bonus, a cart-steer movement primitive), not more shaping on a body that flails in place.

**Velocity bonus.** A small shaping reward (`--velocity-bonus-scale 0.10`) for moving toward the target rather than freezing — counters AB's tendency to collapse into stillness (the "floor episode" attractor).

**Curriculum (warmup / ramp / offset).** Easing AB into difficulty: start with the target arranged in its favor and quietly move it out of reach over training. Controlled by `--curriculum-warmup` (steps before difficulty starts rising), `--curriculum-ramp-end` (step where it reaches full difficulty), and `--curriculum-final-offset` (how far the prize ends up displaced).

**Broad touch vs hand-only touch.** Two contact metrics in the cart env.
*Broad* (`touched_ball*`) fires on a real MuJoCo collision between the ball and
ANY MIMo geom — feet, legs, torso, head, or hands. *Hand-only* (`hand_touched_ball*`)
counts only the 8 hand/finger geoms. Reward, termination, and our "both-touched"
success all use the BROAD metric — so a "success" need not involve a hand at all.
Finding (2026-06-15): both_HAND = 0 across every size AND every ecc bin on the whole
cart line — the task is never *completed* with two hands (though a hand does engage
for one ball, more at high ecc). The two metrics are developmentally distinct, not
redundant: broad = unconditioned "the object is there / world is consistent" contact
(a valid signal); hand-only = the intentional/conditioned reach. both_HAND=0 is a
developmental stage, not a bug — the quantity to track is hand-touch fraction over
training (does incidental encounter mature into intentional reach?).

**Cart-sweep delivery (substrate delivery).** Because the balls are fixed on the
cart's sweep line, the cart carries AB's *body* straight into them, so the ball is
delivered into AB by the moving substrate rather than reached for. The dominant
source of low-eccentricity broad-touch "success". NB the cart geom itself is a
non-colliding visual marker (`contype=0 conaffinity=0`) — it never pushes the ball;
all ball contact is via AB's body geoms or the real (solid) platform. Not purely a
confound to remove: incidental body contact is a legitimate "world-is-consistent"
signal. But to study *reaching*, move balls off the sweep line (forces a lateral
hand reach) or score on hand-only touch.

**Knock-away artifact (glancing-blow).** The mechanism behind the 0.075 "anomaly":
a mid-size ball (radius ~0.070-0.080) is contacted off-centre by the trained reach,
which propels it skidding to the arena wall, out of a no-locomotion creature's reach
envelope. The first touch registers but the second ball can't be recovered, so the
two-ball episode never completes. Geometric (ball mass is unchanged by radius), not
momentum. A manipulation artifact, not a generalization failure.

**Dead band.** A contiguous range of a swept parameter where performance drops while
both shoulders are fine — e.g. the [0.070-0.080] radius band where both-touched falls
to ~0.45-0.55 with 0.053 (~0.95) and 0.090 (~0.75) clean. Distinct from a point
anomaly; only visible if you sample inside the band (the original eval sampled just
0.075, so a band masqueraded as a spike).

---

## Prior-work concepts (from the 2026-06-16 literature scout)

Named ideas from the outside literature that map onto our "integrated but inert" problem. See `LITERATURE_SCOUT_2026_06_16.md` for the full citations.

**Modality dominance / modality competition.** From the supervised multimodal-learning literature (Wang/Tran/Feiszli 2020; Peng et al. OGM-GE 2022): when a network is trained jointly on two input streams, the stream that is *easier* to learn from wins the gradient and the other stays underused — a joint multimodal net can be beaten by the best single modality alone. This is the closest *named* neighbor to AB's "vision integrated but inert": proprio already solves the task, so SAC's gradient flows through proprio and vision is starved. Key difference: in the supervised literature the weak modality is usually still informative and *recoverable by gradient rebalancing*; ours is more severe — the vision latent barely encodes the target at all (representation failure, not mere imbalance), so rebalancing alone may not suffice.

**Privileged teacher → student distillation.** The field's standard recipe for making vision load-bearing when a non-visual channel already solves the task (Learning by Cheating, Chen et al. 2019; RMA, Kumar et al. 2021; Lee/Hwangbo 2020). First train a *teacher* policy that gets privileged information unavailable at deployment (e.g. the exact target vector or ground-truth state); then train a *student* that has only the real sensors (vision) to imitate the teacher's actions. Vision becomes load-bearing because supervision *forces* it to — the step AB's end-to-end run skips. The literature's #1 recognized fix for exactly AB's pathology.

**Asymmetric actor-critic.** An RL training trick (Pinto et al. 2018) where the *critic* receives extra privileged information (full state) during training while the *actor* only ever sees the real sensor observations. Because the critic is used only at training time and discarded at deployment, you can feed it ground-truth it could never have in the real world, giving the actor a better learning signal without cheating at test time. "Asymmetric" = actor and critic see different things. Relevant as a gentler alternative to full teacher→student distillation for getting supervisory pressure onto a vision policy.

**Auxiliary decode loss (auxiliary ball-position decode loss).** A second training objective bolted onto the main RL loss whose only job is to force the encoder to represent a specific variable. You attach a small *decoder* to the vision latent, train it to predict the target's position, and backpropagate that error *into the encoder* — so the encoder is rewarded for putting target position into the latent regardless of whether the RL task gradient bothers to. This is our #2 candidate fix for the representation-failure result, and it targets the encoder directly (cf. the read-only ball-x probe, which only *measures* the failure). Success test: does vision-ablation then become *directional* (sensitive to which side the ball is on) instead of the content-free ecc=0 spike we see now? (Result, R49: refuted — the loss mostly captured distance, lateral probe R² stayed ~0.01, and the predictive-KL term destabilized the encoder. See *winnability* for the reframe.)

**Constrained action space (position-offset control).** The make-or-break ingredient for learning locomotion, from *A Walk in the Park* (Smith/Kostrikov/Levine 2022): instead of the policy outputting raw joint *torques*, it outputs small *offsets around a default pose*, which a position (PD) controller turns into motion (with damping Kd≈10). Their ablation is blunt: with a raw-torque ("unconstrained") action space the agent "cannot make any progress"; with position-offset actions training is stable and a quadruped walks in ~20 minutes. Directly relevant: AB's `mimo_crawler.xml` uses 26 raw-torque `<motor>` actuators and 0 position servos — exactly the unconstrained config the paper shows fails — which is the leading (verified-precondition) hypothesis for why AB's from-scratch locomotion collapsed to stillness / one-move-then-freeze across Phases IV–VI. The fix is a small additive XML/wrapper change, not new research.

**Reverse / start-state curriculum.** The principled, literature-backed way to enforce the *winnability* rule (Florensa et al. 2017/2018). Start every episode with the goal trivially achievable — target spawned close, in reach, and in view — and push start-states / goals outward only as the agent's success rate rises, so the agent always has a winnable episode and difficulty tracks competence. It replaces AB's hand-tuned spawn-cone / eccentricity scaffolds with a mechanism that, by construction, never presents an impossible config. Pairs with automatic goal generation (a generator proposing goals of intermediate, feasible-but-not-mastered difficulty).

---

## Locomotion & imitation (crawler track, 2026-07)

**Crawl-ready default pose (position-offset around a locomotion-adjacent pose).** The single biggest lever that unblocked crawling. In position-offset control the actuator ranges are centred on a *default pose*, and the action nudges joints ±offset around it. "A Walk in the Park" (Smith et al. 2022) shows this only works if the default pose is *already close to the target behaviour* (a quadruped's default is a standing stance one stride from walking). Our old default was flat **prone**, so even wide offsets swept a region that never reached a propulsive stance. Re-centring the ranges on an **arms-forward "commando" pose** (`CRAWL_POSES["arms_fwd"]`) raised hand-driven translation 6× (0.039 m → 0.225 m) and let RL discover real crawling. Lesson: *change the default pose, not just the clamp width.*

**Early termination / tip-termination.** Ending an episode (with a penalty) when the body fails — here, when the dorsal axis tilts past a threshold (`--terminate-tilt-deg 50`). DeepMimic (Peng 2018): early termination "eliminates local optima by penalising the character when on the ground." Without it, lying still / rolling onto one's side and sliding are unpunished optima — exactly the freeze/rock exploits we saw. It is *the* standard tool that makes from-scratch locomotion trainable.

**Specification gaming.** When a policy optimises the literal reward in a way that violates its intent. Our canonical case: a reward on |CoM velocity| (unsigned speed) was maximised by **rocking in place** — high speed, zero net displacement. The fix is a *signed* reward (approach = distance-closed toward the goal) plus structurally blocking the cheap exploit (tip-termination). Rewarding unsigned speed is a documented anti-pattern (DeepMind specification-gaming list).

**Potential-based reward shaping.** A dense shaping reward defined as the change in a potential function (here, the drop in distance-to-ball per step = our `approach` reward). Being a difference of a potential, it gives a smooth gradient toward the goal without changing the optimal policy — the recommended way to guide sparse-reward tasks (and the reward form MIMo's rolling paper used).

**target_obs / privileged target observation.** Appending the ball's position (in the body frame, heading-invariant) to the observation. Directed homing was impossible without it: the approach reward fired only when *random* locomotion happened toward the ball, giving zero directional gradient (mean toward-ball translation exactly 0.000 m). Adding it converted the random walk into directed homing (+0.195 m). It is *privileged* info (given, not perceived); the project's next phase is to have **vision** supply the same bearing.

**DeepMimic / AMP / reference-state initialization (RSI).** The standard tools for gaits pure RL will not discover: imitate a reference motion (DeepMimic, Peng 2018; AMP, Peng 2021 uses an adversarial motion prior). Crucially you do **not** need mocap — hand-authored keyframes work, and often just **RSI** (starting episodes from a few good poses of the target motion) is enough. Held in reserve for us: our affordance fix made the gait discoverable without imitation, so RSI/DeepMimic was not needed — but it is the indicated tool if a gait remains unsampled.

**PPO-vs-SAC for locomotion stability.** SAC (off-policy, replay buffer) discovered directed crawling but never *stabilised* it — every run burst to a good policy then regressed as the buffer drifted (no stable lock-in). PPO (on-policy, no replay drift) trained smooth-monotonically to a plateau and roughly doubled the contact rate (23% → 57%). MIMo's one working whole-body skill used PPO; the lesson is that on-policy stability can matter more than off-policy sample-efficiency for consolidating a sparse-reward motor skill.

**Creeping vs crawling (clinical).** In infant development, "crawling" = belly on the floor (commando/belly-crawl); "creeping" = up on hands-and-knees. Our arms_fwd result is belly-crawl. Worth distinguishing because hands-and-knees creeping demands supporting body weight (strength), which the MIMo rolling paper flags as a hard, separate requirement.

## Vision-phase terms (added 2026-07-04)

**Substrate / capability split** — the measurement contract for "reinforced not destroyed." Report a
policy's competence as two separate things, not one number: the SUBSTRATE (proprioceptive: crawl
gait, posture/tip-rate, displacement, speed) which must be PRESERVED, and the CAPABILITY (vision:
contact rate, distance-toward-ball) which must CLIMB. "Reinforced" = capability climbs while
substrate holds; "destroyed" = substrate falls when vision is added.

**Vision load-bearing gap (pixel-ablation gap)** — contact rate with real pixels minus contact rate
with the pixel block zeroed. A large positive gap that degrades to (not below) the blind baseline
means vision is doing real behavioral work and proprio is recoverable. Stage B: 63.3% → 20.0%
(+43 pts). Distinguish from an ablation gap that drops BELOW baseline (vision made the policy depend
on pixels without adding value — the Stage C / "integrated but inert" pattern).

**Distil-then-RL (distillation-then-reward-finetune)** — the winning recipe of the vision phase:
first SUPERVISE a vision encoder to predict the privileged signal (Stage A, bearing from pixels,
R²=0.84), then WARM-START a reward-trained policy from it (Stage B) so reward only has to USE the
representation, not discover it. Reward ALONE (Stage C) failed to grow vision. Matches the
literature's Learning-by-Cheating / Distillation-PPO recipes.

**Representation failure vs policy-gradient failure** — two distinct reasons vision can be inert.
Representation failure: the encoder never builds the signal (decode R²≈0). Policy-gradient failure:
the signal is decodable but reward doesn't connect it to behavior. The vision phase proved AB's was
the SECOND (R²=0.84 but reward-alone inert), overturning the long-assumed "representation failure."

**Residual vision head (zero-init additive adapter)** — a trainable CNN whose output is ADDED to a
frozen base policy's action, with the final layer initialized to zero so at start the policy == the
base (gait preserved by construction). Vision can then only ADD, never destroy. From "No More Blind
Spots" (Duan 2025) / residual RL / ControlNet zero-init. Used in Stage C (`ResidualVisionPolicy`).

**Winnability preflight (camera visibility check)** — a mandatory pixel-level measurement, before any
vision run, that the target is a clear blob (≥ a few px) across the spawn cone at the true training
resolution. `cam_visibility_preflight.py`. Catches the "vision task is unwinnable because the target
is off-frame / a horizon speck" trap that produced years of false "vision can't learn" nulls.
