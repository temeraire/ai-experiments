# Glossary

Terms that show up in FINDINGS.md, the training scripts, and our conversations. Kept plain-English; each entry aims for "what is it, why does it matter here."

---

## Reinforcement learning (the algorithm side)

**SAC (Soft Actor-Critic).** The learning algorithm we use. It's a modern reinforcement-learning method for problems with continuous actions (like joint torques). "Soft" because it rewards the agent for keeping some randomness in its behavior — it discourages premature lock-in to a single strategy. It's the standard workhorse for robotic-arm tasks in simulation.

**Actor.** The neural network that chooses actions. Input: the observation (proprio + pixels). Output: a distribution over actions. In our case it's a small MLP: obs → 256 → 256 → action.

**Critic.** A second neural network that estimates "how good was that action in that state?" — a scalar value. SAC actually uses two critics and takes the minimum (a trick to stop the agent from over-trusting its own value estimates). The actor trains itself to pick actions the critic rates highly.

**Policy.** The rule that maps observations to actions — i.e., the creature's moment-to-moment behavior strategy. In our code the policy *is* the actor network: feed in the observation vector (proprio + pixels), out comes a distribution over the action vector (joint torques, head commands). "Running the policy" = executing that mapping step-by-step through an episode. SAC's whole job is to shape the policy over training so that actions it picks lead to high reward. When we talk about "vision-ablation sensitivity" (how much the action changes when pixels are zeroed), we are asking: how much does the policy depend on vision?

**Episode.** One trial: reset the environment, let the agent act for up to 200 timesteps or until it touches the target (whichever comes first). Our task gives +10 reward for touching the target.

**Timestep / step.** One `env.step(action)` call — the physics simulator advances by a small amount.

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

**Weight / parameter.** A single learnable number inside the network. The actor has ~3.2M weights total; we freeze most of them and only let a small subset (the pixel-column part of the first layer) train.

**Freezing weights.** Setting `requires_grad = False` so the optimizer won't update them. In v2/v3/v5 we freeze all of the actor's weights except the first-layer pixel columns — proprio's behavior is mathematically preserved.

**Gradient hook / gradient mask.** A PyTorch feature that intercepts gradients as they flow back through a tensor. We use it to zero out the gradient on the proprio columns of the first-layer weight — so even though that tensor is marked trainable, the proprio slice effectively isn't. Belt-and-suspenders on top of freezing.

**Backprop / gradient descent.** The training loop: run the network forward, compute a loss (how wrong the output was), compute the gradient of the loss w.r.t. each weight, nudge each weight a tiny bit in the direction that reduces the loss. Repeat.

**Learning rate.** The size of the nudge. We use 1e-4 (0.0001) for v2–v5.

**Loss function.** A single scalar that measures how badly the model is doing on a batch. The optimizer minimizes it. SAC's actor loss has two parts (task performance and entropy); v5 adds a third (consistency). MSE — mean squared error — is the simplest loss: average of `(prediction − target)²`.

---

## Representation-analysis tools

**R² ("R-squared").** The coefficient of determination. A goodness-of-fit number for regression, in range [−∞, 1]. R² = 1 means the regressor predicts targets perfectly; R² = 0 means it does no better than guessing the mean; negative R² means it's worse than guessing the mean. When we said "vision's target-localization accuracy capped at R² = 0.35," we meant: a linear readout trained to decode target position from the network's hidden activations explained only 35% of the variance in target position.

**Linear probe.** A simple trick for asking "does this internal representation contain information X?" You freeze the main model, train a *linear* readout (one matrix multiply) to predict X from the model's hidden activations, and measure how well it works. If a linear probe gets high R², the info is there and easily extractable. Low R² could mean info is absent — or just tangled up in a non-linear way.

**Ridge regression.** The specific linear-probe regressor we use (`sklearn.linear_model.Ridge`). It's linear regression plus a small penalty on large weights, which prevents overfitting. Nothing exotic.

**CKA (Centered Kernel Alignment).** A similarity measure between two sets of activations from two different networks on the same inputs. CKA = 1 means "these two networks represent the input in essentially the same way (up to a linear reshuffling)." CKA = 0 means "totally unrelated representations." We use linear CKA; the math is `‖Xᵀ Y‖² / (‖Xᵀ X‖ · ‖Yᵀ Y‖)`. Why it matters here: if v5's hidden1 has high CKA with stage1_v3's hidden1, vision is living on proprio's manifold — *additive* interpenetration. Low CKA = vision has carved out its own manifold and ignored proprio's.

**k-NN (k-nearest neighbors).** For each point, find the k closest other points (we used k=5, closeness measured in hidden1 space by Euclidean distance). Used here as a probe: if two observations are neighbors in hidden1 space, are their *actions* also similar? If yes, we have an equivalence class (different inputs → same action → same perception, in the theory's sense).

**Neighbor-consistency ratio.** Our equivalence-class metric: `mean(action-distance among hidden1 neighbors) / mean(action-distance among random pairs)`. Lower = tighter equivalence classes. Stage1_v3 came out at 0.07 (neighbors' actions are 14× more similar than random); v1 agents were ~0.45 (weak class structure).

**k-means.** An unsupervised clustering algorithm. Give it N points and a k; it finds k "center" points such that each observation ends up assigned to its nearest center. We use it to carve hidden1 activations into discrete Π-classes we can inspect.

**Silhouette score.** A measure of how clean a clustering is: high = points are much closer to their own cluster's center than to other centers; low = clusters bleed into each other. Range [−1, 1].

**Spearman correlation (distance correlation).** Correlation between the *ranks* of two variables (not the raw values). We use it to ask "when two observations are close in hidden1 space, are their actions also close?" Spearman is robust to outliers and non-linear monotonic relationships.

**MSE (mean squared error).** Average of squared differences. Our v5 consistency loss is `MSE(h_full, h_blind)` — push the two hidden1 vectors to be numerically close. Same quantity R² implicitly measures.

---

## The simulation stack

**MuJoCo.** A fast, accurate rigid-body physics simulator. Handles the arm, the objects, contact forces, gravity. Bought by DeepMind in 2021 and open-sourced.

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
