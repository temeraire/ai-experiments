# Gap Analysis: Alien Baby RL Stack vs. Standard Practice

*April 2026 — companion to `PRIOR_ART.md`*

*What the literature considers table-stakes that we are not doing — and what that costs us.*

---

## Why this doc exists

`PRIOR_ART.md` answers "what has the field solved?" This doc turns it inward: "what specifically have *we* been omitting that the field considers basic?" Triggered by David's frustration at being told the MirrorWrapper "should have been there from the start" — and the suspicion that other things were similarly missed.

Grounded in code reads of `train_v8.py`, `platform_creature_env.py`, `envs/mirror_wrapper.py`, the v8/v9 XML files, and `PROJECT_STATUS.md` / `FINDINGS.md`.

## TL;DR

Three architectural gaps probably account for the "vision never becomes load-bearing" symptom recorded in PROJECT_STATUS.md:

1. **No CNN encoder.** 3072 raw pixel values are flattened straight into the MLP's first layer alongside 29 proprio scalars. A flat MLP cannot efficiently learn "where is the red ball" from raw pixels — it has to rediscover the same spatial filter independently at every location. DrQ-v2 [Yarats 2021] solves this with a small CNN → 50-dim latent before the MLP. Compatible with the frozen-proprio architecture.
2. **MirrorWrapper missing from follow-on.** The wrapper exists and is correct, but the 16 follow-on envs are built without it (`train_v8.py:174-184`). Even if Stage 1 is run with `--mirror-augmentation`, follow-on reintroduces the lateralization attractor in all 16 workers. Two-line fix.
3. **No VecNormalize.** Proprio values reach ±10 (joint velocities); pixel values are in [0,1]. The pixel columns of the first MLP layer start near zero and compete against order-of-magnitude larger proprio activations. SB3's standard `VecNormalize` would bring both to unit variance before the MLP sees them.

Six secondary gaps (replay buffer too small, no action smoothing, no health-based termination, manual non-adaptive curriculum, no domain randomization, no GPU batch sim) are real but won't unblock vision on their own — see ranked list in §3.

One large strategic gap: **MuJoCo Playground (MJX, Jan 2025)** gives 100×–1000× iteration speed on the same MuJoCo physics. Worth evaluating once the architecture above is fixed.

---

## 1. What We Have

A concise inventory of the current state, with file references, to establish the baseline before the gaps.

### Algorithm

SAC (Soft Actor-Critic), off-policy, single continuous-control environment in Stage 1, 16 parallel environments in follow-on. (`train_v8.py:17`, `train_v8.py:33`, `train_v8.py:276-288`). The training script selects `device="mps"` on Apple Silicon via `_best_device()` (`train_v8.py:36-44`). `learning_starts=5000`, `gradient_steps=1`, `tau=0.005`, `batch_size=256`, `buffer_size=200_000`, `gamma=0.99` (`train_v8.py:276-287`).

### Environment and Observation

`PlatformCreatureEnv` returns a flat float32 vector: 29-dim proprioception concatenated with an optional 3072-dim flattened 32×32 RGB pixel frame (`platform_creature_env.py:214-224`). No normalization wrapper is applied to the observation or reward. No frame stacking; the current time step's single frame is appended raw.

### Network Architecture

SB3's default `MlpPolicy`: `Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU → action head`. For follow-on, `ConsistencySAC` inherits the same policy and adds a consistency auxiliary loss (`FINDINGS.md:v5` section). There is no CNN encoder for the pixel portion of the observation. Pixel values are divided by 255 (`platform_creature_env.py:263`) and flattened directly into the same MLP input layer as the proprio dimensions.

### Symmetry

`MirrorWrapper` exists (`envs/mirror_wrapper.py`) and is correct: per-episode coin-flip that transforms obs and un-transforms actions with proper sign flips for quaternion y/z, angular velocity ωy/ωz, linear velocity vx, and horizontal pixel columns (`mirror_wrapper.py:57-87`). In Stage 1 training it is wired in only when `--mirror-augmentation` is passed at the CLI (`train_v8.py:252-253`). In follow-on training, there is **no MirrorWrapper integration at all** — the `_make_followon_env` factory does not accept or apply it (`train_v8.py:174-184`).

### Reward Shape

Five components: hunger penalty (−0.05/step), edge warning (up to −0.5/step near platform edge), attract gradient (up to +0.3/step when hand is within 0.25 m of ball), contact reward (+200 on touch, terminal), fall penalty (−500, terminal). Optional PBRS shaping (`pbrs_alpha`, default 0) and closure bonus (`closure_bonus_scale`, default 0) (`platform_creature_env.py:35-47`, `platform_creature_env.py:475-518`). No action-smoothing penalty or torque-energy term.

### Curriculum

Manual radius anneal via `RadiusAnnealCallback` or `Stage1RadiusAnnealCallback`, driven by CLI args. Stage 0 → Stage 1 hand-off is a single warm-start from checkpoint. No automated curriculum: task difficulty does not respond to agent performance.

### Domain Randomization

None. Mass, friction, motor gains, and observation noise are fixed across all episodes and all training runs.

### Early Termination

Fall-off-platform detected by torso z < `PLATFORM_Z − 0.2` (`platform_creature_env.py:293`). Ball-lost (v9 only) detected by ball z below platform (`platform_creature_env.py:463-473`). No health-based early termination (e.g., tilt angle, angular velocity threshold) beyond the physical fall.

### MuJoCo Physics

- Integrator: **RK4** in v8 XML (`platform_creature.xml:3`); switched to **Euler** in v9 (`platform_creature_v9.xml:3`).
- Timestep: 0.01 s, 5 substeps per env step → 0.05 s per decision.
- Default joint armature 0.1, damping 2 (`platform_creature.xml:6-7`).
- Default geom friction 0.8/0.3/0.1 (sliding/torsional/rolling); wheel friction overridden to 2.0/0.005/0.0001 (`platform_creature.xml:70-71`).
- No explicit `impratio`, `solref`, or `solimp` overrides — MuJoCo defaults are used.
- No `<equality>`, `<tendon>`, or `<contact>` filter/pairing sections.
- No actuator `forcerange` or `forcelimited` constraints; control limited to `[-1, 1]` but force is uncapped below that.
- No `<site>` sensors for velocity, accelerometer, gyroscope, or frame pose. Torso velocity and quaternion are read directly from `data.qpos` and `data.qvel` rather than from declared sensors (`platform_creature_env.py:245-249`).

### Throughput

Stage 1: single env, ~5 000 FPS (proprio only). Follow-on: 16 parallel SubprocVecEnv, vision active, ~100–400 FPS estimated from comparable setups. No GPU-accelerated batch simulation (MJX/Brax).

---

## 2. Gap-by-Gap Analysis

### Gap 1 — MirrorWrapper Not Wired Into Follow-On

**What we have.** The wrapper exists and is correct for Stage 1 when explicitly invoked. Follow-on training (`train_followon_v8`) ignores it entirely (`train_v8.py:174-184`). The 16 parallel environments in follow-on are built without `MirrorWrapper`.

**Standard practice.** [Abdolhosseini et al. 2019] systematically evaluate four symmetry methods — network symmetry (NET), data duplication (DUP), phase-offset pairing (PHASE), and symmetry loss (LOSS) — and show that any of them outperforms vanilla training on bilateral bodies. More recent work formalizes bilateral symmetry as a morphological group symmetry and enforces it architecturally with equivariant GNN policies [Ordonez-Apraez et al. 2025, MS-PPO 2024]. The data-augmentation approach (mirror obs, un-mirror actions) — exactly our implementation — is the cheapest-to-implement variant and is treated as baseline hygiene in bilateral locomotion work published from 2019 onward.

**Why this matters here.** The ARCHITECTURE.md documents the lateralization attractor explicitly (Section 6.1): Stage 1 policies collapse to solving one hemisphere. Follow-on receives a lateralized Stage 1 policy as its initialization and then trains without MirrorWrapper. Even if Stage 1 is run with the flag, follow-on re-introduces the lateralization risk in 16 envs that have no symmetry pressure. The "breakthrough-then-collapse" pattern across runs is consistent with a policy that learns one hemisphere in Stage 1, transfers it, and then cannot generalize the behavior to the other hemisphere when the ball spawns there in follow-on.

**Confidence that this is load-bearing:** High. The body is bilateral, the task is bilateral, and a lateralization attractor is a documented failure mode in our own telemetry. Not wrapping follow-on is an oversight of the same category as not wrapping Stage 1.

---

### Gap 2 — Observation and Reward Normalization (VecNormalize)

**What we have.** Raw observations are passed to the policy. Proprio values are on heterogeneous scales: joint angles in radians (roughly ±2), joint velocities in rad/s (potentially ±10+), quaternion components in [−1, 1], linear velocity in m/s, torso xy in meters from platform center (±1 m), touch sensors binary {0,1}. The pixel values are in [0, 1] after the /255 division. Reward magnitudes span five orders of magnitude: hunger is −0.05/step, attract is +0.3/step, contact is +200 one-time, fall is −500 one-time. No normalization is applied.

**Standard practice.** The SB3 documentation and the RL Zoo hyperparameter repository treat `VecNormalize` (running mean/std on observations and rewards) as table-stakes for continuous-control locomotion tasks. The standard pattern wraps the `SubprocVecEnv` with `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`. Without it, proprio and pixel values arrive on incompatible scales — a joint velocity of 8 rad/s and a pixel value of 0.03 occupy the same MLP input vector and compete for the same initial-layer weights, with no a priori reason for the magnitudes to be comparable. The reward normalization compresses the 500:0.05 = 10 000:1 range of reward components into something the critic can represent without saturating.

**Why this matters here.** The follow-on observation is `[proprio (29 dims on mixed scales), pixels (3072 dims in [0, 1])]`. The pixel values are systematically smaller than many proprio values. In the first-layer weight matrix, the pixel-column weights start near zero (they are initialized from Stage 1 weights that had no pixel signal and then zero-padded). The proprio columns have order-of-magnitude larger input activations. This imbalance means gradient steps on the pixel columns are competing against a backdrop of large proprio activations, slowing the emergence of any pixel-derived features. `VecNormalize` would bring both modalities to unit variance before the first layer sees them.

**Confidence:** High. This is one of the most reliable fixes in off-the-shelf SB3 locomotion. The absence of it is a plausible contributor to "vision adds nothing" across multiple follow-on runs.

---

### Gap 3 — No CNN Encoder for Pixel Observations

**What we have.** 3072 pixel values (32×32×3) flattened and concatenated directly into the MLP input layer alongside the 29 proprio values. The MLP's first linear layer (obs_dim × 256) processes the combined vector with no spatial processing applied to the image.

**Standard practice.** From DrQ [Yarats et al. 2021, ICLR 2021] and DrQ-v2 [Yarats et al. 2021, arXiv 2107.09645] forward, the standard architecture for visual continuous control is: `CNN encoder (e.g., 4-layer, 32 channels) → latent (50-dim) → MLP critic/actor`. The CNN encodes spatial structure; the MLP operates on the compact latent. Without a CNN, a flat MLP cannot efficiently learn translation-invariant or spatial-covariant features from pixels — it has to re-learn the same feature at every spatial location independently. DrQ-v2 can solve humanoid locomotion from pixels with this encoder; a flat MLP treating pixels as an unordered feature vector cannot.

**Our specific situation is worse than the generic case.** Our image is 32×32 — small, but the ball is typically only a few pixels across. The red ball's position is the critical visual feature. A CNN with a few channels would find it in ~10K training samples by learning a "red pixel cluster" filter. A flat MLP with 3072 input columns has to learn that specific combinations of (column 847, column 848, column 879, ...) correspond to "ball is at position (x, y)" — a combinatorial explosion. The current architecture is not designed to learn what it is being asked to learn.

**Note on the research goal.** The consistency loss and frozen-proprio architecture require vision to "speak through" the hidden layer. A CNN encoder applied before the MLP is compatible with this: freeze proprio columns of the first MLP layer, let the CNN + pixel-MLP columns train. The CNN processes the image into a 50-dim latent; those 50 values feed into the pixel columns of the first layer. The consistency loss still applies. Nothing about the research goal requires a flat pixel representation.

**Confidence:** High. This is the most technically crisp gap: a standard architecture exists specifically for this problem, we do not use it, and the consequence (vision never becoming load-bearing) is exactly what we observe.

---

### Gap 4 — No Action Smoothing or Torque Penalty

**What we have.** No term in the reward penalizes large actions, action changes between steps, or total torque energy. Actions are clipped to ±1 before being passed to `data.ctrl`, and the arm motors have fixed gear ratios (15/15/10) producing a fixed force ceiling, but no per-step cost discourages jitter.

**Standard practice.** MuJoCo locomotion environments in the Gymnasium suite (HalfCheetah, Ant, Hopper, Walker2d) all include a `ctrl_cost` term proportional to the sum of squared actions, typically weighted at 0.1. The practical effect is to discourage high-frequency torque oscillation, which produces jittery, energetically wasteful policies that are harder to transfer to real hardware and easier for the critic to over-exploit. Some works add an explicit action-difference penalty (`α * ||a_t - a_{t-1}||^2`) to enforce temporal smoothness, particularly relevant when training policies that will run on real motors that cannot tolerate high-frequency command reversals.

**Why this matters here.** The creature's arms are acting as paddles — the locomotion primitive is a low-frequency swing. Jitter in the arm joints (rapid oscillation around a mean position) does not displace the body but consumes the policy's action bandwidth and creates contact-force noise in the physics. An action-smoothing penalty would bias SAC toward the low-frequency periodic solutions that produce actual locomotion rather than high-frequency twitching that looks like locomotion to the critic but does not translate to body motion. This is a secondary concern relative to Gaps 1–3, but it is a free fix.

**Confidence:** Medium. Jitter has not been explicitly measured in our runs, so this is inference from general practice rather than direct diagnosis of a known symptom.

---

### Gap 5 — Algorithm Choice: SAC vs. PPO at Scale

**What we have.** SAC, off-policy, 1 (Stage 1) or 16 (follow-on) parallel environments. Total training budget: 250K–500K steps per stage, occasionally extended to 1M. Wall-clock time on M-series Mac: approximately 20–60 minutes per run depending on stage.

**Standard practice in the locomotion community.** The field bifurcated around 2021. For GPU-parallel simulation (IsaacGym/Isaac Lab, MJX), PPO with 4 096–16 384 parallel environments is the dominant choice: it achieves minute-scale convergence for complex quadruped and biped gaits [Hwangbo et al. 2019, Kumar et al. 2021, ETH Zurich legged_gym]. The reason is throughput: PPO is on-policy (no replay buffer management overhead), trivially parallelizes over env workers, and its sample complexity disadvantage vs. SAC is overwhelmed by sheer throughput when you can run 10 000 envs at once.

SAC remains the dominant choice for single-env or small-parallel CPU-based setups and for tasks where sample efficiency matters (e.g., real-robot training). Our setup — 16 envs on CPU + MPS — is exactly the regime where SAC is appropriate. The gap is not "use PPO instead of SAC." The gap is:

1. **We are running SAC at a scale (16 envs) that is well below the parallel-env count where SAC's off-policy advantage is maximized.** Antonin Raffin (SB3 author) documented extensively in 2023 that getting SAC to work on massive parallel simulators requires specific modifications (replay buffer scaling, gradient step scaling); at 16 envs we should be fine, but at 16 envs we are also not getting much of the throughput advantage of having 16 envs vs 1.

2. **If we moved to MJX (see Gap 8), PPO would become the correct algorithm.** MJX can run thousands of parallel MuJoCo instances on GPU; at that scale PPO dominates and SAC's replay buffer becomes a bottleneck.

**Confidence:** Medium-low for "switch to PPO now." High for "SAC + 16 envs is not the bottleneck; the bottleneck is the observation/reward/architecture issues above."

---

### Gap 6 — SAC-Specific Hyperparameter Misconfiguration

**What we have.** `buffer_size=200_000`, `learning_starts=5000`, `gradient_steps=1` for Stage 1 single-env; `gradient_steps=N_ENVS_FOLLOWON` (16) for follow-on (`train_v8.py:276-288`, `train_v8.py:476-483`).

**Standard SAC practice for MuJoCo locomotion.** The SB3 documentation and RL Zoo recommendations for MuJoCo locomotion environments use `buffer_size=1_000_000` (5× what we have), `learning_starts=10_000` (2× what we have). The smaller buffer means old, less-relevant transitions are evicted faster — less diversity in gradient updates. At 200K total training steps (Stage 1) with `buffer_size=200_000`, the buffer fills to capacity and then becomes a pure FIFO; we never have diverse long-term experience for the critic.

The `gradient_steps=16` in follow-on means: for each env step across 16 workers (= 16 env steps in wall time), we do 16 gradient steps. This is a 16:1 update-to-data ratio, which is above the standard 1:1 and can cause instability (critic divergence, Q-value overestimation) without a correspondingly large batch size or a lower learning rate. The learning rate is already reduced to 1e-4 for follow-on (Stage 1 uses 3e-4), but the gradient_steps scaling is the more direct risk.

**Target entropy.** `ent_coef="auto"` is correctly used for auto-tuning. The default target entropy is `-dim(action_space) = -8`, which is SB3's default. For an 8-DOF system where the head pan/tilt commands are effectively position-controlled servos (not torques), the head DOFs may be entropically "free" — the arm DOFs do all the locomotion work. Setting target entropy to `-6` (arms only) might produce tighter arm policies without crushing head exploration, but this is speculative.

**Confidence for buffer size:** High (200K buffer is too small by standard practice). Confidence for gradient_steps instability: Medium (this is a known risk but our learning rate reduction may compensate).

---

### Gap 7 — Curriculum Is Manual; No Performance-Triggered Adaptation

**What we have.** Radius anneal callbacks driven by fixed step counts chosen before training begins. The schedule does not adapt to whether the agent is actually succeeding (`train_v8.py:66-139`). Stage 0 → Stage 1 is a single manual warm-start.

**Standard practice.** Automatic curriculum learning (ACL) adjusts task difficulty based on agent performance. The simplest implementation is a threshold rule: if touch_rate > X over the last N episodes, increase radius by ΔR. More sophisticated versions use Absolute Learning Progress (ALP-GMM), RIAC, or regret-based automatic curricula [Portelas et al. 2020, Robust-PLR 2021]. MuJoCo Playground (2025) implements this as a standard component.

**Why this matters here.** The PROJECT_STATUS.md notes: "7/20 via forward-paddle + attract-reward is a viable non-vision strategy, so vision has no need to develop useful features." A performance-triggered curriculum would not advance the radius until the agent is reliably touching the ball — ensuring that locomotion is truly solved before the task gets harder. Our manual schedule advances radius by time, not by competence, which means the agent is often being asked to reach farther before it can reliably reach at all.

**Confidence:** Medium. This is a process fix, not a mechanism fix. It would make the existing curriculum more reliable but would not unblock vision on its own.

---

### Gap 8 — No Domain Randomization

**What we have.** Fixed physics across all episodes: same mass, same friction, same motor gains, same observation noise level (zero).

**Standard practice for locomotion.** Domain randomization over mass (±20%), joint friction (±50%), motor gain (±30%), and additive observation noise is standard in sim-to-real transfer literature [Tobin et al. 2017, Andrychowicz et al. 2020] and has become common even for sim-only experiments because it prevents the policy from memorizing fixed physics artifacts. MuJoCo Playground (2025) applies domain randomization as part of its default training recipe.

**Why this matters in our context.** Without observation noise, the policy can overfit to precise sensor values that would not be available in a real deployment and that may not generalize across physics variants. More relevantly: our late-training regression pattern (peak at 30–40% of training, decay thereafter) is consistent with overfitting to a narrow region of state space. Adding small observation noise acts as a regularizer that can slow this collapse. This is not the primary bottleneck but is a free robustness improvement.

**Confidence:** Low-medium for "this is causing the regression." Medium for "this would prevent it from getting worse and improve generalization."

---

### Gap 9 — Throughput: No GPU-Accelerated Batch Simulation

**What we have.** CPU MuJoCo via Python `mujoco` bindings, 16 parallel subprocess workers (`SubprocVecEnv`). Each worker runs one instance. Estimated 100–400 FPS in follow-on with vision rendering.

**State of the art.** MJX (MuJoCo XLA) runs the same MuJoCo physics as JAX JIT-compiled operations on GPU or TPU, achieving millions of steps per second for simple environments. MuJoCo Playground (February 2025) demonstrates full humanoid locomotion tasks at 1M+ FPS on modern GPUs, using PPO with 4 096 parallel envs. On our M-series Mac, MJX on the MPS backend provides 10×–100× speedup depending on scene complexity. The primary limit is that MJX currently has incomplete support for contact-heavy scenes with many geoms (the wheel ball-joints and platform contact stack in our XML may stress this limit), but simpler versions of the env would work.

**Relevance.** At 250 FPS × 16 envs = 4 000 env-steps/second, a 250K-step run takes ~62 seconds of wall clock. Iteration time is not the bottleneck. The bottleneck is iteration quality (see Gaps 1–4). Moving to MJX would change the algorithmic choice (PPO becomes dominant at that scale), requiring a larger architectural change. **This gap is real but not the root cause of the breakthrough-collapse pattern.** At our scale, SAC + CPU is appropriate; MJX becomes relevant when we have validated the reward/observation/architecture design and want to run 10× longer experiments cheaply.

**Confidence that this is the priority:** Low. Fix Gaps 1–4 first.

---

## 3. Ranked Recommendations

The following ranking is by expected impact on the specific symptom: *vision never becomes load-bearing; runs that look promising at Stage 1 fail to transfer to follow-on*.

### Rank 1 — Add a CNN Encoder for Pixel Inputs (Gap 3)
**Impact:** Highest. The current flat-MLP treatment of 3072 raw pixel values is architecturally incapable of efficiently learning "where is the red ball." A standard 4-layer CNN (as in DrQ [Yarats 2021]) would take the 32×32×3 image to a 50-dim latent before the MLP sees it. The pixel columns in the first MLP layer would then receive a meaningful 50-dim signal rather than 3072 individually nearly-useless scalars. This is not in conflict with the frozen-proprio / consistency-loss architecture: freeze the proprio columns of the first MLP layer; let the CNN and the pixel-column slice of the first MLP layer train. The consistency loss (`MSE(h_full, h_blind)`) still applies identically.

Implementation: use SB3's `CnnPolicy` with a custom feature extractor that encodes pixels through a small CNN and concatenates the resulting latent with the raw proprio vector before the shared MLP trunk. The frozen-proprio hook applies to the proprio-dimension slice of the combined embedding, as before.

**Confidence:** High. This is the single clearest mismatch between problem structure and architecture.

### Rank 2 — Apply MirrorWrapper in Follow-On (Gap 1)
**Impact:** High. The lateralization attractor documented in Stage 1 does not disappear when follow-on begins. The warm-start policy already has a hemispheric bias, and the 16 follow-on envs have no symmetry pressure. Wrapping the follow-on `SubprocVecEnv` with `MirrorWrapper` requires a two-line change to `_make_followon_env` (add the wrapper). The wrapper is tested and correct.

**Confidence:** High. This was identified as a gap even by the model that introduced the wrapper ("should have been there from the start"). The effort is minimal; the risk of not doing it is the documented hemispheric collapse.

### Rank 3 — Add VecNormalize (Gap 2)
**Impact:** High, especially in combination with the CNN encoder fix. The proprio/pixel scale mismatch is a systematic headwind on every follow-on run. Adding `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` around the `SubprocVecEnv` is a one-line change. The main operational caution: VecNormalize statistics (running mean/std) must be saved alongside the model checkpoint and loaded with it at eval/render time, or the eval observations will be denormalized relative to what the model expects. This is a known SB3 gotcha with a documented solution (save `vec_normalize.pkl` alongside `model.zip`).

**Confidence:** High for "this is absent and it should not be." Medium for "this alone will make vision load-bearing" — the CNN encoder fix is likely needed in conjunction.

### Rank 4 — Increase Replay Buffer to 1 000 000 (Gap 6)
**Impact:** Medium. At 250K steps with `buffer_size=200_000`, the buffer is full by step 200K and evicts early transitions before they can be revisited. Increasing to 1M means the buffer never fills during a 250K run, preserving diversity across the full training history. RAM cost: approximately 1M × obs_dim × 4 bytes; for obs_dim = 3101 (29 + 3072), this is ~12 GB for the full follow-on obs — potentially too large. A reasonable compromise is 500K with `batch_size=512` (more diverse batches from a larger buffer). For Stage 1 (obs_dim=29), a 1M buffer is ~29 MB — trivially feasible.

**Confidence:** High that buffer is too small. Medium on exact size target given RAM constraints.

### Rank 5 — Terminate Early on Excessive Tilt (Gap 7/health criterion)
**Impact:** Medium. The current early termination is fall-off-platform only. An episode where the creature is tilted 45° and oscillating without producing locomotion will run to the 300-step truncation, generating 300 steps of noisy data from a non-representative state. Standard locomotion environments terminate when the torso z drops below a healthy range or when the tilt angle (angle between the up-vector and the world z) exceeds ~60°. Adding `is_healthy = (tilt_angle < threshold)` to the env's step logic, and terminating on `not is_healthy`, would: (a) cut bad episodes short so the buffer is not filled with useless transitions, and (b) provide an implicit pressure against falling over that reinforces the platform design. The v8 platform creature almost never falls off (`0/20` in recent runs), but it may be spending time in degenerate tilted states that are being truncated rather than terminated.

**Confidence:** Medium. This is standard in biped/quadruped tasks but may be redundant here given the stable wheel-body design. Worth adding as a lightweight guard.

---

## 4. MuJoCo Features and Tools We Are Not Using That We Should Consider

Each item below is available in the XML or MuJoCo Python API with minimal effort, and each has a specific rationale for the alien_baby setup.

**`<sensor type="framelinvel">` and `<sensor type="frameangvel">` on the torso body.**
We read torso velocity directly from `data.qvel[0:3]` and `data.qvel[3:6]` (`platform_creature_env.py:247-249`). Declaring these as named sensors in the XML means they appear in `data.sensordata` alongside the joint sensors, making the proprio vector fully sensor-backed. This matters for reproducibility (sensor values can be noise-filtered via `sensor/cutoff`) and for MJX compatibility (MJX reads declared sensors; direct qvel indexing may break in XLA-compiled forward passes). Declaring the sensors also self-documents the observation layout in the XML rather than requiring code inspection.

**`<sensor type="accelerometer">` on the torso.**
An accelerometer site on the torso would provide gravity-referenced linear acceleration, which contains information about tilt that the quaternion provides but in a form that may be easier for the MLP to exploit (the accelerometer reads (0, 0, −g) when upright, deviates as the body tilts). This is a non-redundant observation for tilt detection and is how legged robots sense orientation in practice.

**`actuatorfrc` sensor or `forcerange` on arm actuators.**
Adding `<sensor type="actuatorfrc" name="lsp_force" actuator="left_shoulder_pitch_motor"/>` gives the policy direct readout of the torque it is actually producing (which differs from the ctrl command due to joint velocity backEMF effects). Standard in manipulation tasks. More conservatively, adding `forcerange="-X X"` to each motor actuator sets a hard force ceiling in the physics, preventing extremely high-torque configurations that the `gear` parameter alone does not limit.

**`<option cone="elliptic"/>` (already in v9, missing from v8).**
The v9 XML has `cone="elliptic"` (`platform_creature_v9.xml:3`); v8 uses the default pyramidal cone. Elliptic friction cones are physically more accurate and produce smoother gradients for the wheel contact dynamics. If any v8-XML runs are still launched (Stage 1 uses v8 by default when `--v9` is not passed), they have inferior contact physics.

**`<option integrator="RK4"/>` vs `integrator="Euler"` inconsistency.**
v8 uses RK4; v9 uses Euler (`platform_creature.xml:3`, `platform_creature_v9.xml:3`). RK4 is more accurate for fast dynamics (e.g., arm swing) at the cost of 4 function evaluations per timestep vs. 1 for Euler. The change was undocumented. If Euler was adopted in v9 for speed, the consequence should be checked: with a 0.01 s timestep, Euler is generally stable for the torques in use, but the behavior under RK4 and Euler may differ enough that a policy trained with one will behave differently under the other.

**`<default class>` for arm actuators to share parameters.**
All arm motors share the same gear ratios and ctrl ranges. Declaring a `<default class="arm_motor">` and referencing it reduces XML verbosity and ensures consistent parameter updates. Not a physics gap — purely an engineering hygiene item that reduces the risk of a future parameter change being applied to only some joints.

**`checkerboard` texture on the platform as a visual landmark (v9 removed it).**
The v9 ARCHITECTURE comment notes that the checker texture was removed because the policy was attending to the walls/texture rather than the ball (`platform_creature_v9.xml:28–31` comments). This is the correct fix. A uniform gray platform with a distinctly colored ball (bright red) gives the CNN one learnable feature (red pixel cluster) rather than many (texture edges, wall colors). This design choice is already in v9; mentioning it here confirms it should not be reverted.

**MuJoCo `mjvOption.geomgroup` for render-only geoms.**
Several geoms (wheel_fl_mark, edge markers, face_white, eye_L, eye_R, hair_back) are `contype="0" conaffinity="0"` — physics-invisible but visually present. These geoms appear in the head_cam observation. For a 32×32 camera image, the wheel marker yellow capsules and the red edge markers are visually distinct features at the resolution of the camera. If the CNN learns to key on these decorative elements rather than on the ball, it will have learned a spurious correlation. Using `mjvOption.geomgroup` to disable these decorative geoms from the agent's camera rendering (while keeping them in the experimenter's render) would ensure the head_cam only shows task-relevant features.

**`mujoco.mj_jacSite()` for end-effector Jacobian.**
Not a sensor but a physics API call that provides the Jacobian mapping joint velocities to hand Cartesian velocity. Exposing the hand's Cartesian velocity in the observation (derived from the Jacobian and joint velocities) would give the policy a more direct reach-velocity signal than it can infer from joint positions and velocities alone. This is the kind of observation engineering common in manipulation environments (e.g., FetchReach in OpenAI Gym). Given that the task is "move a hand to a ball," this is a potentially high-value observation with no additional physics cost.

---

## 5. Summary Table

| Gap | Severity | Effort | Fix |
|---|---|---|---|
| No CNN encoder for pixels | Critical | Medium | SB3 CnnPolicy with custom feature extractor |
| MirrorWrapper missing from follow-on | High | Trivial | Two lines in `_make_followon_env` |
| No VecNormalize | High | Low | Wrap `SubprocVecEnv` with `VecNormalize` |
| Replay buffer too small (200K vs 1M) | Medium | Trivial | `buffer_size=500_000` or `1_000_000` |
| No health-based early termination | Medium | Low | Add tilt threshold to `step()` |
| No action smoothing / torque penalty | Low-Medium | Low | Add `ctrl_cost` term to reward |
| No domain randomization | Low-Medium | Medium | Randomize mass/friction at reset |
| No performance-triggered curriculum | Low-Medium | Medium | Touch-rate threshold in callback |
| Inconsistent integrator v8/v9 | Low | Trivial | Unify to Euler in both XMLs |
| Decorative geoms visible in head_cam | Low | Low | Filter geomgroup in agent camera |

---

## See also

- **`PRIOR_ART.md`** — the outward-facing companion. Surveys what the locomotion + motor-learning + perception-theory literature has actually solved, and where Taylor's specific Π framing fits (or doesn't) in contemporary computational work. The "what 'solved' means" section there is the right context for understanding why fixing the gaps below is *necessary but not sufficient* for the project's research goal.
- **`PROJECT_STATUS.md`** — the empirical ground truth this analysis is responding to. The vision-ablation sensitivity numbers (FROZEN 2.29, UNFROZEN 0.85) directly motivate Gap 3 (CNN encoder): high sensitivity means pixels are being read; low utility means they aren't being read *productively*, which is exactly the architectural symptom of "no spatial encoder before the MLP."

---

## References

[Abdolhosseini 2019] Abdolhosseini, F., Ling, H.Y., Xie, Z., Peng, X.B., van de Panne, M. (2019). On Learning Symmetric Locomotion. *Proc. ACM SIGGRAPH MIG 2019*. https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/

[Andrychowicz 2020] Andrychowicz, O.M., et al. (2020). Learning dexterous in-hand manipulation. *Int. J. Robotics Research*.

[ETH legged_gym] Rudin, N., et al. (2021). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. *CoRL 2022*. https://github.com/leggedrobotics/legged_gym

[Haarnoja 2018] Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *ICML 2018*.

[Haarnoja 2019] Haarnoja, T., et al. (2019). Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905.

[MuJoCo Playground 2025] Google DeepMind. (2025). MuJoCo Playground. arXiv:2502.08844. https://playground.mujoco.org/

[MS-PPO 2024] Anonymous. (2024). MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion. arXiv:2512.00727.

[Ng 1999] Ng, A.Y., Harada, D., Russell, S. (1999). Policy Invariance Under Reward Transformations. *ICML 1999*. (PBRS, already used in our codebase.)

[Ordonez-Apraez 2025] Ordonez-Apraez, D., et al. (2025). Morphological symmetries in robotics. *Int. J. Robotics Research*. https://iit-dlslab.github.io/papers/ordonez2025ijrr.pdf

[Raffin 2023] Raffin, A. (2023). Getting SAC to Work on a Massive Parallel Simulator. https://araffin.github.io/post/sac-massive-sim/

[SB3 VecNormalize] Stable Baselines3 documentation. VecNormalize. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

[Tobin 2017] Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS 2017*.

[Yarats 2021] Yarats, D., et al. (2021). Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning (DrQ-v2). arXiv:2107.09645. https://github.com/facebookresearch/drqv2

[Yu 2018] Yu, W., Turk, G., Liu, C.K. (2018). Learning Symmetric and Low-Energy Locomotion. *SIGGRAPH 2018*.
