Here is a systematic, five-phase experimental plan. It moves from **plumbing fixes** (what "should have been there from day one") through **standard baselines** (proving the task is solvable) to **Taylor-specific architectures** (testing interpenetration) and finally **stress tests** (generalization under distortion).

Each phase includes: (1) the theoretical rationale in Taylor’s terms, (2) the exact implementation delta, (3) automated success criteria, and (4) the video/logging protocol.

------

## Executive Summary

| Phase | Goal              | Taylor Concept                | Key Fix / Addition                                    | Kill Criteria                           |
| ----- | ----------------- | ----------------------------- | ----------------------------------------------------- | --------------------------------------- |
| **0** | Stop the bleeding | *Equilibrium before learning* | MirrorWrapper in follow-on, VecNormalize, CNN encoder | Still lateralized; CKA < 0.5            |
| **1** | Prove solvability | *Affordance + drive*          | Standard SAC + CNN (no frozen layers)                 | < 70% touch rate after 200K steps       |
| **2** | Interpenetration  | *Π-manifold preservation*     | Frozen proprio + Consistency loss + CNN               | CKA < 0.95; Ablation sensitivity < 1.0  |
| **3** | Generalization    | *Equivalence class stress*    | Domain randomization + moving ball + spectacles       | Performance collapses > 50% under noise |
| **4** | Reference frames  | *Σ₄ self-generated frames*    | Sparser obs, thousand-brains voting (if prev succeed) | —                                       |

**Automation principle:** Each phase runs 3–5 seeds in parallel. A master script evaluates checkpoints every 50K steps against the criteria. If a seed passes, it renders a video and logs to `trials.db`; if all seeds fail the kill criteria, the phase halts for inspection.

------

## Phase 0: Plumbing Fixes (The "Should Have Been" Baseline)

**Taylor Rationale:** Before testing *interpenetration*, we must ensure the organism’s bilateral symmetry is respected. Taylor’s equivalence classes require symmetric response repertoires; training only one hemisphere destroys the class structure before vision ever arrives.

**Implementation:**

1. **MirrorWrapper in Follow-on** (`train_v8.py:174-184`):
   - Wrap `SubprocVecEnv` with `MirrorWrapper` in `_make_followon_env`, not just Stage 1.
   - *Diff:* Add `env = MirrorWrapper(env)` before returning in `_make_followon_env`.
2. **VecNormalize** (`GAP_ANALYSIS` Gap 2):
   - Wrap with `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`.
   - Save `vec_normalize.pkl` with checkpoints for eval.
3. **CNN Encoder** (`GAP_ANALYSIS` Gap 3 – Critical):
   - Replace `MlpPolicy` with a custom `CnnPolicy` that processes pixels through a 4-layer CNN (output 50-dim) before concatenating with proprio.
   - Crucially: freeze the proprio-processing layers of the *feature extractor*, not just the policy first layer. This maintains the "vision speaks through proprio’s manifold" architecture while allowing spatial feature learning.
4. **Hygiene fixes:**
   - Unify integrator to `Euler` in both XMLs (GAP table item 9).
   - Increase replay buffer to 1M (GAP 6).

**Success Criteria (automated):**

- Run `spawn_hemisphere_probe.py` on Stage 1 and Follow-on checkpoints.
- **Pass:** `touched_left_frac` and `touched_right_frac` both > 40% (symmetric).
- **Kill:** Either hemisphere < 10% (lateralization attractor still present).

**Video Protocol:** Render one left-spawn and one right-spawn episode at Stage 1 best checkpoint. Side-by-side in `phase0_symmetry_check.mp4`.

------

## Phase 1: Standard Baseline (Proving the "Plumbing Goal")

**Taylor Rationale:** Before asking whether vision *interpenetrates* proprio, we must verify that the environment actually affords the behavior—that the sensory-motor system *can* form the equivalence class "ball → reach" given sufficient sensory input. This is the "world of instances" baseline.

**Implementation:**

- **Algorithm:** Standard SAC (no frozen layers, no consistency loss).
- **Architecture:** CNN encoder → 256-dim MLP (full weights trainable).
- **MirrorWrapper:** Active (per Phase 0 fix).
- **VecNormalize:** Active.
- **Reward:** Existing shaped reward (hunger + attract + contact + fall).
- **Training:** 500K steps, 16 parallel envs.

**Success Criteria:**

- **Touch rate:** > 80% on evaluation (20 episodes).
- **Fall rate:** < 5%.
- **Mean time-to-contact:** < 100 steps (from spawn).

**Kill Criteria:**

- < 70% touch rate after 500K steps → Task is harder than Reacher; investigate reward shaping or physics (friction too high?).

**Video Protocol:** Render `phase1_standard_success.mp4` showing the creature locating and touching the ball efficiently using vision.

------

## Phase 2: Taylor-Compliant Interpenetration (The Research Question)

**Taylor Rationale:** This tests whether vision arrives as a *follow-on* that confirms proprio’s existing categories, rather than a parallel channel. The consistency loss enforces that vision’s contribution lives on the proprio manifold (high CKA), while the CNN encoder ensures vision is actually *processing* spatial information (unlike the flat-MLP failure in v8).

**Implementation:**

- **Base:** Phase 1 architecture (CNN + Mirror + VecNormalize).
- **Constraint 1 (Frozen Proprio):** Freeze all weights in the feature extractor and policy that process proprio dimensions (first 29 inputs). Gradient mask at extractor level.
- **Constraint 2 (Consistency Loss):** λ = 0.1, compute MSE between `h_full` (proprio + vision) and `h_blind` (proprio-only) at the first hidden layer after concatenation.
- **Constraint 3 (Warm Start):** Initialize from Phase 1 checkpoint (proprio already knows how to reach).

**Success Criteria:**

- **CKA(Stage 1, Follow-on)** on proprio-only inputs: > 0.95 (interpenetration, not obliteration).
- **Vision-ablation sensitivity:** > 1.0 (L2 norm of action delta when pixels zeroed).
- **Touch rate:** > 75% (slight drop acceptable due to constraints).

**Kill Criteria:**

- CKA < 0.8 → Vision is rewriting proprio; consistency loss too weak or frozen mask broken.
- Ablation sensitivity < 0.2 → Vision not load-bearing; CNN not learning.

**Video Protocol:** Generate `phase2_ablation_demo.mp4`: left panel sees ball, right panel pixels blacked out, showing divergent policies only where vision is necessary.

------

## Phase 3: Stress Testing (The "Spectacles" Experiment)

**Taylor Rationale:** Experiment III (Papert’s spectacles): when the visual field is systematically distorted, the system must reorganize *through motor success*, not recalibrate via visual correction. We simulate "reversing spectacles" by flipping the camera image horizontally and providing a context bit (spectacles on/off).

**Implementation:**

- **Distortion:** 50% of episodes apply `np.fliplr` to the camera observation; add binary input to observation (spectacles flag).
- **Dynamics:** Ball moves with velocity decay (v9 physics) to prevent blind-paddle solutions.
- **Curriculum:** Start with static ball (Phase 2 behavior), anneal to moving ball over 200K steps.
- **Domain Randomization:** Randomize ball color, light position, floor friction ±20% (standard sim-to-real hygiene).

**Success Criteria:**

- **Cross-modal transfer:** After spectacles training, `vision_ablation_sensitivity` with spectacles ON equals sensitivity with spectacles OFF (within 10%).
- **Task success:** > 60% touch rate with moving ball under distortion.

**Kill Criteria:**

- < 40% success with moving ball → Interpenetration is fragile; need more robust consistency loss or longer Stage 1.

**Video Protocol:** `phase3_spectacles.mp4`: split-screen showing the flipped visual input vs. the correct reaching behavior (reaching to the physical left when the flipped image shows it on the right).

------

## Phase 4: Reference Frames (Numenta Extension)

**Taylor Rationale:** Taylor’s Σ₄ (self-generated reference frame) emerges from trunk mobility. Numenta’s Thousand Brains asks whether reference frames are encoded locally in columns. This phase tests if the creature has learned an object-centered reference frame by evaluating whether its policy generalizes to novel ball *orientations* (not just positions).

**Implementation (Conditional):**

- Only if Phase 3 succeeds.
- Replace ball with cube (different affordances).
- Add "Greek Room" obstacles (columns) requiring the creature to navigate around occlusions—testing if the learned representation is allocentric vs. egocentric.
- Optional: Replace SAC with a Hebbian one-shot learner (Monty-style) for comparison, or add a second "column" (parallel policy head) that votes with the first—testing the synchronic-vs-diachronic distinction identified in `PRIOR_ART_NUMENTA`.

**Success Criteria:**

- Zero-shot transfer to cube > 50% touch rate without retraining (generalization of grasping equivalence class).
- In Greek Room, gaze behavior emerges (head tracks ball around columns before reaching).

------

## Automation Framework (`auto_trial_runner.py`)

To run unsupervised, implement:

```python
PHASES = [
    {"name": "plumbing", "script": "train_phase0.py", "criteria": {"symmetry": 0.4}},
    {"name": "standard", "script": "train_phase1.py", "criteria": {"touch_rate": 0.8}},
    {"name": "taylor", "script": "train_phase2.py", "criteria": {"cka": 0.95, "ablation": 1.0}},
    {"name": "stress", "script": "train_phase3.py", "criteria": {"moving_success": 0.6}},
]

for phase in PHASES:
    for seed in [1, 2, 3]:
        run_training(phase, seed, checkpoint_freq=50000)
        if not check_criteria(phase, seed):
            log_failure(phase, seed)
            break  # Halt phase for human review
    render_video(phase, best_seed)
    log_to_db(phase, metrics)
```

**Logging:** Every checkpoint saves:

- `cka_score.pkl`
- `hemisphere_balance.json`
- `ablation_sensitivity.json`
- `video.mp4` (overhead + head-cam split-screen)

**Decision Rules:**

- If Phase 0 fails → Fix XML/mirror_wrapper, do not proceed.
- If Phase 1 fails → Task is poorly specified; revisit reward shaping (add hunger penalty from GAP 7).
- If Phase 2 fails → Reduce CNN capacity or increase λ consistency (vision overpowering proprio).
- If Phase 3 fails → Expected; document generalization limits. Phase 4 is optional research.

------

## Relation to Prior Art

This plan directly addresses your three literature questions:

1. **Locomotion-to-contact solved?** Phase 1 uses the standard recipe from `PRIOR_ART` (SAC + CNN + dense reward + symmetry). If Phase 1 fails, the issue is task specification, not algorithmic novelty.
2. **Bilateral symmetry?** Phase 0 implements the MirrorWrapper as "table stakes" per Abdolhosseini et al. 2019 and the MS-PPO 2024 work on morphological symmetry.
3. **Taylor tested elsewhere?** No prior computational test of Π interpenetration via load-bearing-vision exists (per `PRIOR_ART` §3). Phase 2 is the novel contribution; Phase 3 tests the critical "spectacles" generalization that distinguishes Taylor feature-fusion.

The "breakthrough-then-collapse" cycle (Section 9) is broken by strictly ordering interventions: **plumbing first** (fix architecture), **then baseline** (prove solvability), **then theory** (constrain learning). Previously, you were trying to test interpenetration on a plumbing-leaky system (flat MLP, no normalization, asymmetric training), which confounds architectural failure with theoretical failure.

**Next step:** Implement Phase 0 (the three line fixes: MirrorWrapper, VecNormalize, CNN encoder skeleton). Once those three files are committed, the automated runner can execute the sequence without further permission.