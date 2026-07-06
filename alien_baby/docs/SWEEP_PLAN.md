# Autonomous Trial Sweep — Plan

*April 2026 — proposal only, awaiting greenlight before implementation*

Companion to `PRIOR_ART.md`, `GAP_ANALYSIS.md`, `PRIOR_ART_NUMENTA.md`. Reads those for justification of any specific technique mentioned below; this doc is the *strategy and execution mechanics*, not a re-do of the literature.

---

## Goal

Run a structured, automatic sequence of training experiments — most-likely-success first, working out toward less-likely — that systematically tests the fixes identified in `GAP_ANALYSIS.md` and the Taylor-grounded pressure variants. Each run logs metrics, saves checkpoint videos, and writes a sweep-level results row. Long enough to run overnight unattended; short enough per-experiment that we can kill bad runs early.

## Design principles

- **Plumbing before novelty.** Phase 0 fixes (MirrorWrapper-in-follow-on, VecNormalize, replay buffer, health termination) ship as a baseline before any new science is run. These are "should have been doing this all along" items from GAP_ANALYSIS — not experimental.
- **One variable per experiment, then combinations.** Each Wave 1 experiment isolates one fix; Wave 2 combines the survivors with environmental pressure. We don't run 20-way grids; we run 3–5 experiments per wave, three seeds each.
- **Kill clearly-failing runs early.** Standard kill criterion: zero touches over last 5 evals at step ≥ 100K. Saves hours per dead branch.
- **Each experiment ties to a Taylor concept.** Plumbing fixes serve goal (1) — making the experiment *runnable*. Pressure variants serve goal (2) — vision-load-bearing in the Π sense. Listed explicitly per wave.
- **Stay in the existing stack.** Python, SB3, MuJoCo, the existing `train_v8.py` and `render_v8.py`. No Hydra, Ray Tune, or new framework. Add a thin sweep runner; do not rebuild training.
- **No commits without your review.** Each phase produces a branch + PR; you greenlight before merge.

---

## Phase 0 — Plumbing baseline (must ship before any sweep)

A single PR containing the GAP_ANALYSIS Tier-1 fixes. None of these are experiments; all are corrections.

| Fix | Source | Effort |
|---|---|---|
| Wire `MirrorWrapper` into `_make_followon_env` | GAP_ANALYSIS Gap 1 | 2 lines |
| Wrap follow-on `SubprocVecEnv` with `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` and persist `vec_normalize.pkl` alongside checkpoints | GAP_ANALYSIS Gap 2 | ~10 lines + checkpoint hook |
| Bump `buffer_size` from 200K → 500K (proprio-only Stage 1 unchanged) | GAP_ANALYSIS Gap 6 | 1 line |
| Add tilt-angle early termination (terminate when |tilt| > 60°) | GAP_ANALYSIS Gap 5 | ~5 lines in `platform_creature_env.py` |
| Add `ctrl_cost = 0.001 * sum(action²)` to reward | GAP_ANALYSIS Gap 4 | 2 lines |

**Validation before Wave 1**: re-run the existing follow-on baseline (current PROJECT_STATUS reference: 7/20 unfrozen) with these fixes applied. If it gets *worse* than 7/20, we have a regression to chase before launching the sweep. If it stays at or above 7/20, baseline is healthy and we proceed.

**Taylor justification**: Plumbing — these are corrections to what should already have been there. They don't test Taylor; they make the test runnable.

---

## Phase 1 — Sweep runner architecture

A single Python file: `alien_baby/sweeps/run_sweep.py`. ~200 lines, no dependencies beyond what's already installed.

### What it does

1. Reads a YAML sweep file (e.g., `sweeps/wave1.yaml`) defining a list of experiments.
2. For each experiment:
   - Creates `runs/<sweep_id>/<exp_id>/`
   - Launches `train_v8.py` as subprocess with the experiment's CLI args
   - Streams stdout to `runs/<sweep_id>/<exp_id>/train.log`
   - Reads TensorBoard scalars periodically; on each new checkpoint, calls `render_v8.py` to save `checkpoints/<step>.mp4`
   - Applies kill criterion (see below); on kill, terminates subprocess and records reason
3. After each run completes:
   - Runs a standardized eval producing `touch_rate`, `fall_rate`, `mean_speed`, `vision_ablation_sensitivity` over 50 episodes
   - Appends one row to `runs/<sweep_id>/results.csv`
   - Saves a final-checkpoint render
4. After all experiments: writes `RUN_REPORT_<sweep_id>.md` with a summary table + per-experiment notes + video links. Plays the done-sound.

### Kill criterion (per-experiment)

Killed early if **all** of:
- `step ≥ 100_000`
- `touch_rate` over last 5 evals == 0
- `tilt_angle_p95 > 30°` (creature is degenerate, not just learning slowly)

Or if explicitly: `train_v8.py` exits non-zero (crash).

### Sweep file schema (YAML)

```yaml
sweep_id: wave1
total_steps: 250_000
seeds: [42, 1337, 2024]
experiments:
  - id: 1A_mlp_baseline
    description: "Tier-0 plumbing only, MlpPolicy. Reference baseline."
    args: { policy: MlpPolicy, mirror: true, vecnorm: true }
  - id: 1B_cnn_unfrozen
    description: "DrQ-v2-style CNN encoder, unfrozen."
    args: { policy: CnnPolicy, mirror: true, vecnorm: true, frozen_proprio: false }
  - id: 1C_cnn_frozen
    description: "Same as 1B but frozen-proprio + consistency loss."
    args: { policy: CnnPolicy, mirror: true, vecnorm: true, frozen_proprio: true }
```

### What I am explicitly *not* building

- No hyperparameter optimizer (Bayesian, Optuna, etc.). Sweeps are hand-curated waves, not blind grids.
- No distributed orchestration. Experiments run serially on this Mac. With 16 envs/run and ~30 min/run, a 9-experiment wave (3 exp × 3 seeds) finishes in ~4.5 hours.
- No web dashboard. TensorBoard + the Markdown report is sufficient.

---

## Phase 2 — Wave structure

Each wave runs only after the previous wave completes and you've reviewed the report. Waves get progressively less likely to succeed; we stop early if Wave 1 or Wave 2 already produces vision-load-bearing behavior.

### Wave 1 — CNN encoder (most likely to fix vision-load-bearing)

The single biggest architectural mismatch from GAP_ANALYSIS. A flat MLP cannot efficiently learn "where is the red ball" from 3072 raw pixels; DrQ-v2's standard 4-conv → 50-dim latent is the off-the-shelf fix.

| Exp | What | Taylor justification |
|---|---|---|
| 1A | Plumbing-only baseline (`MlpPolicy`) | Control. Reference for all subsequent comparisons. |
| 1B | `CnnPolicy` with custom feature extractor (4 conv layers → 50d latent → concat with proprio → MLP head). Unfrozen. | Vision is *given the architectural means* to develop spatial features — necessary precondition for any equivalence-class formation that depends on visual structure. |
| 1C | Same as 1B but frozen-proprio (consistency loss `λ·MSE(h_full, h_blind)`) | "Protect proprio" principle — vision must *confirm* what proprio established. Tests whether forced compatibility produces or blocks Π. |

3 seeds × 3 experiments = 9 runs × 250K steps × ~30 min ≈ **4.5 hours**.

**Success criterion for the wave**: at least one experiment where `vision_ablation_sensitivity > 1.0` AND `touch_rate ≥ baseline (7/20)`. Means vision is being used *and* not at the cost of the existing competence.

**If Wave 1 succeeds**: the plumbing-and-architecture fix was sufficient. Stop. Write up. The Taylor test (vision-load-bearing) is now runnable on the v8 stationary-target env. Move to v9 to harden.

**If Wave 1 fails (vision still not load-bearing)**: blind paddling is still a viable strategy at 7/20. The architecture is fine; vision has no *need* to develop. Move to Wave 2 — apply pressure.

### Wave 2 — Pressure variants (vision becomes instrumentally necessary)

This is the "pressure not bribery" wave. Each environment is engineered so that blind strategies *fail*, forcing the policy to use vision.

| Exp | What | Taylor justification |
|---|---|---|
| 2A | v9 moving-target env (ball spawns with random velocity, rolls until caught or falls), best Wave 1 architecture | "M (forward-paddle) must FAIL unless modified by visual evidence of ball trajectory." (PROJECT_STATUS §v9.) Splits the too-broad equivalence class. |
| 2B | Visual-occlusion zones — half the platform blocks the head_cam (e.g., a wall-like fixture); ball can spawn either side. Best Wave 1 arch. | Forces gaze-and-position planning: the creature must *move* to disambiguate. Direct test of sensorimotor contingencies. |
| 2C | Multi-ball discrimination — red ball (reward), blue ball (penalty). Same env otherwise. Best Wave 1 arch. | Pure visual class. The proprio modality cannot distinguish; vision *is* the categorization. The cleanest possible test of "vision-derived equivalence class." |

3 seeds × 3 experiments = ~4.5 hours.

**Success criterion**: any experiment where `vision_ablation_sensitivity > 1.5` AND `touch_rate > random-policy floor`. The high sensitivity bar (vs. 1.0 in Wave 1) reflects that pressure should *force* vision use, not just permit it.

**If Wave 2 succeeds**: we have the experimental setup that operationalizes Π. Write up.

**If Wave 2 fails**: the architecture or the algorithm has a deeper limitation. Proceed to Wave 3.

### Wave 3 — Architectural alternatives (longer-shot)

If Waves 1 and 2 don't produce vision-load-bearing behavior, the limitation is structural and we explore alternatives drawn from PRIOR_ART and PRIOR_ART_NUMENTA.

| Exp | What | Source / justification |
|---|---|---|
| 3A | SDR bottleneck encoder (sparse activation, top-k = 2% of latent) between CNN and MLP. | PRIOR_ART_NUMENTA §4.1. SDRs are Numenta's principled answer to representation structure; tests whether sparse coding helps cross-modal binding. |
| 3B | Hierarchical CPG-PD low-level + RL high-level. Replace direct joint torque actions with paddle-stroke parameters; an embedded CPG executes the stroke. | PRIOR_ART §2 (Ijspeert). The locomotion literature's standard answer to "small action space." Reduces what the RL policy has to learn. |
| 3C | Domain randomization + observation noise (mass ±20%, friction ±50%, motor gain ±30%, obs Gaussian σ=0.05). | PRIOR_ART §1 (Hwangbo, Tobin). Standard regularization; tests whether overfitting to fixed physics is part of the late-training regression in PROJECT_STATUS. |

3 seeds × 3 experiments = ~4.5 hours.

### Wave 4 — Strategic platform shift (gated; only if Waves 1–3 plateau)

Not part of the standard sweep. A separate, larger decision: port the env to MuJoCo Playground (MJX). 100×–1000× iteration speed on the same MuJoCo physics, letting us run the entire sweep in minutes instead of hours and try ideas that are currently too expensive.

This is a 1–2 day port, not a sweep wave. Proposed only if Waves 1–3 fail to produce vision-load-bearing behavior — at that point, the question is whether we're under-trained, and MJX is the way to find out cheaply.

---

## Phase 3 — Local LLM cross-check (optional, gated by user)

After each wave, the runner emits a `WAVE_<n>_FOR_REVIEW.md` containing: the sweep config, the results CSV as a table, and per-experiment 3-line summaries. This file gets piped to local gemma 4 (via `ollama generate`) with a prompt asking for: (1) which experiment looks most promising and why; (2) what it would change about the wave design; (3) what it thinks the bottleneck is.

The runner then writes a final `WAVE_<n>_DECISION.md` with my recommendation, gemma's response, and a synthesized recommendation. You read both and authorize the next wave.

This is optional. If you don't want the gemma loop in the v1, the runner just writes the report and waits.

---

## Phase 4 — Reporting and persistence

End of each wave produces three artifacts in `runs/<sweep_id>/`:

- `results.csv` — one row per (experiment, seed)
- `RUN_REPORT_<sweep_id>.md` — narrative summary, video links, my recommendation for next wave
- `checkpoints/` — per-experiment per-step renders (MP4)

End of the full sweep produces a top-level `SWEEP_RESULTS.md` aggregating all waves.

All committed to a branch `feature/sweep-<sweep_id>` so we have a permanent record.

---

## Estimated cost

- Phase 0 implementation: ~2–3 hours of code work + 1 baseline validation run (~30 min).
- Phase 1 implementation: ~3–4 hours of code work for the runner.
- Wave 1: ~4.5 hours unattended.
- Wave 2: ~4.5 hours unattended (only if Wave 1 needs a follow-up).
- Wave 3: ~4.5 hours unattended (only if Wave 2 also needs follow-up).
- Wave 4 (MJX port): 1–2 days, only if needed.

Total before the first overnight sweep can launch: ~half a day of code work + ~half hour validation.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CnnPolicy's MPS implementation has known quirks for SB3 — could crash or be slow. | Quick smoke test in Phase 0 validation. If MPS is broken for CnnPolicy, fall back to CPU for Wave 1; still ~1–2 hours per run, acceptable. |
| Kill criterion too aggressive — might kill slow learners that would have converged. | Conservative thresholds (zero touches at 100K is a strong signal). Killed runs are recorded with reason; you can override and re-run any. |
| Disk usage from MP4 checkpoints. | Save 1 MP4 per 50K-step checkpoint; ~5 MP4s/run × 9 runs/wave = ~45 videos/wave. At ~5 MB each = ~225 MB. Trivial. |
| Sweep runner crashes mid-wave, losing intermediate state. | Each experiment writes its row to `results.csv` on completion; runner is restartable from where it left off (skips experiments whose row already exists). |
| One bad fix in Phase 0 silently regresses the baseline. | Run the baseline validation *before* launching Wave 1. If touch_rate drops below 5/20 with Phase 0 applied, we triage Phase 0 before sweeping. |

---

## What I need from you to proceed

Three explicit yes/no decisions:

1. **Greenlight Phase 0 (plumbing PR)** — apply the GAP_ANALYSIS Tier-1 fixes and validate baseline. No new science. Single PR for your review.
2. **Greenlight Phase 1 (runner)** — write `run_sweep.py` and the YAML sweep configs. Single PR for review. No experiments launched yet.
3. **Greenlight Wave 1 launch** — only after Phase 0 is merged and the runner is reviewed. This is the first overnight unattended sweep with `--dangerously-skip-permissions`.

Optional decision:
4. **Phase 3 (gemma cross-check)** — wire the local-LLM consult, or skip for v1?

I'll wait for explicit yes on (1) before any code changes.
