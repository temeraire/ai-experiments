# Phase I Results — MICOA β=0.1 (R36, 80K steps)

Run tag: `mimo_phase_i_R36_micoa_beta0.1`
Branch: `feature/phase-i-micoa` (unpushed)
Date: 2026-05-21 (started ~23:53 PDT, completed ~00:34 PDT, ~40 min wall-clock)
Commits on branch: 4 (see git log on the branch)

## TL;DR

**Architecture works mechanically; the corner forms; but vision is still not
load-bearing in behavior. β=0.1 was likely too high — encoders learned to
agree (KL → 0) too quickly, before either had learned anything useful from
its channel. Recommended next iteration: β ≈ 0.01–0.03.**

The auto-extend-to-250K trigger does NOT fire. The pre-approved criterion
was "sigma decreasing AND ablation delta > 0.05". We hit the first but
fell well short of the second (0.0019 actual, threshold 0.05). Per the
pre-approval, R36 stops at 80K and waits for human review.

---

## Diagnostic scalars

### sigma_combined (lower = corner forming)

| t      | sigma_combined | Δ% from t=8K |
|--------|----------------|--------------|
|  8K    | 0.7105         |  —           |
| 12K    | 0.6981         |  -1.7%       |
| 16K    | 0.6883         |  -3.1%       |
| 20K    | 0.6784         |  -4.5%       |
| 24K    | 0.6682         |  -5.9%       |
| 28K    | 0.6625         |  -6.8%       |
| 32K    | 0.6541         |  -7.9%       |
| 40K    | 0.6426         |  -9.5%       |
| 48K    | 0.6211         | -12.6%       |
| 56K    | 0.5667         | -20.2%       |
| 64K    | 0.5525         | -22.2%       |
| 72K    | 0.5170         | -27.2%       |
| 80K    | **0.4516**     | **-36.4%**   |

Monotone decreasing (with one wobble at 60K/68K). The PoE + symmetric-KL
mechanism is doing exactly what it was designed to do: each encoder is
landing on a tighter, more confident region of Z over training.

### kl_agreement (lower = encoders' distributions agree)

| t   | kl_agreement |
|-----|--------------|
|  8K | 0.813        |
| 12K | 0.035        |
| 16K | 0.006        |
| 20K | 0.001        |
| 24K | 0.002        |
| 80K | 0.001        |

KL **crashed in the first 4K steps** of training (0.81 → 0.001) and stayed
there. The encoders are nearly identical Gaussians for every observation.
This is the warning sign — see "β too high" below.

---

## Task performance (deterministic eval, 20 seeds, static balls, offset 0.15)

Using `final_model.zip` (80K) loaded via `MICOASAC.load` against the same
env config as training.

|                                | best_model (5K, pre-MICOA) | final_model (80K, post-MICOA) |
|--------------------------------|----------------------------|--------------------------------|
| mean_reward                    | +127.1                     | **-746.1**                     |
| both-touched                   | 18/20  (90%)               | **6/20 (30%)**                 |
| only-one-touched               | 2/20                       | 14/20                          |
| neither                        | 0/20                       | 0/20                           |

**The policy regressed badly.** A near-perfect static-ball policy (5K) was
turned into a half-broken one (80K). The KL agreement pressure dominated
the actor's learning signal: the encoders converged on a representation
the actor could no longer use effectively.

The 5K "best" model was saved before MICOA's KL pressure ramped up
(learning_starts=10K). After that, the actor's loss landscape was deformed
by the agreement step and the task policy degraded.

---

## Vision ablation — the headline test

Both numbers way below the >0.05 success threshold:

| metric                                  | value    | threshold | pass? |
|-----------------------------------------|----------|-----------|-------|
| Single-step action L2 delta (mean)      | 0.00195  | > 0.05    | ✗     |
| Episode both-touched, normal pixels     | 6/20     |  —        |       |
| Episode both-touched, pixels ZEROED     | 5/20     |  —        |       |
| Episode delta (normal − zeroed)         | +1       | > +4      | ✗     |

**Zeroing all 6144 pixel inputs at every step changes the agent's actions
by less than 0.2% of typical action magnitude.** The agent is not using
vision. The encoders agree (KL → 0), and the agreement object got tighter
over training (σ shrinking), but the agreement is on a representation the
policy ignores.

Compare to Phase H R33 and R35 (the closest "vision present" baselines):
both had ablation deltas around 0.02. MICOA's ablation delta (0.0019) is
*worse* — vision is even more inert here than in plain SAC + StereoCrawlerCNN.
That is a counterintuitive but coherent finding: forcing the two encoders to
output the same Gaussian early in training removed the natural pressure for
the vision encoder to encode anything pixel-specific. They both encode
"whatever proprio thinks" and the policy reads only proprio.

---

## Diagnosis: β=0.1 was too high

The KL plateau at ~0.001 from t=12K onward (out of ~10⁵ training steps)
is the smoking gun. The encoders collapsed to near-identical distributions
in the first 2K training steps of MICOA pressure — far faster than the
policy was learning to USE vision for anything. By the time the policy
started shaping its loss, vision was already a redundant copy of proprio,
not an independent constraint.

The σ_combined decrease (-36.4%) reflects the encoders becoming more
confident overall, not vision learning to confirm something proprio
already knows. It's both encoders being trained to be more confident
together — confidence in agreement, not in correct grounding.

This matches the design comment in `micoa_architecture.py`:

> beta=0.1 is the starting point. Diagnostic guidance:
>   kl_agreement drops fast → beta may be too high (forced agreement)

The pre-approved auto-tune sequence (0.1 → 0.3 → 1.0) was designed for
the opposite failure mode ("kl_agreement never moves → beta too low").
**Going to β=0.3 would be strictly worse**, so the overnight loop did NOT
auto-escalate. It stopped at β=0.1 and is leaving the call to the human.

---

## Recommended next iteration

1. **Lower β.** Try β ∈ {0.03, 0.01}. Watch the KL trajectory: ideally
   KL should drop, but slowly enough that the actor's task pressure
   shapes vision before the encoders collapse.
2. **Optional: ramp β.** Start with β=0 (no KL pressure) for the first
   30K steps, then ramp to a small β. This gives vision time to develop
   task-relevant features before the agreement pressure is applied.
3. **Optional: critic-only KL.** Currently the KL is applied to a
   separate optimizer over the shared extractor. Consider applying it
   ONLY when sigma_combined is large (gate the loss on σ > threshold)
   so it doesn't fire once agreement is already reached.

The Phase II anticipation idea (vision predicts proprio's *next* state)
is still worth pursuing, but Phase I needs to actually demonstrate the
corner is load-bearing first.

---

## What overnight did

- Wired MICOA into `train_crawler.py` (--micoa, --micoa-beta).
- Discovered and fixed 4 bugs (all in `OVERNIGHT_LOG.md`):
  1. SB3 logger.record without tensorboard_log writes nowhere visible.
  2. `policy.features_extractor` is None for SAC; correct path is via
     `policy.actor.features_extractor`. Affected both the callback AND
     `MICOASAC.train()` — the KL backward had been a silent no-op.
  3. `last_mu_*` stored from forward cannot be reused for backward
     (graph freed). Need a fresh forward inside train().
  4. SB3 `EvalCallback` save crashes on transient macOS FS timeouts
     (Spotlight/iCloud holding the file briefly).
- Switched MICOA policy to `share_features_extractor=True` (architecturally
  correct: one set of encoder weights, not two unrelated copies).
- Smoke-tested each fix (KL dropping monotonically across 400 steps).
- Ran R36 to completion (80K steps, ~40 min wall-clock).
- Ran touch eval + vision ablation eval — wrote this writeup.
- Did NOT extend to 250K (pre-approved criterion failed).
- Did NOT push or open a PR.

## Files

- `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1/final_model.zip` — 80K final
- `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1_best/best_model.zip` — 5K best (pre-MICOA)
- `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1.log` — full training log (~7K lines)
- `alien_baby/results/mimo_phase_i_R36_micoa_beta0.1_{noscalar,silentkl,savecrash}*` — archived failures of intermediate launches; can be deleted
- `alien_baby/OVERNIGHT_LOG.md` — turn-by-turn record of the overnight session
- `alien_baby/agents/micoa_architecture.py` — the architecture
- `alien_baby/visualization/eval_phase_i.py` — eval script (uses final_model)

## Visual render

After R36 completed I added `--vision` and `--memory-obs` flags to
`render_crawler_cart.py` and rendered two deterministic episodes
(seeds 0 and 1) from R36's `final_model.zip`. Gemini description of
seed 0: *"agent reaches but does not consistently contact the target;
blue ball ignored entirely; repeats the same half-reach motion."*
Matches the 6/20 both-touched eval.

---

# R37 — same config, β=0.03 (re-run on 2026-05-21 morning)

## Why R37

R36's diagnosis ("β=0.1 too high, KL collapsed before vision learned
anything") suggested lower β might preserve task performance. R37 is
R36 with one change: `--micoa-beta 0.03`. Same seed, same env, same
80K step budget.

## Headline numbers

|                                   | R36 (β=0.1) | R37 (β=0.03) | Δ              |
|-----------------------------------|-------------|--------------|----------------|
| sigma_combined end                | 0.4516      | 0.4634       | ~equal         |
| sigma trajectory                  | monotone    | non-monotone (0.40 → 0.46 wobble at 68K-80K) | R37 more data-responsive |
| KL final                          | 0.0014      | 0.0147       | R37 ~10× higher |
| KL noise                          | constant ~0.001 | oscillates 0.001–0.015 | R37 alive, R36 dead |
| Eval @ 80K mean_reward            | -835        | +121         | +956           |
| Eval @ 80K ep_length              | 1607        | 674          | -933           |
| Deterministic both-touched (20 seeds) | **6/20**   | **18/20**    | +12            |
| Vision ablation single-step delta | 0.00195     | 0.00225      | +0.0003        |
| Episode-level both, pixels normal | 6/20        | 18/20        | +12            |
| Episode-level both, pixels zeroed | 5/20        | 16/20        | +11            |
| Pixel-zero delta                  | +1          | +2           | +1             |

## What β=0.03 fixed

**Task performance is fully preserved.** R37 hits 18/20 both-touched —
matching the pre-MICOA 5K-best policy from R36 (also 18/20). The KL
pressure no longer destroys the actor's task representation. Reward
went from -835 to +121.

R37 seed-0 video, Gemini description: *"agent reaches with right arm
and grasps red sphere, then bends and reaches left toward blue sphere
and grasps it. Loses balance after acquiring both."* The fall-over
isn't a MICOA problem — it's the well-known cart-mode posture issue
from Phase H.

## What β=0.03 did NOT fix

**Vision is still not load-bearing.** Single-step ablation delta went
from 0.00195 → 0.00225 — a ~15% relative improvement, but still ~22×
below the 0.05 success threshold. Episode-level pixels-zeroed both-
touched is 16/20 vs 18/20 normal: zeroing every pixel changes outcomes
in only 2/20 episodes. The policy navigates by proprio + memory_obs.

## Diagnosis: confirmation ≠ anticipation

The pattern that emerges across R36 and R37:

- The PoE+KL mechanism *does* form a corner. σ shrinks; the two
  Gaussians overlap.
- But the corner forms on whatever proprio already encoded, because
  proprio gets a strong RL gradient (it directly maps to task-relevant
  body state) and vision does not.
- Vision learns to mirror proprio's encoding — the cheapest way to
  satisfy the KL pressure. Mirroring is parameter-free agreement,
  it does not require pixels to mean anything.
- Once vision mirrors proprio, ablation has no effect: the policy is
  already getting the same Z from proprio alone.

This is exactly the "anticipation" gap noted at the bottom of
`micoa_architecture.py`:

> Confirming what proprio already knows is not enough. Real visual
> utility is ANTICIPATORY: vision should predict contact before it
> happens, not just confirm it after.

Symmetric KL between (μ_p(t), σ_p(t)) and (μ_v(t), σ_v(t)) at the same
timestep has no asymmetry that would push vision to do anything proprio
can't already do. The architecture needs vision to be trained against
proprio's *future* state, not its present one.

## Verdict and next moves

R37 also fails the pre-approved success criterion (ablation > 0.05).
Two coherent next experiments, ranked:

1. **Temporal predictive loss (Phase II).** Train vision to encode now
   the Gaussian that proprio will encode one step ahead. This is the
   anticipation step from the original sketch. Requires storing
   (μ_v(t), σ_v(t)) alongside transitions in the replay buffer so the
   loss can be computed against (μ_p(t+1), σ_p(t+1)) sampled later.
2. **β-ramp** (cheaper, but unlikely to be sufficient). β=0 for the
   first 30K steps so vision learns whatever it can from the critic's
   gradient alone, then ramp β to 0.03. May give vision time to develop
   pixel-specific features before mirroring becomes the cheap option.
   Worth trying as a one-night experiment; (1) is the architecturally
   correct fix.

A third option — accepting that RL signal alone is insufficient and
introducing an explicit prediction loss (e.g. visual forward dynamics:
given current frame, predict next-step proprio) — is essentially (1)
expressed differently. Whichever framing makes the implementation
cleaner is the one to pick.

## Files (R37)

- `alien_baby/results/mimo_phase_i_R37_micoa_beta0.03/final_model.zip`
- `alien_baby/results/mimo_phase_i_R37_micoa_beta0.03.log`
- `alien_baby/results/videos/R37_micoa_beta0.03_final_seed{0,1}_off0.15.mp4`
- Eval JSON above; also re-runnable via
  `python -m alien_baby.visualization.eval_phase_i --run-tag mimo_phase_i_R37_micoa_beta0.03 --ball-speed 0.0 --offset 0.15`
