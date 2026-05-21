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

---

# R38 — Phase II: pure temporal predictive loss

## Config

Same env as R36/R37. `--micoa-beta 0.0 --micoa-pred-beta 0.1`. Vision is
pulled toward `proprio(t+1)` with proprio detached. No same-time
symmetric KL. 80K steps, seed 42.

## Headline result: the ablation barrier broke

|                                  | R36 (β_sym=0.1) | R37 (β_sym=0.03) | **R38 (β_pred=0.1)** | threshold |
|----------------------------------|----------------|-------------------|----------------------|-----------|
| **Vision ablation L2 delta**     | 0.0019         | 0.0023            | **0.6563**           | > 0.05    |
| Episode both-touched, normal     | 6/20           | 18/20             | 6/20                 | —         |
| Episode both-touched, pixels zeroed | 5/20         | 16/20             | 6/20                 | —         |
| Episode-level pixel-zero delta   | +1             | +2                | **+0**               | > +4      |
| Eval @ 80K mean_reward           | -835           | +121              | -757                 | —         |

**For the first time across 9 vision experiments, the single-step ablation delta
is non-trivial.** 0.66 is 350× higher than R36/R37 and 13× past the
pre-approved 0.05 threshold. Vision is genuinely encoding information
that changes the agent's actions, step by step.

## But — active is not the same as productive

The episode-level outcome is unchanged when pixels are zeroed (6/20
both-touched either way). The single-step deltas don't accumulate into
better task performance. Vision is **active** (it moves the actions)
but not **productive** (it doesn't move the *outcomes*).

This is a real, novel failure mode worth naming. Across the entire
Phase G/H/I (R36/R37) history, vision was *silent* — actions were the
same with or without it. R38 made vision *loud* without making it
useful. The KL pressure successfully forced the vision encoder to
carry information proprio doesn't, but the actor hasn't learned to
use that information to choose better actions.

Task performance regressed to R36 levels (6/20, -757 reward) because
the actor has to share the latent space with an encoder whose output
keeps shifting (the kl_pred loss kept growing — see next section).

## Diagnostic scalars

### sigma_combined — fastest drop yet

| t   | R36    | R37    | R38         |
|-----|--------|--------|-------------|
|  8K | 0.7105 | 0.7105 | 0.7105      |
| 24K | 0.6682 | 0.6684 | 0.4971 (-30%) |
| 48K | 0.6211 | 0.6145 | 0.4118 (-42%) |
| 80K | 0.4516 (-36%) | 0.4634 (-35%) | **0.1468 (-79%)** |

R38 ends with σ_combined less than a third of R36/R37. The PoE corner
is dramatically tighter — *and* it is **earned**, not forced by
mirroring (see kl_agreement below).

### kl_agreement — the smoking gun for "not mirroring"

This is the **same-time** symmetric KL — measured as a diagnostic
(β_sym = 0, so no gradient pressure on it).

| t   | R36          | R37          | R38                   |
|-----|--------------|--------------|-----------------------|
|  8K | 0.81         | 0.81         | 0.81                  |
| 24K | 0.0022       | 0.0018       | **0.63**              |
| 48K | 0.0007       | 0.0004       | **0.98**              |
| 80K | 0.0014       | 0.0147       | **84.10**             |

R36/R37: same-time KL crashed to ~0 in the first 4K of training and
stayed there — vision encoded the same Gaussian as proprio at every
moment. **R38: same-time KL grew by 100× over training.** Vision's
distribution at time t is now *very* different from proprio's at time
t. This is the architectural pre-condition for ablation to matter:
vision encodes something proprio does not.

### kl_pred — the loss that was actually optimized

The optimizer target started at ~1.18 and ended around 116–158 with
high variance. The predictive loss grew during training rather than
shrinking.

Why: the KL formula `KL(N(μ_v,σ_v) || N(μ_p_next,σ_p_next))` has terms
in `1/σ_p_next²` and `log(σ_p_next/σ_v)`. As σ_p shrinks (proprio
becoming more confident about its own next state), the KL blows up
unless vision can match with near-zero precision — which is hard
because pixels carry less direct information about a 20ms-ahead
proprio state than proprio itself does.

This is a known instability in VAE-style KL losses: when the prior
becomes tight, the posterior matching it becomes nearly impossible.
The growing kl_pred is *not* the architecture working — it's the
architecture being asked to do something the loss formulation makes
hard.

## Diagnosis: right shape, wrong horizon

R38 is the *first* experiment in this codebase where vision became
load-bearing in any measurable sense. That is real progress on the
deepest pathology of Phase G/H. The architecture's predictive bone
is in the right place.

But predicting `t+1` (= 20 ms of simulated time) only teaches vision
to encode "what proprio will feel in the next breath." That is a
useful but very weak signal: for our task, "the next breath" is barely
different from "now" — same posture, same cart position, ball still
where it was. The actor doesn't get *long-horizon* information from
vision that would tell it which ball to commit to reaching.

The pre-rendering conversation already named the fix: **predictive
horizon should scale with distance to contact**. Vision's job is to
encode `proprio(t + tau)` where tau is on the order of
*time-to-interaction*, not one timestep. 50 steps ahead = 1 sim-second
= roughly the cart-traversal timescale of the curriculum.

## Recommended R39

**Multi-horizon predictive loss.** Sum predictive KLs at several
look-aheads:

```
loss = β_1 * KL(v(t) || p(t+1).detach())
     + β_5 * KL(v(t) || p(t+5).detach())
     + β_25 * KL(v(t) || p(t+25).detach())
     + β_50 * KL(v(t) || p(t+50).detach())
```

Implementation cost is low: sample several `(o_t, o_{t+k})` pairs from
the replay buffer instead of one. Each horizon's KL adds its own
gradient pressure; vision must encode something useful at all of them
simultaneously, which forces a richer representation than t+1 alone.

We should also consider scaling the β to stabilize the loss against
the precision blow-up — divide each term by `1 + σ_p_next` or clamp
the σ_p log-term. The t=80K σ_combined of 0.147 means encoders are
*very* tight; a regularizer that keeps σ in a healthier range may
help.

## Files (R38)

- `alien_baby/results/mimo_phase_ii_R38_predictive_only/final_model.zip`
- `alien_baby/results/mimo_phase_ii_R38_predictive_only.log`
- `alien_baby/results/videos/R38_predictive_only_final_seed{0,1}_off0.15.mp4`
- Eval JSON inline above
- Re-run: `python -m alien_baby.visualization.eval_phase_i --run-tag mimo_phase_ii_R38_predictive_only --ball-speed 0.0 --offset 0.15`

---

# R39 — Phase III: multi-horizon predictive loss

## Config

Same env as R36/R37/R38. New flag: `--micoa-pred-horizons "1:0.03,5:0.05,25:0.1,50:0.15"`.
Four horizons (20 ms, 100 ms, 500 ms, 1 sim-sec), weighted toward longer
horizons because (a) the long ones carry the cart-traversal-timescale
information vision could plausibly add and (b) k=50 KL is naturally
harder to satisfy by trivial mirroring.

Also tightened the encoder σ clamp from `log_sigma ∈ [-4, 2]` → `[-2, 2]`
(so σ ∈ [0.135, 7.4] instead of [0.018, 7.4]) to address R38's
precision blow-up where kl_pred grew unbounded as σ_p shrank.

## Headline result: went backwards on load-bearing

| metric                            | R36   | R37   | **R38** | **R39** | threshold |
|-----------------------------------|-------|-------|---------|---------|-----------|
| Vision ablation L2 delta          | 0.0019| 0.0023| **0.66**| **0.035**| > 0.05    |
| Episode both-touched, normal      | 6/20  | 18/20 | 6/20    | 3/20    | —         |
| Episode both-touched, pixels zeroed | 5/20 | 16/20 | 6/20    | 4/20    | —         |
| Pixel-zero delta (normal − zeroed)| +1    | +2    | +0      | **−1**  | > +4      |
| Eval @ 80K mean_reward            | -835  | +121  | -757    | -951    | —         |

Two unpleasant findings:

1. **Ablation delta is 18× lower than R38** (0.035 vs. 0.66). The
   multi-horizon experiment moved away from the only configuration
   so far that crossed the load-bearing threshold.

2. **Vision is now slightly anti-productive.** Pixels-zeroed both-
   touched is 4/20 vs. pixels-normal 3/20 — zeroing vision *improves*
   the outcome by 1 episode. R36/R37/R38 all had vision either
   inert or neutral. R39 has vision actively hurting.

## Diagnostic scalars

### σ_combined — controlled, no R38-style collapse

| t   | R36   | R37   | R38         | R39          |
|-----|-------|-------|-------------|--------------|
|  8K | 0.71  | 0.71  | 0.71        | 0.71         |
| 32K | 0.65  | 0.65  | 0.61        | 0.49 (then ↑ to 0.57 @36K) |
| 80K | 0.45  | 0.46  | **0.147**   | 0.33         |

R39's σ_combined trajectory is **non-monotonic and oscillating** —
drops to 0.45 at 28K, bounces back to 0.57 at 36K, oscillates between
0.39 and 0.61, lands at 0.33. The σ-clamp is doing its job — no
collapse to 0.147 — but the oscillation suggests the encoders are
being pulled in conflicting directions by the four horizons and never
converging.

### kl_pred per horizon — bounded, no explosion

R39's per-horizon KLs stayed in [0.3, 2.5] throughout training.
Compare R38 where kl_pred grew unbounded to 100+. The σ clamp solved
the instability. But — see ablation result — solving the instability
also lost the load-bearing signal.

### kl_agreement (diagnostic since β_sym=0)

| t   | R36         | R37         | R38     | R39     |
|-----|-------------|-------------|---------|---------|
|  8K | 0.81        | 0.81        | 0.81    | 0.81    |
| 32K | 0.0014      | 0.0014      | 1.02    | 3.21    |
| 80K | 0.0014      | 0.0147      | **84.1**| 1.55    |

R39 sits in the middle: encoders are NOT mirror-collapsing (kl > 0.8
throughout) but also not exploding apart (kl < 5 most of training).
This is exactly what the architecture *should* look like in principle.
But the policy isn't using vision's differentiation usefully.

## Diagnosis: three suspect knobs were turned at once

Between R38 (load-bearing but broken) and R39 (not load-bearing, more
broken), three things changed simultaneously:

1. **σ clamp tightened** (R38: σ ∈ [0.018, 7.4] → R39: [0.135, 7.4]).
   R38's high ablation delta may have ridden on the pathology — σ_p
   shrinking to 0.018 let the vision encoder confidently broadcast
   *something specific* into the policy, even if that something was
   the chaos of an unsatisfiable loss. The clamp killed the explosion
   *and* the broadcast.

2. **Total β tripled** (R38: 0.10 → R39: 0.33 across 4 horizons).
   Three times more KL pressure deforming the actor's task gradient.
   Task performance dropped from -757 (R38) to -951 (R39).

3. **Multi-horizon may be interfering with itself.** Vision is being
   pulled toward proprio at four different futures. The σ oscillation
   (0.45 → 0.57 → 0.39 → 0.61 → 0.33) suggests the gradients don't
   compose cleanly — each horizon's pressure is undoing the others'.

We changed three variables and the outcome is worse. Need to
disambiguate.

## Recommended R40

**Isolate the σ clamp variable first.** R40 = R38 setup (single horizon
t+1, β=0.1) but WITH the tighter σ clamp from R39. If ablation drops
from 0.66 to <0.05, the σ clamp is what killed the load-bearing
signal in R39 — and the fix is to allow vision more precision (relax
the σ_v clamp specifically) while regularizing proprio's precision
(keep σ_p ≥ 0.135 to prevent the explosion).

If R40's ablation stays high (~0.5), the multi-horizon setup itself
is the problem, and the next experiment should pick one horizon and
weight (e.g. k=25 alone at β=0.1) and see if a long-horizon-only loss
preserves the load-bearing signal R38 found.

In either case, **task regression** is a separate problem the σ clamp
doesn't address. Even if R40 recovers ablation delta, the policy still
needs a way to use vision's signal *productively*. That may require:
- much smaller β (vision pressure as a perturbation, not a force)
- β-ramp (let actor learn task first, then add vision pressure slowly)
- splitting the predictive loss into a separate critic-side gradient
  (so vision pressure doesn't deform the actor's policy directly)

## Files (R39)

- `alien_baby/results/mimo_phase_iii_R39_multihorizon/final_model.zip`
- `alien_baby/results/mimo_phase_iii_R39_multihorizon.log`
- `alien_baby/results/videos/R39_multihorizon_final_seed{0,1}_off0.15.mp4`
- Re-run eval: `python -m alien_baby.visualization.eval_phase_i --run-tag mimo_phase_iii_R39_multihorizon --ball-speed 0.0 --offset 0.15`
