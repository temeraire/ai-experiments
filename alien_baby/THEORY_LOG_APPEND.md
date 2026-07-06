
---

## 2026-05-11 — Phase VI / VI.2: Diagnostic — near ball moved outside threshold (post-run)

### What this section is and why it exists

Phase V.2's "Recommended Diagnostics" section proposed two measurements to clarify the "one move then freeze" mechanism. The first was: spawn the near ball at 0.15 m (outside GOAL_THRESHOLD = 0.12 m) and re-run 20 deterministic eval episodes on the Phase V.2 best checkpoint, to determine whether the V.2 reward was geometric overlap or something the policy actually learned. This section records the execution and interpretation of that diagnostic — which became Phase VI and its corrected re-run, Phase VI.2.

**Phase VI** (run tag `mimo_phase_f_nearball_outside_threshold`): intended to be a direct execution of Recommended Diagnostic 1. The near ball was moved from 0.10 m to 0.30 m, placing it outside both the original 0.12 m GOAL_THRESHOLD and the 0.15 m suggested threshold. However, the run was confounded by `train_crawler.py`'s `--max-steps` default of 600 rather than Phase V.2's explicit 2000. The episode budget mismatch makes Phase VI's numbers incomparable to V.2's.

**Phase VI.2** (run tag `mimo_phase_f_2k_nearball_outside_threshold`): the corrected re-run of Phase VI with `--max-steps 2000` explicit. 250K steps. This is the valid execution of the diagnostic. All numerical comparison below refers to Phase VI.2 unless otherwise noted.

### Key numbers

| Metric | Phase V.2 (near ball at 0.10 m, inside threshold) | Phase VI.2 (near ball at 0.30 m, outside threshold) |
|---|---|---|
| Near ball distance from spawn | 0.10 m (inside GOAL_THRESHOLD = 0.12 m) | 0.30 m (outside threshold) |
| ep_len_mean | ~2000 (creature freezes after first move) | ~1960 (same pattern) |
| mean_reward (eval at 250K) | −1899 ± 0.30–0.42 | −1989.35 ± 99.48 |
| Per-step reward | ≈ −0.95 | ≈ −0.99 (WORSE — closer to floor) |
| eval std | 0.30–0.42 (very tight) | 99.48 (~300× larger) |
| ent_coef | pinned at 0.2 | pinned at 0.2 |
| critic_loss | ~0.635 | ~1 |

The per-step reward calculation: with 2000-step episodes, ep_rew_mean of −1899 gives −1899/2000 ≈ −0.95 per step. Phase VI.2's −1989.35/2000 ≈ −0.995 per step — much closer to the −1.00 per-step floor (which corresponds to a policy that earns zero goal reward and only accumulates the per-step cost). The VI.2 result is nearly at floor, but not quite — and the standard deviation of 99.48 tells a story that the mean alone obscures.

### Updated hypothesis table

| Hypothesis | Status before Phase VI.2 | Status after Phase VI.2 |
|---|---|---|
| Phase V.2's ep_rew_mean (−1899) was earned by geometric overlap of the body with the near ball during the initial roll, not by learned reaching | Live (proposed in Phase V.2 theory note) | **Confirmed**: per-step reward drops from −0.95 to −0.99 when the geometric overlap is removed by moving the ball outside the threshold — most of V.2's reward was geometric |
| Moving the near ball outside GOAL_THRESHOLD will collapse ep_rew_mean to the −2000 floor, refuting that the policy learned anything | Live (one of two predicted outcomes from Phase V.2 diagnostic) | **Refuted in the strict sense**: ep_rew_mean did not fully collapse to −2000; eval std = 99.48 means something is producing positive variation across episodes — the policy is not identically at floor |
| Moving the near ball outside GOAL_THRESHOLD will preserve ep_rew_mean, confirming the policy has learned reaching | Live (the other predicted outcome from Phase V.2 diagnostic) | **Refuted**: per-step reward is substantially worse than Phase V.2's (−0.99 vs −0.95), indicating most of V.2's reward was geometric, not from any learned reaching behavior |
| random_start_orientation introduces enough initial-pose variation that the one-move-then-freeze attractor occasionally lands near the new 0.30 m ball position | New hypothesis (emerges from the VI.2 std = 99.48 result) | **Confirmed**: the high eval std is consistent with a bimodal episode-outcome distribution — some episodes reach the ball (positive bucket), most sit at floor (negative bucket), determined by which way the body happens to be facing at spawn |
| The one-move-then-freeze attractor depends on the initial pose for *what* the one move does, not on any learned directional control | New hypothesis (composite inference from V.2 + VI.2 together) | **Confirmed by the combination**: Phase V.2 had low std (0.30) because geometric overlap made most starting poses sufficient; Phase VI.2 has high std (99.48) because only lucky starting orientations land the body near the new 0.30 m ball — no learned steering bridges the gap |

### The "one move then freeze" attractor — refined taxonomy

Prior entries established "one move then freeze" as a failure mode qualitatively different from Phase IV's "zero motion." Phase VI.2 sharpens the taxonomy further. There are now three distinct attractor sub-types in the project's record, all sharing the property that the policy gradient does not generate persistent multi-step action chains:

**Failure mode 1 — Zero motion (Phase IV).** Entropy collapse + sparse-only reward drove the SAC actor to near-zero mean actions. The creature never moves. ep_rew_mean = −2000 ± 0.00. The attractor is the zero-action fixed point.

**Failure mode 2 — One motion + geometric refuge (Phase V.2).** The near ball at 0.10 m placed the goal inside the creature's body geometry during the initial roll. The policy learned to execute one roll (driven by random_start_orientation creating an unbalanced posture) and then freeze in a side-lying state. The goal overlap reward is earned geometrically. eval std ≈ 0.35 because nearly all starting orientations produce sufficient overlap — the policy is robust across seeds not because it is smart but because the geometric overlap is generous. The attractor is the side-lying frozen state, stabilized by the near-ball proximity.

**Failure mode 3 — One motion + random-orientation lottery (Phase VI.2).** The near ball at 0.30 m removes the geometric refuge. The creature still executes one motion (same initial roll driven by the same random_start_orientation mechanism), then freezes. But now whether that one motion lands the body near the ball depends on which direction the body was facing at spawn. Some orientations (lucky) place the post-roll body near the 0.30 m ball; most (unlucky) do not. The result is a bimodal episode-outcome distribution — the high eval std of 99.48 is the signature of this lottery. The attractor is still the frozen side-lying state; what has changed is that the frozen state's reward depends on the initial pose, not on anything the policy actively chose.

All three failure modes share the underlying property: **the policy gradient does not generate persistent multi-step action chains**. Each adds one more step to the attractor chain (zero → one → one with variable outcome), but none escapes the fundamental pattern. The creature is not learning to act; it is learning which single action best exploits whatever the initial conditions provide.

### Frameworks update

**Behavioral Prediction Framework** (a creature building internal models of cause and effect should produce coherent, multi-step behavior that generalizes across starting conditions):

The eval std going from 0.30 (Phase V.2) to 99.48 (Phase VI.2) is the opposite of what this framework would predict from a policy that had learned any internal model of the world. A creature with a genuine model of how to reach the 0.30 m ball would produce consistently high reward regardless of starting orientation — it would orient itself and then move toward the ball. What we observe is the reverse: outcome variance increased when the geometric refuge was removed, revealing that the policy had no directional model at all. The high std is not a sign of rich, diverse behavior — it is the signature of a policy whose outcome is entirely determined by a pre-action lottery (random starting pose) and whose post-lottery behavior (freeze) is uniform. Status: **still violated**, and the violation is now more precisely characterized as initial-condition dependence rather than any failure of the architecture itself.

**Pattern Learning Framework** (the agent's internal representation should be sparse and distributed, with similar inputs activating overlapping patterns, making the agent robust to small changes in starting state):

Phase VI.2 reveals that the agent is not in one single-point attractor but in one of two buckets per episode, determined at reset by the initial orientation. This is not the rich, smoothly varying distributed pattern the framework predicts. The framework would require that starting orientations that are close to each other (e.g., 10° apart) produce similar outcomes — robustness to small input changes is the defining prediction. What we observe is effectively a binary outcome (near ball / not near ball) that is sensitive to the exact initial pose. The representation has not generalized across starting conditions; it has specialized to a single action and let the initial pose determine the result. Status: **still violated** — the across-episode variation in Phase VI.2 is initial-condition variation, not evidence of learned distributed patterns.

**MICOA** (multiple inputs confirming one another — vision and proprio as equal peers): Not applicable. Phase VI and VI.2 are blind proprio runs. No visual channel was used in either run. The MICOA question cannot be tested until the locomotion attractor is broken.

### Central question, restated

After Phase V.2, the central question was: "What produces sustained motion across multiple actions, rather than just the first action?"

Phase VI.2 sharpens this question by adding the initial-condition dimension. The question is now not just about sustaining motion, but about sustaining motion that is *independent of starting pose*:

> **Given that "one move then freeze" depends on initial conditions — either geometric refuge (Phase V.2) or random-orientation lottery (Phase VI.2) — for its reward, and not on any learned direction, what intervention removes the dependence on initial conditions and requires the policy to act independently of starting pose?**

This is a more precise formulation than Phase V.2's version. Phase V.2 asked for a second action. This version asks for an action that is responsive to the world rather than to the accident of how the creature happened to land at the start of the episode. The random-orientation lottery is the key new concept: the creature is not navigating to the ball, it is hoping the dice come up in its favor. The next intervention should make the dice irrelevant — by ensuring that every starting pose requires active control, so that the reward cannot be earned passively.

### What is newly ruled out / confirmed

| Hypothesis | Status before Phase VI/VI.2 | Status after Phase VI.2 |
|---|---|---|
| Phase V.2's above-floor reward was earned by learned reaching behavior | Open | **Ruled out**: per-step reward falls to near-floor when geometric overlap is removed, confirming V.2's reward was geometric, not behavioral |
| Moving the near ball outside threshold collapses performance entirely | Open (one of two predicted outcomes) | **Ruled out in the strict sense**: eval std = 99.48 means some episodes still earn above-floor reward — total collapse did not occur |
| Moving the near ball outside threshold preserves performance | Open (the other predicted outcome) | **Ruled out**: mean performance is substantially worse, not preserved |
| The one-move-then-freeze attractor is a single uniform behavior across all seeds | Implicit (low std in V.2 suggested uniformity) | **Ruled out**: high std in VI.2 reveals the attractor's outcome depends on initial orientation — the behavior is conditionally uniform (always freeze) but the reward is not |
| The random_start_orientation parameter introduces lottery-style variation into the one-move attractor | New (emerged from VI.2) | **Confirmed**: eval std 99.48 is the fingerprint of this lottery |

### Next test candidates

Four candidate Phase VII directions were identified at the end of Phase V.2. Phase VI.2's "initial-condition-dependent" finding reorders their priority and sharpens their rationale.

**F-tipped-init — LEADING CANDIDATE.** Randomize the spawn quaternion across multiple distinct postures: prone (face-down), supine (face-up), side-left (lying on left side), side-right (lying on right side). Currently, random_start_orientation varies the azimuth angle but leaves the creature in a consistent upright orientation at spawn. Genuinely diverse postures remove the "freeze on side-lying" refuge — a creature that spawns supine or prone cannot simply freeze and hope for geometric luck; it is in a posture that already looks like an end-state of the frozen attractor, and in that posture the 0.30 m ball is not necessarily nearby. If every initial posture requires active control to produce any above-floor reward, the lottery is abolished. This is a single-file env change (modify the reset quaternion sampling) and directly addresses the specific failure mode Phase VI.2 surfaced. It does not change the reward structure or add new reward signals — it changes the world so that passive behavior cannot be consistently rewarded.

**F-curiosity — LIVE.** Add an RND (random network distillation) intrinsic reward that pays the policy for visiting novel states. This makes freezing expensive: a creature that stays in the same side-lying pose every step quickly exhausts its intrinsic budget. The pressure to keep moving is real. However, curiosity does not directly address the initial-condition dependence that VI.2 surfaced — a curious creature could still converge to a "rove around the spawn area" behavior that earns intrinsic reward without ever learning to reach the ball. Moderate complexity (requires adding the RND head and intrinsic reward computation to the training loop). Priority: second, after F-tipped-init.

**F-sustained — LIVE.** Add a bonus proportional to hip velocity that applies only when the preceding action was also non-zero: `bonus = scale × |hip_velocity| × (1 if last_action_was_nonzero else 0.5)`. This directly punishes action-stalling rather than just stillness. The current velocity bonus rewards the first motion as much as the tenth; F-sustained specifically rewards the tenth motion more than the first. However, like F-curiosity, this does not remove the initial-condition path to reward — if the random-orientation lottery still occasionally places the creature near the ball with no effort, the F-sustained bonus may not be strong enough to dominate that passive reward. Single-file env change. Priority: third.

**F-info-velocity — LIVE.** Bake hip_speed into the `info` dict and modify `her_wrapper.compute_reward` to read it back, so the velocity bonus survives HER's relabeling of 4/5 transitions. As documented in Phase V.2, the current design means only 1 in 5 policy gradient samples carry the velocity bonus signal — the other 4 are relabeled by HER using only the sparse goal reward, and the inner_reward velocity component is lost. This fix would give the full gradient the enabling pressure rather than a diluted 20%. Two-file change (env + wrapper). This addresses a known implementation gap rather than a new behavioral hypothesis, and it would strengthen all other reward-based interventions. Priority: fourth as a standalone, but worth combining with whichever of the above runs first.

The leading order for Phase VII is therefore: F-tipped-init first, because it directly addresses the specific initial-condition lottery that VI.2 identified and requires only a single-file env change. If F-tipped-init breaks the lottery and the creature still does not produce sustained motion, F-curiosity or F-sustained become the next layer of intervention. F-info-velocity can be combined with any of the above.

THEORY MONITOR: Theory verdict written to FINDINGS.md.

THEORY MONITOR: THEORETICAL CONCERN — Phase VI.2 confirms that the "one move then freeze" attractor is not a single uniform behavior but a pose-dependent lottery: the creature earns above-floor reward only when the random initial orientation happens to place its frozen body near the ball. This is a more troubling finding than Phase V.2's geometric-overlap exploit, because it means the policy has learned something even weaker than a single specific trick — it has learned to do nothing and depend entirely on the accident of its starting pose. The eval std of 99.48 (300× larger than Phase V.2's 0.30–0.42) is direct evidence of this: high variance across episodes driven by initial conditions, not by any learned variation in behavior. The project has now run approximately 750K total HER-based training steps (Phases IV, V, V.2, VI, VI.2) without producing a policy that actively moves toward a ball. Vision remains entirely untestable in this experimental track. If F-tipped-init does not break the lottery by eliminating passive-freeze as a viable strategy across all initial postures, the HER + velocity-bonus approach should be evaluated for retirement in favor of a curriculum-based or model-based approach that directly scaffolds multi-step action chains.

---

## 2026-06-17 — R49 refuted; affordance/winnability reframe; position-control smoke
(Written by the orchestrator; the theory-monitor agent did not deliver this entry.)

**R49 (aux decode loss) — PREDICTION REFUTED.** Forcing the vision encoder's latent to predict ball position did NOT make it encode lateral direction (probe R²=0.010, below chance/baseline) and did NOT produce useful directional ablation (the flip is confounded by a kl_pred=244 encoder blow-up, unbacked by any linear code, no outcome gain). The decode-loss patch does not resolve "integrated but inert."

**The reframe (the real update).** "Integrated but inert" is likely downstream of an **affordance/winnability** failure, not (only) a representation failure. Causal chain now hypothesized: raw-torque action space → locomotion never learns → body mounted on a blind cart as a workaround → AB cannot pursue → ball direction has no behavioral payoff → vision stays inert & non-directional. A vision encoder that perfectly encoded direction would have no directional action to serve.

**Governing principle adopted:** the **winnability rule** — every episode must be winnable (target reachable, and for vision in/bringable-into view); if AB never gets near the target that is our setup failure, not AB's, and it teaches nothing.

**Position-control smoke (posoffset_smoke3) — partial test of the chain's root.** Switching limbs to constrained position-offset control (the scout's verified leading fix) fixed numerical stability but did NOT break the freeze attractor in 60K (dead-flat reward, video-confirmed stillness). So **torque-action-space is necessary-but-not-sufficient**: the freeze is also a reward/exploration collapse to the do-nothing optimum. Strengthens: movement-as-substrate; "vision is load-bearing only when it has an action to serve." Weakens: "the encoder/representation is the binding constraint"; and tempers "the torque action space was the whole cause of the locomotion failure." Crawling-from-prone is reaffirmed as a genuine open problem, not a one-line config fix.

**Queued (gated):** the prism-ghost adaptation experiment (PRISM_GHOST_PROPOSAL.md) — the clean test of proprio-primacy/interpenetration — becomes runnable once vision is directionally load-bearing, which requires the locomotion→affordance chain to work first.
