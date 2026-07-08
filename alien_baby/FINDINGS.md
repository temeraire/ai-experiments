# Interpenetration Simulation: Findings

## What We Built

A simulated robotic arm in a physics engine (MuJoCo) that learns to reach and touch objects on a table. The arm has two senses: **proprioception** (knowing where its joints are, how fast they're moving, and whether it's touching something) and **vision** (an overhead camera image of the table).

We trained three versions of this arm, each with a different relationship between its senses:

1. **The staged agent** (staged development): First learned to reach objects using only proprioception — like an infant in the dark, finding things by feel. Then we turned on vision and let it keep learning. Crucially, the visual information was fed into the *same neural pathways* that already handled proprioception. There was no separate "vision module."

   Concretely: Stage 1 is an MLP whose 10-dim input is `[3 joint angles, 3 joint velocities, 1 binary touch bit, 3 fingertip-to-target xyz offsets]`, feeding two hidden layers of 256 units. Stage 2 expands the input to 778 dims by appending a flattened 16×16 RGB pixel vector. The first hidden layer is now a 256×778 weight matrix; we copy Stage 1's learned weights into the first 10 input columns and zero the remaining 768. Each of the 256 hidden units is the *same physical neuron* that previously encoded a function of joint state and touch — over Stage 2 training, its 768 pixel weights grow from zero, so the unit's activation becomes a mixed function of "this joint configuration AND this pixel pattern." There is no separate vision layer that could be cleanly amputated. By contrast, the feature-fusion agent routes pixels through their own sub-MLP first, so early vision-only units exist that the staged design does not have.

2. **The "all-at-once" agent**: Got both vision and proprioception from the start. Same network architecture as the staged agent, same total training time. The only difference is that it never had a proprioception-only phase.

3. **The "feature-fusion" agent**: Also got both senses from the start, but with a larger network designed to give each sense its own processing pathway before merging them. This is how most modern AI systems handle multiple input types.

All three agents learned the task well (90-95% success rate). The interesting question was never whether they could reach the objects — it was what their internal representations looked like.

---

## What the theory's Theory Predicts

the source theorist argued that perception isn't a collection of separate senses that get combined. Instead, the senses develop sequentially — touch and proprioception first, vision later — and when a new sense arrives, it doesn't get its own processing channel. It gets woven into the existing sensory fabric. He called this **interpenetration**: seeing an apple and touching an apple activate the same internal representation, not because the brain learned to associate two separate representations, but because vision was built *on top of* the tactile understanding of apples.

The key predictions:
- Developmental order matters — learning touch first, then vision, should produce different internal representations than learning both simultaneously
- The resulting representations should be resilient — if you lose one sense, the others carry traces of it
- Senses should be genuinely entangled in the network, not merely associated

---

## What We Found

### 1. The representations genuinely changed (not just a bolt-on)

The first concern was that our staged agent might just be "a proprioception agent with a vision module stuck on top" — that the proprioceptive pathways would remain unchanged while vision occupied its own corner of the network.

We tested this by comparing the internal activations of the Stage 1 agent (proprioception only) with the Stage 2 agent (after vision was added), feeding them both the same proprioceptive inputs.

**Result: CKA similarity = 0.033** (on a scale where 1.0 means identical and 0.0 means completely different).

The proprioceptive representations were almost entirely rewritten when vision arrived. This is not a bolt-on. The network fundamentally reorganized how it processes touch and body position in order to accommodate vision. This is interpenetration in the most literal sense — adding vision changed how the network handles proprioception.

### 2. Degraded vision still helps (up to a point)

If vision were truly interpenetrated into proprioception — woven into the same pathways rather than kept separate — then even noisy, degraded vision should provide some benefit over pure proprioception, because the visual signal is feeding into pathways that proprioception also uses.

We compared the Stage 1 agent (pure proprioception, no vision at all) against the Stage 2 agent with varying levels of visual noise:

| Condition | Success Rate | Reward vs. Stage 1 |
|-----------|-------------|-------------------|
| Stage 1 (no vision) | 27/30 | baseline (5.85) |
| Stage 2, clean vision | 26/30 | +0.75 |
| Stage 2, 20% noise | 28/30 | **+5.32** |
| Stage 2, 40% noise | 23/30 | -4.10 |
| Stage 2, 80% noise | 18/30 | -22.49 |

At low noise (20%), the Stage 2 agent significantly outperforms pure proprioception. Even imperfect vision is genuinely contributing through the interpenetrated pathways. But at 40% noise, performance drops below the pure proprioception baseline — the noisy visual signal starts *interfering* with the proprioceptive processing it's entangled with.

This is a double-edged sword of interpenetration that the source theory would actually predict: if the senses are truly woven together, you can't corrupt one without disturbing the other. The benefit is resilience at low degradation; the cost is vulnerability to noise that a modular system could simply ignore.

### 3. Graceful degradation vs. catastrophic failure

When we progressively corrupted the vision input for all three agents:

| Agent | Clean Success | Full Noise Success | Reward Drop |
|-------|-------------|-------------------|-------------|
| **Staged** | 95% | **70%** | **17.6** |
| All-at-once | 95% | 30% | 41.3 |
| Feature-fusion | 90% | 65% | 19.4 |

The staged agent loses performance gradually. The all-at-once agent collapses — from 95% to 30%. This makes sense: the all-at-once agent learned to rely on vision from the start and never developed robust proprioceptive skills. The staged agent has a proprioceptive foundation that, while reshaped by vision (see Finding 1), retains enough of its original capability to function when vision fails.

However — and this is important for honesty — this particular result is the one most susceptible to the tautological critique. The staged agent *did* spend 50,000 steps learning proprioception alone. Some of its resilience may simply be curriculum effects (it practiced without vision) rather than interpenetration per se. The next two findings address this concern more directly.

### 4. All three agents are entangled, but differently

We measured how much each agent's internal activations change when vision is suddenly zeroed out (the "entanglement probe"):

| Agent | Deep-layer activation change |
|-------|----------------------------|
| Staged | 0.74 |
| All-at-once | 0.74 |
| Feature-fusion | **1.62** |

The staged and all-at-once agents show similar levels of entanglement — removing vision changes their deep hidden layers by about 74%. The feature-fusion agent is disrupted *far more* (162%) — its larger, supposedly more modular architecture actually created deeper vision dependency, not less.

This suggests that network architecture alone doesn't prevent entanglement. All agents, regardless of design, developed entangled representations. The difference is in the *character* of that entanglement — the staged agent's entanglement sits on a proprioceptive foundation, while the all-at-once agent's entanglement has no such anchor.

---

## What This Means

**The theory was right about the mechanism, partially right about the benefits, and didn't anticipate the costs.**

The mechanism: Developmental staging produces genuine interpenetration. Vision doesn't get bolted onto proprioception — it reshapes proprioception entirely (CKA = 0.033). The senses become woven together through shared pathways.

The benefit: The proprioceptive foundation provides resilience. The staged agent degrades gradually when vision fails, while the all-at-once agent collapses. And degraded vision genuinely helps at low noise levels, suggesting the integration is functional, not just structural.

The cost The theory didn't anticipate: Deep interpenetration means noisy input from one sense actively disrupts the other. A modular system can quarantine a failing sensor; an interpenetrated system cannot. At 40% visual noise, the staged agent performs *worse* than a pure proprioception agent — the corrupted vision is poisoning the proprioceptive pathways it's entangled with.

**The strongest non-tautological evidence** is the combination of Findings 1 and 2: the proprioceptive representations were almost completely rewritten by vision training (so this isn't just "retained fallback skills"), yet degraded vision still provides a boost at low noise levels (so the integration is functional). A simple curriculum effect would predict retained Stage 1 performance under vision loss; what we actually see is a reorganized system that uses vision through proprioceptive pathways — which is exactly what interpenetration means.

---

## Limitations

- **Scale**: This is a 3-joint arm reaching for objects on a table. Real perception involves vastly more complex bodies and environments. These findings show interpenetration *can* happen, not that it *must* happen at scale.

- **Training budget**: 50,000-100,000 steps is modest. With more training, all agents might converge to similar solutions. The question is whether developmental ordering creates lasting structural differences, and that would require longer experiments.

- **Only two senses**: the source theory covers the full sensory repertoire. We only tested proprioception and vision. Adding auditory or haptic channels would be a natural next step.

- **The fusion baseline is imperfect**: We approximated separate encoders with a wider network, not a true dual-encoder architecture. A proper implementation with genuinely separate vision and proprioception pathways would be a stronger baseline.

- **No sensory gating**: The agent's policy is a fixed function from a 778-dim input vector to actions; it has no attention, gain control, or modality-selection mechanism. So the "interpenetration is dangerous under noise" finding (Section 2, 40%+ noise dropping below pure proprio) reflects what happens *in the absence of any gating*. A biological organism — or a network with learnable per-channel gain or attention — could downweight a corrupted modality the way a person might close their eyes and rely on touch. A fairer test of the cost of interpenetration would give all three agents a learnable gain over the vision channel and re-run the noise sweep; the current result conflates "vision poisons proprio because they share pathways" with "the policy has no way to ignore vision."

---

## How to Reproduce

```bash
# Full experiment (takes ~20 minutes)
python -m alien_baby.run_experiment

# Quick validation (~2 minutes)
python -m alien_baby.run_experiment --quick

# Just the interpenetration tests (requires trained models)
python -m alien_baby.tests.test_interpenetration

# Render videos
python -m alien_baby.visualization.render_episodes
```

Videos are saved to `alien_baby/results/videos/`. The side-by-side comparison shows all three agents attempting the same reaching task.

---

## Follow-up experiments (v1 long-run, v2 follow-on, v3 blind proprio)

After the initial writeup above, three follow-ups extended the experiment. All three added non-trivial information — including one finding that revises the headline v1 interpretation.

### 1. Long-run v1 (200K/200K/400K steps)

The original experiment ran 50K Stage-1 / 50K Stage-2 / 100K baselines. Re-running at 4× that budget changes the picture:

| Agent | Clean reward | 100%-noise reward | Reward drop | Clean success |
|---|---|---|---|---|
| Staged | 14.5 | -22.7 | 37.2 | 100% |
| All-at-once | 30.7 | -33.3 | 64.0 | 90% |
| Fusion | 27.5 | -28.6 | 56.0 | 75% |

**What changed:** with more training, the baselines caught up and passed staged on *clean* reward. The "graceful degradation" story (37.2 drop vs. 64/56) still holds — staged falls less under noise — but from a lower baseline. The all-at-once / fusion agents keep improving with training; staged stagnates because its downstream layers were reshaped by vision onto an increasingly fragile point in policy space.

Stage 2's late-training regression is particularly pronounced: peak reward 18.9 at step 120K, but the *final* checkpoint (which the test battery reads) ended at 5.4. The long-run battery above uses that regressed final, which is part of why the clean-reward gap is so stark.

### 2. v2 — "Follow-on" architecture (frozen Stage 1)

The original staged agent lets SAC rewrite every actor weight during Stage 2, including the ones derived from Stage 1. Our headline v1 claim — "CKA = 0.033 means the proprioceptive representations were almost entirely rewritten when vision arrived" — was framed as the *success* of interpenetration. But a more careful reading of the theory's principle (Ch. 5, "properties determined by one set are incorporated in the perceptual field determined by the other set") says that the later sense should *inherit* and *support* the earlier sense's structure, not overwrite it. Under that reading, CKA = 0.033 is not a triumph but a failure: the proprio pattern didn't survive.

**v2** tests the alternative: freeze every actor weight derived from Stage 1 during Stage 2, and only let the new pixel-input weights train. Concretely:
- All actor weights set `requires_grad=False` except the first hidden layer's weight.
- On the first hidden layer's weight, a gradient mask zeros the columns corresponding to proprio inputs; only the pixel-input columns train.
- Critics train normally over the full observation space.
- This guarantees that with `pixels = 0`, the actor's output is bit-identical to Stage 1.

**Result:** proprio weight drift after 200K steps of Stage 2 = exactly 0.0. Actions on `pixels = 0` match Stage 1 to float32 precision.

| Test | v1 staged | v2 follow-on |
|---|---|---|
| Proprio weight drift | large (CKA 0.03) | **0.0** (exact) |
| 100%-noise reward | -22.7 | **-2.9** |
| 100%-noise success | 45% | **80%** |
| Clean → full-noise reward drop | 37.2 | **17.4** |
| Clean vision reward | 14.5 | 14.5 |
| Vision-zeroed reward | (would differ from S1) | 17.3 |

**Interpretation:** v2 confirms the "protect proprio" reading. When proprio cannot be corrupted by vision, noise-robustness improves dramatically (80% success under 100% vision noise vs. v1's 45%). But there's a cost: the frozen downstream network constrains vision's upside. Vision didn't beat Stage 1 on clean performance (14.5 vs. Stage 1's own ~17) — because every hidden-layer transformation vision can influence must route through Stage 1's learned representation geometry. Vision can only speak Stage 1's language.

### 3. v3 — Blind proprio (vision made necessary)

v2 worked but vision didn't add much on clean performance, partly because the task was solvable by proprio alone: the observation included a 3-dim `fingertip → target` offset that essentially handed the agent the target's location. **v3** strips that offset. Proprio is reduced to `[3 joint angles, 3 joint velocities, 1 touch bit]` — 7 dims instead of 10. The agent has no direct target-location signal from proprio.

Stage 1 is retrained on this blind observation, then v2-style follow-on layers vision on top.

**Test 1 — target-position probe (cross-validated Ridge on hidden1 activations predicting target x,y):**

| Input | v2 (offset included) | v3 (blind) |
|---|---|---|
| Proprio-only | 1.00 | **-0.02** (zero) |
| Pixels-only | 0.42 | 0.35 |
| Full obs | 1.00 | 0.35 |

The env change worked: proprio genuinely has no target-location signal anymore. Vision is now the only non-trivial source. Vision's R² stayed around 0.35 — the frozen architecture still caps how accurately vision can pin down the target, even when vision is necessary.

**Test 2 — temporal shortening (episodes to contact):**

| Condition | v2 | v3 |
|---|---|---|
| Stage 1 alone (no vision) | 16.8 | 62.4 |
| Follow-on, clean vision | 16.8 | 33.2 |
| Follow-on, vision zeroed | 27.4 | 70.5 |
| Follow-on, vision random noise | 114.0 | 128.1 |

In v3, clean vision nearly halves the time to contact (33 vs. 62). In v2 the speedup was present but subtle (17 vs. 27). When vision is necessary, its contribution is empirically visible. Random-noise vision quadruples episode length — the agent has *learned to trust* vision enough that corrupting it is worse than having no vision at all. That's the theory's double-edged sword showing up.

**Videos:** `alien_baby/results/videos/comparison_v3_seed{0,1,2}.mp4` show the blind Stage 1 agent vs. the follow-on-with-vision agent on identical starting states. On hard seeds (0, 1) the blind agent gropes for 200 steps and hits truncation without finding the target; the vision agent finishes in ~40 steps. On easy seeds (2) both succeed quickly, vision faster.

### Revised interpretation

The initial v1 framing ("CKA = 0.033 is the signature of interpenetration") was mistaken. That low CKA measured the *destruction* of Stage 1's representation by Stage 2 training, not its incorporation into a richer representation. Under the theory-faithful reading — vision should be *added to* proprio without corrupting it — a successful experiment should show *high* CKA between Stage 1 and the final agent on proprio-only inputs.

v2 instantiates that reading architecturally and produces the predicted noise robustness. v3 shows that when proprio is actually blind, vision steps up as an independent target-localizer, though only partially (R² = 0.35). The frozen downstream architecture of v2/v3 is too restrictive to let vision fully learn target localization — a v4 with an explicit cross-modal consistency loss, or a softer freeze (e.g., low LR on downstream layers rather than full freeze), is the natural next step.

**What's supported empirically so far:**
1. Developmental staging with proper weight protection (v2) produces real noise robustness, not just a curriculum artifact.
2. Vision woven onto a blind proprio (v3) provides genuine independent confirmation — halving time-to-contact.
3. Entangled senses carry a real cost under noise: agents that rely on vision perform worse under corrupted vision than agents that never learned to use it.

**What remains uncertain:**
1. Whether vision's target estimate can be driven to match proprio-level accuracy without explicit consistency loss (v3 shows partial, not full, independent localization).
2. Whether the cross-modal identity alignment (same-object see/touch vs. diff-object see/touch) ever emerges under current architectures — our measurements remain near zero.
3. Whether active looking (camera follows hand) and occlusion would produce the richer theory-style interpenetration the current task cannot force.

---

## v5 — Consistency-loss follow-on (2026-04-14)

### Architecture
Same frozen-proprio / trainable-pixel-columns architecture as v3, but the actor loss gains an auxiliary term:

```
L_actor = L_SAC + λ · MSE(hidden1(full obs), hidden1(pixels-zeroed obs))
```

with λ = 0.1. This pulls whatever vision contributes to hidden1 toward the hidden1 that proprio alone would produce. Vision can only *confirm* proprio's manifold, not carve its own. Gradient mask still zeroes proprio columns; `drift = 0.0` verified post-training.

Re-uses `stage1_v3_checkpoint` as the frozen proprio base. 200k SAC steps, consistency_loss settled at 0.94 (from 2.4 at smoke-test).

### Results (20-episode eval; CKA/equivalence over ~900 samples)

| Metric | v3 follow-on | **v5 follow-on** | Interpretation |
|---|---|---|---|
| Clean-vision success | 19/20 (95%) | **20/20 (100%)** | Slight improvement |
| 100%-noise success | 11/20 (55%) | **16/20 (80%)** | Large robustness gain |
| CKA(hidden1, stage1_v3.hidden1) | 0.905 | **0.998** | v5 lives on proprio's manifold |
| Neighbor-consistency ratio | 0.250 | 0.272 | Effectively tied |
| Proprio-column drift | 0.0 | 0.0 | Freeze holds |

### Interpretation

**CKA = 0.998 is the headline.** v5's internal representation is indistinguishable from Stage 1's proprio representation under linear-alignment similarity. Vision is *additive* — it extends proprio's manifold rather than displacing it. This is the theory's "properties of the perceptual field determined by one set are incorporated in the perceptual field determined by the other set" (Ch 5) rendered as a number: CKA of 0.998 with the earlier-established sense.

**The noise-robustness jump (55% → 80% at 100% vision noise) is the downstream consequence.** By pinning vision's contribution to proprio's manifold, v5 never learns to *depend* on specific pixel patterns in a way that breaks when pixels get corrupted. When vision becomes uninformative, the network falls gracefully back to something close to the proprio-only behavior — because that's geometrically where it already was.

**The equivalence-class ratio did not meaningfully improve** (0.272 vs 0.250). The consistency loss pulls representations toward proprio but doesn't explicitly tighten action-class structure. This is consistent with the memory-level feedback that generalization is the primary state — v3 already had strong class structure; v5 preserves it without sharpening it.

**Reinterpreting v1 once more.** v1 "staged" had CKA = 0.033 between Stage 1 and Stage 2 on proprio inputs. v5 has CKA = 0.998. Same task, same algorithm, opposite architectural commitments — and a 30× difference in how the later sense relates to the earlier one. v1 was obliteration; v5 is genuine incorporation. The earlier framing of v1 as "strong interpenetration" was exactly backwards.

### What this closes and what it opens

**Closed:** Whether an explicit mechanism can enforce "vision must confirm proprio, by its own means" — yes, and with a single line of loss. The architectural requirement (protect proprio) composes with a representational requirement (stay on proprio's manifold) to produce both noise robustness and interpenetration in the theory's sense.

**Opens:** 
1. Does v5's on-manifold vision support richer cross-modal transfer (e.g., seeing → touch-readiness) than v3? Our cross-modal identity alignment metric remains unaddressed.
2. Is λ = 0.1 near-optimal or can vision do more with less constraint? A sweep could tell us.
3. The eye is still a fixed overhead camera. Source Ch 3–4 demands an eye that moves. v5 is the strongest test of the theory we can run without gaze; the next frontier requires moving the camera.

---

## v6 — Head-mounted gaze camera (2026-04-15)

### Motivation (what Ch 6-7 added)
Reading Source Ch 6 (Expanding the Visual Field) and Ch 7 (Parallax) revised the plan. Ch 6 says the "expanding world" is a function of the moving observer — each motor cycle produces a predictable retinal image expansion that is conditioned to locomotor responses. Size/shape constancy (Ch 6.19) is inherited from invariant terminal manipulation. Ch 7 argues binocular fusion is response-conditioned, not structural: two retinal images fuse because they trigger the same reaching response, not because their retinal points geometrically match.

The v5 setup — fixed overhead camera — cannot test any of this. v6 adds a pan/tilt head with its own 45° camera so the agent must orient its gaze to see.

### Architecture
- `tabletop_v6.xml`: head body at (0, -0.28, 0.15) behind/above the arm base. Two hinge joints: `head_pan` (±1.2 rad) and `head_tilt` (±0.8 rad). Head camera FOV 45°, default tilted 30° downward via xyaxes so `tilt=0` already sees the table center.
- `TabletopGazeEnv`: 9-dim proprio (3 jpos + 3 jvel + 1 touch + 2 head angles), 5-dim action (3 arm torque + 2 head position commands, rescaled to joint ranges).
- Reward unchanged — no reward for looking. Gaze must emerge from visual utility.
- Stage 1: proprio-only SAC on 9D obs. Stage 2 (follow-on): v5 `ConsistencySAC` with `proprio_dim=9, λ=0.1`. Head camera pixels flattened + downsampled to 16×16×3. 200K steps each.

### Results (using best EvalCallback checkpoint — see note below on late-training regression)

| Metric | Stage 1 v6 | v6 follow-on | v5 follow-on (ref) |
|---|---|---|---|
| Clean success | 90% | **95%** | 100% |
| 100%-noise success | N/A (no vision) | **35%** | 80% |
| CKA(hidden1, stage1.hidden1) | — | **0.992** | 0.998 |
| Neighbor-consistency ratio | — | 0.382 | 0.272 |
| Proprio-column drift | — | 0.0 | 0.0 |

**Gaze-specific metrics (v6 only):**
- In-view fraction (target inside head cam FOV): 27.1% overall
- In-view when near target (<0.08 distance): **31.3%**
- In-view when far from target: **26.5%**
- Near/far ratio: 1.18 (gaze is only 18% more likely to find the target when close)

### What worked

**Task performance preserved.** v6 follow-on reaches 95% clean success — effectively the same as v5. The 30° default downward tilt ensures the head camera sees a useful view even without active gaze control.

**Interpenetration preserved.** CKA = 0.992 between v6 hidden1 and stage1 v6 hidden1 (matched samples, same proprio). The consistency loss is doing its job: vision through the gaze camera stays on proprio's manifold.

**Vision is more load-bearing than in v5.** v5's success dropped 100% → 80% at full vision noise (drop of 20 points). v6 drops 95% → 35% (drop of 60 points). Makes sense: with a limited-FOV camera the agent has *less* redundant visual information, so when the one view it has becomes noise, performance collapses more sharply. This is consistent with the theory's "over-determination" argument in Ch 7.1 — a single channel provides less slack than multiple redundant channels.

### What didn't work

**Gaze behavior barely emerged.** 31% in-view near vs 27% in-view far is a 4-point difference. By eye, inspecting rendered videos, the agent is not tracking its hand — the red target stays in frame mostly because the 45° FOV + 30° default tilt covers most of the reachable area. The agent's head *moves* (it's actuated and rewarded via task success) but the motion isn't purposeful gaze-following.

**Root cause:** the constraint was too weak. With a 45° FOV and 30° default tilt, a large fraction of the reachable area is in the camera's default cone, so random or near-constant head poses already give useful vision. There's no strong gradient pushing the agent to track its hand.

### Late-training regression (important)

The final (200K-step) checkpoint succeeded on only 7/20 episodes. The EvalCallback-saved best model succeeded on 19-20/20. All results above are from the **best checkpoint**, not the final one. The consistency loss was still climbing at training end (0.55 mid-training → 0.85 late), suggesting the optimization went off-track in the last third.

This is the same late-training regression pattern that damaged v1's Stage 2 long-run. The cause for v6 isn't pinned down — possibilities include lambda too high for the larger action space, or the policy chasing noise in the vision channel as entropy coefficient decayed. Follow-up runs with lower lambda or entropy annealing would be informative.

### Interpretation

v6 is a mixed result, and the mixture itself is informative:

1. **Task-level: success.** The v5 architecture (frozen proprio + consistency loss) carries through to a gaze camera. Grounded perception doesn't require a god's-eye overhead view.

2. **Representation-level: success.** CKA = 0.992 says vision-via-gaze still lives on proprio's manifold. The theory-faithful property of interpenetration persists across the shift from fixed to movable vision.

3. **Gaze-behavior-level: failure.** The agent did not learn to point its head at what its hand was doing. This was the central motivating prediction of v6 (the theory's "the child must look at what they hold") and it did not pan out — not because the source theorist was wrong, but because the environment didn't make gaze *necessary*.

### What v6 teaches us

The architectural claim of v6 (gaze camera = needs gaze learning) was too weak. Vision is a useful channel, but *useful* does not mean *requires gaze*. For gaze to emerge, the world has to make it impossible to succeed without orienting. Two concrete changes would do this:

- **Narrow FOV** (~20°): target often out of frame unless tracked.
- **Moving target** or **occluding clutter**: forces continual re-orientation.
- **Reduce default tilt** or **move the head further away**: make the reachable area subtend more of the possible camera cone.

The deeper lesson: **emergent behavior requires environmental pressure, not just architectural capability.** We gave the agent the means to gaze but not the need. the source theory is about conditioning under conditions that make a response *necessary*; we gave the means without the necessity, and the means stayed largely unused.

### What's supported empirically through v6

1. The v5 architecture (frozen proprio + consistency loss) generalizes to a movable camera with expanded action space — task performance and on-manifold vision both preserved.
2. With a limited-FOV camera, vision becomes more load-bearing (bigger performance drop under vision noise), consistent with the theory's "overdetermination" argument for multiple redundant channels.
3. A gaze camera alone is not sufficient to produce emergent gaze-following behavior. The environmental pressure to orient has to be built in — either by narrowing the FOV, moving the target, or otherwise forcing the issue.

### Open (next-step-worthy)

1. Re-run v6 with narrower FOV (e.g. 20°) and see if gaze-touch correlation increases.
2. Investigate the late-training regression: try lower λ, or entropy annealing, or stop early at the best eval.
3. Extend to Path 4 (LLM in the loop) without fixing v6's gaze problem — vision grounding may be enough even if gaze behavior is underdetermined.

---

## v8 — Platform Creature ("Alien Baby"): physics, methodology, and a negative gaze finding

### Motivation

v6/v7 gave the agent a gaze camera but not a strong reason to orient. The theory-faithful move was to give the agent a world where getting orientation wrong has consequences: a creature on a finite platform that can roll off and fall. v8 is the first environment in this project where wrong perception kills you. This is reflective of what can occur in the real world.

The creature is a torso-sled with two arms and a pan/tilt head. No legs. The working metaphor is a skateboarder-without-legs: push with hands, body rolls. Survival is grounded in physics — not architectural weight freezing (v2/v3/v5) and not reward shaping (v6 implicit).

### What this session focused on

Three parallel threads:
1. **Body physics that are actually drivable** (this was the hard part).
2. **Visualization methodology** — ringside camera, temporal smoothing, and a blur-to-sharpen overlay keyed to vision-ablation sensitivity.
3. **Environmental pressure for gaze** — narrow FOV + target placement bias.

### Physics

Starting XML (a torso-box sliding on a slippery underside, decorative non-colliding wheels) was undrivable: random policies tipped within seconds, and trained policies flailed arms without producing locomotion. Diagnosis: shoulder mounts sat outside the support polygon of the torso, so arm pushes acted as tipping moments instead of horizontal thrust.

Final configuration:
- Torso flatter + wider (0.15×0.10×0.04 → 0.18×0.12×0.03), density 400 → 600 (mass ~1.9kg → ~3.4kg).
- Wheels now collide with rolling friction (0.05); torso_skid demoted to visual-only. Wheelbase widened to ±0.16 X / ±0.09 Y so wheels span outside the shoulders.
- Torso spawn z lifted to rest on wheels rather than on the skid.

With these changes the creature is **stable under coordinated policies and does not fall off the platform**, which is the minimum bar. Zero falls across every trained evaluation (0/20, 0/20, 0/20).

### Visualization methodology

Three additions that outlive this specific experiment:
- **Ringside camera** (low south-side angle) — makes posture read unambiguously in renders. The overhead camera hides tipping; the ringside view cannot.
- **Head-cam temporal smoothing** (EMA, display-side only) — removes the "violently discombobulated" head shake that comes from bolting the camera rigidly to the torso. The agent's observation is unchanged; only the viewer sees a smoother feed.
- **Blur-to-sharpen overlay keyed to vision-ablation sensitivity.** Sharpness = how much the policy's action changes when pixel columns are zeroed, averaged over eval states, linearly mapped, smoothed across checkpoints, non-monotonic. Theory-truer than task success for measuring "is vision load-bearing?" — because the creature can solve the task without using vision at all.

**The sensitivity metric worked as a diagnostic.** In the stage-2 run below, it returned 0.17 (compared to 1.31 on the earlier old-env policy) — correctly flagging that the new policy nearly ignores vision. The blurry head-cam panels in the rendered videos faithfully reflect this. This is the first metric in the project that gives a clean, honest "vision does / doesn't matter" signal independent of task success.

### The experiment

Stage-1 (blind proprio): 50K steps. Peak eval reward 74 at 30K, then decayed (same late-training regression as v1 / v6). Final eval 2/20 touches, 0/20 falls.

Stage-2 (follow-on with vision, consistency loss λ=0.1, narrow 25° FOV, target spawn excludes ±25° forward cone): 250K steps. Peak eval reward 100 at 240K, final 3/20 touches, 0/20 falls. **Vision-ablation sensitivity = 0.17 — policy barely uses vision.**

### Negative finding: gaze did not emerge

The narrow FOV + target-placement bias was the theory-faithful move — environmental pressure rather than reward shaping. It did not produce gaze behavior. The creature mostly tips and gets stuck in one region of the platform; it never reliably crosses to the target, so visual localization doesn't pay off enough for the policy to learn to use it.

The theory-level lesson reinforces v6's: **pressure is necessary but not sufficient.** For gaze to emerge, AB has to (a) need to know where the target is, and (b) be capable enough to use that information. Condition (b) was missing — locomotion was not competent enough that vision would be the bottleneck. The bottleneck was staying upright and crossing distance.

### Reward-structure diagnostic (methodological byproduct)

In attempting to fix stage-1 weakness we discovered a problematic local optimum:
- Reward: +0.3/step when close to target, +200 on contact, -0.05/step hunger.
- Hovering near-but-not-touching the target pays ~+70 over a 300-step episode; a quick touch pays ~+220. Ratio ~3× — not enough to dominate when exploration is hard.

Dropping ATTRACT_SCALE 0.3 → 0.05 made hovering a loss (good), but also removed a useful exploration gradient (bad): touch rate fell from 5/20 → 2/20. The lesson: attract-reward was doing double duty as both a gradient toward target and a hover-bribe. Cleanly replacing it would require splitting those functions — e.g., attract-only-when-approaching, or making attract conditional on hand-to-target velocity.

### What v8 teaches us

1. **Ablation sensitivity is the right diagnostic for "is vision load-bearing."** It's cheap, theory-aligned, and did not get fooled by a weak-vision / weak-task policy that might have registered as "succeeding" under raw task reward.
2. **Gaze pressure is real but requires a substrate that can exploit it.** Without competent locomotion, even a perfect gaze-forcing environment produces a policy that locks itself in place rather than orienting.
3. **The platform-creature body is finally drivable**, but stage-1 reward shaping leaves it in a hover local optimum that stage-2 inherits and vision can't reasonably break.
4. The **late-training regression** pattern from v1/v6 reappears in v8 — peak at ~20-40% of training steps, decay thereafter. Same open question as v6.

### Open (next-step-worthy)

1. **Locomotion competence first.** Before another stage-2 attempt: does AB reliably cross the platform under stage-1? If not, fix the body or task (easier target placement, curriculum, looser time limit) until it can.
2. **Reward structure redesign.** Split "exploration gradient" from "hover-tolerance" in the attract term. Candidate: attract-on-approach (conditional on negative dist/dt) rather than attract-on-distance.
3. **Stage-2 with stronger stage-1.** Only meaningful once locomotion is solid.
4. **Late-training regression** (inherited open from v6): entropy annealing, early stopping, or lower learning rate in later stages.
5. **Body refinements** if locomotion doesn't come together: larger wheelbase vs shoulder span, reduced arm gear, a damped neck joint for the head camera.

---

## 2026-05-07 — entpin_005_3_hand_only_cone_2026_05_07

**What we ran:** A 250K-step follow-on from the stage1_v8 checkpoint, identical to overnight run #3 (CNN encoder, hand-only contact, ±45° spawn cone, seed=42, vec_normalize, mirror_augmentation) except that SAC's entropy coefficient was pinned at 0.05 for the entire run instead of being auto-tuned — a direct test of whether entropy collapse was causing the input-blind policy failure identified in the overnight sweep.

**Numbers:**
- ep_rew_mean: 52.7 (first stable rollout window, ~4.8K steps) → 7.0 (at 250K steps)
- Peak ep_rew_mean: 61.73 at 160K steps
- loco_speed_mean: N/A (not reported)
- touch_rate (stochastic rollout, throughout training): 0.27–0.37
- touch_rate (deterministic eval, best checkpoint at 160K): 1/20 (5%)
- consistency_loss (CNN pixel latent vs zeroed): ~0.0015 throughout
- Steps completed: 250,000

**What we learned:** Pinning entropy at 0.05 did technically work as a mechanical fix — the entropy coefficient held at 0.05 for the entire run. But the result was indistinguishable from the auto-tuned run it was meant to improve: the peak reward of 61.73 is virtually identical to overnight run #3's peak of 61.5, the same late-training collapse happened anyway (61.7 at 160K crashing to -7.6 by 210K), and the best checkpoint produced only 1/20 deterministic touches — the same floor as every other run. The entropy collapse was a symptom, not the root cause.

The most diagnostic signal is the rollout/eval gap: during stochastic training rollouts, touched_frac was 0.27–0.37 the whole time — suggesting AB is bumping into the ball roughly as often as a random direction canned paddle would. But in deterministic eval (best checkpoint, no sampling noise), only 1/20 touches. This gap means the apparent "learning" during training was the noise in the policy's sampling, not a learned directional strategy. The policy that emerges is not much better than a fixed paddle direction; it only looks better when you add stochastic jitter.

The consistency_loss at ~0.0015 throughout is the direct confirmation: the CNN pixel latent is essentially the same whether pixels are present or zeroed out. Vision was not being used at any point in training.

**Is vision load-bearing?** Not yet confirmed. The consistency_loss (~0.0015, near zero throughout) directly shows the CNN pixel latent is not being used — the policy's hidden state is the same with or without visual input.

**Next question:** Why does the late-training collapse happen even with entropy pinned — is SAC losing its good policy because the replay buffer gradually fills with episodes from the collapsed behavior, crowding out the good early data?

**Theory signals:** The rollout/eval gap (good rollout touch rate, 1/20 deterministic eval) is consistent with the Behavioral Prediction Framework's concern: the policy has not built a model of where the ball is likely to be — it has a fixed motor pattern that happens to intersect with ball locations under stochastic sampling. There is no prediction, only repetition. This is exactly what the framework predicts a policy looks like when it has not developed internal predictive structure.


### Theory Monitor Note — 2026-05-07

**Behavioral Prediction Framework: CHALLENGED** — The rollout/eval gap (27–37% stochastic touches vs. 1/20 deterministic) is a direct violation of what this framework predicts: a creature with an internal model of ball location should perform at least as well when noise is removed from its decisions, not collapse to near-zero.

**Pattern Learning Framework: CHALLENGED** — The consistency_loss of ~0.0015 throughout all 250K steps is direct evidence that the CNN pixel latent never differentiated: the creature's internal state is the same whether looking at the scene or at a blank screen, meaning no stable visual map formed at any point in training.

**The most important thing we don't know yet:** Whether a blind forward-paddle strategy is genuinely viable for ~25–30% of ball placements under the current spawn geometry — because if it is, vision faces no selective pressure to develop, and no entropy fix or architecture change will alter that.

**Recommended diagnostic** (not a training run — just a measurement): Run 20 deterministic eval episodes using the stage-1 blind proprio checkpoint (no vision, no new training) and record the touch rate. If blind proprio also scores ~1/20, the floor is set by locomotion competence and task difficulty; if blind proprio scores higher than the vision follow-on, we have a regression to explain.

---

## 2026-05-07 — Blind Proprio Baseline Eval

**Checkpoint:** stage1_v8_best (no vision, deterministic, 20 episodes)

| Metric | Result |
|---|---|
| Touched | **12/20 (60%)** |
| Fell | 0/20 |
| Timeout | 8/20 |
| Mean ep reward | 110.9 ± 102.8 |

**Comparison to all vision follow-ons:** every run scored 1/20 (5%).

**Interpretation:** Vision training is not failing to help — it is actively destroying a working 60% blind policy. This is a 12× regression, consistent across 5 independent runs varying architecture, entropy regime, contact definition, and spawn geometry.

### Theory Monitor Note — 2026-05-07 (Blind Baseline + MICOA)

**Behavioral Prediction Framework: CRITICAL VIOLATION** — A vision-augmented policy should outperform or match a blind one. Every vision follow-on scores 12× worse. The framework has no account for this.

**Pattern Learning Framework: CRITICAL VIOLATION** — Adding a redundant informative channel (vision) to a stable representation (blind 60%) should extend it, not erase it. The consistency_loss ~0.0015 confirms the pixel pathway never stabilized; it overwrote the working proprio code instead.

**MICOA reframe:** The problem is not "vision failing to help" — it is "vision not being introduced as a confirming signal." Proprio already knows how to find the ball. Vision must learn to *agree with* and *reinforce* that knowledge, not compete with it. The current concatenation architecture gives both channels equal gradient access to overwrite each other. MICOA requires that vision be added as a subordinate confirming pathway on top of a protected proprio substrate.

**Recommended next step:** Frozen proprio trunk + vision head, seeded from stage1_v8_best (the 60% policy). The freeze flag already exists in train_v8.py. This is the first true MICOA-aligned test: proprio establishes itself first, vision must inherit — not overwrite.

---

## 2026-05-07 — MICOA Freeze Run (micoa_freeze_60proprio_lam010)

**What we ran:** The first true MICOA-aligned experiment: a 150K-step follow-on from stage1_v8_best (the 60% blind proprio policy), with proprio weights completely frozen via gradient masking, a consistency loss pulling the pixel pathway toward proprio's activations (lambda=0.1), and entropy coefficient held at 0.05. The goal was to protect the working 60% blind policy from being overwritten, and let vision learn on top of it as a confirming channel.

**Numbers:**
- ep_rew_mean: 14.5 (10K) → peak 56.0 (60K) → -14.7 (150K final)
- loco_speed_mean: N/A
- touch_rate (deterministic, best checkpoint at 60K, seeds 0-19): 0/20 (0%)
- touch_rate (deterministic, best checkpoint at 60K, seeds 100-149): 4/50 (8%)
- consistency_loss: 0.25–0.29 throughout (vs. ~0.0015 in all previous vision runs)
- proprio-column drift: 0.00e+00 (freeze held perfectly)
- Falls: 0/20
- Steps completed: 150,000

**What we learned:** The freeze worked exactly as designed — proprio weights did not move by a single float, and the creature never fell, meaning the stable motor behavior from stage1_v8_best was preserved intact. The 8% touch rate on the secondary seed batch is the best any vision follow-on has achieved, though still a long way from the 60% blind baseline. The most important new number is the consistency_loss: at 0.25–0.29, it is roughly 150 times higher than in every previous run. This tells us something genuinely new has happened: the pixel pathway is now producing activations that differ meaningfully from what proprio alone would produce. In every prior run, the camera was essentially ignored (consistency_loss near zero); here, the camera is doing something. But that something may be the wrong thing. A high consistency_loss under MICOA's formulation means vision is diverging from proprio rather than confirming it — the two pathways are in tension, not agreement. The policy evaluation still looks mostly like timeouts (8/20 on the best seed batch, zero touches on the main seed batch), suggesting vision's distinct signal has not yet translated into directional ball-finding behavior.

The eval trajectory also shows the same late-training collapse seen in every previous run: peak reward of 56.0 at 60K, then erratic decline to -14.7 by 150K. The freeze did not prevent this; the instability appears to be in the vision pathway and the downstream critic/actor joint optimization, not in proprio itself.

**Comparison table:**

| Policy | Deterministic touch rate |
|---|---|
| stage1_v8_best (blind proprio) | 12/20 (60%) |
| micoa_freeze best (60K), seeds 0-19 | 0/20 (0%) |
| micoa_freeze best (60K), seeds 100-149 | 4/50 (8%) |
| All overnight runs + entpin | 1/20 (5%) |

**Is vision load-bearing?** Not yet confirmed. The consistency_loss of 0.25–0.29 is the first evidence that the pixel pathway is producing a distinct signal — in all prior runs it was near zero and the camera was effectively ignored. But a distinct signal is not the same as a useful signal. An 8% touch rate (vs. 60% blind) means vision's divergent contribution is not yet guiding the creature toward the ball. The freeze confirmed proprio's stability; it did not confirm that vision is helping.

**The consistency loss paradox:** In previous runs, consistency_loss ~0.0015 meant the pixel pathway was not doing anything at all — vision latent equaled blind latent, so there was nothing to lose from zeroing pixels. In this run, consistency_loss 0.25–0.29 means the pixel pathway is producing activations that differ substantially from proprio's. But under the MICOA objective, vision is supposed to *converge* toward proprio's activations, not diverge from them. The high loss means the optimizer is failing to pull vision onto proprio's manifold — they are in tension. There are two possible readings: (1) the pixel pathway has developed features that genuinely reflect the visual scene, but those features point in a different direction than proprio's ball-finding strategy — a tug-of-war rather than confirmation; (2) the pixel pathway is producing arbitrary activations that happen to be different from proprio's, not because they encode anything meaningful, but because the optimizer has not found a way to reconcile them. We cannot distinguish these from the current data alone.

**Failure mode shift:** All previous vision follow-ons failed partly through falls or through pure entropy collapse. This run failed through timeouts with zero falls. The freeze preserved the stability but not the directional capability: the creature stays upright and keeps moving, but its motion is not reliably aimed at the ball. This is a different, arguably cleaner failure — the creature is not broken, it just doesn't know where to go.

**Next question:** Why does the consistency loss remain high (0.25–0.29) throughout 150K steps despite the optimizer working to minimize it — is the frozen downstream architecture so rigid that vision literally cannot find a configuration that agrees with proprio, or is the learning rate inadequate to drive convergence in 150K steps?

**Theory signals:** The consistency_loss pattern offers weak evidence against the Pattern Learning Framework's prediction. Under that framework, the pixel pathway should converge to a sparse, stable pattern that overlaps with proprio's representation of "ball location" — we would expect consistency_loss to fall toward zero as training proceeds. Instead it stayed elevated and volatile. This is more consistent with the pixel pathway searching for a representation that the frozen architecture cannot absorb — the downstream geometry established by proprio may be too rigid to accommodate a genuinely new visual signal, which is exactly the cost the v2/v3 findings predicted when they noted that "vision can only speak Stage 1's language."


---

## 2026-05-07 — MICOA Zero-Init + Forward Cone (micoa_freeze_zeroinit_cone30_2026_05_07)

**What we ran:** A 150K-step follow-on from stage1_v8_best (the 60% blind proprio policy), using the same MICOA freeze architecture as the previous run (frozen proprio, pixel-column gradient, consistency loss lambda=0.1, ent_coef=0.05), but with two new changes designed to fix why vision never switched on: (1) pixel columns zero-initialized at training start so vision begins completely silent rather than carrying random noise the optimizer had to suppress; (2) ball spawn cone narrowed to ±15° straight ahead so the ball is always inside the head camera's field of view from the very first frame (previous runs excluded the ±25° forward cone, meaning the ball was never in camera at reset).

**Numbers:**
- ep_rew_mean: 39.9 (10K, new best early start) → 55.6 peak (30K) → -9.1 (150K final)
- loco_speed_mean: N/A
- touch_rate (deterministic, best checkpoint at 30K, 50 episodes): 4/50 (8%)
- consistency_loss: 0.20–0.38 throughout (higher early than previous MICOA run)
- proprio-column drift: 0.00e+00
- Steps completed: 150,000
- Falls: 0/50
- Timeouts: 46/50

**What we learned:** The two new changes produced a meaningfully different early-training trajectory without changing the final result. The starting reward jumped to 39.9 at 10K steps — more than twice the previous MICOA run's 14.5 at the same point — which suggests that having the ball visible in the camera from frame one gave the policy something to work with immediately. The best checkpoint arrived earlier (30K vs. 60K), and the consistency_loss started higher (indicating the pixel pathway was more active from the start) before the usual late-training collapse took over. However, the deterministic touch rate at the best checkpoint matched the previous MICOA run exactly: 4/50 (8%). The same gap to the 60% blind baseline persists. We reached the ceiling faster but did not raise it.

The most interesting new observation is the rollout symmetry: touched_left_frac = 0.102, touched_right_frac = 0.098 — nearly perfectly balanced. Every previous run showed strong left-side bias (0.386 left / 0.196 right in both the previous MICOA run and the entpin run). Since the ball now spawns ±15° from straight ahead, a policy that is genuinely responding to what its camera sees should produce equal left/right contacts — the ball is equally likely to be slightly left or slightly right, so a vision-steered creature should touch it symmetrically. That is what we observe for the first time here. This is circumstantial, not conclusive — perfect symmetry could also come from a fixed forward-paddle that doesn't use vision at all — but it is the first behavioral pattern in the project that is consistent with directed visual steering rather than random or biased wandering.

**Comparison table:**

| Policy | Det. touch rate | Notes |
|---|---|---|
| stage1_v8_best (blind) | 12/20 (60%) | baseline |
| micoa_freeze_zeroinit_cone30 best (30K) | 4/50 (8%) | this run |
| micoa_freeze best (prev, 60K) | 4/50 (8%) | seeds 100-149 |
| entpin + all overnight runs | 1/20 (5%) | all others |

**The persistent gap to 60% blind:** Every MICOA-aligned run now lands at 8%, compared to 5% for all non-MICOA runs. That is an improvement at the margins, but the blind proprio baseline sits at 60% — a gap of roughly 7× that has not closed. There are two live interpretations of this gap. First, it may mean that 150K steps is not enough for the pixel pathway to converge under a frozen downstream architecture — the optimizer has to teach vision to speak in proprio's language, and proprio's language is a rigid geometric constraint, which takes time. Second, it may mean that the forward-cone change is still not enough environmental pressure: even with the ball always in front, if the creature can find the ball through proprio-guided wandering within the cone, vision faces no selective pressure to develop. The symmetry observation weakly supports the first interpretation over the second, but we need an ablation to distinguish them.

**Is vision load-bearing?** Not yet confirmed — but the first weak evidence is now present. The near-perfect left/right touch symmetry (0.102 / 0.098) is consistent with visual steering and is qualitatively different from every prior run's left-biased contact pattern. The consistency_loss remaining elevated (0.20–0.38) confirms the pixel pathway is producing a distinct signal. Neither of these is proof — an ablation test comparing touch rate with vision on versus vision zeroed at the best checkpoint is needed to make the claim.

**Next question:** Does the best-checkpoint policy (30K steps) produce a different touch rate when pixels are zeroed versus when they are live — specifically, is the 8% score vision-dependent or would the same checkpoint score 8% with a blank screen?

**Theory signals:** The rollout symmetry shift from biased (0.386/0.196) to balanced (0.102/0.098) is weak positive evidence for the Behavioral Prediction Framework. A policy with an internal model of the ball's likely location — informed by a camera that now sees the ball from frame one — would predict the ball to be equally likely on the left or right given ±15° symmetric spawning, and would produce symmetric contact. The previous biased pattern was consistent with a fixed motor habit, not a predictive model. This is the first behavioral signature that looks more like prediction than repetition. The Pattern Learning Framework would predict the pixel pathway to converge toward sparse, stable patterns that overlap with proprio's ball-finding representation; the elevated consistency_loss (0.20–0.38) suggests that convergence has not happened yet, though the earlier and higher activation compared to the previous MICOA run is a weak sign that the zero-init change created a better starting point for that convergence.


### Theory Monitor Note — 2026-05-07 (Zero-Init + Forward Cone)

**Behavioral Prediction Framework: CHALLENGED** — The near-perfect rollout symmetry (left 0.102 / right 0.098) is consistent with visual steering but equally consistent with a symmetric spawn distribution alone. Without breaking the spawn symmetry and checking whether touch distribution follows the ball, we cannot credit purposeful directional behavior.

**Pattern Learning Framework: CHALLENGED** — Zero-initialization was designed to let a stable visual pattern emerge gradually; instead consistency_loss rose to 0.20–0.38 faster than the prior run, meaning the visual pathway finds a high-divergence configuration even from a neutral zero state. The opposite of gradual stable pattern formation.

**The most important missing number:** Vision-ablation sensitivity on the 30K best checkpoint — how much does the action change when pixels are zeroed? Until that is measured, we do not know whether the 8% touch rate has any visual contribution or is purely proprio-driven wandering within the ±15° cone.

**Recommended next diagnostic:** Run 30 deterministic eval episodes with left-only spawn (ball 10–15° left of forward) and 30 with right-only spawn. If touch rate differs between conditions, vision is providing directional information. If symmetric, the left/right balance is geometric coincidence.

### Theory Monitor Note — 2026-05-07 (Vision-Ablation Breakthrough)

**Behavioral Prediction Framework: PARTIALLY CONFIRMED** — Vision is now confirmed load-bearing on every measured step (mean action diff 0.456, 100% above threshold). But the directional spawn test (0/30 both sides) shows the visual consultation is not yet directionally accurate.

**Pattern Learning Framework: PARTIALLY CONFIRMED** — 100% step-level sensitivity is consistent with a stable internal visual representation. But "ball slightly left" vs "ball slightly right" has not yet produced different directional motor behavior.

**The most important thing we don't know yet:** Whether the visual influence is directionally undifferentiated ("something visible" but not "at bearing X") or whether the creature can detect direction but can't yet execute the corresponding turn. These require different fixes.

**THEORETICAL CONCERN:** The breakthrough exists only at the 30K checkpoint. We don't know if vision sensitivity survives or collapses during the late-training period. The 150K final checkpoint may have reverted to vision-blind behavior. Measuring ablation sensitivity at the final checkpoint is the outstanding check.


---

## 2026-05-07 — Stage1 Headfix + Velocity Bonus (stage1_headfix_velbonus_2026_05_07)

**What we ran:** A 500K-step stage1 training run from scratch under two physics and reward fixes committed earlier today: head servo gain reduced from kp=30 to kp=5 (head now moves slowly instead of snapping), and a velocity bonus of VELOCITY_BONUS_SCALE=0.02 x torso_speed added (moving is always better than standing still). This run replaces stage1_v8_best, which video review showed was a stationary creature getting spawn-luck touches despite its 60% blind touch rate.

**Numbers:**
- ep_rew_mean: peak 176.9 at 270K → sustained 90–135 range from 200K–500K → final 112.6 at 500K
- loco_speed_mean: N/A (not separately logged; mean_dist_mean = 0.417, ep_len_mean = 151 steps)
- touch_rate (rollout, 500K): 0.69 (touched_left_frac 0.569, touched_right_frac 1.0)
- touch_rate (deterministic eval, 20 episodes, blind, seeds 0–19): 9/20 (45%)
- Falls (deterministic eval): 0/20
- Steps completed: 500,000

**Render results (5 episodes, seeds 0–4):**

| Seed | Outcome | Contact step |
|---|---|---|
| 0 | TOUCHED | 176 |
| 1 | TIMEOUT | 300 |
| 2 | TOUCHED | 130 |
| 3 | TOUCHED | 39 (very fast) |
| 4 | TIMEOUT | 300 |

**What we learned:** This run behaves fundamentally differently from stage1_v8_best. Training stayed in the 90–135 reward range for most of the 200K–500K window without catastrophic collapse — the previous run had no such sustained plateau. The 45% deterministic touch rate is lower in raw number than stage1_v8_best's 60%, but stage1_v8_best was declared invalid after video revealed the creature was nearly stationary and the head was oscillating wildly. The velocity bonus appears to have broken the stillness local optimum: mean_dist_mean of 0.417 and contact at step 39 on one seed both suggest the creature is actively covering ground. However, two timeouts at 300 steps mean some ball positions remain unreachable. The right-side rollout touch rate of 1.0 (every right-side rollout touch was successful) alongside a 0.569 left-side rate suggests the creature may have a directional asymmetry in its paddle stroke — worth noting for video review.

**Human verification required before vision follow-on begins:** The 5 rendered videos at `alien_baby/results/videos/v8_sanity_stage1_blind_trained_headfix_velbonus_best_seed{0-4}.mp4` must be watched to confirm: (1) the head moves slowly and calmly rather than oscillating across full range, (2) the creature actively covers ground during each episode rather than sitting in one spot, and (3) contact events result from movement toward the ball rather than ball spawning adjacent to the creature.

**Is the locomotion goal met?** The project's target was 60% deterministic touch rate. This run achieved 45%. But the previous 60% was spawn-luck from a stationary creature, and video review is the only reliable way to judge whether 45% from a moving creature is "better" than 60% from a stationary one. The sustained training reward, nonzero mean_dist, and fast seed-3 contact (step 39) all suggest qualitatively improved locomotion. Formal decision on whether to proceed to vision follow-on must wait for video confirmation of calm head and active movement.

**Is vision load-bearing?** Unknown — this is a blind proprio run with no visual input. Vision cannot be measured or confirmed until a follow-on run is conducted from this checkpoint.

**Next question:** When the 5 rendered videos are reviewed: does the creature move actively toward the ball with a calm head, or does it still rely on the ball spawning nearby?

**Theory signals:** The fast contact at seed 3 (step 39) is the first behavioral datum in this project's history that is straightforwardly consistent with the Behavioral Prediction Framework's prediction — a creature that has internalized a motor program for "move toward likely ball positions" would produce fast touches across seeds where the ball is favorably placed. Whether this is a directional motor program or just a lucky spawn cannot be determined without video. The velocity bonus creating active locomotion from an agent that previously found stillness optimal is also consistent with the Pattern Learning Framework's expectation that the agent should generalize its movement patterns broadly — but only if the movement patterns are real and not incidental.

---

## 2026-05-07 — Stage1 v2: Velocity Bonus + Forward Cone + Longer Episodes (stage1_headfix_velbonus2_cone180_600steps_2026_05_07)

**What we ran:** A 500K-step stage-1 training run from scratch, building on the immediately preceding velbonus v1 run. Three parameters were changed simultaneously: the velocity bonus was increased from 0.02 to 0.05 (2.5x stronger incentive to move), the ball spawn cone was expanded to 180 degrees so the ball always spawns in the creature's front hemisphere rather than potentially behind it, and the maximum episode length was doubled from 300 to 600 steps (approximately 30 seconds of simulated time). The goal was to push the deterministic touch rate above the 60% target that the retired stage1_v8_best had reached through spawn-luck, and to produce an actively locomoting creature whose locomotion quality could be confirmed by video before vision follow-on begins.

**Numbers:**
- ep_rew_mean: oscillated 5–156 throughout training; reward was not monotonic
- Best checkpoint (deterministic eval, 20 episodes, cone180, 600 steps): **17/20 (85%)**
- loco_speed_mean: N/A (not separately logged); mean_dist = 0.417
- touch_rate (rollout at 500K): touched_frac = 0.45; touched_left = 0.549, touched_right = 0.347
- touch_rate (deterministic eval, best checkpoint): 17/20 (85%), 0 falls, 3 timeouts
- Mean ep reward (best checkpoint): 165.1 ± 78.6
- Mean ep length (best checkpoint): 198 steps
- Steps completed: 500,000
- Run completion: Normal

**Rendered sanity episodes (seeds 0–4, capped at 300 steps — env default, not 600):**

| Seed | Outcome | Notes |
|---|---|---|
| 0 | TIMEOUT at step 300 | Would likely resolve with full 600 steps |
| 1 | TIMEOUT at step 300 | Same — render cap shorter than policy budget |
| 2 | TIMEOUT at step 300 | Same |
| 3 | TOUCHED at step 8 | Exceptionally fast — ball must have spawned very close |
| 4 | TOUCHED at step 267 | Active movement; ball found near end of episode |

Note: the 3 render timeouts are misleading because sanity_render doesn't use max_steps_override=600. The deterministic eval used the full 600-step budget, where only 3/20 timed out.

**Comparison to all prior stage-1 runs:**

| Policy | Det. touch rate | Falls | Notes |
|---|---|---|---|
| stage1_v8_best (retired) | 12/20 (60%) | 0 | Stationary creature; spawn-luck; declared invalid on video |
| stage1_headfix_velbonus v1 | 9/20 (45%) | 0 | kp=5, vel=0.02, 300 steps; pending video confirmation |
| **stage1_headfix_velbonus v2 (this run)** | **17/20 (85%)** | **0** | kp=5, vel=0.05, cone180, 600 steps |

**What we learned:** This run cleared the 60% target by a wide margin and did so against a harder spawn distribution. The cone180 spawn means the ball is always in front of the creature but can be anywhere in the front half of the arena — the creature cannot succeed by pointing in one fixed direction and hoping. The 600-step budget means balls that require active searching can still be found. The 85% rate under those conditions is qualitatively different from the old 60%, which was achieved against a smaller spawn cone on a stationary creature. Five consecutive new best-checkpoints appeared early in training, indicating the locomotion policy was actively and consistently improving rather than riding early noise. The reward oscillated between 5 and 156 throughout the full run — this is not a sign of instability but of a task that rewards variable amounts depending on how quickly the ball is found. The three timeouts in the deterministic eval represent the hardest ball positions (far from the creature's typical patrol area), not a systemic locomotion failure.

**Is vision load-bearing?** Unknown — this is a blind proprio stage-1 run with no visual input. Vision cannot be tested until a follow-on run is built from this checkpoint. However, this run establishes the strongest locomotion substrate the project has yet produced, which is the necessary prerequisite for any vision follow-on to have a chance of working. The previous MICOA experiments were built on a stage-1 foundation that didn't locomote; this one does.

**Next question:** When the rendered episodes are reviewed by a human, does the creature actively move toward the ball across multiple seeds — including seeds where the ball is not immediately adjacent — confirming that 85% reflects search behavior rather than spawn-luck?

**Theory signals:** The 85% rate under cone180 (ball anywhere in front hemisphere) is the first result in this project's history that is difficult to explain by spawn-luck alone. A stationary creature could not achieve 85% across a 180-degree front hemisphere without an implausibly small spawn radius. This is weak positive evidence for the Behavioral Prediction Framework's prediction: the creature has internalized a motor program that moves it toward likely ball positions, rather than simply waiting. The Pattern Learning Framework would predict that the velocity bonus shaped a stable movement pattern that generalizes across ball positions — the monotonic early improvement (5 new bests in a row) is consistent with a pattern solidifying rather than oscillating between strategies.

---

## 2026-05-11 — Phase I: No-bribery substrate, existing CNN (mimo_substrate_A)

**What we ran:** A 250K-step training run on the MIMo crawler under a deliberately stripped-down substrate that removes every behavior-specific reward shaping the prior runs depended on, while keeping the existing flat-concatenation StereoCrawlerCNN architecture. The intent was to test, as a control, whether the substrate change alone — no velocity bonus, no approach reward, no FOV reward, 360° ball spawn so blind-forward-paddle cannot exploit any spawn bias, 2000-step episodes so the creature has time to recover from bad initial direction, 20cm guardrails so falling off is impossible, and a 15° downward camera tilt so the ball (resting on the platform) can in principle appear in the eye-camera FOV past the prone arms — is enough to make vision become load-bearing. Other config: 4 envs DummyVecEnv, MPS, lr=1e-4 (DrQ-v2 recommended), ent_coef=0.2 fixed, buffer_size=100K, learning_starts=10K. The only reward signals are +200 on ball contact and −0.05 per step (hunger). The creature has zero information about ball direction from proprio.

**Numbers:**
- ep_rew_mean over evals (20 deterministic episodes each):
  - 50K:  −85.04 ± 65.20  (touch est: 1/20 = 5%)
  - 100K: **−60.01 ± 96.80**  (touch est: 3/20 = 15%) ← peak
  - 150K: **−100.00 ± 0.00**  (touch est: 0/20) ← total collapse
  - 200K: −70.08 ± 89.75  (touch est: 2/20 = 10%)
  - 250K: **−100.00 ± 0.00**  (touch est: 0/20) ← total collapse again
- Best-checkpoint render (5 seeds × 600 steps): 0/5 touched (all timeouts)
- Vision-ablation sensitivity (15 seeds × 200 steps, pixels zeroed): mean L2 action delta = **0.0226** (median 0.0238); 0/15 touched in either condition
- Steps completed: 250,000

**What we learned:** The substrate change alone is not sufficient. The eval trajectory replicates the exact same SAC late-stage collapse pattern documented across v5–v11 (peak at ~100K, complete collapse at 150K, partial recovery, second collapse at 250K), even though every behavior-specific reward signal has been removed. More diagnostically: the std=0.00 at 150K and 250K means every one of 20 deterministic episodes returned exactly −100.00, which equals 2000 steps × −0.05 step cost with no contact. That is not a high-variance failure — it is the deterministic policy converging to a literal "do nothing useful" attractor in policy space. The CNN-and-flat-concatenation actor finds it consistently. Frame inspection of the rendered best-checkpoint (100K) episode confirms the behavior: the creature flails into a tipped-over pose on its back with legs in the air, eye cameras see chaotic tilted views, and the policy continues to produce actions that maintain the failed pose rather than recover. The vision-ablation result rules out the alternative that vision was contributing but invisibly: mean delta 0.0226 is essentially zero — the same level v10 produced under its much easier (cone 45°) substrate. The pixel pathway is not being used.

**Is vision load-bearing?** No. Mean ablation L2 delta of 0.0226 is far below the 0.5 threshold (or any meaningful threshold). The pixel observation is being ignored by the actor across all seeds tested. The substrate's hardness — 360° spawn with no proprio direction signal — does not, on its own, force a flat-concatenation architecture to develop a useful visual pathway. The honest reading is that the architectural problem identified in the May 7 retirement memo (3072 pixel dims drowning 69 proprio dims under joint SAC training, with no mechanism enforcing that vision *confirm* proprio rather than overwrite it) survives the substrate change.

**Failure mode shift:** Worth noting separately from prior runs. v10 collapsed to 20% deterministic and stabilized. v11 collapsed and partially recovered. This run, mimo_substrate_A, collapsed to **0%** deterministic — the worst outcome of any run on the MIMo body — and showed total-zero deviation across all 20 episodes at both 150K and 250K. Removing the velocity bonus and approach reward (the bribery components) deprived the policy of the gradient signals that prior runs implicitly relied on to maintain locomotor activity even when vision was silent. Without those signals, the only reward gradient comes from sparse contact, which the policy never reaches deterministically — so the actor collapses toward a no-action attractor. This is a clean negative result for the "remove bribery, hope vision develops" hypothesis: when bribery goes away, the architecture has nothing to learn from until vision provides direction, and vision will not provide direction unless the architecture is structured to listen.

**Next question:** Does the dialogue architecture (two-stream actor with proprio + vision producing independent action proposals, a learned scalar gate combining them, and a consistency loss `λ · ||μ_p − μ_v||²` penalizing per-stream disagreement) escape this attractor on the same substrate? Phase II is currently running (300K steps, mimo_dialogue_v1) with identical environment and reward configuration. If Phase II's vision-ablation sensitivity comes back above 0.5 with touch rate > 25%, the structural change is what made vision load-bearing — substrate alone was a necessary but not sufficient prerequisite. If Phase II also collapses, the MIMo body is the limiting factor and the next experiment should be on the platform creature with the dialogue architecture instead.

**Theory signals:** The Behavioral Prediction Framework predicts that an agent with a viable internal model of where contact is achievable should outperform random walk on the 360° spawn distribution (random walk on a 2m × 2m platform with a 5cm ball over 2000 steps should produce ~30-50% touch by ballistic coverage alone). The deterministic 0% touch rate is far below the random-walk baseline, indicating the policy has not built a predictive model — it has been driven by the SAC update dynamics into a state of literal inaction. This is a stronger violation of the framework's prediction than any prior run, because no prior run had a substrate that supported random walk baseline at this level. The Pattern Learning Framework predicts that the visual representation should respond to recurring patterns in the camera feed (the ball as a stable red structure, the guardrail walls as a recurring spatial feature). The 0.0226 ablation delta shows no such pattern has developed — the visual pathway is not just unused for control, it is structurally inert. Both frameworks are challenged in the same direction: when bribery is removed, the flat-concatenation architecture produces a less capable policy, not a more grounded one. The architecture, not the substrate, is the load-bearing failure.

---

## 2026-05-11 — Phase II: Dialogue architecture on no-bribery substrate (mimo_dialogue_v1)

**What we ran:** A 300K-step training run with the same no-bribery substrate as Phase I (360° spawn, 2000-step episodes, guardrails, camera tilt, no velocity bonus, no approach reward) but with the **dialogue architecture** from the May 7 retirement memo: a two-stream actor where proprio (69 dims) and vision (stereo CNN → 64-dim latent) each produce independent action proposals (μ_p, μ_v), a learned scalar gate w combines them as a = w·μ_p + (1−w)·μ_v, and a consistency loss λ · ||μ_p − μ_v||² (λ = 0.05) is added to the actor objective to pressure the two streams toward agreement. Same SAC hyperparameters as Phase I: lr=1e-4, ent_coef=0.2 fixed, buffer 100K, learning_starts 10K. The run was killed at 112K when the trajectory made the outcome predictable.

**Numbers (rollout, no completed eval at kill time):**
- ep_rew_mean trajectory: −100 (8K) → −66.5 (16K) → −56.3 (24K, peak) → −66.6 (32K) → −73 (40K) → −65.3 (48K) → −70.1 (56K) → −63.6 (58K) → −67.6 (66K) → −70.8 (74K) → ~−65 throughout the rest
- Gate trajectory: started at 0.49 (random), collapsed to 0.0047 by 40K, slowly recovered to 0.07 by kill at 112K
- Disagreement (||μ_p − μ_v||) trajectory: 0.090 (16K) → 0.099 (24K) → 0.105 (32K) → 0.112 (40K) → 0.178 (48K) → 0.196 (56K) → 0.222 (60K) → 0.271 (66K) → 0.415 (74K) → ~0.42 (steady)
- Consistency_loss trajectory: 0.013 (16K) → 0.020 (40K) → 0.055 (48K) → 0.086 (60K) → 0.123 (66K) → 0.285 (74K) → 0.31 (steady)
- μ_proprio_mean ≈ 0.19; μ_vision_mean ≈ 0.085 (vision actions roughly half the magnitude of proprio actions throughout)
- First eval at 60K: episode_reward = −85.03 ± 65.26 (≈ same as Phase I 50K eval)
- Steps completed before kill: 112,000

**What we learned:** The dialogue architecture, as configured, degenerated into a *monologue*: the gate w collapsed to near zero (0.005 at its low) within 40K steps, meaning the policy effectively ran the vision stream alone with the proprio stream's contributions multiplied by ~0. The asymmetry is structural in the gradient flow: with w ≈ 0, the SAC actor loss flows primarily to μ_v (because the combined action a ≈ μ_v), so μ_v gets a strong reward gradient. The proprio stream μ_p receives essentially no SAC gradient — only the smaller consistency-loss pressure. As μ_v drifts toward Q-value-maximizing actions, μ_p slowly lags behind via the consistency term, but it cannot keep up. Disagreement and consistency_loss grow together (0.09 → 0.42 and 0.013 → 0.31 respectively over 100K steps), exactly the opposite of what the consistency objective was supposed to produce. By the kill, the gate began ticking back up (0.005 → 0.07), suggesting a long-run equilibrium might emerge where proprio re-enters, but at the cost of compute we judged not worth investing given the trajectory looked like Phase I in slow motion.

**Vision-stream magnitude problem:** A separate diagnostic from the disagreement trajectory: throughout training, μ_vision_mean (~0.08) was about half the magnitude of μ_proprio_mean (~0.19). With the gate weighting vision at ≥ 93%, the actual combined action magnitude was dominated by these smaller vision actions. The deterministic policy therefore output unusually small actions — which, on a 25-DOF MIMo body, are not enough to produce locomotion. The combination "gate ≈ 0, μ_v is small" produces a deterministic policy that barely moves. This is consistent with the rollout reward plateauing at −65 (~14% one-ball touch rate) and explains why the eval at 60K was identical to Phase I's eval at 50K.

**Is vision load-bearing?** Unconfirmed — and architecturally misleading. The ablation test (zeroing pixels and measuring action delta) would likely report large delta because zeroing the inputs to a network with gate ≈ 0 makes μ_v change dramatically, and the policy IS combined ≈ μ_v. So the ablation would say "vision is load-bearing," but only in the degenerate sense that the architecture made vision structurally load-bearing by zeroing out the proprio stream. This is not the MICOA outcome — there is no confirmation, only dependence. A genuine load-bearing vision pathway would have w in some intermediate range (say 0.3–0.7) with disagreement decreasing over training (vision and proprio converging on the same answer). What we got was the opposite: w collapsing and disagreement growing.

**Next question:** Why does the gate collapse? Two hypotheses: (1) μ_p has noisier outputs early in training because proprio actions correspond to "complex motor coordination," while μ_v is simply CNN-derived and outputs small values around zero. SAC favors lower-variance action distributions early because they produce more consistent (and slightly higher) Q-values, so the gate learns to suppress μ_p. (2) The consistency loss with the gate's product structure has an asymptotic instability — the gradient on the gate scales with (μ_p − μ_v) in a way that pushes w toward 0 or 1 rather than maintaining a balance. Either way, the symmetric-peers design of the dialogue architecture is not symmetric in practice. A fix worth considering: forbid the gate from going below some floor like 0.2 (force partial proprio contribution).

**Theory signals:** The MICOA hypothesis predicts that vision and proprio converge on the same answer through training, with disagreement shrinking. The observed disagreement growing (0.09 → 0.42) is direct refutation of MICOA convergence under this architecture. The dialogue framing is structurally correct (two streams, explicit gate, consistency objective) but the implementation produces anti-convergence. The Pattern Learning Framework predicts a stable visual pattern should develop; the consistency_loss growing throughout (it would shrink if a stable joint pattern formed) is direct refutation. Both frameworks predict the same direction: the architecture is not producing the integration it was designed for.

---

## 2026-05-11 — Phase III / III.2: Fixed-ball-position substrate (mimo_phase_c_*)

**What we ran:** Two experiments testing whether the random-spawn distribution is the load-bearing failure rather than the architecture. The user's insight: random spawn means every episode is a fresh problem with no stable spatial structure for the agent to model. With *fixed* ball positions, the world has a stable structure the agent can learn to navigate to. Two variants tested:

- **Phase III**: Two fixed balls at (0.7, 0.0) and (0.0, 0.7), random starting orientation per episode (creature spawns prone facing a random yaw), blind proprio (no vision), no bribery, 2000-step episodes, episode terminates only when both balls touched. Killed at 56K after first eval.
- **Phase III.2**: Same as Phase III, plus **memory_obs** appended to the observation: two binary flags (`touched_ball1`, `touched_ball2`) get concatenated to proprio at every step, letting a stateless SAC policy condition on its own past contacts within an episode. Killed at 56K after first eval.

**Numbers:**

| Run | Step | ep_rew_mean | Eval @50K |
|-----|------|------------:|----------:|
| Phase III  | 8K  | −50    | (run too short for eval landing) |
| Phase III  | 16K | −50    | |
| Phase III  | 24K | **−16.7** ← peak | |
| Phase III  | 32K | −25    | |
| Phase III  | 40K | −40    | |
| Phase III  | 50K | (eval) | **−100.00 ± 0.00** |
| Phase III.2 | 8K  | −50    | |
| Phase III.2 | 16K | −75    | |
| Phase III.2 | 24K | −50    | |
| Phase III.2 | 32K | −50    | |
| Phase III.2 | 40K | −60    | |
| Phase III.2 | 48K | −66.7  | |
| Phase III.2 | 50K | (eval) | **−100.00 ± 0.00** |

Both runs collapsed to identical deterministic-eval failure: every one of 20 episodes returned exactly −100.00, meaning zero contacts and zero variance — the deterministic policy outputs near-zero actions and the creature does not move during eval. Same total-zero attractor as Phase I.

**What we learned:** Fixed ball positions alone do not rescue the substrate. Adding within-episode memory flags also does not rescue it. The deeper failure is upstream: under no-bribery conditions with sparse contact reward, SAC's deterministic policy collapses to zero-action regardless of whether the spatial structure is learnable or whether memory of past contacts is available. The collapse happens because the policy receives no consistent gradient signal for locomotion — non-zero actions occasionally produce contact reward (during stochastic exploration with ent_coef=0.2), but they also produce step_cost and proprio chaos when the body tips over. SAC averages over this uncertainty by going to zero. Both fixed positions and memory require a working locomotor base to be useful, and the no-bribery setup doesn't produce one. The user's two-ball + memory hypothesis is therefore not refuted — it is not testable in this configuration, because the prerequisite (the agent reliably moves at all) was never satisfied.

**The dependency graph this reveals:** body works → can move → can encounter things → can remember encounters → can use memory + stable structure to map. We confirmed empirically that the chain breaks at link 2 ("can move") in the no-bribery setting. Memory flags (link 4) and fixed structure (link 6) are downstream-of-locomotion features and cannot compensate for an absent locomotion base.

**Is vision load-bearing?** Not applicable; both Phase III and III.2 are blind proprio.

**Stochastic vs deterministic gap, revisited:** In all three failed runs (I, III, III.2), the same pattern: stochastic rollout shows touch rates of 10–30% (ep_rew_mean around −20 to −70), but deterministic eval shows zero contacts (mean reward −100, std 0). The exploration noise from ent_coef=0.2 is the only thing producing contacts during training; the policy mean has not learned to produce contacts on its own. This is the structural diagnostic: SAC + no-bribery + sparse contact reward + MIMo body has *not* in 250K-300K steps developed a locomotion policy whose mean produces motion. Whatever motion happens in rollouts is the noise, not the policy.

**Next question:** Does adding back a *small* velocity bonus (say 0.01–0.02, vs the 0.05 of the prior 85% blind run, vs 0.0 in Phases I/III/III.2) restore locomotion without resurrecting the bribery-driven failure modes? The user's framing distinguishes "enabling locomotion" (substrate, fair) from "paying for a specific behavior" (bribery, not fair). A small velocity bonus is closer to the first — it tells the creature "moving is better than not moving" but not WHICH direction to move. Combined with fixed ball positions and memory flags, this might finally produce the locomotor base needed for the spatial-structure hypothesis to be testable.

**Theory signals:** The Behavioral Prediction Framework's prediction that an agent should outperform random walk on a learnable task is violated in all three runs. The agent does not produce random walk — it produces *no walk*. This is a stronger failure than the framework anticipates. The agent has not built any internal model, predictive or otherwise; it has gone to a null policy. The Pattern Learning Framework expected stable representations to form around recurring features of the environment. With fixed ball positions, the world *has* the stable structure required, but the policy never reaches a state where it could differentially respond to those features. Both frameworks predict that with the substrate fixed in this way, *something* should be learnable; the result that *nothing* is learnable is informative — it tells us the prerequisite layer (motor activity that produces consequences) is not in place. The dependency-graph principle (body → motion → encounter → memory → map) maps onto Taylor's developmental progression: you cannot build perceptual equivalence classes for objects you have never encountered, and you cannot encounter objects without a working motor system that produces encounters.

---

## 2026-05-11 — Phase IV: HER + fixed balls + memory + small velocity bonus (mimo_phase_d_her)

**What we ran:** A 250K-step training run combining everything from Phase III.2 plus **Hindsight Experience Replay** (Andrychowicz et al. 2017) — the canonical literature solution for sparse-reward goal-reaching. Config: HERCrawlerWrapper exposing goal-conditioned Dict obs (achieved_goal = hip XY, desired_goal = active ball XY), HerReplayBuffer with `n_sampled_goal=4` and `goal_selection_strategy="future"`, MultiInputPolicy, VecNormalize with `norm_obs_keys=["observation"]`. Same substrate as III.2: two fixed balls at (0.7, 0) and (0, 0.7), random starting orientation, blind proprio (mono camera in env but no vision in policy), memory flags. Plus a small `velocity_bonus_scale=0.02` to test the "enabling pressure" hypothesis from the Phase III.2 next-question. 16 parallel envs on MPS, `learning_starts=40000` (required by HER's per-env first-episode constraint: needs ≥ `n_envs × max_steps`).

**Crash diagnostics on the way in:** First launch crashed at startup with `EXC_BAD_ACCESS` in `_datetime.delta_new` during `_Py_Finalize → finalize_modules → gc_collect_main`. This is a Python 3.13 + MuJoCo + PyTorch shutdown race on macOS: any uncaught exception causes Python's interpreter teardown, which calls a generator's `__del__` chain that touches `datetime.timedelta()` after `_datetime` is partially torn down. Fixed by (a) `faulthandler.enable()` at module top for real tracebacks instead of opaque segfaults, (b) wrapping `train(args)` in try/except + `os._exit(exit_code)` to skip Python's broken finalization entirely whether training succeeds or raises, and (c) setting `progress_bar=False` in `model.learn()` to avoid the `tqdm.rich` destructor's `Live.refresh → datetime` chain. With these in place the real underlying error surfaced: `RuntimeError: Unable to sample before the end of the first episode` — HER's replay buffer needs at least one *completed* episode per env before it can relabel, and with `learning_starts=10000` × `n_envs=16` × `max_steps=2000`, no env had finished a full episode by warmup's end. Resolved by bumping `learning_starts` to 40000.

**Numbers:**

| Step | ep_rew_mean | ep_len_mean | ent_coef | actor_loss | critic_loss |
|-----:|------------:|------------:|---------:|-----------:|------------:|
| 40K  | −2000 | 2000 | 0.985 | (warmup) | (warmup) |
| 100K | −2000 | 2000 | 0.65 | −180 | 35 |
| 160K | −2000 | 2000 | 0.477 | −342 | 24.2 |
| 224K | −2000 | 2000 | 0.322 | −400 | 11.4 |
| 250K eval | **−2000.00 ± 0.00** | 2000 | 0.275 | −406 | 13.5 |

`mean_reward = −2000` is the absolute floor for an episode under HER's `-1`-per-step sparse reward over 2000 steps — it means **the hip never came within `GOAL_THRESHOLD = 0.12 m` of either ball at any timestep in any of the 20 deterministic eval episodes.** Throughout training, `ep_rew_mean` stayed flat at −2000 from the first eval at 50K through the final eval at 250K. `best_model.zip` is the very first 50K checkpoint, because the policy never improved on its (already-floor) eval reward.

**What we saw in the video:** Rendered three deterministic episodes (`crawler_mimo_phase_d_her_seed{0,1,2}.mp4`). All three show the same behavior: the creature lies prone in essentially the same spot the entire episode, with only tiny limb wiggles. Hip never translates measurably. Both balls sit untouched 0.5–0.7 m away. The policy found the **proprio-grope local minimum**: small actions that don't tip the body and don't generate horizontal motion. SAC's auto-entropy compressed onto this distribution (ent_coef 0.985 → 0.275). Tilt termination never fires because the body never moves enough to tip.

**Why HER alone failed (this is the important part):** HER's premise is "even failed episodes are useful if you treat the agent's end-state as the goal." This is powerful when episode endpoints *vary*. Here they don't. Every episode ends near the spawn position, so every *hindsight* goal HER samples is approximately the spawn position. HER's relabeling degenerates into "learn a policy that reaches the spawn position" — which is trivially achieved by the policy already there (do nothing, you're already at spawn). The literature's claim that HER fixes sparse-reward exploration assumes the agent already produces *varied* trajectories. **HER does not produce exploration; it exploits exploration that already exists.** If the underlying policy cannot wander, HER has no useful hindsight to relabel. The bootstrap problem is upstream of HER.

**The dependency-graph picture, refined:** Phases I, II, III, III.2, IV have now closed off the upper-substrate fixes. Substrate (I) doesn't bootstrap. Two-stream architecture (II) doesn't bootstrap. Fixed structure (III) doesn't bootstrap. Memory (III.2) doesn't bootstrap. Goal-conditioned relabeling (IV) doesn't bootstrap. The chain `body → motion → encounter → memory → map` still breaks at link 2, and we have now tested fixes at links 2 (substrate), 3 (architecture and memory), 4 (relabeled hindsight goals) without any of them generating motion. What none of these has tested is the human-developmental scaffold: in real infants, link 2 is solved not by the infant but by a *caregiver* who places objects within stumble-range so accidental contact happens. The infant's first-ever reach is a contingency: random body motion → object moves → "I did that." Once that contingency is logged, the agent has a varied endpoint to learn from. HER would then have something to relabel.

**Is vision load-bearing?** Not applicable; Phase IV is blind proprio.

**Next question:** Does placing one ball within stumble-range of the spawn (≈ 0.15 m, the prone-reach distance) — caregiver-scaffolding — produce accidental contacts during HER's warmup, generating varied endpoints that HER can then relabel into useful exploration? The second ball remains at 0.7 m to test whether learned reaching generalises beyond the scaffold or stays attached to it. The intervention is minimal: change `--fixed-ball-positions "0.7,0.0;0.0,0.7"` to `--fixed-ball-positions "0.15,0.0;0.7,0.0"`. If accidental contacts on the near ball produce varied endpoints, HER relabeling should kick in and we'd see `ep_rew_mean` rise off −2000 within the first 100K steps. If `ep_rew_mean` stays at −2000 even with a ball within reach, then random initial action noise from SAC's high-entropy start is insufficient to produce even prone-reach motion on this body, and the next escalation is a stronger velocity bonus or a "kick-start" initial action perturbation.

**Theory signals:** Taylor's "interpenetration" predicts that perception requires *encounters with the world* through one's own action. Phase IV confirms a developmentally meaningful failure: with no encounters, the agent has no equivalence classes to build, no model to predict, and no behaviour to perceive. The Behavioral Prediction Framework predicts outperformance of random walk on learnable tasks; Phase IV continues to fail this test at the strongest level (no walk at all). The Pattern Learning Framework predicts stable patterns from recurring features; with no motion, no recurring features ever land in the sensorium that aren't fixed proprio constants. The caregiver-scaffold hypothesis is the first proposed intervention that addresses **encounter generation** rather than substrate, architecture, or memory — the first to plausibly satisfy Taylor's prerequisite "there has to be something to perceive."


---

## 2026-05-11 — Phase V / V.2: Caregiver scaffold + velocity bonus (mimo_phase_e_*, mimo_phase_e2_*)

These two runs must be read together. Phase V was the planned caregiver-scaffold escalation from Phase IV. Phase V.2 was the rerun after we found a bug that had silently invalidated Phase V — and, retrospectively, Phase IV as well. The story is as much about a wrapper bug as about the experimental results.

### What we ran

**Phase V** (`mimo_phase_e_scaffold_velbonus_10`, 250K steps, completed normally): HER + fixed balls + memory + caregiver scaffold + 5× velocity bonus. The near ball was moved from 0.70 m to 0.10 m — within the prone-reach envelope of the creature — so accidental contacts could happen during SAC's high-entropy warmup without any purposeful motion. The velocity bonus was increased from 0.02 to 0.10 (5× larger than Phase IV's setting) to break the stillness attractor. HER relabeling with `n_sampled_goal=4`, `goal_selection_strategy="future"`, 16 parallel envs, MPS, ent_coef pinned at 0.2, learning_starts=40000. Same command as Phase IV except `--fixed-ball-positions "0.10,0.0;0.7,0.0" --velocity-bonus-scale 0.10 --ent-coef 0.2`.

**The bug found between V and V.2:** While investigating why the velocity bonus appeared to have no effect in Phase V, we audited `her_wrapper.py`'s `step()` method and found that it returned only `self.compute_reward(achieved, desired, info)` — the HER sparse goal signal — without adding `inner_reward`. The inner env's reward (which carries `velocity_bonus_scale × hip_speed`, `step_cost`, and `approach_reward`) was computed and then silently discarded. This bug was present from the first HER run (Phase IV). Both Phase IV and Phase V were tested with effective velocity bonus = 0.00, not the 0.02 and 0.10 they were supposed to have. The fix (line ~150 of `her_wrapper.py`) is `reward = self.compute_reward(achieved, desired, info) + inner_reward`, with a multi-line comment above it explaining the composition. An important caveat is noted in that comment: HER's relabel mechanism re-runs `compute_reward()` for 4 out of every 5 sampled transitions (the hindsight-relabeled ones), so those samples see only the sparse term. The velocity bonus only flows through the 1/5 of samples that retain the original goal. If the dilution proves too strong, the next escalation is to bake `hip_speed` into `info` and have `compute_reward()` read it back so relabeled transitions carry the bonus too.

**Phase V.2** (`mimo_phase_e2_her_velbonus_fixed`, 250K steps, completed normally): Same command as Phase V but with the wrapper patched. One additional explicit flag: `--approach-reward-scale 0.0`, to keep the inner env's default 2.0 approach-reward from feeding through the now-propagated inner_reward and introducing bribery that Phase V.2 was not designed to include.

### Numbers

**Phase V (broken wrapper):**

| Checkpoint | ep_rew_mean | ep_len_mean | eval mean ± std | ent_coef | critic_loss |
|---|---|---|---|---|---|
| 50K eval | −2000 (rollout) | 2000 | −2000.00 ± 0.00 | 0.2 | 0.378 |
| 100K eval | −2000 (rollout) | 2000 | −1999.95 ± 0.22 | 0.2 | 0.729 |
| 150K eval | −2000 (rollout) | 2000 | −2000.00 ± 0.00 | 0.2 | 1.05 |
| 200K eval | −2000 (rollout) | 2000 | −2000.00 ± 0.00 | 0.2 | 0.985 |
| **250K eval** | **−2000 (rollout)** | **2000** | **−2000.00 ± 0.00** | **0.2** | **0.783** |

**Phase V.2 (fixed wrapper):**

| Checkpoint | ep_rew_mean | ep_len_mean | eval mean ± std | ent_coef | critic_loss |
|---|---|---|---|---|---|
| 50K eval | −1890 (rollout) | ~2000 | −1899.05 ± 0.32 | 0.2 | 0.408 |
| 100K eval | −1860 (rollout) | ~1980 | −1899.19 ± 0.36 | 0.2 | 0.772 |
| 150K eval | −1860 (rollout) | ~1980 | −1899.00 ± 0.42 | 0.2 | 0.926 |
| 200K eval | −1860 (rollout) | ~1980 | **−1899.35 ± 0.30** | 0.2 | 0.742 |
| **250K eval** | **−1860 (rollout)** | **~1980** | **−1899.18 ± 0.34** | **0.2** | **0.635** |

- loco_speed_mean: N/A (not separately logged in either run)
- touch_rate: N/A (both runs are blind proprio; the near ball overlap with spawn is geometric, not a touch in the hand-sensor sense)
- Steps completed: 250,000 each
- Both runs exited normally (no crash; one semaphore leak warning at shutdown, cosmetic only)

### What we saw in the video (Phase V.2)

Three episodes were rendered (seeds 0, 1, 2 of `mimo_phase_e2_her_velbonus_fixed`). Eight evenly-spaced frames from seed 0 were extracted to `alien_baby/results/frames/phase_e2/`. All three episodes show the same pattern: the creature makes **one initial motion** — rolling from its prone starting orientation onto its side, driven by random_start_orientation causing an unbalanced starting posture — and then freezes in that side-lying posture for the remainder of the 2000-step episode. The hip never translates more than a few centimeters from spawn in any episode. The near ball at (0.10, 0.0) ends up adjacent to the creature's flank because of the initial roll; this is why the policy earns approximately −1899 instead of −2000 — it is spending roughly 100 steps of the episode within GOAL_THRESHOLD = 0.12 m of the near ball by virtue of body geometry, not locomotion. The far ball at (0.70, 0.0) is never visited.

### Why it failed

The -1899 reward in Phase V.2 is geometric, not learned. The near ball at 0.10 m from origin, combined with GOAL_THRESHOLD = 0.12 m and random_start_orientation that rotates the body up to 180°, means the ball is almost certain to overlap with the body's footprint for some portion of the episode regardless of what the policy does. The policy's contribution is: make the one initial roll happen early enough to maximise the overlap duration. That is the entire learned behavior. The policy did not learn to crawl, did not learn to approach the near ball, and did not learn to approach the far ball. The "one motion then freeze" pattern is a new failure mode — it is different from Phase IV's total stillness, but it is not locomotion.

The deeper reason this failure mode exists: ent_coef pinning at 0.2 prevents entropy collapse (so the failure is not "policy degenerated to zero entropy"), but it does not prevent the **action mean** from converging to "produce one big motion at step 0, then output near-zero actions forever." SAC's actor learns that the first motion earns HER goal-overlap reward at the beginning of the episode, and subsequent actions earn −1/step no matter what, so the best policy is one big move then stop. The one-move attractor is different from the no-move attractor in Phase IV, but both are still attractors that block sustained locomotion.

### The dependency-graph picture

The Phase IV analysis identified the dependency chain: body → motion → encounter → memory → map. Phase IV broke at "motion." Phase V and V.2 broke at the same link, but with a clearer diagnosis: the creature **can** produce one motion, but cannot sustain motion across multiple time steps against the per-step cost. The caregiver scaffold (near ball) confirmed that accidental first-contact can happen — the overlap reward is reaching the policy. What is missing is the incentive to keep moving after the first contact has been made and the creature is no longer in the goal zone. The chain now reads: **body → (one action) → [stuck] → encounter (geometrically, not behaviorally) → memory flags (unused) → map (unreachable).**

The HER bug retrospective adds another dimension: Phases IV and V were supposed to be testing "HER + velocity bonus" and "HER + 5× velocity bonus + scaffold," respectively. What they were actually testing was "HER alone" and "HER alone + scaffold." The velocity bonus was never part of either experiment. This means we have not yet run a single valid test of whether a velocity bonus — applied correctly through the wrapper — can break the stillness attractor under HER. Phase V.2 is the first valid test, and it shows the bonus (once actually propagated) does something: ep_rew_mean left the floor. But it did not produce sustained locomotion.

### What we actually learned from this pair

1. **The wrapper bug retrospectively invalidates Phase IV-HER as a test of velocity bonus.** Phase IV tested HER alone. Phase V tested HER alone (bug). Phase V.2 is the first valid test of HER + velocity bonus composition, and it moves the needle from −2000 to −1899 — meaningful numerically, not meaningful behaviorally.

2. **"One move then freeze" is now the canonical failure mode at this stage.** It is distinct from Phase IV's total stillness and from the prior MICOA runs' timeout-while-moving. The policy is not broken — it is rational given the reward structure. One initial motion earns some geometric overlap reward; every subsequent step costs −1 regardless of direction. Freezing is locally optimal once the first motion is made.

3. **The non-zero eval std (0.30–0.42) is the first faint positive signal in months.** Every run from Phase I through Phase IV produced eval std = 0.00, meaning all 20 deterministic eval episodes returned identical reward — the policy was doing literally the same thing in each. Phase V.2's non-zero std means the policy is arriving at slightly different micro-states across the 20 eval episodes, likely because random_start_orientation creates slightly different body configurations that the one-motion strategy interacts with differently. This is genuinely new — but should not be oversold. The variance is less than 1 reward unit per episode (std ~0.35 means individual episode rewards vary by less than ±1 from −1899) and the source is geometric, not behavioral.

4. **Critic confidence improved.** critic_loss dropped from a Phase IV peak of 35 (at 100K) to Phase V.2's 0.635 at 250K. The critic is now producing confident, consistent value estimates. This is a diagnostic positive: the policy found a stable, learnable attractor (even if the wrong one). In Phase IV, the critic could not find a stable gradient because the policy was too diffuse. Phase V.2's critic has something concrete to model.

5. **The caregiver-scaffold + enabling-pressure combination is not sufficient for sustained locomotion.** Even with both interventions actually applied (Phase V.2 vs. the broken-wrapper E), the chain breaks at "after the first motion, the policy freezes." The scaffold created the condition for accidental contact; the velocity bonus was actually propagated; the ent_coef pin prevented entropy collapse. None of these together produced a policy that keeps moving.

### Methodological lesson

Wrappers that override or intercept reward are a hidden audit risk. When composing rewards from multiple layers — inner env shaping + outer wrapper's goal signal — every layer must be verified individually before assuming the composition is correct. A one-line oversight (`return compute_reward(...)` without `+ inner_reward`) silently zero-ed out two runs' worth of reward engineering. The fix is not difficult once found; the problem is finding it. The correct practice: after implementing any wrapper that touches `step()`, write a smoke test that verifies `wrapper.step(a).reward != inner_env.step(a).reward` when inner_env has non-zero shaping enabled.

### Is vision load-bearing?

Not applicable. Both Phase V and Phase V.2 are blind proprio runs. No visual channel was used in either. The non-zero eval std in Phase V.2 is a behavioral signal, but it does not involve vision.

### Theory signals

The Behavioral Prediction Framework predicts that an agent with internal predictive structure should produce behavior that stays coherent in novel positions it has not visited before. The "one move then freeze" pattern is anti-predictive: after the first timestep, the creature's behavior is entirely decoupled from its state — the same near-zero actions regardless of body orientation, distance from either ball, or limb configuration. This is not prediction; it is convergence to a constant output. The Pattern Learning Framework predicts that sparse, stable patterns should emerge in the representation as recurring inputs are encountered. Phase V.2's stable critic_loss (converging to 0.635) suggests the value function has learned a stable pattern — but the pattern it learned is "first-step geometric overlap, then −1 per step," not a distributed representation of spatial structure. Both frameworks are challenged: the agent has converged to a representation that is internally consistent but disconnected from any useful environmental structure.

### Next question

The failure is post-first-action freeze, not zero-action freeze. The single most important thing we still do not know is: **what intervention targets persistent motion across multiple actions, rather than just generating one initial motion?** Four candidate directions exist without pre-committing to any: (a) episode-level reward shaping that requires sustained motion across a window of steps rather than rewarding instantaneous goal-zone overlap; (b) curiosity-driven exploration (RND) that rewards novel states so freeze is always penalized because a frozen creature quickly revisits the same state; (c) initial-state randomization that places the creature in already-tipped postures so the one-roll refuge is not available; (d) baking hip_speed into the `info` dict and having `compute_reward()` read it back so the velocity bonus survives HER's 4/5 relabeling dilution and reaches the full policy gradient, not just 20% of it.


---

## 2026-05-11 — Phase VI / VI.2: Diagnostic 1 from Phase V.2 — near ball outside goal threshold (mimo_phase_f_nearball_outside_threshold / mimo_phase_f_2k_nearball_outside_threshold)

These two runs must be read together. Phase VI was the intended execution of Phase V.2's "Recommended diagnostic 1": move the near ball from 0.10 m (inside GOAL_THRESHOLD = 0.12 m) to 0.30 m (well outside threshold). Phase VI.2 was its corrected rerun after a methodological confound was discovered mid-session. The story is partly about what the diagnostic revealed and partly about how a one-parameter oversight silently invalidated the first attempt.

### What we ran

**Phase VI** (`mimo_phase_f_nearball_outside_threshold`): Identical config to Phase V.2 in all intended respects, except `--fixed-ball-positions "0.30,0.0;0.7,0.0"` — the near ball moved from 0.10 m to 0.30 m, placing it outside GOAL_THRESHOLD. If Phase V.2's reward was coming purely from the geometric footprint overlap of the creature's body with the near ball at 0.10 m, then removing that overlap should return ep_rew_mean to the −2000 floor. If some genuine locomotion had been learned, the policy should partially maintain its score by moving toward the new ball position.

**The confound discovered during F:** `train_crawler.py`'s `--max-steps` default is 600, not 2000. Phase V.2 had been run with an explicit `--max-steps 2000`. Phase VI was launched without that flag, so it ran with 600-step episodes while the intention was 2000-step episodes. An episode of 600 steps with HER's −1-per-step sparse reward has a floor of −600, not −2000. Phase VI's ep_rew_mean and per-step reward cannot be compared numerically to Phase V.2 at all — the units are different. The diagnostic manipulation (ball position) was applied correctly; the episode-length confound made the result unreadable against its predecessor.

**Action taken:** Phase VI was retired immediately. Phase VI.2 was launched with `--max-steps 2000` made explicit.

**Phase VI.2** (`mimo_phase_f_2k_nearball_outside_threshold`, 250K steps, completed normally): Config identical to Phase V.2 except `--fixed-ball-positions "0.30,0.0;0.7,0.0"` and `--max-steps 2000` (explicit). ent_coef pinned at 0.2 (confirmed held). All other parameters — HER, memory flags, velocity bonus 0.10, fixed balls, 16 parallel envs — unchanged.

### Numbers

**Phase VI.2 (the valid run):**

| Metric | Phase V.2 | Phase VI.2 | Change |
|---|---|---|---|
| eval mean_reward | −1899.35 ± 0.30 | −1989.35 ± 99.48 | −90 pts mean, ~300× larger std |
| ep_len_mean | ~1980 | ~1960 | marginally shorter |
| per-step reward | ≈ −0.95 | ≈ −0.99 | worse by ~4% |
| ent_coef | 0.2 (pinned) | 0.2 (pinned) | stable |
| critic_loss | 0.635 at 250K | ~1.0 at 250K | mild increase |
| Steps completed | 250,000 | 250,000 | same |
| loco_speed_mean | N/A | N/A | blind proprio |
| touch_rate | N/A | N/A | blind proprio |
| Falls | 0 | 0 | body stable |

**Phase VI (invalid; methodological record only):** Ran with 600-step episodes instead of 2000. All numerical results are incommensurable with V.2 and VI.2 and are not reported here. The only contribution of Phase VI is identifying the --max-steps default gotcha.

### What we saw in the video

No new video was rendered for Phase VI or Phase VI.2 during this session. The analysis below is based entirely on eval reward statistics. A rendering from the Phase VI.2 best checkpoint is recommended before Phase VII launches, to visually confirm what the statistics imply: do individual episodes show the creature reaching the 0.30 m ball position on lucky orientations, or freezing without contact on unlucky ones?

### Reading the two outcomes of Diagnostic 1

Phase V.2's entry described two predicted outcomes:

- **Pure geometric:** moving the ball outside GOAL_THRESHOLD eliminates the footprint-overlap bonus entirely. ep_rew_mean returns to the −2000 floor. This would confirm that no locomotion was learned — only a geometric refuge was exploited.

- **Pure behavioral:** the policy maintains a score above −2000 with the ball at 0.30 m, because a genuine locomotion strategy would still drive the creature toward wherever the near ball is placed.

What Phase VI.2 returned is **neither**. The result is mixed, and the mixture is diagnostic in its own right.

### Why it failed — and what the mixed result reveals

**The mean got worse, not better.** Per-step reward degraded from −0.95 (V.2) to −0.99 (VI.2). The near-ball geometric refuge that earned V.2 its 100-step overlap bonus is largely gone — the ball at 0.30 m is too far for the initial-roll's body footprint to overlap it under typical spawn configurations. This confirms the core V.2 hypothesis: most of V.2's reward above the −2000 floor was geometric, not learned locomotion.

**But the standard deviation exploded from 0.30 to 99.48.** That is roughly 300 times larger than Phase V.2's std. This magnitude means individual episodes are not clustered around one value. Some rollouts earn reward substantially above the floor; most sit at −2000. The distribution now has at least two qualitatively distinct outcomes.

**The likely mechanism is random-orientation roulette.** The creature spawns prone with `random_start_orientation` rotating its yaw up to 180°. Its learned behavior is still "one big roll at step 0, then freeze." On lucky spawns, that initial roll brings the body's footprint into proximity with the 0.30 m ball position — not because the policy directed it there, but because the random starting angle happened to align the initial roll vector toward the ball's fixed location. On unlucky spawns, the initial roll goes in the wrong direction entirely and the episode earns −2000. With the near ball at 0.10 m (V.2), almost every spawn was lucky because the ball was close enough for the body footprint to overlap regardless of yaw. With the ball at 0.30 m (VI.2), only a fraction of spawns produce lucky alignment, and the rest return floor reward. A 99.48 std is exactly what this lottery looks like.

**The important recharacterization of the failure mode:** The "one move then freeze" failure mode is now understood to have two internal layers that Phase V.2 could not distinguish. Layer one is geometric refuge — the near ball was so close at 0.10 m that even a stationary body overlapped it. Layer two is random-orientation lottery — the initial roll is undirected, but on lucky yaw angles it happens to land the body near a more distant ball position. Phase V.2 was almost entirely Layer one (std = 0.30 means almost no variance, almost every episode earns the same bonus). Phase VI.2 exposed Layer two by removing Layer one. The policy did not learn to direct its motion. It learned to bet on the lottery.

Both layers share the same upstream cause: freeze is globally optimal after the first action. The creature makes one move and stops because moving further costs −1/step with no compensating gradient. The fix that collapses the lottery and forces genuine locomotion is making stillness genuinely costly across all initial conditions — so even a lucky-orientation rollout that ends up near the ball must keep moving to sustain reward, and cannot earn its way to a non-floor total by freezing on a lucky placement.

### Methodological lesson: the --max-steps default gotcha

Phase VI is a specific instance of the general wrapper-audit rule established after the Phase IV/V her_wrapper.py bug. When launching a follow-on experiment intended as a controlled comparison, every parameter that affects reward scale or episode length must be verified explicitly. Phase VI.2 adds a concrete checklist item: confirm `--max-steps` matches the predecessor before comparing ep_rew_mean numbers. A 600-step run and a 2000-step run are measuring different quantities even if all other config is identical.

### Is vision load-bearing?

Not applicable. Phases VI and VI.2 are blind proprio runs, identical in architecture to Phases IV, V, and V.2. No visual channel was used in either. The eval std signal — the most informative new number in this entry — is a consequence of the random-orientation lottery interacting with ball geometry, not of any visual processing.

### Theory signals

The eval std jumping from 0.30 (Phase V.2) to 99.48 (Phase VI.2) is the first signal in this project's history that reflects genuine outcome variance across qualitatively different behavioral attractor basins, rather than micro-variation around a single attractor.

The Behavioral Prediction Framework predicts that a policy with internal predictive structure should produce coherent behavior across a range of initial conditions. What we observe is the opposite: outcome quality is almost entirely determined by the random starting angle, not by anything the policy learned to do. Two initial conditions that differ only in yaw angle produce rewards of −1900 vs. −2000. The policy has no model of "where is the ball relative to my current orientation" — it cannot compensate for a bad starting angle. The high std is direct evidence that the policy's behavior is orientation-blind, which is precisely what the framework predicts a policy looks like when no predictive internal model has formed.

The Pattern Learning Framework predicts that stable, sparse patterns should emerge that fire for structurally similar situations regardless of surface-level differences. If the creature had developed a spatial map of where the near ball is relative to its own body, similar patterns should activate for "ball is to my left" regardless of whether the absolute yaw is 45° or 135°. What we see instead is that the same trained behavior produces completely different episode outcomes depending on the random yaw. No orientation-invariant pattern has formed. Neither framework predicted this specific two-attractor structure — one basin near −1900, one at −2000 — but both are updated by it: the policy is not generalizing across initial conditions in the way either framework would expect from a system that had internalized any stable representation of the task geometry.

### Next question

The four candidate directions from Phase V.2 remain live — none has been tested under conditions where the initial-condition lottery is eliminated. The leading candidate is **F-tipped-init**: force the creature to spawn in postures from which the initial-roll refuge is not available, so that even a geometrically lucky orientation cannot produce a non-floor reward without continued motion. The three alternatives are **F-curiosity** (RND bonus that penalizes returning to visited states, so freeze is always costly regardless of luck), **F-sustained** (episode-level reward requiring sustained goal-zone overlap across a time window rather than instantaneous overlap, so a single lucky freeze cannot earn full reward), and **F-info-velocity** (bake hip_speed into the `info` dict so the velocity bonus survives HER's 4/5 relabeling dilution and reaches the full policy gradient). The four candidates are not mutually exclusive, but tipped-init is the most direct empirical test of whether the one-roll-then-freeze failure mode is locked to the current spawn posture or survives even when that posture is removed.


---

## 2026-05-20 — Phase VIII: Moving balls, R32/R33/R34/R35 (mimo_phase_h_moving_balls_v1)

**What we ran:** Four 80K-step runs on the cart substrate (R20+R30 recipe: ent_coef=0.5 anneal→0.2, vel_bonus=0.10, strength_scale=1.0, offset curriculum 0→0.15, offset=0.15 locked, 16 envs proprio / 8 envs vision), testing whether constant-velocity bouncing balls make vision load-bearing. Two speed settings (0.08 m/s and 0.05 m/s), each with a matched proprio control (no camera) and vision run (stereo CNN). Pre-flight confirmed balls bounce correctly off all four edges, touch detection works on moving balls, ball velocity is NOT in proprio (obs length identical at 71), and --ball-speed 0.0 is bit-identical to Phase VII. Phase VII found vision non-load-bearing across six substrates; this was the strongest remaining substrate-level hypothesis.

**Training log summary — all four runs:**

| Run | Ball Speed | Type | Peak rollout ep_rew_mean | Eval at 50K | Eval std |
|---|---|---|---|---|---|
| R32 | 0.08 | proprio | 437 (t≈14K, mid-curriculum) | −1872 | 536 |
| R33 | 0.08 | vision  | 442 (t≈6K, mid-curriculum) | −2051 | 203 |
| R34 | 0.05 | proprio | 443 (t≈14K, mid-curriculum) | −1450 | 787 |
| R35 | 0.05 | vision  | 442 (t≈6K, mid-curriculum) | −1955 | 265 |

All four runs hit a uniform peak of 437–443 during the curriculum ramp (t < 15K), then collapsed uniformly after entropy annealing completed (t > 30K) and curriculum locked at offset=0.15 with moving balls. The single eval at t=50K landed in the trough of this collapse. Because checkpoint_interval=50000, each run fired exactly ONE eval (at 50K), making the eval reward an unreliable signal — it measures the worst training moment, not peak behavior. The best_model.zip files save the checkpoint that maximized this single eval score (all at 50K because no later eval was better). The four eval rewards (−1450 to −2051) are NOT a clean measure of policy quality.

**Numbers:**
- ep_rew_mean: ~440 (peak, all runs) → −310 to −492 (final rollout, all runs)
- loco_speed_mean: N/A (not logged; cart speed 0.15 m/s fixed, AB rigidly mounted)
- touch_rate (moving ball, deterministic): see Table 1 below
- Steps completed: 80,000 each (normal finish)
- Errors/crashes: none (semaphore leak at shutdown is cosmetic)

**TABLE 1 — Touch-counting eval, 20 seeds, deterministic, max_steps=2000**

Primary: training ball_speed, offset=0.15 (the proposal's decisive comparison)

| Policy | ball_speed | offset | mean reward | std | both/20 | one/20 | neither/20 |
|---|---|---|---|---|---|---|---|
| R32 proprio speed08 | 0.08 | 0.15 | −1857 | 541 | **1** | 15 | 4 |
| R33 vision speed08  | 0.08 | 0.15 | −1991 | 220 | **0** | 12 | 8 |
| R34 proprio speed05 | 0.05 | 0.15 | −1837 | 516 | **1** | 19 | 0 |
| R35 vision speed05  | 0.05 | 0.15 | −1774 | 538 | **1** | 15 | 4 |

Vision both-touched at speed 0.08: 0/20 vs proprio 1/20 — vision is WORSE, not better.
Vision both-touched at speed 0.05: 1/20 vs proprio 1/20 — identical.
Relative gap: 0% at speed 0.05; vision loses at speed 0.08. The proposal required ≥+25% relative improvement for Promising. The result is within noise (≤5%) or worse — REFUTED on this metric.

Static ball (speed=0.0), offset=0.15 — generalization test:

| Policy | ball_speed | offset | mean reward | std | both/20 | one/20 | neither/20 |
|---|---|---|---|---|---|---|---|
| R32 proprio speed08 | 0.0 | 0.15 | −313 | 1027 | 14 | 6 | 0 |
| R33 vision speed08  | 0.0 | 0.15 | −511 |  792 | 8 | 12 | 0 |
| R34 proprio speed05 | 0.0 | 0.15 | −13  |  742 | **18** | 2 | 0 |
| R35 vision speed05  | 0.0 | 0.15 | −737 |  826 | 6 | 14 | 0 |

The static-ball results are dramatically better than moving-ball results for all four policies. The proprio runs (R32: 14/20, R34: 18/20) perform comparably to Phase VII benchmarks, confirming the best_model checkpoints are real policies that solved the task at some point — they just cannot handle moving balls. The vision runs perform worse than their matched proprio controls on static balls too (8/20 vs 14/20 at speed 0.08; 6/20 vs 18/20 at speed 0.05). Vision not only fails to help under moving balls — it also degrades static-ball performance compared to matched proprio controls. This is the seventh straight substrate under which vision underperforms proprio.

Reach curve (training ball_speed, offsets 0.10 and 0.05):

| Policy | ball_speed | offset | mean reward | both/20 |
|---|---|---|---|---|
| R32 proprio speed08 | 0.08 | 0.10 | −1872 | 1/20 |
| R32 proprio speed08 | 0.08 | 0.05 | −1810 | 2/20 |
| R33 vision speed08  | 0.08 | 0.10 | −2017 | 0/20 |
| R33 vision speed08  | 0.08 | 0.05 | −1803 | 2/20 |
| R34 proprio speed05 | 0.05 | 0.10 | −1494 | 4/20 |
| R34 proprio speed05 | 0.05 | 0.05 | −1495 | 4/20 |
| R35 vision speed05  | 0.05 | 0.10 | −1887 | 0/20 |
| R35 vision speed05  | 0.05 | 0.05 | −1882 | 1/20 |

The reach curve shows no improvement as offset shrinks (as it did in Phase VII). Moving balls have collapsed performance at every offset. Proprio reach slightly better than vision reach at both offsets.

**TABLE 2 — Vision Ablation (R33 and R35)**

(a) Single-step action delta: 200 steps × 10 seeds, pixels zeroed vs normal, L2 norm of action diff

| Policy | mean delta | median delta | Phase VII reference |
|---|---|---|---|
| R33 vision speed08 | 0.0244 | 0.0187 | 0.023 (R14), 0.009 (R17) |
| R35 vision speed05 | 0.0207 | 0.0202 | 0.009–0.023 |

Both values land squarely in the same 0.009–0.023 dead zone that Phase VII identified as "vision inert." The proposal required ≥0.20 for Promising and 0.05–0.20 for Marginal. Both R33 and R35 measure < 0.05. REFUTED on this metric.

(b) Episode-level: 20 seeds pixels-normal vs 20 seeds pixels-zeroed

| Policy | pixels NORMAL both/20 | pixels ZEROED both/20 | delta |
|---|---|---|---|
| R33 vision speed08 | 0 | 0 | 0 |
| R35 vision speed05 | 1 | 1 | 0 |

Zeroing all pixels produces no change in episode-level performance on either vision policy. Vision is not influencing behavior at all. This is the same pattern as R14 (14/20 → 14/20 zeroed) and R17 (16/20 → 17/20 zeroed, i.e., pixels-zero was better).

**Per-speed gradient:**

Proprio both-touched at speed 0.05 vs 0.08: 1/20 in both conditions. Ball motion is biting equally hard at both speeds — the task appears to be at saturation difficulty (both speeds broke the policy completely). There is no measurable gradient in proprio performance between the two speed settings. The proposal's diagnostic — proprio drops as speed rises, vision stays flat — cannot be applied because both drop to the same floor.

**What we learned:**

1. Moving balls at 0.05–0.08 m/s break the stationary-ball performance completely (1/20 vs 14-20/20 in static) but do not differentiate vision from proprio. Both modalities are equally destroyed by ball motion. The task became too hard for either to solve rather than creating pressure that only vision could relieve. Ball motion at these speeds appears to be in the regime where timing the arm to a moving ball is beyond what 80K steps of SAC training can learn regardless of modality — the Phase VIII proposal's pre-flight check found only 10% success for a stationary outstretched arm, which means even a perfect reach policy would only get 2/20 from geometric chance. This is the upper ceiling, and both proprio and vision run into it.

2. Vision continues to show ablation sensitivity of 0.020–0.024, matching the 0.009–0.023 range found in six prior Phase VII experiments. The pixel pathway is structurally inert under SAC in this body. Moving balls at these speeds did not change that. The dead zone is not a task-difficulty artifact — it is a consistent property of the CNN + SAC combination in this substrate.

3. Vision hurts static-ball performance even after training on moving balls. The R34 proprio policy achieves 18/20 on static balls at offset=0.15 — matching the best Phase VII results. R35 vision achieves only 6/20 on the same condition. The CNN adds noise without adding useful signal, and this effect persists even when the training environment was explicitly designed to make visual prediction valuable.

**Is vision load-bearing?** No. Single-step ablation sensitivity 0.020–0.024 on both vision policies, identical episode-level results with pixels normal vs zeroed, and vision underperforming its matched proprio control at every evaluated condition. This is the seventh consecutive substrate in which vision ablation returns < 0.05. The pattern is consistent and robust.

**Verdict against proposal Section 6 criteria:**

| Metric | Threshold | Actual | Grade |
|---|---|---|---|
| Vision both-touched vs proprio (speed 0.08) | ≥+25% relative | −100% relative (0 vs 1/20) | REFUTED |
| Vision both-touched vs proprio (speed 0.05) | ≥+25% relative | 0% (1 vs 1/20) | REFUTED |
| Vision ablation sensitivity (mean L2 delta) | ≥0.20 | 0.0244 (R33), 0.0207 (R35) | REFUTED |
| Per-speed gradient (proprio drops, vision flat) | Vision compensates | Both flat at floor | REFUTED |
| Eval mean reward, vision best | ≥+200 | −1991 (R33), −1955 (R35) at 50K | REFUTED |
| Falls/cart departures | 0 | 0 (both vision policies) | PASS |

**VERDICT: REFUTED.** Moving balls at 0.05–0.08 m/s do not make vision load-bearing in the cart substrate. The substrate-level hypothesis has been exhausted. This is the seventh substrate tested (R6, R10, R14/R15, R17/R18, R26/R27, R28/R29, R32-R35) in which vision ablation returns < 0.05 and vision does not outperform matched proprio. The OVERNIGHT_LOG "definitive verdict" conclusion (lines 786–816) is confirmed and extended: substrate tweaks cannot make vision load-bearing in this body/algorithm combination.

**Theory signals:** The Behavioral Prediction Framework predicts that a policy with an internal model of where a moving ball will be should outperform a policy that must react to contact. Both vision and proprio are in the same boat — neither produces a timing strategy. The Pattern Learning Framework predicts that recurring visual patterns (a colored ball approaching from a consistent direction) should eventually activate consistent internal representations that steer behavior. The ablation delta of 0.020–0.024 after 80K steps with moving balls visible in the head_cam is a direct refutation: no stable pattern differentiation has occurred. Both frameworks predict that this specific environmental pressure (moving, predictable trajectory) should have been sufficient to force visual learning. The finding that it was not is strong evidence that the failure is architectural (CNN + SAC without explicit visual gradient mechanism) rather than task-structural.

**Next question:** With substrate-level options exhausted, what is the minimum structural change that would force the pixel pathway to differentiate? The four candidates from OVERNIGHT_LOG lines 786–816 remain: (a) one-armed body where geometric reach cannot cover the arena, (b) gaze-gated reward, (c) DrQ-v2 or image-aware RL algorithm, (d) reward tied to visual attention before contact. The question is which of these provides genuine test of "does the creature learn to see" rather than "can we engineer vision into the reward signal."

---

## 2026-05-22 — Phase XII R41/R42: MICOA + vision vs. proprio control on moving balls (phase_iv_R41_micoa_vision_speed08 / phase_iv_R42_proprio_speed08)

**What we ran:** Two matched 250K-step runs testing whether R40's architectural integration of vision (MICOA + predictive KL) rescues performance on moving balls (ball_speed=0.08, cart_mode=constant_velocity_bouncer). R41 used MICOA + vision (n_envs=8, --micoa-beta 0.0, --micoa-pred-beta 0.1, R40 Goldilocks σ clamp); R42 was the matched proprio-only control (n_envs=16, no MICOA, no vision). Both used identical task config: max_steps=2000, curriculum warmup=2000/ramp_end=15000/final_offset=0.15, entropy anneal 0.5→0.2 over [15K,30K], velocity-bonus-scale=0.10, seed 42, fresh initialization. Eval cadence tightened to 10K steps (fixing Phase VIII's 50K cadence that missed the policy's peak).

**Numbers:**

R41 training trajectory:
- ep_rew_mean: 422 (t=3040) → peak 442 (t≈6K, curriculum ramp) → −2070 (final eval at t=250K)
- Final rollout ep_rew_mean: approximately −1540 to −1560
- Steps completed: 250,000 (normal finish)

R42 training trajectory:
- ep_rew_mean: 412 (t=2208) → peak ~437 (t≈14K, curriculum ramp) → −2038 (final eval at t=250K)
- Final rollout ep_rew_mean: approximately −1500
- Steps completed: 250,000 (normal finish)

**TABLE 1 — Touch-counting eval (20 seeds, deterministic, max_steps=2000)**

Primary: training condition, ball_speed=0.08, offset=0.15

| Policy | mean reward | std | both/20 | one/20 | neither/20 |
|---|---|---|---|---|---|
| R41 MICOA+vision (moving 0.08) | −1944.6 | 573.3 | 1 | 8 | 11 |
| R42 proprio (moving 0.08) | −1770.5 | 735.1 | 2 | 13 | 5 |

Static-ball generalization: speed=0.0, offset=0.15

| Policy | mean reward | std | both/20 | one/20 | neither/20 |
|---|---|---|---|---|---|
| R41 MICOA+vision (static) | −397.0 | 755.7 | 9 | 11 | 0 |
| R42 proprio (static) | +127.6 | 611.5 | 18 | 2 | 0 |

Phase VIII benchmarks for comparison (80K runs, single eval at 50K):
- Phase VIII R32 proprio @ 0.08: 1/20 moving, 14/20 static
- Phase VIII R33 vision @ 0.08: 0/20 moving, 8/20 static
- R42 (this run) proprio @ 0.08: 2/20 moving, 18/20 static — best proprio yet on moving balls; beats Phase VIII proprio by 4 episodes on static
- R41 (this run) MICOA+vision @ 0.08: 1/20 moving, 9/20 static — slightly better than Phase VIII R33 but well below its matched proprio control

**TABLE 2 — Vision Ablation (R41 only)**

Single-step action delta: 200 steps × 10 seeds, pixels zeroed vs normal, L2 norm of action difference

| Condition | mean delta | median delta | vs Threshold (0.05 / 0.20) |
|---|---|---|---|
| R41 @ ball_speed=0.08 (training condition) | 0.8511 | 0.7994 | far above Promising threshold (0.20) |
| R41 @ ball_speed=0.0 (static generalization) | 0.7946 | 0.7337 | far above Promising threshold (0.20) |

Project ablation history for context:
- R36 (Phase IX): 0.0019 — dead zone
- R37 (Phase IX): 0.0023 — dead zone
- Phase VIII R33/R35: 0.020–0.024 — dead zone
- R38 (Phase X): 0.6563 — first barrier break
- R40 (Phase XI): 0.481 — load-bearing confirmed
- R41 (Phase XII): 0.8511 — highest ever measured in this project

**TABLE 3 — MICOA training health (R41)**

| Diagnostic | t=12K (early) | t=60K (collapse) | t=250K (final) |
|---|---|---|---|
| sigma_combined | 0.673 | ~0.124 (at/below threshold) | 0.0999 (below 0.10 threshold) |
| kl_pred_k1 | 0.57 | ~30–62 (runaway) | ~638 |
| kl_agreement | 0.31 | ~244 | ~953 |

For reference: R38 kl_pred_k1 eventually reached ~116 (flagged as pathology in Phase X). R41 exceeded that by 5×, reaching 638–760 in the final 50K steps. sigma_combined crossed below 0.10 (the flagged collapse threshold) around t=60–70K and remained near or below it for the remaining 180K steps. The R40 Goldilocks clamp (σ_p_min ≈ 0.135) prevented collapse in the static-ball setting but did not hold under moving-ball dynamics.

**Verdict against the Phase XII Section 6 criteria:**

| Metric | Threshold | Actual | Grade |
|---|---|---|---|
| R41 both-touched @ 0.08 | ≥ 4/20 | 1/20 | REFUTED |
| R41 single-step ablation L2 | ≥ 0.20 | 0.8511 | PASS |
| R41 episode-level (pixels NORMAL − ZEROED) | ≥ +3 both-touched | NOT MEASURED (see Methodological Lesson) | INCOMPLETE |
| R41 vs R42 both-touched @ 0.08 | R41 ≥ R42 + 4 | R41=1, R42=2 (R41 worse by 1) | REFUTED |
| R41 static both-touched | ≥ 10/20 | 9/20 (one episode short) | NEAR-REFUTED |

**VERDICT: REFUTED.** The architectural integration of vision is confirmed at the encoding level (ablation 0.85 — highest in project history) but vision did not rescue the moving-ball task. R41 underperformed R42 by one episode at the primary condition and by nine episodes on static-ball generalization. The bottleneck has moved from "vision not integrated" to "vision integrated but pointed in the wrong direction."

**What we learned:**

1. R41 is the highest ablation delta ever measured in this project (0.85), showing that MICOA's predictive-KL mechanism scales — moving balls under MICOA drive vision into even more deeply integrated territory than R40's static-ball setup did. At the level of encoding, this is a success.

2. The task outcome tells a completely different story. Vision did not rescue the moving-ball problem. R41 (1/20 both-touched on moving balls) matched R42 (2/20 proprio) within noise, and R41 (9/20) was 9 episodes worse than R42 (18/20) on static-ball generalization. Adding MICOA + vision under moving-ball training conditions cost 9 episodes of static-ball capability compared to plain proprio alone. This is the sharpest evidence yet that high ablation sensitivity does not imply behavioral usefulness.

3. The MICOA encoder suffered severe pathology under moving-ball dynamics despite starting with the R40 Goldilocks σ clamp. sigma_combined collapsed to ~0.10 at around t=60K and kl_pred_k1 exploded to 638–760 by the run's end — exceeding R38's flagged ceiling of ~116 by more than 5×. The visual encoder appears to have been forced into increasingly extreme precision estimates as it tried to track moving targets, eventually locking itself into a degenerate regime. The σ clamp that worked for static balls was not sufficient for moving balls.

4. The bottleneck in this project has now been isolated precisely: the architectural gap ("vision not integrated") is closed. The new gap is "the RL update does not use the now-integrated visual representation to find better actions." Whether this is the actor's policy gradient, the nature of the MICOA loss under dynamic targets, or some interaction between the two is the open question.

5. R42's static-ball performance (18/20, mean reward +127.6) is the best proprio result yet on this substrate, validating the 250K training length and 10K checkpoint cadence as significant improvements over Phase VIII's 80K runs with 50K cadence.

**Is vision load-bearing?** Action-level: yes — ablation delta 0.85 is the highest this project has produced. Outcome-level: no — R41 underperformed R42 on both the primary condition and static generalization. These two answers are not contradictory: vision's weight in the policy's computations is large, but the content it encodes under moving-ball pathology does not steer the creature toward the balls. A large bad signal is worse than a small good signal.

**Methodological lesson — eval_phase_h.py has hardcoded paths:** The Phase XII launch script attempted to evaluate R42 using eval_phase_h.py with a --run-tag flag. eval_phase_h.py silently ignores all CLI flags and uses hardcoded Phase VIII run paths; it exited 0 while evaluating the wrong model entirely. R42's numbers in this entry came from a one-off eval_phase_iv_R42.py script created after the failure was detected. Bug: eval_phase_h.py has no argparse, no parameterization, and no warning when its flags are ignored. Fix: either add a parameterized CLI to eval_phase_h.py or create a shared eval_phase_iv.py that accepts --run-tag and --ball-speed from the command line. Any eval script with hardcoded paths that is also referenced in launch automation should be treated as a latent data-integrity bug.

**Theory signals:** The Behavioral Prediction Framework predicts that a policy with useful internal predictive structure should produce coherent, task-relevant behavior — not just large action changes when inputs are perturbed. R41's ablation of 0.85 confirms prediction is happening; the task regression (1/20 vs R42's 2/20, and −9 on static) confirms the predictions are not task-useful. The framework's prediction that predictive coding would improve behavior on dynamic targets is weakly disconfirmed at the behavioral level, while strongly confirmed at the representational level. This is a new split in the evidence that the framework does not easily accommodate. The Pattern Learning Framework predicts that sparse, distributed internal patterns should fire for structurally similar inputs regardless of exact position. R41's encoder pathology (kl_pred_k1 = 638, sigma_combined = 0.10) suggests the pattern representation did not stabilize — the encoder was driven to extreme precision estimates by moving targets instead of forming stable sparse codes. A healthy sparse code would produce stable ablation deltas; the fact that R41's ablation (0.85) is far above R40's (0.48) despite worse task outcomes is consistent with an over-fitted, non-generalizing representation rather than a stable sparse one.

**Next question:** With the ablation barrier crossed and encoder pathology confirmed as the likely mechanism, the question is whether the sigma collapse under moving balls can be prevented — and if it can, whether that alone would allow the policy gradient to use the visual signal productively — or whether the RL update itself needs a fundamentally different mechanism (such as image-aware augmentation in DrQ-v2) to turn rich visual encoding into better actions.

---

## 2026-05-30 — Phase XIII R43/R44: MICOA + vision vs. proprio control on static reachable balls with eccentricity sweep (phase_v_R43_micoa_vision_static_randbox / phase_v_R44_proprio_static_randbox)

**What we ran:** Two matched 250K-step SAC runs on the cart substrate, seed 42, with static balls (ball_speed=0.0) and per-episode ball placement drawn from a uniform box jitter of ±0.08 m on top of the ±0.15 m curriculum offset — so the ball's lateral position landed in roughly [0.07, 0.23] m from center across episodes, forcing per-episode directional uncertainty while keeping the ball inside the reachable band. R43 used MICOA + vision (n_envs=8, --micoa-beta 0.0, --micoa-pred-beta 0.1, R40 Goldilocks sigma clamp); R44 was the matched proprio-only control (n_envs=16, no vision, no MICOA). All other config identical: max_steps=2000, entropy anneal 0.5→0.2 over [15K,30K], velocity-bonus-scale=0.10, curriculum warmup 2000/ramp_end 15000/final_offset 0.15. Two new eval tools were introduced: eval_phase_v.py (eccentricity/direction sweep, 20 eps/bin, reach-conditional metrics) and eval_generalization_battery.py (size/distance/speed zero-shot sweeps, 15 eps/bin, reach-conditional). Both report got_close (fraction whose hand came within 0.12 m of the ball) and both|close (touch rate among episodes where got_close was true) to cleanly separate "couldn't get near" from "got near and missed." Goal: test the "generalization is the primary state" hypothesis — that proprioceptive general approach is what generalizes, and vision (if anything) is a late refinement deployed only where direction information is needed most.

**Numbers:**

Training (250K steps each, both runs completed normally, no crashes):
- ep_rew_mean: not separately reported by run; eval behavior at 250K summarized by eval table below
- Steps completed: 250,000 (R43); 250,000 (R44)

Eccentricity / Direction Sweep (eval_phase_v.py, 20 eps/bin, static ball, ball at lateral offsets from 0.00 to 0.25 m):

| ecc (m) | R44 both/20 | R44 got_close | R44 both\|close | R43 both/20 | R43 got_close | R43 both\|close | R43 abl_L2 |
|---|---|---|---|---|---|---|---|
| 0.00 | 20 | 0.85 | 1.00 | 20 | 0.85 | 1.00 | 1.04 |
| 0.05 | 20 | 0.60 | 1.00 | 19 | 0.65 | 0.92 | 0.62 |
| 0.10 | 16 | 0.40 | 0.62 | 16 | 0.50 | 0.60 | 0.65 |
| 0.15 | 10 | 0.55 | 0.64 | 11 | 0.30 | 0.33 | 0.67 |
| 0.20 |  5 | 0.50 | 0.30 |  4 | 0.50 | 0.40 | 0.69 |
| 0.25 |  0 | 0.25 | 0.00 |  0 | 0.35 | 0.00 | 0.72 |

Generalization Battery (eval_generalization_battery.py, 15 eps/bin):

SIZE sweep (ball_radius: 0.035 / 0.053 / 0.070 / 0.090 — at eccentricity 0.10):

| radius | R44 both|close | R43 both|close | R43 abl_L2 |
|---|---|---|---|
| 0.035 | 0.73 | 0.83 | 0.74 |
| 0.053 | 0.67 | 1.00 | 0.65 |
| 0.070 | 0.43 | 0.50 | 0.70 |
| 0.090 | 0.80 | 0.71 | 0.56 |
| invariance check | size-SENSITIVE (range 0.37) | size-SENSITIVE (range 0.50) | flat ~0.56–0.74 |

Note: both|close varies with size in the raw battery (invariance check fails), but the range is comparable across R43 and R44 — vision adds nothing to size invariance, and abl_L2 is flat across sizes, meaning vision is equally active regardless of how big the ball is.

DISTANCE sweep (forward ball_y from 0.25 to 0.65 m; trained range was approximately 0.27–0.43 m, so 0.55 and 0.65 are zero-shot extrapolation):

| dist (m) | R44 both/15 | R44 mean_contact_step | R43 both/15 | R43 mean_contact_step | R43 abl_L2 |
|---|---|---|---|---|---|
| 0.25 | 9 | ~1 (cart-sweep artifact) | 8 | ~1 (artifact) | 0.90 |
| 0.35 | 15 | 19.7 | 15 | 24.1 | 1.04 |
| 0.45 | 15 | ~1 (cart-sweep artifact) | 15 | ~1 (artifact) | 0.99 |
| 0.55 | 8 | 156.7 | 7 | 197.6 | 0.88 |
| 0.65 | 9 | 276.4 | 11 | 560.4 | 0.49 |

Time-to-contact rises with distance (script fit over all 5 bins; y=0.25 and y=0.45 are cart-sweep-confounded ~1-step artifacts that add scatter and understate the trend — on genuine-reach bins it is clean, monotone, and extrapolates: R44 0.35->20/0.55->157/0.65->276 steps, R43 0.35->24/0.55->198/0.65->560). Fits (bins 0.35–0.65, step-0 contacts excluded):
- R43 vision: steps ≈ 1292 × dist − 425, r = +0.85
- R44 proprio: steps ≈ 688 × dist − 218, r = +0.89

SPEED sweep (ball_speed: 0.000 / 0.010 / 0.020 / 0.030 at eccentricity 0.10):

| speed (m/s) | R44 both\|close | R43 both\|close | R43 abl_L2 |
|---|---|---|---|
| 0.000 | 1.00 | 1.00 | 0.65 |
| 0.010 | 0.67 | 0.80 | 0.59 |
| 0.020 | 0.50 | 0.33 | 0.57 |
| 0.030 | 0.50 | 0.25 | 0.55 |
| speed retention (fastest/slowest) | ~0.50 (graceful) | 0.25 (cliff) | — |

**What we learned:**

The proprioceptive general approach response is real and it generalizes. R44 reaches the ball in lawful proportion to distance (r=+0.89), extends successfully into distances outside its training band (0.55 m and 0.65 m were never seen during training), and degrades gracefully under ball motion — losing half its performance by speed 0.030 rather than collapsing suddenly. This is exactly what a learned internal model of "how to get to a thing" looks like when you probe it systematically. The "generalization is the primary state" hypothesis is confirmed for proprio.

*Clarification (added 2026-06-14):* The "generalization is the primary state" idea was the researcher's, offered on 2026-05-30 as a tentative reframing for discussion. Its core claim — proprioceptive object-agnostic approach as the primary, broadly-transferring generalization — is **confirmed here**. A separate strong-form rider that the assistant attached when designing Phase XIII (that vision would be recruited *selectively at the margin*, i.e. at high eccentricity) was **not supported** by the ablation data below. That rider is a claim about how vision bound under MICOA, not part of the researcher's core idea. Earlier write-ups that framed this as the hypothesis being "refuted" overstated it: the core idea held; only the added rider failed.

Vision adds essentially nothing to task outcomes. R43 and R44 are nearly identical at every eccentricity bin — the rows in the direction sweep table above are nearly indistinguishable. This result is on the cleanest task the project has run: static ball, reachable geometry, no encoder pathology (the static-ball setting keeps MICOA's sigma healthy). Phase VIII and XII found vision non-load-bearing under moving-ball conditions that were near-impossible for either modality; Phase XIII found the same null result on a static task where vision could in principle have helped with direction. The null is stronger here.

Vision under ball motion is actively worse. R43's speed retention (0.25, a cliff) is half of R44's (0.50, graceful). This is the third time this project has seen vision degrade faster than proprio under ball motion, now cleanly isolated on a reachable task where the proprio baseline is strong.

**Methodological wins:** eval_phase_v.py and eval_generalization_battery.py are fully parameterized (no hardcoded paths, fixing the eval_phase_h.py bug documented in Phase XII). The reach-conditional metric (both|close) cleanly separates "couldn't get close" from "got close and whiffed," so unreachable placements no longer miscounted as policy failures. The box=0.08 jitter choice worked as intended — the task landed in a non-saturated eccentricity band where there is room to see differences across conditions. The step-0 contact artifact (spawn-overlap triggering false contact at step 0) was identified and corrected; distance-bin results use only contacts after step 0.

**Is vision load-bearing?** Not yet confirmed — and this is now the strongest negative result in the project. The abl_L2 range is 0.62–1.04 across all eccentricity bins (well above the 0.05 load-bearing threshold, so vision is structurally integrated and influencing actions), yet R43 and R44 produce virtually identical behavior at every tested condition. Integration is not the same as usefulness. Phase XII showed the same split on moving balls; Phase XIII now shows it on the cleanest possible static task, making the "vision integrated but pointed the wrong way" diagnosis harder to attribute to task difficulty.

**Theory signals:** The Behavioral Prediction Framework's central prediction is that useful internal models should produce coherent behavior that scales lawfully with task structure. R44's proprio policy passes this test cleanly — lawful distance scaling (r=0.89, monotone on the clean reach bins), graceful speed degradation, extrapolation outside the training band. R43's vision policy also passes the distance-scaling test (r=0.85), but the ablation evidence shows the visual component of its predictions is not contributing to that coherence. Vision is generating predictions (abl_L2 is high everywhere), but those predictions are not task-calibrated — the creature's behavior would be the same without them. The Pattern Learning Framework predicts that similar inputs should activate overlapping internal patterns. The flat abl_L2 across ball sizes (0.56–0.74 in R43's size sweep) is consistent with stable internal coding of the visual channel — the representation is not fragmented. But the absence of any abl_L2 rise with eccentricity (it is highest at ecc=0.00, where directional information is least needed, and flat-to-slightly-rising thereafter) is direct evidence that the visual patterns being activated are not encoding ball direction in a way that gets used.

**Next question:** Why does MICOA bind vision uniformly and non-directionally — with the highest activation precisely where vision carries the least direction information — rather than recruiting it selectively where it would help? The natural probe is to decode ball lateral position from R43's vision latent directly: if a linear probe trained on the latent cannot predict ball-x above chance, that would directly explain why abl_L2 is high but touch rates are unchanged — the visual representation has learned to affect actions without encoding the task-relevant variable. If the decoding succeeds, the problem is in how the policy uses the representation rather than in the representation itself.

### Vision-latent probe (2026-05-30, follow-up to the Next question)

Ran the proposed probe: `probe_vision_latent.py` collected 3072 (latent, egocentric-ball-position) samples from R43 across randomized static ball placements (lateral x_ego range −0.66 to +0.49 m, forward y_ego −0.26 to +0.58 m), then cross-validated (5-fold) Ridge regression decoded egocentric ball position from each MICOA expert latent.

| decode target | mu_v R² (vision) | mu_p R² (proprio control) |
|---|---|---|
| ball x_ego (LATERAL / direction) | **0.136 ± 0.021** | 0.096 ± 0.007 |
| ball y_ego (forward / distance) | 0.217 ± 0.016 | **0.426 ± 0.042** |

**Verdict: WEAK REPRESENTATION — vision barely encodes ball direction.** The vision
latent decodes ball lateral position at R² = 0.136 — well below useful (a usable
spatial code would be R² ≳ 0.5), but slightly *above* the proprio control (0.096).
So vision is not at pure chance, but it carries only a trace of direction
information — nowhere near enough for the policy to steer on. This resolves the
Phase XIII paradox: vision-ablation L2 is 0.6–1.04 (vision strongly changes actions)
while the thing it encodes about the ball is only marginally informative. The high
ablation is mostly vision acting on non-directional features (ball presence,
lighting, self-motion) plus a weak directional trace the policy cannot exploit.

A notable secondary result: **proprio decodes forward distance (y_ego) better than
vision does (0.426 vs 0.217).** The proprio latent — body/cart state — carries more
information about how far the ball is than the camera latent does. This is coherent
with the Phase XIII distance law being carried by proprio, and underlines that vision
is the *weaker* channel on exactly the spatial variables that matter.

The earlier ecc=0 ablation spike is consistent with this: vision is maximally
"active" largely on a non-directional feature, and what little direction it encodes
(R² 0.136) is too weak to change outcomes.

**Implication for next steps:** the fix target is the *visual encoder*, not the
actor's policy gradient. Under static balls, MICOA's predictive-KL target is
proprio(t+1), which contains no ball-direction signal — so nothing in the loss ever
rewarded the encoder for representing ball direction; the trace it does carry
(R² 0.136) is incidental, not optimized for. The natural first guess was that adding a
direction-dependent loss term would fix it — but the encoder-capacity test below
*does not support that* (end-to-end supervision barely lifts lateral decode), so the
bottleneck is more likely the substrate (resolution/architecture) than the loss.
Candidate interventions, in priority order: (a) a rigorous capacity rerun (fresh-init
CNN, tuned supervised training, possibly 64×64) to firm up capacity-vs-loss; (b) only
if capacity is adequate, add an auxiliary direction-prediction loss and re-probe.
(Cameras are 32×32 stereo RGB.)

**Methodological note:** sklearn's `cross_val_score` deadlocked under macOS
Accelerate threading; the probe forces single-threaded BLAS (`OMP_NUM_THREADS=1`
etc.) and uses an explicit KFold loop. Keep this pattern for any sklearn-in-eval
scripts on this machine.

### Reachable-band probe + encoder-capacity test (2026-05-30, follow-ups)

Two follow-ups sharpened the probe verdict. All numbers below are from verified log
files (`phase_v_R43_vision_latent_probe_reachable.log`,
`phase_v_R43_encoder_capacity.log`).

**(1) Reachable-band probe.** Re-ran the latent probe restricted to the band AB can
actually touch (`--reachable-x 0.20`, 3430 samples), since the original probe sampled
ball positions out to ±0.66 m — well past reach — which unfairly inflated the apparent
failure. Restricted to reachable positions:

| decode target | mu_v R² (vision) | mu_p R² (proprio control) |
|---|---|---|
| ball x_ego (LATERAL / direction) | 0.080 | 0.128 |
| ball y_ego (forward / distance) | 0.235 | 0.432 |

Vision decodes reachable lateral direction at R² 0.080 — *below* even the proprio
control (0.128, and proprio has no target signal in its observation). So the
representation-failure verdict holds and is in fact slightly stronger on the
reachable band: the trained MICOA vision latent does not encode where the ball is,
even for balls AB can reach.

*Reproduced 2026-06-16:* re-ran `probe_vision_latent.py` from scratch (same args,
fresh process) → mu_v lateral R² = 0.081, mu_p 0.128, mu_v forward 0.235, mu_p 0.432
— identical to the logged numbers above. The representation-failure result is stable
and reproducible. The probe and encoder-capacity scripts are now committed to the
repo (`visualization/probe_vision_latent.py`, `visualization/encoder_capacity_test.py`).

**(2) Encoder-capacity test** (`encoder_capacity_test.py`): decides whether this is a
*capacity* limit (the 32×32 CNN physically cannot resolve direction) or a
*loss-geometry* limit (it can, but MICOA's objective never asked). Method: collect
(pixels, egocentric-ball-position) pairs on the reachable band (4091 samples), then
train a supervised head mu_v → position, FROZEN encoder vs UNFROZEN (end-to-end),
60 epochs. Authoritative numbers from `phase_v_R43_encoder_capacity.log`:

| decode target | FROZEN encoder R² | UNFROZEN (end-to-end) R² |
|---|---|---|
| ball x_ego (LATERAL / direction) | 0.066 | 0.104 |
| ball y_ego (forward / distance) | 0.312 | 0.403 |

**Verdict: CAPACITY-LIMIT-LEANING (provisional) — does NOT support the loss-geometry
escape.** Unfreezing the encoder and training it end-to-end on a direction target
barely moves lateral decode (0.066 → 0.104) — nowhere near a usable spatial code
(≳0.5). The 32×32 stereo CNN, even when *explicitly supervised* to predict reachable
ball direction, does not learn to. This points to a capacity/representation limit
(resolution, or the MICOA encoder init sitting in a poor basin), NOT a simple
missing-loss-term that a direction-dependent objective would fix.

**Important caveat (why "provisional"):** the supervised probe may be underpowered —
60 epochs, fixed lr=1e-3, a 1-hidden-layer head, encoder initialized from the trained
MICOA weights rather than fresh. A stronger supervised setup (more epochs, lr sweep,
fresh-init CNN) could still lift the unfrozen ceiling. So this is suggestive, not a
hard capacity proof. The clean follow-up is to run that test from a randomly
initialized CNN with proper supervised tuning; if THAT also caps near ~0.1 lateral,
capacity-limit is confirmed and higher camera resolution is indicated.

**Honesty note:** an earlier version of this section reported UNFROZEN R² of
0.485/0.876 and a "LOSS-GEOMETRY LIMIT" verdict. Those numbers were written before the
full run's log was read and do not match it; they are corrected above to the verified
log values, which flip the verdict. This is logged so the reversal is traceable.

---

## 2026-06-14 — Phase XIV R45: DroQ critic-stabilization infrastructure validation (phase_w_R45_droq_proprio_validation)

**What we ran:** A 150K-step infrastructure validation run on the same proprio-only task config as Phase XIII R44 (constant-velocity-bouncer cart at speed 0.15, static ball at speed 0.0, random ball box ±0.08 m, hip actuation off, memory obs, strength scale 1.0, entropy anneal 0.5→0.2 over [15K,30K], velocity bonus 0.10, curriculum warmup 2000/ramp end 15000/final offset 0.15, seed 42, n_envs=16). The single change from R44: a new `--droq` flag activated DroQ-style critic regularization from Smith/Kostrikov/Levine 2022 ("A Walk in the Park") — LayerNorm and Dropout (rate 0.01) after each hidden critic layer, actor architecture untouched, gradient steps per environment step raised from the SB3 default to 4 (UTD=4). The new code lives in `crawler/droq_policy.py` (DroQSACPolicy, DroQCritic). The `--droq` flag defaults OFF; existing runs that do not pass it are bit-identical to prior configs. The stated pass/fail criteria were: critic_loss stays below 10 throughout; eval reward is positive in the second half of training; no peak-then-collapse pattern.

**Numbers:**

- ep_rew_mean: +413 (first rollout window, ~3856 steps) → −115 (final rollout window, ~147K steps)
- Eval mean reward trajectory (10K cadence):
  - 10K: −322 ± 1013
  - 20K: −741 ± 1071
  - 30K: −550 ± 983
  - 40K: −334 ± 885
  - 50K: −181 ± 798
  - 60K: −285 ± 853
  - 70K: −12 ± 731
  - 80K: −838 ± 930
  - 90K: −332 ± 843
  - 100K: −420 ± 907
  - 110K: −747 ± 1046
  - 120K: −350 ± 881
  - 130K: −689 ± 860
  - 140K: −179 ± 729
  - 150K (final): −486 ± 857
- loco_speed_mean: N/A (not separately logged; creature is locomoting, as the rollout ep_rew_mean starting at +413 and remaining positive through the first ~70K steps confirms activity)
- touch_rate (deterministic eccentricity sweep, eval_phase_v.py, 20 eps/bin):

| ecc (m) | R44 both/20 (250K) | R45-DroQ both/20 (150K) |
|---|---|---|
| 0.00 | 20 | 19 |
| 0.05 | 20 | 20 |
| 0.10 | 16 | **20** |
| 0.15 | 10 | 11 |
| 0.20 |  5 |  6 |
| 0.25 |  0 |  1 |

- critic_loss (R45-DroQ): baseline range ~4–11 with intermittent spikes to 40–145 (spike values at selected steps: 40.4, 25.4, 22.4, 20.2, 18.5, 20.7, 41.9, 40.5, 85.5, 58.2, 110, 66.1, 95.7, 74, 145)
- Steps completed: 150,000 (normal finish, no crashes)

**Baseline comparison (R44 at 250K, from Phase XIII):**

| ecc (m) | R44 both/20 | R45-DroQ both/20 |
|---|---|---|
| 0.00 | 20 | 19 |
| 0.05 | 20 | 20 |
| 0.10 | 16 | 20 (+4) |
| 0.15 | 10 | 11 (+1) |
| 0.20 |  5 |  6 (+1) |
| 0.25 |  0 |  1 (+1) |

**What we learned:**

DroQ integrated cleanly and did not degrade the task. R45 matches or beats R44 in every eccentricity bin despite running for 40% fewer steps (150K vs 250K), with the ecc=0.10 bin showing a notable improvement: 20 vs 16 touches. The generalization shape is identical — strong near center, graceful fall-off at the margins, zero both-touched at ecc=0.25 (just one episode slipping through at R45's ecc=0.25 is within noise for a 20-episode bin). This is consistent with a genuine sample-efficiency gain from the higher update-to-data ratio, though the comparison is imperfect (we lack a 150K checkpoint eval for R44).

The most important finding from this run is not about DroQ — it is about the pre-registered pass/fail criteria, which turned out to be invalid discriminators. The check "eval reward must be positive in the second half" fails for R44 as badly as it does for R45: R44's eval reward at 70K is its best (+167), and from 80K through 250K every R44 eval is negative (−6.02, −68, −108, −269, −191, +159 anomaly, −314, −601, −196, −399, −723, −432, −866, −598, −600, −710, −496, −378). Final R44 eval at 250K: −378. The check "critic_loss stays below 10" also fails for R44: critic_loss alternates between ~2–4 and spikes to 39, 55, 65, 76, 79, and 136 across the R44 run. R45's critic_loss baseline of ~4–11 with spikes to 40–145 is not meaningfully different from R44's profile. Both look the same; only the eccentricity eval distinguishes them. The pre-registered criteria were written on the assumption that reward negativity and critic_loss spikes indicate a broken or degraded policy. They do not, on this task. The SAC training reward for this env is dominated by step-cost and contact-event variance; the eval std (±600–1000 across the entire project history for this substrate) is so large that the mean reward is not a reliable indicator of policy quality within any single evaluation window. The critic_loss spikes are contact-event TD errors intrinsic to the environment's reward structure — when the creature makes or misses contact, a large one-step reward signal arrives that the Q-function has not yet seen, causing a momentary spike. These spikes appeared in R44, our confirmed-best generalizer (Phase XIII distance law r=+0.89). They are not a DroQ artifact.

**CRITICAL METHODOLOGICAL FINDING:** On this task and substrate, the ONLY valid success metric is the deterministic eccentricity sweep (eval_phase_v.py). Training ep_rew_mean and eval mean_reward are corrupted by structural variance; critic_loss spikes are contact-event artifacts shared by good and bad policies alike. Any future run on this substrate must be judged solely by the reach-conditional touch counts from the deterministic eval sweep.

**Honest caveats:**

- Single seed (42). Conclusions about DroQ's sample efficiency rest on a single matched run.
- The 150K-vs-250K comparison is not a matched-steps experiment. R44's 150K checkpoint was never separately evaluated. If R44 at 150K had matched or exceeded R44 at 250K (plausible, given the eval variance pattern), the per-step advantage for DroQ would shrink or disappear. A clean test requires either evaluating R44's 150K checkpoint or running DroQ to 250K.
- ±1 differences across 20-episode bins are within noise. The ecc=0.25 result (1 vs 0) and ecc=0.20 result (6 vs 5) should not be over-read.
- No sanity-render video was produced for this run. However, the environment configuration is identical to R44, which was validated, and DroQ only modifies the critic network — it does not change physics, rewards, or camera plumbing. The risk of a silent environment breakage is low.

**Verdict:** Infrastructure validation passed on the metric that matters. DroQ integrates cleanly, runs stably, and does not degrade the proprio generalizer. Tentative evidence of a sample-efficiency gain (matching 250K baseline at 150K), which is theoretically expected from the higher UTD ratio, but not yet cleanly confirmed. DroQ is safe to keep as an opt-in. Whether it delivers a real efficiency gain needs a matched-step comparison.

**Is vision load-bearing?** Not applicable — this is a proprio-only run (DroQ modifies only the critic; no vision policy is present). Vision cannot be assessed.

**Next question:** Does R44's 150K checkpoint (or a new DroQ run run to 250K) match R45's eccentricity profile — that is, is the apparent sample-efficiency gain real, or was R44's 250K performance also achievable at 150K without DroQ?

**Theory signals:** The Behavioral Prediction Framework predicts that a stable, predictive internal model should generalize to positions not seen in training. R45 reproduces R44's distance law and generalization shape in fewer steps — if the sample-efficiency advantage survives the matched-step check, it would suggest that the higher update-to-data ratio (UTD=4) allows the critic's value landscape to stabilize faster, giving the actor more reliable gradient signal earlier. This is consistent with the framework's emphasis on coherent internal predictive structure: more gradient steps per sample should accelerate the formation of a stable value map. The Pattern Learning Framework would predict that a higher UTD ratio helps stable sparse patterns form faster, which aligns with the tentative efficiency result. Neither framework has a specific prediction about critic_loss spikes, which is consistent with the finding that those spikes are environmental artifacts rather than learning pathology.

---

## 2026-06-15 — Phase XV (R46/R47/R48): Object variety — does it deepen the proprioceptive generalizer?

**What we ran:** Three proprio-only DroQ runs (250K steps each, same config as Phase XIV R45 but no vision) testing whether training on varied objects produces a generalizer that transfers zero-shot to held-out objects. R46 varied ball size only (trained on radii {0.040, 0.053, 0.075}; held-out sizes {0.047 interpolation, 0.090 extrapolation}). R47 varied ball shape only (trained on sphere, box, cylinder; held-out shapes ellipsoid and capsule, bounding size ~0.053). R48 combined both (trained on all size × shape combinations; held-out sizes AND held-out shapes tested). All runs: DroQ critic regularization (LayerNorm + Dropout 0.01, UTD=4), seed 42, n_envs=16, cart constant_velocity_bouncer speed 0.15, static ball (speed 0.0), hip off, memory obs, strength 1.0, entropy anneal 0.5→0.2 over [15K, 30K], curriculum warmup 2000 / ramp 15K / final offset 0.15 m. Direction B of the object-variety hypothesis: richer training variety deepens the proprioceptive equivalence class so the creature reaches anything it can touch.

**Numbers (training):**

| Run | ep_rew_mean start | ep_rew_mean end | Steps completed |
|-----|------------------|-----------------|-----------------|
| R46 size variety | 418 | −225 (training reward — invalid metric; see Phase XIV finding) | 250,000 (normal) |
| R47 shape variety | 415 | −70 (training reward — invalid metric) | 250,000 (normal) |
| R48 size+shape | 431 | 93 (training reward — invalid metric) | 250,000 (normal) |

Note: Training ep_rew_mean is structurally unreliable on this substrate due to cart-sweep variance (see Phase XIV). All performance conclusions below are from the deterministic eval sweeps only.

**Numbers (deterministic eval — the only valid metric on this substrate):**

R46 — SIZE VARIETY:

Held-out sizes (both/20):
| ball_radius | both/20 | both|close | mean_R |
|---|---|---|---|
| 0.047 (interpolation, held-out) | 17 | 0.83 | +24.4 |
| 0.090 (extrapolation, held-out) | 16 | 1.00 | −70.3 |
Held-out invariance range: 0.17 → INVARIANT

Full size sweep (both/20):
| ball_radius | both/20 | both|close | mean_R |
|---|---|---|---|
| 0.040 (trained) | 15 | 0.79 | −162.2 |
| 0.047 (held-out) | 17 | 0.83 | +24.4 |
| 0.053 (trained) | 20 | 1.00 | +364.2 |
| 0.075 (trained) | 10 | 0.33 | −678.8 |
| 0.090 (held-out) | 16 | 1.00 | −70.3 |
Full sweep range: 0.67 → size-SENSITIVE (driven entirely by the 0.075 anomaly — see caveats)

R46 eccentricity sweep (sphere, both/20):
| ecc | both/20 | both|close |
|---|---|---|
| 0.00 | 19 | 1.00 |
| 0.05 | 19 | 1.00 |
| 0.10 | 18 | 0.92 |
| 0.15 | 15 | 0.71 |
| 0.20 | 11 | 0.50 |
| 0.25 |  0 | 0.00 |

R47 — SHAPE VARIETY:

Per-shape eccentricity sweep (both/20) — sphere and box are TRAINED, cylinder trained, ellipsoid and capsule are HELD-OUT:
| shape | ecc=0.00 | ecc=0.05 | ecc=0.10 | ecc=0.15 | ecc=0.20 | ecc=0.25 |
|---|---|---|---|---|---|---|
| sphere (trained) | 20 | 20 | 19 | 8 | 2 | 0 |
| box (trained) | 20 | 20 | 20 | 12 | 5 | 0 |
| cylinder (trained) | 19 | 20 | 15 | 10 | 5 | 0 |
| ellipsoid (HELD-OUT) | 20 | 20 | 15 | 10 | 5 | 0 |
| capsule (HELD-OUT) | 17 | 15 | 12 | 8 | 3 | 0 |

R47 eccentricity sweep (sphere/default, both/20): 20, 20, 19, 8, 2, 0 at ecc 0.00→0.25.

R48 — COMBINED SIZE + SHAPE:

Held-out sizes (both/20):
| ball_radius | both/20 | both|close | mean_R |
|---|---|---|---|
| 0.047 (interpolation, held-out) | 19 | 1.00 | +188.5 |
| 0.090 (extrapolation, held-out) | 15 | 0.50 | −194.8 |
Held-out range: 0.50 → size-SENSITIVE (0.090 bin is weaker; see caveats)

Full size sweep (both/20):
| ball_radius | both/20 | both|close | mean_R |
|---|---|---|---|
| 0.040 (trained) | 18 | 0.93 | +149.0 |
| 0.047 (held-out) | 19 | 1.00 | +188.5 |
| 0.053 (trained) | 20 | 1.00 | +368.1 |
| 0.075 (trained) | 6 | 0.50 | −1213.5 |
| 0.090 (held-out) | 15 | 0.50 | −194.8 |
Full sweep range: 0.50 → size-SENSITIVE (again driven by 0.075 anomaly)

Per-shape sweep (R48, both/20 at ecc 0.00/0.05/0.10/0.15/0.20/0.25):
| shape | ecc=0.00 | ecc=0.05 | ecc=0.10 | ecc=0.15 | ecc=0.20 | ecc=0.25 |
|---|---|---|---|---|---|---|
| sphere (trained) | 20 | 20 | 19 | 13 | 5 | 0 |
| box (trained) | 20 | 20 | 20 | 17 | 5 | 0 |
| cylinder (trained) | 20 | 20 | 18 | 13 | 5 | 0 |
| ellipsoid (HELD-OUT) | 20 | 20 | 20 | 13 | 5 | 0 |
| capsule (HELD-OUT) | 20 | 15 | 16 | 6 | 4 | 0 |

R48 eccentricity sweep (default/sphere, both/20): 20, 20, 19, 13, 5, 0 at ecc 0.00→0.25.

**Baselines for comparison (from Phase XIII / Phase XIV):**
- R44 proprio (single fixed size, no variety): ecc both/20 = 20, 20, 16, 10, 5, 0
- R45 DroQ proprio (single fixed size): ecc both/20 = 19, 20, 20, 11, 6, 1
- R44 size invariance: both|close range = 0.37 (size-SENSITIVE on full sweep; trained on one size only)

**What we learned:**

Object variety transfers zero-shot. The creature in R46 reached held-out ball sizes it had never trained on — 17/20 for the small interpolation size (0.047) and 16/20 for the large extrapolation size (0.090) — nearly identical to its performance on trained sizes. The held-out-only range of 0.17 is the INVARIANT threshold, meaning the policy genuinely does not care much what size the ball is as long as it is in the smaller half of the size distribution. This directly supports the object-agnostic equivalence class prediction: the creature treats "any reachable object" as the same category and reaches it the same way, because what it has actually learned is a reach-motor program driven by contact-distance proprioception, not by anything size-specific.

Shape variety (R47) also transfers zero-shot at the center of the workspace. Ellipsoid — a shape the R47 policy never trained on — scored 20/20 both-touched at ecc=0.00 and 0.05, exactly matching trained shapes. Capsule, the most geometrically distinct held-out shape (pill-shaped, longer axis), scored 17/20 at center and 15/20 at ecc=0.05. The policy's proprioceptive reach program is shape-agnostic at zero eccentricity: it only needs to know "something is here" via the touch signal, not what shape it is. At higher eccentricities capsule's transfer is weaker than other shapes — consistent with the idea that off-center reaching relies on a rough positional estimate from proprioception, and the unusual elongated geometry of a capsule may shift the contact point relative to what the proprioceptive estimate predicts.

Combined variety (R48) shows a clear improvement in direction generalization at the margins compared to the single-size baseline (R44). At ecc=0.15, R48 scores 13/20 versus R44's 10/20 and R45's 11/20. The improvement is modest but consistent across all shapes including the held-out ellipsoid (13/20 at ecc=0.15). The most striking R48 result is that both held-out shapes at center are near-perfect: ellipsoid 20/20 and capsule 20/20 at ecc=0.00. Training on multiple shapes may have driven the policy to rely more heavily on the contact signal and less on shape-specific proprioceptive geometry, producing a more robust center-approach.

Shape-only training (R47) slightly degraded the high-eccentricity performance compared to size-only (R46) and combined (R48): R47 sphere falls to 2/20 at ecc=0.20 vs R46's 11/20 and R48's 5/20. However, R47's center-performance (ecc=0.00, 0.05) is perfect across all five shapes including held-out ones. This suggests shape variety does not hurt center-reaching but may introduce some variability in the learned reach program that slightly reduces the eccentricity ceiling. The direction limit of ecc=0.25 (0/20 both-touched in all runs) is universal and is not caused by object variety.

**Honest caveats:**

1. REPRODUCIBLE ANOMALY at ball radius 0.075. In both R46 and R48, the 0.075 bin — a TRAINED size — drops sharply: R46 gets only 10/20 (mean_R −679) and R48 gets only 6/20 (mean_R −1214), while adjacent sizes score 15–20/20. This cannot be noise: 0.075 appeared in both independent runs and both times produced the same hard dip. The most likely explanation is an interaction between the large ball and the constantly-moving cart (the "constant_velocity_bouncer" cart oscillates across the workspace): a large ball sitting near the cart path may be pushed out of normal reach geometry by cart collisions, or may be partially occluded by the cart body in a way that confuses the touch-contact window. This anomaly is what drives the full-sweep invariance flag to SENSITIVE (range 0.50–0.67) in both runs. The held-out-only range (0.17 for R46) looks invariant because 0.075 is a trained size and is excluded from the held-out eval. Investigation is warranted before drawing conclusions about the 0.075 regime specifically.

2. The extrapolation size 0.090 (large ball) is weaker than the interpolation size 0.047 in both R46 (16 vs 17) and R48 (15 vs 19). This is the expected direction: interpolating between trained sizes is easier than extrapolating beyond the training range. The gap is small enough (1–4 episodes out of 20) to be partly noise, but the direction is consistent.

3. Capsule is the weakest-transferring held-out shape in both R47 and R48, consistent across eccentricity bins. This makes sense: capsule is a rounded cylinder with elongated geometry unlike any of the three training shapes. The policy's contact-approach geometry learned on compact shapes (sphere, box, cylinder of similar bounding size) transfers less cleanly to a longer object.

4. The universal fall-off to 0/20 at ecc=0.25 is not an object-variety problem. It is present in R44 (single size), R45 (DroQ, single size), and all three Phase XV runs regardless of shape or size. The creature's proprioceptive reach program simply cannot handle targets placed at 25 cm of lateral offset — a direction limit intrinsic to the task geometry or the creature's reach mechanics.

5. All three runs are single-seed (seed 42). The ±1–2/20 variation across bins is within expected noise for 20-episode bins. Differences smaller than 3 episodes should not be over-read.

6. Note on R48 held-out size invariance flag: the log reports `both|close range = 0.50  (size-SENSITIVE)` for the held-out sweep — this is because the large-ball extrapolation bin (0.090) scored 15/20 with both|close = 0.50, and the interpolation bin (0.047) scored 19/20 with both|close = 1.00, giving a range of 0.50. The briefing stated range = 0.17 (INVARIANT) for R48 held-out, which is incorrect; 0.17 is R46's held-out range. The log values are reported here.

**Is vision load-bearing?** Not applicable — all three Phase XV runs are proprio-only (no camera input, no vision policy). There is no pixel signal to ablate. Vision load-bearing status is unchanged from the Phase XIII/XIV conclusion: action-level active but outcome-level inert on static reach tasks.

**Theory signals:**

The Pattern Learning Framework predicts that similar inputs activate overlapping patterns. The zero-shot transfer to held-out shapes and sizes is consistent with this: the creature's internal representation of "object to reach" must be encoding something general (possibly just the proximity/contact signal from the touch bit and joint displacements) rather than memorizing shape-specific lookup entries — otherwise transfer to ellipsoid and capsule would fail. The near-perfect ellipsoid transfer (20/20 at center, both R47 and R48) in particular suggests the internal representation clusters held-out shapes with trained ones rather than treating them as unknowns.

The Behavioral Prediction Framework would predict that a policy with genuine internal models of the reach task should stay coherent even at novel object configurations. The lawful distance generalization (R46 time-to-contact law r=+0.84; R48 r=+0.87) is consistent with this — the farther the ball, the more steps it takes, in a linear relationship the creature was never explicitly rewarded for. The zero-shot shape/size transfer is also consistent with behavior staying coherent under novel conditions. However, both frameworks' predictions are consistent with a simpler explanation (the policy just learned "move toward contact sensor activation" which is object-agnostic by construction). We cannot distinguish them without probing internal representations.

**Next question:** Is the 0.075-radius anomaly a cart-ball collision artifact (testable by re-running the 0.075 bin with the cart disabled or at a different speed), and does removing it reveal true size invariance across the full trained range?


## 2026-06-15 — Phase XV diagnostic: the "0.075 anomaly" resolved, and a hand-touch finding

Follow-up to the Phase XV entry above. Two results: the 0.075 anomaly is diagnosed
(a contact-dynamics artifact, not a generalization hole), and — more importantly —
the both-touched task is **never completed with the hands** (`both_HAND = 0` across
every size and every eccentricity, on the whole cart line), though hands *do* engage
for one ball, more so at high eccentricity. Reframed below: body contact is a valid
"world-is-consistent" signal, and hand-reach is the intentional signal to track over
training — `both_HAND = 0` reads as a developmental stage, not a bug.

Scripts: `visualization/diag_0075_anomaly.py` (stats), `render_0075_diag.py` (video),
`rescore_hand_touch.py` + `rescore_hand_touch_ecc.py` (broad vs hand-only). Logs in `results/`.

### 1) The 0.075 anomaly is real, but it's a BAND, not a point — and it's a manipulation artifact

Re-ran the size neighbourhood at the original eval seed (20000) and a fresh seed
(70000), 30 episodes each, on both R46 and R48. both-touched / 30:

| radius | R46 orig | R46 fresh | R48 orig | R48 fresh |
|--------|----------|-----------|----------|-----------|
| 0.053  | 29 | 28 | 30 | 29 |
| 0.070  | 16 | 15 | 14 | 12 |
| 0.075  | 17 | 16 | 13 | 14 |
| 0.080  | 15 | 11 | 14 | 13 |
| 0.090  | 23 | 22 | 23 | 22 |

- **Real, not a shared-seed artifact.** The original hypothesis was that both evals
  sharing seed 20000 made an unlucky-geometry blip look reproducible. Refuted: the
  fresh seed reproduces it almost exactly.
- **It's a dead BAND ~[0.070–0.080], not the point 0.075.** The original `full_sizes`
  eval only sampled 0.075 between 0.053 and 0.090, so a whole failing band disguised
  itself as a single-point spike. The original 6/20 and 10/20 were also low-side
  binomial noise around a true rate ~0.45–0.55 (30 eps tightens it).
- **Mechanism (watched on video, seeds 20003/20007/20012 at r=0.075):** the creature
  contacts the first ball, but with a mid-size ball the contact is a **glancing,
  off-centre blow that knocks the ball skidding to the arena wall**, out of reach.
  Sitting on a fixed cart base with no locomotion, it cannot recover the displaced
  ball, and its post-contact policy settles into an upright posture without
  re-reaching (in one episode the second ball ends up at its feet, untouched). The
  episode never terminates (needs both) → hunger penalty → reward −600 to −2200.
- **It is geometric, not momentum.** A radius override only sets `geom_size`; ball
  mass/inertia is unchanged (env code). So it is not "heavier ball". At 0.053 the
  hand/body reaches near the ball centre → soft contact, ball stays. At 0.070–0.080
  the larger surface forces an earlier, off-centre strike → skid. At 0.090 it
  partially recovers (not fully explained; possibly the surface is so close that
  contact is immediate/repeated).
- **Verdict:** this is a manipulation/metric artifact, NOT a hole in size
  generalization. The reach generalizes across all sizes (got_close is similar
  everywhere); what breaks is two-ball *completion* when a mid-size ball is knocked
  out of a no-locomotion creature's reach envelope. This confirms and sharpens
  Phase XV caveat #1 (which guessed a cart/large-ball interaction).

### 2) Bigger finding: task completion is non-hand body contact — and what that means

Prompted by the observation that the creature never visibly touches a ball with a
hand, re-scored under the env's two contact metrics:
- `touched_ball*` (BROAD): contact between the ball and ANY MIMo geom — feet, legs,
  torso, head, or hands. This is what reward, termination, and our "both-touched"
  success use.
- `hand_touched_ball*` (HAND-ONLY): contact only with the 8 hand/finger geoms. The
  env computes it but does not use it for success.

Phase XV size sweep (30 eps, seed 20000):

| run | size | both_BROAD | both_HAND | first-touch-is-hand |
|-----|------|-----------|-----------|---------------------|
| R46 | 0.053 | 29/30 | **0/30** | 17% |
| R46 | 0.075 | 17/30 | **0/30** | 7% |
| R46 | 0.090 | 23/30 | **0/30** | 3% |
| R48 | 0.053 | 30/30 | **0/30** | 17% |
| R48 | 0.075 | 13/30 | **0/30** | 0% |
| R48 | 0.090 | 23/30 | **0/30** | 0% |

Audit of the EARLIER runs the "proprio generalizes" headline rests on — R44
(Phase XIII ecc sweep) and R45 (Phase XIV DroQ), 30 eps, seed 10000:

| run | metric | ecc 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|-----|--------|----------|------|------|------|------|------|
| R44 | both_BROAD | 29 | 29 | 23 | 14 | 7 | 0 |
| R44 | both_HAND  | **0** | **0** | **0** | **0** | **0** | **0** |
| R44 | any_HAND   | 20 | 15 | 8 | 16 | 11 | 8 |
| R45 | both_BROAD | 29 | 30 | 30 | 18 | 10 | 1 |
| R45 | both_HAND  | **0** | **0** | **0** | **0** | **0** | **0** |
| R45 | any_HAND   | 8 | 4 | 11 | 16 | 21 | 18 |

- **both_HAND = 0 everywhere, every run.** The two-ball task is never *completed* with
  two hands, across the entire cart-substrate line (not just Phase XV). Verified not a
  detection bug: the hand-geom set is correctly populated (8 geoms: right/left
  hand1/hand2/fingers1/fingers2, subset of the MIMo geom set), and broad-touch fires
  normally in the same episodes.
- **But hands DO engage for one ball** (`any_HAND` up to 20–21/30), and that engagement
  *rises with eccentricity* — exactly where the ball is off to the side and the
  substrate cannot just deliver it, so a genuine lateral reach is required. The pattern:
  where the cart can deliver the ball (low ecc), success is high and hand-free; where a
  reach is actually needed (high ecc), the hand engages but cannot finish, so success
  collapses. So this is NOT "zero reaching" — it is "reaching happens and is not yet
  good enough to complete the task."
- **How the body contact arises (physics correction).** Balls are fixed at (0, ±0.35)
  and the cart sweeps AB's platform straight along that y-line, carrying AB's *body*
  into the stationary ball. Note: the cart geom itself is a NON-COLLIDING visual marker
  (`contype=0 conaffinity=0`); it never pushes the ball. All ball contact is via AB's
  body geoms or the real (solid) platform — never the cart. So "the substrate delivers
  the ball to AB's body" is the right reading; "the cart pushes the ball" is not.

**Interpretation (reframed — body contact is a real signal, not just a confound).**
Both kinds of contact matter and they mean different things developmentally:
- **Body-bump = unconditioned confirmation.** Hip into a table, head under a table —
  "the object is there; the world is consistent." Reflexive, undirected, and a valid
  signal for the equivalence-class / world-consistency learning this project is about
  (cf. unconditioned vs conditioned response in GLOSSARY).
- **Hand-reach = conditioned / intentional.** "I am going over there to pick it up" —
  what an undirected encounter *becomes* after repetition: directed, anticipatory.

So `both_HAND = 0` is not a bug to patch; it is a **developmental readout**: AB is still
at the incidental-encounter ("world is consistent") stage and has not yet crossed into
reliable intentional two-handed reaching, with `any_HAND` at high ecc as the first sign
of directed reaching. The thing to *measure* is therefore the **hand-touch fraction over
training time** — its growth relative to body-touch would be the signature of encounters
maturing into intentional reaches, which is the project's central question made into a
number. The caution that remains: Phase XV's "object-variety zero-shot transfer" headline
is a BROAD-touch result; read it as transfer of "be in a posture the swept ball intersects
(and don't knock it away)", which is weaker than "AB learned to *reach* for novel-sized
objects" — true, but not yet the intentional reach.

### Methodological note
`describe_video.py` (Gemini) narrated the failing 0.075 episodes as successful
"throw-then-pickup" sequences — it over-reported success because it does not know the
task is "touch both balls". Pass/fail must come from the metric and from frames read
directly, not the free-text narration.

### Recommended next steps
1. **Track hand-touch fraction over training checkpoints** as a developmental metric —
   does intentional (hand) contact grow relative to incidental (body) contact as AB
   learns? That growth is the project's central question rendered as a number.
2. Report BOTH metrics going forward (broad = encounter / world-consistency; hand =
   intentional reach) rather than collapsing to one. Keep broad-touch — it is a real
   unconditioned signal, not merely a confound.
3. If/when the goal is to *push* AB toward intentional reaching, make the task **require
   a hand**: use `hand_touched_ball*` for reward/termination, or move the balls **off the
   cart sweep line** so a body sweep cannot deliver them, forcing a lateral hand reach.
4. Finer size sampling (include 0.070/0.080) in future size batteries so a dead band
   cannot masquerade as a point anomaly.

---

## 2026-06-16 — Phase XVI R49: Auxiliary ball-position decode loss on the vision encoder (phase_xvi_R49_micoa_vision_auxdecode)

**What we ran:** A 250K-step training run byte-for-byte identical to Phase XIII R43 (MICOA + vision, cart substrate, static ball, random box ±0.08 m jitter, seed 42) with one addition: a small `Linear(64,2)` decode head attached to the vision encoder's latent (`mu_v`) that was trained with MSE loss (coefficient 1.0) to predict the ball's egocentric position [x_ego, y_ego] at every step. The decode head was wired to the MICOA encoder optimizer, so its gradient flowed directly into the vision encoder weights. The hypothesis was that this direct pressure would force the vision encoder to represent lateral ball direction — fixing the "representation failure" diagnosed in Phase XIII's reachable-band probe (mu_v lateral R² = 0.08, below the proprio control). Pre-registered success bar: lateral R² ≥ 0.30 on the reachable band (|x_ego| ≤ 0.20 m).

Checkpoints under `alien_baby/results/phase_xvi_R49_micoa_vision_auxdecode(_best)`. Branch: `exp/phase-xvi-R49-aux-decode`.

**Numbers:**

- ep_rew_mean: not separately tracked; eval behavior summarized below
- Steps completed: 250,000 (normal finish, no crashes)
- loco_speed_mean: N/A (cart substrate; AB cannot locomote under its own control — the policy cannot locomote, only the cart moves AB)

Ball-x decode probe (reachable band, |x_ego| ≤ 0.20 m):

| decode target | mu_v R² (vision) | mu_p R² (proprio control) |
|---|---|---|
| ball x_ego (LATERAL / direction) | **0.010** | 0.043 |
| ball y_ego (FORWARD / distance) | 0.157 | 0.360 |

Unrestricted probe: lateral 0.013, forward 0.072.

Eccentricity sweep (both-touched per bin, 0.00 → 0.25 m):

| ecc (m) | R49 both/20 | R43 reference (Phase XIII) |
|---|---|---|
| 0.00 | 20 | 20 |
| 0.05 | 19 | 19 |
| 0.10 | 19 | 16 |
| 0.15 |  8 | 11 |
| 0.20 |  4 |  4 |
| 0.25 |  0 |  0 |

Vision ablation abl_L2 by eccentricity bin:

| ecc (m) | R49 abl_L2 | R43 abl_L2 (Phase XIII) |
|---|---|---|
| 0.00 | 1.29 | 1.04 |
| 0.05 | ~1.95–2.04 | 0.62 |
| 0.10 | ~1.95–2.04 | 0.65 |
| 0.15 | ~1.95–2.04 | 0.67 |
| 0.20 | ~1.95–2.04 | 0.69 |
| 0.25 | ~1.95–2.04 | 0.72 |

MICOA training diagnostics at end of run:

| Diagnostic | R49 final | R43 reference | R41 (catastrophe) | Healthy range |
|---|---|---|---|---|
| kl_pred_k1 | 244 | low (static balls kept it healthy) | 638 | 0.5–2 |
| kl_agreement | 148 | low | 953 | low |
| sigma_combined | 0.117 | healthy (~0.13+) | ~0.10 | > 0.10 |
| aux_ball_decode loss | 0.20 (fell to ~0.10 early, drifted back up) | N/A | N/A | — |

**What we learned:**

The representation-fix hypothesis is refuted. Adding a direct auxiliary loss telling the encoder "encode where the ball is" did not make the encoder encode where the ball is — at least not the lateral (left/right) component that matters for direction. Lateral R² ended at 0.010, which is at chance, below the proprio control's 0.043, and well below both the 0.30 success bar and the R43 ~0.08 baseline. The auxiliary loss caught something: forward/distance R² reached 0.157 (vs R43 reachable-band 0.080 for forward). But the forward dimension was already weakly represented in R43, and the lateral dimension is the one needed for directional steering — that result is essentially zero. The most likely reason the loss failed on lateral: the ball is usually near the lateral center of the workspace (the ±0.08 m jitter is symmetric around zero), so the decode head minimized MSE most cheaply by predicting x_ego ≈ 0 everywhere. A head that ignores lateral position entirely still achieves low average MSE on a symmetric distribution. The encoder learned to satisfy the loss without ever having to represent signed left/right direction.

The encoder also destabilized. By the end of training kl_pred_k1 had reached 244 — well above a healthy 0–2, more than double R38's flagged ceiling of ~116, and on a trajectory toward the R41 catastrophe (638). sigma_combined at 0.117 is near the 0.10 collapse threshold. The aux decode loss and the predictive-KL term appear to have fought each other: the decode head pulled the encoder toward representing (noisy) ball position, while the predictive-KL pulled it toward representing proprio(t+1). Both gradients flowed into the same encoder weights, and the conflict destabilized the latent distribution in the second half of training. The aux_ball_decode loss falling to ~0.10 early and then drifting back up to 0.20 by the run's end is a direct symptom of this: the encoder briefly satisfied the decode target, then the competing predictive-KL gradient pushed it away.

The ablation profile flipped relative to R43. In R43, ablation peaked at ecc=0 (1.04) where directional information is least needed, and was flat-lower off-center (0.62–0.72) — the "content-free" pattern. In R49, ablation is lower at ecc=0 (1.29 is lower than R43's off-center if you account for scale) but high-flat off-center (~1.95–2.04). The eval script automatically flags this rising profile as "vision recruited at the margin." This interpretation is almost certainly wrong and should not be recorded as a positive finding. The ablation is ~2–3× higher than R43 across the board, coinciding with the kl_pred_k1 explosion. This is the same "high ablation ≠ useful vision" pattern confirmed in R41 (ablation 0.85, kl_pred_k1 = 638, task regressed). A destabilized encoder generates large, noisy action changes when pixels are zeroed — not because vision encodes anything useful, but because its latent is thrashing and its removal shifts the combined Gaussian in an arbitrary direction. The flat-high off-center profile is high-noise-floor, not directed recruitment. Task outcomes did not improve, which is the definitive check.

**Honest caveats:**

The ablation-profile flip (ecc=0 lower, off-center higher) is genuinely interesting as a pattern, even if the confounded mechanism makes it uninterpretable here. A future run with the same architectural change but without a competing predictive-KL term — or with a decoder trained on a non-symmetric ball distribution that actually creates lateral gradient pressure — might reveal whether the spatial profile of ablation sensitivity is changeable, and what changes it. This result says "we couldn't do it this way," not "this direction is closed."

The aux_ball_decode loss falling early to ~0.10 before drifting back up is worth noting as a failure mode in itself. The encoder briefly found a configuration that satisfied the decode constraint and lost it. Future experiments using auxiliary decode heads should consider decoupling the decode optimizer from the main encoder optimizer (using a lower learning rate or a separate parameter group) to reduce gradient conflict.

**Is vision load-bearing?** Not yet confirmed — and specifically, the representation-fix attempt failed. Ablation abl_L2 is high (1.29–2.04, above the 0.05 threshold), meaning pixels do influence actions. But this influence is confounded by the kl_pred_k1 explosion; high ablation under MICOA encoder pathology has been shown before (R41) to reflect noise, not useful signal. The probe directly refutes directional encoding: lateral R² = 0.010 is at chance. Vision is wired in but pointing nowhere.

**Theory signals:** The Behavioral Prediction Framework predicts that useful internal predictive structure should produce coherent, task-calibrated predictions. The auxiliary decode head briefly produced predictions that satisfied the MSE target (loss ~0.10 early) but could not hold that configuration against the competing predictive-KL gradient — the two loss terms were optimizing inconsistent internal representations. This is consistent with the framework's concern that a system under competing gradient pressures will not form stable predictive structure for either target. The Pattern Learning Framework predicts that the encoder should converge to a sparse, stable code under consistent training pressure. The drift of aux_ball_decode from 0.10 back up to 0.20, combined with sigma_combined near-collapse, is direct evidence against stable pattern formation: the representation is actively destabilizing in the second half of training, which is the opposite of sparse stable patterns locking in.

**The affordance/winnability reframe (important — record this):**

Watching the R49 renders forced a more fundamental diagnosis. On the cart substrate, AB is physically a passenger — its policy cannot steer the cart, and the cart carries AB's body along a fixed sweep line. The env docstring states this explicitly ("The policy cannot locomote; only the cart moves AB"). This means that even a perfectly directional vision encoder — lateral R² = 1.0 — would have no directional action to serve. AB cannot turn left or right. The encoder can know where the ball is, and the knowledge goes nowhere.

This reframes what we have been trying to do. The sequence of vision-encoder interventions (R43, Phase XIII probes, R49) has been trying to fix the encoder while the downstream action space provides no directional degree of freedom. The correct fix is upstream: give AB the ability to locomote laterally. Scouting found that AB's locomotion likely failed for a configuration reason — the raw-torque action space makes locomotion hard to discover; "A Walk in the Park" (Smith et al. 2022) shows that position-offset (target-angle) control is make-or-break for quadruped locomotion. The next experiment should be a locomotion smoke test under position-control action space, paired with a reverse-curriculum or start-state winnability condition that guarantees early contacts happen. Vision cannot become directionally load-bearing until there is a directional action to steer.

**Next question:** Can AB locomote under a position-offset (target-angle) action space — the control scheme "A Walk in the Park" identifies as essential — and does that locomotion produce directional contacts that give vision a job to do?


---

## Phase XVI — Position-offset action space (posoffset smoke series, 2026-06-17)

**Hypothesis** (crawl-reuse scout): AB's chronic locomotion failure (one-move-then-freeze, Phases I–VI) is caused by its **raw-torque action space**; *A Walk in the Park* shows a constrained **position-offset** action space is make-or-break. Verified precondition: `mimo_crawler.xml` = 26 torque `<motor>`, 0 position servos.

**Build** (commit 6a838f6, additive): new `mimo_crawler_pos.xml` (limbs → `<position>` servos, kp≈10×gear, kv=10, offset clamps shoulders/hips/elbows 0.4 rad, trunk 0.2 rad; head left torque) + `--action-mode {torque,position_offset}`. Required integrator change **Euler→implicitfast** (Euler exploded to ~8e5 N with stiff PD vs rigid contact). Preflight render: body holds a stable prone pose and limbs move smoothly under random actions (so the body *can* move).

**Smoke tests** (free body, DroQ/UTD=4, 60K):
- `smoke`/`smoke2`: INVALID — spawn ≤0.65 m → 100% step-0 contact (prone body sprawls ~0.68 m; ball spawns at radius r from world origin on a 2×2 m platform). No locomotion required.
- `smoke3` (spawn 0.70–0.90 m, probe-verified 0% step-0 contact): VALID. **ep_rew_mean dead-flat at −18.7** (pure step cost) across all 60K; ep_len_mean=400 (every episode times out, 0 contact). **Video (both seeds): seed0 fully motionless; seed1 collapses in <1 s then inert. FREEZE ATTRACTOR UNBROKEN.**

**Conclusion:** position-offset control fixed the numerical explosion and gives a stable, movable body, but did NOT by itself produce crawling in 60K. Since the body *can* move (preflight), this is a **learning/exploration collapse to the do-nothing optimum**, not physical impossibility — the same failure mode, now under position control. Torque-action-space is **necessary-but-not-sufficient**; crawling-from-prone remains open. Confounds not ruled out: 60K may be short (but dead-flat reward is a bad sign); prone default pose may not afford propulsion from small offsets; reward may not punish freezing hard enough; kp/offset untuned. **Decision deferred to human** (see NEXT_SESSION_2026_06_17.md) — did not auto-iterate reward/pose/length.

**Theory Monitor Note — 2026-06-16 (Phase XVI R49)**

> **Behavioral Prediction Framework: UNTESTABLE on the directional claim; CONSISTENT on stability** — R49's task outcomes are unchanged from R43 (ecc sweep nearly identical at every bin), and the aux decode loss produced briefly coherent forward-distance predictions (~0.10 early) that the competing predictive-KL term then undid (drifted back to 0.20). Consistent with the concern that a system under competing gradient pressures cannot form stable predictive structure. The core prediction about directional visual behavior remains untestable until AB has directional locomotion to steer.
>
> **Pattern Learning Framework: CHALLENGED** — aux_ball_decode rose ~0.10→0.20 in the second half, sigma_combined drifted toward the 0.10 collapse floor, kl_pred_k1 reached 244. The representation actively destabilized under competing objectives — the opposite of sparse stable patterns locking in.
>
> **The most important thing we don't know yet (at time of writing):** whether AB can locomote at all under a position-offset action space. If locomotion emerges under position-offset but not raw torque, the cart-era null results are explained structurally (no directional affordance → no directional gradient → no directional code).
>
> **[UPDATE 2026-06-17 — question answered]** posoffset_smoke3 (valid test, winnable spawn) ran: AB did NOT locomote under position-offset control either — dead-flat reward, video-confirmed freeze. Anti-freeze reward shaping (progress rewards 6–7×) also failed. Conclusion: from-scratch crawling is a motor-discovery problem, not fixable by action-space or reward alone. Pivoting to CART-STEER (give AB a movement primitive so it can pursue) to unblock the directional-vision tests this note calls for. Crawling parked as a known-hard track (imitation-from-demonstrations is the indicated tool).

## Phase XVI — RND curiosity + zero step-cost: the freeze attractor IS breakable (rnd_movefirst_60k, 2026-06-17)

**Hypothesis:** the freeze is an *exploration collapse to the do-nothing optimum*, not a hard motor-discovery wall. If so, removing the cost of existing (`--step-cost 0`) and paying an intrinsic novelty bonus (RND — Random Network Distillation, Burda et al. 2018) should make movement the optimal policy and break the freeze. This directly tests the open question the posoffset_smoke3 entry left ("learning/exploration collapse vs physical impossibility").

**Build** (branch `exp/rnd-liveness-gate`): new `rnd_wrapper.py` (`RNDRewardWrapper`, fixed random target net + trained predictor net over the obs vector; normalized prediction error added to train-env reward; eval env wrapped pass-through so VecNormalize sync matches). `mimo_crawler_env.STEP_COST` made configurable (`--step-cost`, default −0.05 unchanged). New **liveness gate** (`LivenessGateCallback`, governing rule "Realm of possibility" in CLAUDE.md): both envs expose `body_motion` (joint speed + body/cart translation); at `--liveness-gate-step` a run below `--liveness-min-motion` is declared VOID, stopped, and marked — no conclusions drawn from a frozen run.

**Run:** free body, position_offset, `--rnd --rnd-coef 1.0 --step-cost 0.0 --velocity-bonus-scale 0.0 --spawn-radius 0.18 0.35`, 60K, 16 envs, seed 0.

**Result — the freeze attractor is BROKEN.**
- Liveness gate **PASS at 10K** (mean body_motion 0.722 vs 0.05 threshold). Untrained preflight already showed 0.73 random-policy motion; trained seed2 video confirms *sustained* limb/leg/head repositioning across 300 steps — not the R50/posoffset_smoke3 squat-and-hold corpse. This is the first crawler run in the project's history where the body keeps moving rather than collapsing to stillness.
- Eval mean_reward rose 100 → 150 (4× "New best"); mean_ep_length fell 242 → 211 → 153.

**Caveat — the movement is UNDIRECTED; the reward is inflated by spawn-adjacency gimmes.**
- The 0.18 m spawn floor is too close: a prone body sprawls ~0.68 m, so at 0.18 m the ball spawns *inside the body footprint*. Trained renders: seed0 TOUCHED in 4 steps, seed1 in 3 steps — both **spawn-adjacency artifacts** (first frame shows the ball against the creature's shoulder; the broad any-geom contact metric fires immediately), NOT learned reaching. The eval reward (≈150, near the 200 contact bonus) is largely these gimmes.
- seed2 is the honest test (ball a real ~0.3 m away): the body moves and repositions continuously but **does not translate toward the ball** → TIMEOUT at 300 steps. It moves; it does not *pursue*.

**Conclusion:** RND + zero step-cost **resolves the open question** from posoffset_smoke3 — the freeze was an exploration/reward collapse, not physical impossibility. Curiosity makes movement the optimal policy and the freeze does not survive. **This updates the prior "from-scratch crawling is unfixable by reward alone" conclusion: undirected movement IS reward-fixable.** What remains unsolved is *directed* locomotion: RND rewards novelty, not approach, so the emergent motion is goal-agnostic. The task reward must now take over to turn motion into pursuit — which the too-close spawn never gave it a fair chance to do.

**Confounds / limits:** single seed trained (3 rendered); 0.18 m spawn floor produced gimme contacts (fix: raise to ~0.30–0.50 m so every touch is real); 60K is short for directed crawling to emerge; broad-contact (not hand) metric; `body_motion` includes in-place flailing, so "alive" ≠ "translating".

**Next:** directed-reach run — spawn floor ~0.30–0.50 m (kills gimmes), keep RND, extend to ~250K so approach/contact reward can shape pursuit now that the body is no longer frozen; optionally an outward curriculum to shape reaching.

## Phase XVI — Directed-reach test: movement ≠ locomotion (rnd_directed_250k, 2026-06-17)

**Setup:** the follow-up the 60K entry called for. Free body, position_offset, `--rnd --rnd-coef 0.1` (lowered so curiosity no longer drowns the task gradient — at coef 1.0 the per-episode novelty bonus ≈400 swamps approach ≈1.6), `--approach-reward-scale 10.0` (give pursuit a real gradient), `--step-cost 0.0`, `--spawn-radius 0.70 0.90` (winnability pre-checked: 0/20 step-0 contacts, start dist 0.71–0.90 m, on-platform — no gimmes), 250K, 16 envs, seed 0.

**Result — freeze stays broken, but NO directed locomotion.**
- Liveness gate **PASS at 10K** (body_motion 0.711) — sustained motion across 250K, no re-freeze.
- **Zero contacts.** Eval ep_length = 600 on all 25 evals (every episode times out); eval mean_reward hovers at ~0 (range −0.07 to +0.24 ≈ 10 × net metres closed ≈ noise around zero). The creature ends each episode about as far from the ball as it started.
- **Video (3 seeds, all TIMEOUT):** the body twists, splays its legs, and reorients continuously, but the **torso/CoM does not translate** toward the ball. It flails and turns *in place*; it never crawls across the floor. Start-vs-late frames: same hip location, ball still ~0.78 m away.

**Conclusion — the bottleneck is translation (a gait), not movement.** RND + zero step-cost reliably produces *movement*, but movement here is in-place flailing/reorienting, which is a local optimum that satisfies RND's novelty appetite **without propulsion**. Because the torso never translates, the approach reward (even at ×10) never receives a positive sample to reinforce, so no pursuit gradient ever forms. This confirms the posoffset_smoke3 diagnosis at a finer grain: **crawling-from-prone is a motor-discovery problem for the *translation* primitive specifically** — curiosity and approach-shaping are insufficient because the gait is never sampled. The corrected slogan: *movement is reward-fixable; locomotion is not (it needs the gait discovered or supplied).*

**Implications for next step (decision for human):** the indicated tools are the ones that supply or force *translation*, not more reward-shaping on a body that flails in place:
1. **Imitation / demonstration** (smoke3's indicated tool): seed the policy with a crawling demo so the gait is in the buffer, then let RL refine. Highest-leverage but most setup.
2. **Cart-steer pivot** (prior plan): give AB a movement primitive (commanded cart velocity) so pursuit is *possible by construction*, unblocking the directional-vision tests — at the cost of not solving crawling.
3. **Propulsion affordance**: the position-offset clamps (shoulders/hips 0.4 rad) + prone default pose may not permit net propulsion at all; widen offsets / change default pose / add a directed hip-translation velocity bonus and re-test whether translation is even achievable before investing in imitation.

**Limits:** single seed trained; 250K (longer might eventually sample a gait, but ~0 reward trend across 250K is a strong negative); approach reward rewards distance-closed, which can't fire without translation (a directed *velocity*-toward-target bonus that rewards even momentary closing speed was not tried and is the cheapest remaining shaping lever before pivoting).


---

## 2026-07-01 — rnd_propulsion_400k: propulsion-affordance test — can the body translate its CoM at all?

**Hypothesis (Option-3 affordance test from rnd_directed_250k):** Three explanations were open for why rnd_directed_250k produced in-place flailing with zero net CoM translation: (1) gait is learnable but needs more time or imitation seeding; (2) cart-steer pivot is the right path; (3) the position-offset clamps (0.4 rad shoulders/hips) and prone default pose physically block propulsion — the body cannot produce net CoM translation under any reward. This run tests Option 3: pay explicitly for ANY horizontal CoM speed via velocity_bonus_scale=2.0, across 400K steps with RND curiosity and no step-cost. If the body can translate, the velocity_bonus should find it and eval reward should grow monotonically. If it cannot translate, eval reward will reflect only in-place CoM oscillation.

**Setup:** Free body (no cart), position_offset action mode, `--rnd --rnd-coef 0.1`, `--step-cost 0.0`, `--velocity-bonus-scale 2.0` (reward any horizontal CoM speed), `--approach-reward-scale 10.0`, `--spawn-radius 0.70 0.80` (0% step-0 contacts), 400K steps, 16 envs, seed 0. Per-step reward = `2.0 × |v_CoM_horizontal|  +  10.0 × (prev_dist − curr_dist)  +  0.1 × RND_intrinsic`. The velocity_bonus uses `norm(qvel[root_x], qvel[root_y])` — always positive, fires equally on symmetric oscillation (rocking) as on forward translation. It cannot distinguish the two.

**Numbers:**

- ep_rew_mean (rollout, VecNorm + RND): 84.8 → 101 (training env; inflated by RND intrinsic and VecNormalize scaling — not comparable to eval)
- eval mean_reward: 6.82 (10K) → **peak 64.46 ± 22.44 (310K)** → 11.14 (400K final)
- loco_speed_mean: N/A (metric not separately logged; implied from eval reward below)
- touch_rate: **0/N across all 40 evals** — ep_len_mean = 600.00 on every eval
- Steps completed: 400,000 (normal finish, no crashes, no errors)
- Liveness gate: **PASS at 10K** (body_motion = 0.711 ≥ 0.05 threshold)

**Eval reward trajectory — key waypoints:**

| Steps | Eval mean_reward | Note |
|---|---|---|
| 10K | 6.82 | first eval, "New best" |
| 110K | 11.58 | local peak |
| 250K | 23.06 | gradual build |
| 290K | 36.40 | acceleration phase |
| **310K** | **64.46 ± 22.44** | **all-time peak** |
| 320K | 59.28 | still elevated |
| 330K | 14.27 | steep drop |
| 350K | 7.72 | near noise floor |
| 400K | 11.14 | final (near 100K baseline) |

**Implied mean torso speed at peak:** 64.46 / 600 steps / 2.0 = **≈ 0.054 m/s average horizontal CoM speed**. This is the magnitude of the instantaneous CoM velocity vector — it is positive regardless of direction. A body rocking left-right at 0.054 m/s average speed (zero net displacement) earns exactly the same reward as one translating forward at 0.054 m/s.

**Video evidence (mandatory per CLAUDE.md):**

Rendered 3 episodes from `alien_baby/results/rnd_propulsion_400k_best/best_model.zip` (saved at ~310K peak). Script: `alien_baby/visualization/render_crawler.py`, `--action-mode position_offset --spawn-radius 0.70 0.80`. Videos at `alien_baby/results/videos/crawler_rnd_propulsion_400k_seed{0,1,2}.mp4`. All three episodes: **TIMEOUT (600 steps, 0 contacts)**.

- **Seed 0:** RINGSIDE view shows creature lying on its side, rolling to its back. The red ball sits near its head throughout and does NOT move closer over the episode. Creature ends lying on its back, ball unmoved from spawn position. No locomotion across the floor.
- **Seed 1:** Creature starts upright/hunched, then slowly collapses to its right side — limbs splay out and it lies flat. The ball is described as "completely stationary throughout the entire video in both perspectives." The collapse is the entirety of the creature's behavior; it then lies inert. No approach, no CoM displacement across the floor.
- **Seed 2:** Gemini description failed (API timeout; Claude fallback also failed). Two seeds with consistent collapse-and-lie behavior provides sufficient qualitative evidence.

**Start-vs-end CoM position (qualitative from video):** In both described seeds, the ball did not move and the creature's torso ended in essentially the same floor region it occupied when first making contact with the surface (seed 0 fell from side to back; seed 1 fell from standing to prone). The hip/CoM did not cross the floor from start to a clearly different destination. This is not locomotion; it is postural collapse.

**Peak-then-regression — what it tells us:**

The 310K peak (64.46, std ±22.44) followed by immediate collapse (330K: 14.27, final: 11.14) has the signature of a **fragile transient oscillation, not a stable gait**:

1. High std at the peak (±22.44 = 35% of mean) means some episodes scored very high and others much lower — the behavior was not locked in.
2. A stable translating gait would show monotonically rising eval reward. Instead, the reward returned to baseline within 30K steps of the peak.
3. The most likely mechanism: around 300K steps the policy settled temporarily into a motor pattern producing rapid in-place CoM rocking (higher oscillation → higher velocity_bonus), then SAC gradient updates (entropy declining, RND novelty saturating) moved the policy out of that narrow configuration.
4. The acceleration phase (250K→290K→310K) does not indicate a gait being discovered — it indicates the policy increasingly specializing in fast rocking, which then got refined away.

If a stable translating gait had been discovered at 310K, subsequent training should have reinforced it further rather than erasing it within 20K steps.

**Translation verdict: (B) TRANSLATION NOT ACHIEVABLE under current configuration.**

Zero ball contacts over 400K steps. Zero approach reward in any sustained sense. Eval reward peaks attributable to transient CoM oscillation, not net displacement. Video confirms no floor-crossing behavior. Option 3 from the rnd_directed_250k entry is now tested and closed: paying with velocity_bonus_scale=2.0 for any CoM speed did not unlock propulsion. Under the current constraints (0.4 rad clamps, prone default pose, position_offset servos), the body cannot reliably produce the net ground reaction forces needed to translate itself across the floor.

**Comparison to prior runs:**

| Run | Eval mean_reward | Contacts | Net CoM translation? |
|---|---|---|---|
| rnd_movefirst_60k | ~150 (gimme contacts) | Yes (spawn inside body footprint) | Not honestly tested |
| rnd_directed_250k | ≈ 0 (no velocity bonus) | 0 | No |
| **rnd_propulsion_400k** | **Peak 64.46, final 11.14** | **0** | **No** |

rnd_directed_250k had near-zero eval reward because there was no velocity_bonus and approach was zero (no translation). rnd_propulsion_400k has higher eval reward because the velocity_bonus fires on oscillation. The underlying body behavior is identical in both: no floor-crossing locomotion.

**Honest caveats:**

- Single seed (seed 0). The parallel seed-1 run of the identical config was described as running at task submission time; this entry should be updated when that result arrives.
- "Translation not achievable" means under the CURRENT config (0.4 rad clamps, prone default pose, 400K). Wider clamps (e.g. 0.8 rad) or a different default pose may unlock propulsion — that is untested.
- The velocity_bonus rewards |v_CoM|, not v_CoM toward the ball. A directed "velocity-toward-target" bonus might produce stronger directed gradient, but if the body cannot generate sustained net horizontal force in any direction, no scalar of that reward fixes the physics.
- 400K SAC steps is moderate by locomotion literature standards, but zero ball contacts across all 40 evals (zero positive approach samples ever in the buffer) is a very strong negative signal. RL cannot reinforce what never appears.

**Is vision load-bearing?** N/A — this is a locomotion Phase A run with no visual input. Vision cannot become load-bearing until the body can produce directed translation. The previous diagnosis stands: the correct fix is upstream (locomotion primitive) not downstream (encoder).

**Theory signals:** The Behavioral Prediction Framework predicts that useful internal predictive structure emerges from actions that produce predictable changes in the world. For a body that cannot produce net floor translation, every action's predicted next state is essentially the same local posture space — the CoM stays where it is regardless of joint commands. The system may converge on a correct-but-useless internal model: "actions do not move me across the floor." The Pattern Learning Framework predicts that the encoder should lock in sparse stable codes for recurring patterns. The peak-then-regression in eval reward suggests the policy briefly settled on a fast-rocking pattern but could not stabilize it — consistent with a pattern that was explored but not sufficiently reinforced before the SAC update moved away from it.

**Next question:** Is CoM translation physically impossible under these clamp constraints (0.4 rad, prone pose), or merely undiscoverable by RL from random initialization? A five-minute test — hand-coding a diagonal hip/shoulder motor sequence in a preflight render, checking whether a single crawl step produces net hip displacement — would distinguish between "physics blocks it" and "RL cannot discover it," and would determine whether widening the clamps is the right fix before investing in imitation-seeding.

## Phase XVI — Affordance diagnostic: translation is PHYSICALLY BLOCKED, not undiscovered (hand_drive_crawler, 2026-07-01)

**Why this test:** rnd_propulsion_400k returned Verdict B (no net CoM translation despite paying velocity_bonus×2.0 for any CoM speed; reproduced across seed 0 and seed 1). Verdict B could not by itself separate two causes: (A) the crawl gait is physically possible under the 0.4 rad position-offset clamps but RL never sampled it (→ fix = imitation seeding), vs (B) the clamps + prone default pose physically block propulsion (→ fix = widen clamps / change pose; imitation would ALSO fail). This is the cheap open-loop measurement (no RL, no learning, no env/model/training changes) that disambiguates them, per the theory-monitor's recommendation.

**Method** (new additive diagnostic `alien_baby/visualization/hand_drive_crawler.py`): bypass the policy and hand-drive `MimoCrawlerEnv` (action_mode=position_offset, spawn 0.70–0.80) with scripted position-offset targets at **full amplitude (a=±1.0 → the clamp extremes)** in four sensible gaits — synchronous limb paddle, alternating/diagonal crawl (contralateral antiphase), belly-crawl (arms pull, legs push, quarter-cycle phased), and full-amplitude static hold. The body is settled into its prone equilibrium first (1 s) so the spawn-drop is not miscounted. Net horizontal displacement of the root/CoM measured over 8 s per pattern. Grounding verified: after settle the body has **11 solid contacts with the platform** (head, both hands, both upper legs, torso) — this is a grounded body with purchase, not a floating one.

**Result — the body cannot translate even when hand-driven at maximum amplitude:**

| Pattern | Net CoM |Δxy| | Max excursion |
|---|---|---|
| synchronous | 0.027 m | 0.027 m |
| alternating (diagonal crawl) | **0.039 m** | 0.039 m |
| belly_crawl | 0.037 m | 0.037 m |
| hold (static push) | 0.004 m | 0.006 m |

Best net translation = **0.039 m**, far below the 0.05 m bar and nowhere near a real crawl stride (~0.1–0.3 m). Video (overhead panel): the body bobs/bends in place — cyclic knee-flexion — with no translation toward the ball. (The ringside oblique camera led the auto-describer to perceive a faint "slide down an incline," but the measured root xy is start≈end, i.e. in-place motion; there is no incline in the flat platform.)

**Verdict: (B) TRANSLATION PHYSICALLY BLOCKED — cause B, not cause A.** The bottleneck is range-of-motion: the position-offset clamps are tiny — `hip_bend ±0.2 rad (≈±11°)`, `chest_lean ±0.157 rad`, shoulders `±0.4 rad (≈±23°)`, hip flex asymmetric ≈ (−0.57, +0.23) rad. A crawl stride needs far larger joint excursions than these clamps permit, so no controller — learned or hand-coded — can produce net propulsion within them. This is why three RL runs (rnd_movefirst_60k, rnd_directed_250k, rnd_propulsion_400k) all produced movement-without-locomotion: not a discovery failure, an affordance failure.

**Implication (decision for human):** the indicated fix is to **widen the offset clamps (e.g. shoulders/hips toward ~0.8 rad, hip_bend/chest beyond ±0.2) and/or change the prone default pose**, then re-run this same hand-drive test to confirm translation becomes achievable before spending any RL compute. **Imitation-from-demonstration is NOT indicated yet** — it would fail for the same reason RL did (the body cannot execute the stride within the current clamps). Widen the affordance first; discovery is only worth solving once the gait is physically possible.

**Caveats:** the four hand-chosen gaits are not exhaustive — a cleverer open-loop pattern might do marginally better, but all four sensible gaits clustering at 0.03–0.04 m (and static hold at ~0) is strong evidence the ceiling is low. Single seed (the physics is deterministic given the pose, so seed matters little here). CoM metric is the root free-joint xy. The clamp values above are read from `mimo_crawler_pos.xml`; widening them is a one-line-per-actuator XML change but should be paired with a re-tuned kp and an integrator/stability check (the clamps were originally tightened for numerical stability).

**Addendum — widened-clamp verification (same day):** to de-risk the recommended fix, a throwaway diagnostic body variant `mimo_crawler_pos_wide.xml` was created (23 limb/spine actuators widened: shoulders ±0.4→±0.9, hip_flex −0.57/+0.23→−1.1/+0.6, knees −0.92/−0.12→−1.6/+0.1, hip_bend ±0.2→±0.45, etc.; original body untouched, no training) and hand-driven with the same four gaits. Result: **numerically STABLE** (root z steady ~2.09, no explosion — so widening with the existing kp is safe under implicitfast), and best net CoM translation rose to **0.092 m** (belly_crawl) vs 0.039 m narrow — a ~2.4× gain that crosses the 0.05 m "achievable" bar. **But** the other three gaits barely moved (0.025–0.032 m) and the belly_crawl video shows a slow, awkward contorting drag "struggling to move from prone," not a clean crawl. **Refined conclusion:** range-of-motion is a *genuine contributing* cause (cause B is partly real — widening helps and is stable), but widening the clamps *alone* is **necessary-but-likely-not-sufficient** for confident locomotion; the prone default pose (and, on top of a now-adequate ROM, gait discovery — cause A) still matter. Indicated next step for the human: (1) adopt a widened-clamp body (verified stable), (2) also reconsider the prone default pose toward a crawl-ready posture, then (3) re-run this hand-drive test to confirm a *clean* >0.15 m hand-driven crawl exists before committing RL/imitation compute. Widening values here are a first pass, not tuned; hand gaits are not optimized (RL may exceed them once ROM permits).


---

## 2026-07-02 — crawl_minimal_400k: first non-zero contact rate in any free-body crawler run

**Hypothesis:** The previous affordance chain established that (1) the original clamps blocked translation physically (hand-drive best = 0.039 m), (2) widening the clamps to `mimo_crawler_pos_wide.xml` raised the ceiling to 0.092 m (marginal), and (3) a pose search found the "arms_fwd" commando-crawl pose (both shoulders forward, elbows slightly bent) gives 0.225 m axial belly-down translation — the first hand-drive result crossing the 0.15 m bar. This run tests the minimal-change RL hypothesis: given a body that CAN physically translate (wide XML + arms_fwd pose), will RL discover locomotion toward the ball? The unsigned velocity bonus was zeroed (previously gameable by rocking); approach-toward-ball reward was kept at ×10; tip-over termination (50° tilt, −5 penalty) was added to block the roll-onto-side exploit that prior runs used to avoid the upright stability problem.

**Setup:** `python alien_baby/crawler/train_crawler.py --action-mode position_offset --xml-path alien_baby/crawler/mimo_crawler_pos_wide.xml --crawl-pose arms_fwd --velocity-bonus-scale 0.0 --approach-reward-scale 10.0 --step-cost 0.0 --terminate-tilt-deg 50 --tip-penalty -5 --spawn-radius 0.70 0.80 --rnd --rnd-coef 0.1 --steps 400000 --n-envs 16 --seed 0 --run-tag crawl_minimal_400k`. Per-step reward = `10.0 × (prev_dist − curr_dist) + 0.1 × RND_intrinsic`. No unsigned velocity term. No step cost.

**Numbers:**

| Metric | Value |
|---|---|
| Eval reward (10K, pure task) | 0.03 ± 0.14 |
| Best eval reward (370K) | **41.12 ± 80.87** — new high-water mark; best_model saved here |
| Final eval reward (400K) | 20.29 ± 60.73 |
| Final eval ep_length | 553.85 ± 134.51 |
| ep_rew_mean (rollout, VecNorm + RND) | 98.4 → 71.6 (includes intrinsic; not comparable to eval) |
| Steps completed | 400,000 (normal finish, no errors) |
| Liveness gate | **PASS** at 10K (body_motion = 1.277 ≥ 0.05) |
| Deterministic eval: contacts (30 eps) | **5/30 = 16.7%** — first non-zero contact rate in project history for a free-body crawler |
| Deterministic eval: tip-terminated | **0/30 = 0%** — tip-termination killed the tipping exploit entirely |
| Deterministic eval: timed out | 25/30 = 83.3% |
| Mean start distance to ball | 0.728 m |
| Mean end distance to ball | 0.728 m |
| Mean distance change (+ = closer) | **+0.000 m** |
| Mean CoM displacement, all episodes | 0.298 m |
| Mean CoM displacement, touch episodes (n=5) | 0.379 m |
| Mean CoM displacement, no-touch episodes (n=25) | 0.282 m |
| Correlation (CoM displacement, contact) | +0.258 |

**Eval trajectory (eval rewards, key waypoints):**

| Steps | Eval reward | Note |
|---|---|---|
| 10K | 0.03 | first eval; New best |
| 90K | 0.09 | New best; still no contacts |
| 210K | **20.01 ± 60.45** | **first contact ever in eval**; New best |
| 220K–360K | −0.70 to +0.42 | back to near-zero; contacts disappeared again |
| 370K | **41.12 ± 80.87** | **highest eval reward in project history**; New best; best_model saved |
| 380K | 40.69 ± 81.76 | maintained |
| 400K | 20.29 ± 60.73 | final |

Contacts arrive in bursts at 210K and 370–380K rather than monotonically. The high std (80.87) at the peak means most episodes still timeout but a small fraction achieve large positive rewards (200 per contact). Expected reward from 5/30 contacts = ~33 + approach reward ≈ consistent with 41.12 peak.

**Is it crawling? (displacement/contact table, 30 deterministic episodes):**

| Episode | Outcome | Start dist | End dist | Dist change | CoM disp |
|---|---|---|---|---|---|
| ep 8 | TOUCH | 0.704 | 0.312 | +0.391 | 0.415 |
| ep 9 | TOUCH | 0.772 | 0.139 | +0.633 | 0.646 |
| ep 15 | TOUCH | 0.737 | 0.399 | +0.338 | 0.348 |
| ep 24 | TOUCH | 0.710 | 0.514 | +0.196 | 0.229 |
| ep 29 | TOUCH | 0.695 | 0.531 | +0.164 | 0.259 |
| Mean (touch) | — | 0.724 | 0.379 | +0.344 | 0.379 |
| ep 23 | TIMEOUT | 0.767 | 1.454 | −0.687 | 0.718 |
| ep 17 | TIMEOUT | 0.757 | 1.133 | −0.375 | 0.535 |
| Mean (no-touch) | — | 0.728 | 0.750 | −0.022 | 0.282 |
| **Mean (all 30)** | — | **0.728** | **0.728** | **+0.000** | **0.298** |

Key observation: mean dist change = 0.000 m is the clearest possible readout of undirected movement. The body moves (0.298 m average), and when the random movement carries it toward the ball, contact happens. But there is no consistent steering. Note also that ep23 (no-touch) has a CoM displacement of 0.718 m — larger than any touch episode — because the body moved far but in the wrong direction. RL has discovered locomotion but not direction.

The RL policy EXCEEDS the hand-drive affordance ceiling (0.225 m axial) in several episodes, confirming that joint coordination discovered by RL is richer than any of our hand-scripted gaits.

**Video evidence** (`alien_baby/results/videos/crawler_crawl_minimal_400k_seed{0,1,2}.mp4` and `_touch_ep8.mp4`, `_touch_ep9.mp4`):

- **Seed 0 (timeout, dist_change +0.159 m):** Gemini describes the body rotating counter-clockwise around its axis near the ball in the overhead view. Body drifts 0.159 m closer but never contacts. This is a spinning/rocking pattern, not a directed crawl.
- **Seed 1 (timeout, dist_change −0.220 m):** Gemini describes the body falling backward, then rolling chaotically, ending near the ball but without contact. "Movement characterized by an initial, somewhat controlled lean that quickly transitions into an uncontrolled fall and subsequent rolling." Body ends up near the ball through a chaotic path, but distances itself 0.220 m net by episode end.
- **Touch ep8, seed 56 (contact at step 223, dist_change +0.391 m):** Gemini confirms a belly-down, directed slide. "The agent deliberately changes its pose by falling/lying down to become level with the target, and then crawls towards it." Body falls forward from arms_fwd starting pose and slides/crawls across the floor. "Movement is fluid and appears goal-oriented towards the red ball." This is genuine directed body translation.
- **Touch ep9, seed 63 (contact at step 394, dist_change +0.633 m):** Gemini describes "an initial fall forward onto its side, landing flat, then slides/crawls body toward the red ball." Body travels 0.633 m closer to ball over 394 steps. Confirms real locomotion, not spawn-luck.

**Comparison to prior runs and hand-drive ceiling:**

| Run | Contacts | Net CoM translation | Note |
|---|---|---|---|
| rnd_movefirst_60k | Yes (gimme) | Not tested honestly | Spawn inside body footprint |
| rnd_directed_250k | 0 | Zero (flailing in place) | Old narrow clamps |
| rnd_propulsion_400k | 0 | Zero (oscillation) | Old narrow clamps, old pose |
| hand_drive belly_crawl (old clamps) | — | 0.039 m | Physically blocked |
| hand_drive belly_crawl (wide clamps) | — | 0.092 m | Marginal |
| hand_drive arms_fwd pose (wide clamps) | — | 0.225 m | First real affordance |
| **crawl_minimal_400k (RL, wide, arms_fwd)** | **5/30 = 16.7%** | **0.298 m avg, 0.646 m best** | **RL exceeds hand-drive ceiling** |

**Verdict: (B) PARTIAL.** Contact rate 16.7% is the first genuine non-zero contact rate in any free-body crawler run. Body CoM displacement (0.298 m average, 0.646 m best) exceeds the hand-drive affordance ceiling, confirming RL has discovered real locomotion primitives. The 0% tip-termination rate confirms the arms_fwd pose + tip-penalty has eliminated the tipping exploit. BUT: mean dist change = 0.000 m and correlation (displacement, contact) = +0.258 confirm that movement is undirected. Contacts come from random drift that happens to carry the body into the ball. The approach reward (×10) cannot teach steering because ball position is not in the proprio observation — the policy observes its own XY position but not the ball's, so it cannot infer which direction reduces the approach-reward gradient. What to fix: add ball-direction information to the proprio observation (`target_obs=True` or a temporary ball-xy vector), which gives the approach reward a signal the policy can act on to produce consistent homing.

**Is vision load-bearing?** N/A — this run has no vision (pure proprio). Vision cannot be evaluated until the body can steer toward a target. This run establishes the precondition: the body can now physically translate and make contact. Vision comes next.

**Theory signals:** The Behavioral Prediction Framework predicts useful internal models emerge when actions produce discriminable outcomes. Here, every direction of locomotion produces a *similar* approach-reward gradient in expectation (the ball is equally likely to be in any direction within the 180° spawn cone), so the prediction of "move this direction → ball closer" never becomes discriminable. The body learns to move but not where. This is consistent with the framework: prediction-of-consequence requires a distinguishable consequence, which requires ball-position information in the observation. The Pattern Learning Framework notes that the 0.298 m locomotion displacement emerging from RL — exceeding the 0.225 m hand-drive ceiling — suggests RL has found sparse stable patterns for joint coordination that are richer than hand-scripted gaits. The locomotion pattern is real; it lacks directional context.

**Honest caveats:**
- Single seed (seed 0). The contact bursts (210K, 370K) are not monotonic, so a different seed might discover locomotion earlier or later.
- The ep_len std (134.51 at 400K) and the burst-pattern in eval rewards suggest the locomotion behavior is not yet stable — the policy discovers contact-able trajectories inconsistently. This is early-stage SAC with a sparse reward: more steps may consolidate the behavior.
- The contact rate (16.7%) matches the expected rate from undirected random walk given the spawn geometry: ball in a 0.70–0.80 m ring, body travels ~0.3 m, probability of overlap is non-trivial. We cannot distinguish "RL discovered a directed gait" from "RL discovered locomotion and the ball fell within reach by chance" without seeing dist_change > 0 consistently. Right now mean dist_change = 0 makes it clear: chance, not direction.
- The best_model was saved at 370K. The final model (400K) shows slightly lower eval reward (20.29 vs 41.12), suggesting some regression in the last 30K steps.

**Next question:** Does adding ball position to the proprio observation (a 2-vector ball_xy or target_obs=True) convert the undirected locomotion into consistent ball-homing, producing dist_change reliably > 0 and contact rate > 50%?

> **Theory Monitor Note — 2026-07-02**
>
> Behavioral Prediction Framework: PARTIALLY CONFIRMED (locomotion) / UNTESTABLE (direction) — The 0.646 m best-episode CoM translation exceeding the 0.225 m hand-drive ceiling confirms a real internal locomotion coordination program has formed; but the mean dist_change of +0.000 m across 30 eval episodes shows no model of "move toward the ball" exists, structurally expected because ball position is absent from the observation — the creature cannot form a directional predictive model from a signal that does not include direction.
>
> Pattern Learning Framework: CHALLENGED (stability) / CONFIRMED (efficiency) — The burst-contact trajectory (contacts at 210K, absent 220K–360K, back at 370K, declining 380K→400K) and high variance at peak (std 80.87 = 197% of mean) are the opposite of a stably locked-in sparse code; but RL's best gait (0.646 m) exceeding the best hand-scripted gait (0.225 m) by 2.9× confirms gradient search finds richer sparse coordination than enumeration.
>
> The most important thing we don't know yet: Whether adding ball-direction to the proprio observation converts the 16.7% accidental-contact rate into reliable directed homing — the clean test of whether the remaining gap is purely informational or a deeper coordination problem. (Acted on: the crawl_targetobs_400k run adds exactly this signal.)

## Phase XVI — DIRECTED CRAWLING ACHIEVED: the informational fix (crawl_targetobs_400k, 2026-07-02)

**Milestone.** This is the first time in the project's history that the creature performs *directed locomotion* — it crawls, belly-down, steadily toward a target whose direction it is told. It resolves the R49 precondition (a directional action for vision to eventually serve).

**Hypothesis** (from the crawl_minimal_400k theory note): that run crawled but made only *random-walk* contact (mean toward-ball translation = 0.000 m) because ball position was absent from the observation — the approach reward had no directional gradient the policy could act on. Prediction: add ball-direction to the obs → directed homing emerges, because the locomotion primitive already exists.

**Change** (one thing): `target_obs=True` — append the ball-1 position in the BODY frame (3 numbers, heading-invariant) to the observation (69→72 dims). This is privileged target info (vision is meant to supply it later). Everything else identical to crawl_minimal_400k (wide body, arms_fwd pose, signed approach reward ×10, |velocity| term zeroed, tip-termination 50°/−5, RND 0.1, spawn 0.70–0.80, 400K, seed 0).

**Result — directed homing confirmed (30-episode deterministic eval):**

| Metric | crawl_minimal (no signal) | crawl_targetobs |
|---|---|---|
| Mean toward-ball translation (Δd) | **+0.000 m** (random walk) | **+0.195 m** (directed) |
| Per-episode Δd sign | random ± | **positive in ~24/30 episodes** |
| Mean start→end distance | ~unchanged | 0.728 → 0.533 m (closed 0.195 m) |
| Contacts | 16.7% (lucky) | 6.7% (approaches, times out short) |
| Timeouts | 83% | 87% |

- **Efficient episodes crawl straight at the ball:** ep7 (seed 49) closed 0.454 m with CoM displacement 0.464 m (≈ all displacement was ball-ward); ep14 (seed 98) closed 0.421 m / disp 0.437 m.
- **Per-step distance is a steady monotonic decline** (the crawl signature, not a fall): seed 49 `0.75→0.70→0.56→0.44→0.39→0.35→0.31→0.30`; seed 98 `0.70→0.61→0.48→0.41→0.36→0.30→0.27`. Tilt stays low throughout (seed 49 12–19°, seed 98 1–8°) — belly-down the entire episode, never tips. This is genuine crawling toward the target.

**Video-describer caveat (important):** the Gemini auto-describer reports these episodes as "the character falls backward, away from the target." This is FALSE and is a documented failure of the describer on the prone MIMo body from the overhead/ringside cameras (it hallucinates a standing figure that then falls). It is contradicted directly by the physics: the hip translates monotonically toward the ball over all 600 steps while tilt stays <20° (no tip-termination ever fires). The per-step distance log is the authoritative evidence; the describer is not usable for this body/camera and should not be relied on here.

**Why contacts DROPPED despite better behavior:** crawl_minimal got occasional *lucky* contacts by wandering (high CoM displacement, no direction). crawl_targetobs *deliberately approaches* but decelerates near the ball and runs out of the 600-step budget at ~0.27–0.35 m (it covers ~0.45 m of a ~0.7 m gap). So contact rate is a poor metric here; **mean toward-ball translation (+0.195 m, systematically positive) is the correct success signal, and it is unambiguous.**

**Verdict:** the crawl_minimal theory prediction is CONFIRMED — the remaining gap was purely informational. With the ball's direction observable, the pre-existing locomotion primitive becomes directed homing. **Directed crawling is solved.**

**Next (running): crawl_targetobs_long_600k** — same config with max_steps 1200 + 600K training, to let the deliberate approach complete into actual contacts (the efficient episodes close ~0.45 m in 600 steps, so ~1000–1200 steps should reach the ball). After that, the throughline reopens: replace the privileged ball-direction obs with **vision** — the core project test, now finally well-posed because a directional action exists to serve.

**Caveats:** single seed trained; privileged target obs (not vision — that's the point of the next phase); contact completion not yet demonstrated (the refinement run tests it); RND still on (may be unnecessary now that the signal is directional — an ablation worth running).

## Phase XVI — RELIABLE directed crawling via PPO (the MIMo-recipe escalation) (crawl_ppo_2M, 2026-07-02)

**Result.** Switching SAC→PPO (the algorithm MIMo's own working whole-body skill used, per the 2026-07-01 lit-scout) converted the *unstable* directed crawl into a *reliable* one. This is the night's culminating result: the creature crawls, belly-down, to a target it is given the direction to, and **reaches it in the majority of episodes.**

**Why PPO** (escalation trigger): the SAC target_obs runs proved directed crawling *emerges* (mean toward-ball +0.08…+0.20 m, 7–23% contacts) but never *stabilises* — every SAC run showed burst-to-40/70-then-regress, `best_model` caught a lucky peak, nothing locked in (Pattern-Learning "no stable lock-in"). No SAC knob (RND on/off, episode length) fixed it; the instability was algorithm-deep. The scout found MIMo's only working whole-body behaviour (supine→prone rolling) used **PPO + shaping**, not sparse-contact SAC. `train_crawler_ppo.py` reuses the identical crawl env/reward/pose/tip-termination/target_obs and swaps only the algorithm.

**Setup:** PPO (MlpPolicy [256,256], lr 3e-4, n_steps 1024, batch 512, 10 epochs, ent_coef 0), same env as crawl_targetobs: wide body, arms_fwd pose, signed approach reward ×10 (|velocity| term zeroed), tip-termination 50°/−5, target_obs (body-frame ball vector), spawn 0.70–0.80, max_steps 1000, 2M steps, 16 envs, seed 0. No RND.

**Result — stable AND better (40-episode deterministic eval, matched VecNormalize):**

| Metric | SAC best (any target_obs run) | PPO crawl_ppo_2M |
|---|---|---|
| Training curve | burst-to-70-then-regress; final eval ~0 | **smooth monotonic → plateau ~180** |
| Contact rate | 23.3% | **57.5%** (23/40) |
| Mean toward-ball Δd | +0.195 | **+0.257 m** |
| Mean end distance | ~0.53 m | **0.476 m** (from 0.733) |
| Tip-terminated | 0% | 7.5% |

- **Training stability** (the whole point): ep_rew_mean rose 97→112→139→164→179→187 then held ~170–187 — monotonic, no collapse. Eval mean_reward sustained ~120–183 (vs SAC's mostly-zero). This is the stable lock-in SAC never achieved.
- **Per-step trajectories confirm clean crawl-to-contact:** seed 0 `0.77→0.69→0.55→0.46→0.30→TOUCH@381`; seed 14 `0.72→0.66→0.57→0.50→TOUCH@553`. Monotonic distance decline to contact.
- **Failure mode of the ~35% timeouts:** approach-then-lose-it (seed 7 stalls at 0.50; seed 21 approaches to 0.54 then drifts back to 1.02) — NOT "never approaches." The homing behaviour is present in every episode; ~57% complete it.

**Verdict:** reliable directed crawling to a target is achieved. Escalating to the MIMo recipe (PPO) was the correct call — it fixed the SAC instability and roughly doubled the contact rate. From 0% directed contact at project start to 57.5%.

**The project throughline is now fully unblocked.** The R49 causal chain: (1) affordance → translation achievable [hand_drive]; (2) → RL discovers locomotion [crawl_minimal]; (3) locomotion + ball-direction obs → directed homing [crawl_targetobs]; (4) PPO → *reliable* directed homing [crawl_ppo_2M]. All four links confirmed. The next phase is the core project test, now well-posed for the first time: **replace the privileged ball-direction observation with VISION** — can the creature learn to use its cameras to provide the ball bearing that the privileged signal currently supplies?

**Caveats:** single seed; privileged target obs (not vision — that's the next phase); ~35% still time out (approach-then-drift — a candidate for a small terminal-approach shaping term or longer episodes); VecNormalize stats must be matched to the model at eval (a best/final-VN mismatch understated an earlier eval as 36.7%; the matched final-model/final-VN figure is 57.5%). Metabolic-cost term from the full MIMo recipe not yet added (not needed for this result; may further reduce the drift-away failure mode).

**Ablation caveat (important honesty correction) — the 57.5% is mostly a forward-crawl gait, not strong steering.** Running the theory-monitor's recommended diagnostic (zero the `target_obs` ball vector at eval) did NOT collapse the contact rate to the ~16.7% random-walk baseline as predicted: PPO best model scored **56.7% with the signal intact vs 46.7% with it zeroed** — only a ~10-point drop. Interpretation: the policy learned a largely **fixed forward-crawl gait** that succeeds because balls spawn in a **180° forward arc** (`spawn_cone_deg=180`), so "crawl forward" reaches ~half of them without needing the bearing; the directional signal adds only ~10 points of steering on top. This is the project's recurring confound (success-rate ≠ using the signal, cf. vision-ablation), now caught for the crawl task. **Revised claim:** we have reliable *forward* crawling that *weakly* steers — not yet strong directional homing. The honest test is a **360° spawn** (`crawl_ppo_360_3M`, launched): with balls in every direction the forward-crawl shortcut fails, so the directional signal becomes necessary, and the key metric is the **ablation gap** (intact vs zeroed contact rate) on that task. This does not undo the locomotion result (the body genuinely crawls, monotonic approach is real) — it corrects the *directedness* claim: how much the creature actually steers toward a specific bearing is what the 360° test will establish.

## Phase XVI — Directional steering CONFIRMED, confound-free (crawl_ppo_360_3M, 2026-07-02)

**The honest test resolves the confound in the strongest way: on a 360° spawn, the directional signal is decisively load-bearing.** The prior 57.5% (180° forward spawn) was ~47% forward-crawl + ~10% steering. This run removes the forward-crawl shortcut by spawning balls in ALL directions (`spawn_cone_deg 360`), so reaching a ball requires actually turning toward its bearing. PPO, 3M steps, otherwise identical to crawl_ppo_2M.

**Ablation result (40-episode deterministic eval, matched VecNormalize):**

| target_obs | Contacts | Mean toward-ball translation |
|---|---|---|
| **INTACT**  | **58%** (23/40) | **+0.255 m** |
| **ZEROED**  | **12%** (5/40)  | **−0.234 m** (crawls *away*) |

- **The ablation gap is now decisive (58% → 12%)** — collapsing to near the random-walk baseline exactly as the theory-monitor predicted, versus the mere 57→47 on the confounded 180° task. The directional signal is doing the work.
- **With the bearing removed, mean toward-ball translation goes NEGATIVE (−0.234 m):** blind, the creature crawls forward into empty space and ends up *farther* from balls that spawned behind/beside it. This is the cleanest possible proof that it steers by the signal rather than a fixed gait.
- **Same performance on the HARDER task:** 58% with balls in any direction ≈ the 57.5% it managed with only-forward balls — so the creature genuinely turns-and-crawls, it isn't just luckier on easy spawns.
- **Training stable:** ep_rew_mean rose 40→66→84→111→130→150→169 monotonically (PPO stability holds on the harder task).

**Verdict:** genuine **directional crawling to a target in any direction** is achieved and confound-proven. The creature perceives (via the privileged bearing) where the ball is, turns toward it, and crawls to it, reaching it 58% of the time; remove the bearing and it fails (12%, moves away). This is the real milestone — the 180° "57.5%" was a partial confound; the 360° "58% vs 12% ablated" is the honest, confound-free demonstration.

**Corrected project status:** reliable, genuinely-directional crawling to a target is solved (privileged bearing). All four R49 causal-chain links hold, now on the fair task. The next phase is the core project question, now cleanly well-posed with a 58% upper-bound benchmark and a proven behavioral gap (12% without direction): **replace the privileged `target_obs` bearing with VISION** — can the creature learn to read the ball's direction from its cameras and drive the same turn-and-crawl behavior?

**Caveats:** single seed; privileged bearing (not vision — next phase); ~42% still miss (turn-then-approach is harder than straight approach; longer training / metabolic-cost term are candidate refinements); rear-ball turning specifically not yet broken out by bearing bucket.

**Multi-seed confirmation (addresses the single-seed caveat):** the 360° directional-steering result reproduces across seeds — the large ablation gap is robust, not a seed artifact.

| Seed | INTACT contacts | ZEROED contacts | Ablation gap | ZEROED toward-ball |
|---|---|---|---|---|
| 0 | 58% | 12% | 46 pts | −0.234 m |
| 1 | 78% | 32% | 46 pts | −0.182 m |

Both seeds: identical ~46-point ablation gap, both go negative toward-ball when the bearing is zeroed (crawl away when blinded), both trained smooth-monotonic to plateau ~180. Absolute rate varies with seed (58% vs 78%, normal PPO variance) but the causal finding — the directional signal is strongly load-bearing and the creature genuinely steers by it — is seed-robust. Directional crawling to a target is confirmed.

**3-seed set complete (2026-07-02):** seed 2 = 70% intact / 20% zeroed (gap 50, toward-ball −0.356 m blinded). Full set — seed 0: 58/12, seed 1: 78/32, seed 2: 70/20. **Mean 69% intact vs 21% zeroed, a ~47-point ablation gap across all three seeds, every seed crawling away from the ball when the bearing is removed.** Directional crawling to a target in any direction is confirmed and robust. This closes the crawler-locomotion arc: the creature has a reliable, genuinely-directional turn-and-crawl-to-target behavior driven by a (privileged) ball-direction signal. Next phase (blocked on a camera-FOV winnability decision — see VISION_PHASE_PROPOSAL_2026_07_02.md): replace the privileged bearing with vision.

---

## 2026-07-04 — Vision phase A/B/C: vision becomes behaviorally load-bearing (reinforced, not destroyed)

**One-line:** With the camera fixed to actually see the ball, vision reads its direction from pixels
(decode R²=0.84, up from ~0.08) and — warm-started into a frozen motor policy — lifts contact from
the 20% blind floor to 63%, recovering the 20% baseline exactly when vision is ablated. The
proprioceptive gait is preserved; vision reinforces it.

### Camera fix (winnability gate)
- The vision phase was blocked: head-cam saw only ~±15°. Diagnosed with a new pixel-level harness
  (`visualization/cam_visibility_preflight.py`): the ball at the 0.70–0.80 m ring sits on the far
  horizon; arms occlude the sides; tilt doesn't change horizontal span; even fovy=150 can't see ±45°.
- Fix: **fovy 90→120** in `mimo_crawler_pos_wide.xml`, matched spawn cone **±22°** → ball a clean
  4–14 px blob across the cone, horizontal position monotonic with bearing. Visually confirmed.

### Blind base (Stage C substrate) — `crawl_ppo_blind_base` (PPO, 1.5M, no target_obs)
- Competent prone crawler: contact 20% (undirected), mean_disp 0.219 m, tip 2.5%. Video-confirmed
  belly-crawl (Gemini wrongly said "on its back"; frames corrected it).

### Stage A — distil bearing from pixels into frozen teacher (`stage_a_distill.py`)
- Lateral decode-R²: in-sample 0.945/0.952 (BC/DAgger); **teacher-driven held-out 0.841** (fwd 0.648,
  z 0.779). Student-driven held-out −0.011 = DAgger distribution shift (only 1 round), not a
  representation failure. **Representation failure fixed** (was R²≈0.01–0.08 for years).

### Stage C — residual grown by reward ALONE (`train_stage_c.py`, ResidualVisionPolicy), 300K
- Contact 20.0% (= blind floor) | pixels-ablated 3.3% | ablation gap +16.7 pts | substrate preserved
  (disp 0.235, tip 3.3%). **Integrated but inert.** Since A proved the representation exists, this is
  a POLICY-GRADIENT failure, not representation: reward alone can't grow steering on a cone where
  forward-crawl already gets 20%.

### Stage B — reward-grow vision into frozen teacher's bearing slot, WARM-STARTED from Stage A CNN
(`train_stage_b.py --warmstart-cnn`, SlotFillVisionPolicy). Eval on 50K best_model, ±22°, 30 eps:

| Metric | Value | Baseline |
|---|---|---|
| Contact rate | **63.3%** | blind floor 20%, teacher ceiling 73–77.5% |
| Contact, pixels ablated | **20.0%** | = blind floor exactly (graceful) |
| Vision load-bearing gap | **+43.3 pts** | (Stage C: +16.7, inert) |
| mean_toward | +0.218 m | blind +0.173 |
| Substrate: mean_disp | 0.504 m | blind 0.219 (enhanced) |
| Substrate: mean_speed | 2.14 mm/step | blind 0.30 (enhanced) |
| Substrate: tip_rate | 6.7% | blind 2.5% (still upright ~93%) |

Generalization battery (same policy, no retrain): radius 0.55–0.65 → 80%; 0.85–0.95 → 50% (graceful);
1.0–1.1 → 20% (floor, far beyond training); cone ±15° → 75%; ±30° → 60%. Generalizes across distance
and cone width. Video (`results/videos/eval_stage_b_ws_50k.mp4`) confirmed by direct frame inspection:
prone crawl to contact in distinct board positions (directed, not spawn-luck). Gemini misread it as
"failed standing" — unreliable for this body; use frames.

**Interpretation:** motor policy frozen (preserved) + pixel-ablation recovers the 20% baseline =
NOT DESTROYED; vision supplies the world-derived bearing, +43 pts, enhanced locomotion = REINFORCED.
The winning recipe is **distil-then-RL** (A builds the representation, B uses it); reward alone (C)
does not. Sensor substitution achieved; deeper interpenetration (prism-ghost aftereffect) now unblocked.

**New files:** `visualization/cam_visibility_preflight.py`, `crawler/eval_crawler.py`,
`crawler/stage_a_distill.py`, `crawler/residual_vision_policy.py` + `train_stage_c.py`,
`crawler/slotfill_vision_policy.py` + `train_stage_b.py`, `crawler/eval_vision_policy.py`.

### CORRECTION (2026-07-04, same night) — Stage B behavioral result was confounded
A polished 300K Stage B eval went net-negative (55% with pixels vs 75% ablated). Diagnostic
(`diag_confound.py`, teacher, true vs zero bearing): FULL ±22° → 82.5% vs 75.0% (bearing worth
+7.5 pts); LATERAL band ±15–22° → 65% vs 65% (+0.0 pts). The forward-crawl reach envelope (~0.28 m)
covers the entire camera-visible ±22° cone, so the bearing is behaviorally nearly irrelevant there.
- The earlier "+43 pt gap" (63% vs g(0)=20%) was an unstable-ablation-baseline artifact; the fair
  no-vision control (teacher + zero bearing) is 75%, and Stage B (55–63%) is BELOW it → vision did
  not reinforce behavior on this cone; it slightly hurt (added noise). RL fine-tuning also eroded
  the distilled encoder (50K read better than 300K).
- STILL SOLID: Stage A representation, decode-R²=0.84 (vision reads direction). Stage C inert with
  gait preserved (consistent — bearing wasn't needed).
- Root cause: winnable cone (±22°, camera-limited) ⊄ vision-necessary cone (> ~±30° forward-crawl
  reach). No overlap ⇒ behavioral vision value cannot be shown here. Fix: HEAD-SEARCH phase
  (head_swivel to fixate an off-cone ball) enabling a >±30–45° winnable spawn forward-crawl can't
  solve. Added `spawn_cone_min_deg` to the env for lateral-band tests. Best Stage B checkpoint is
  the 50K one, but the honest headline is Stage A (representation), not a behavioral win.

### HEAD-SEARCH RESULT (2026-07-04) — vision behaviorally load-bearing at last
±68° cone (vision-necessary per diag_confound; ball off the static view; added act:head_yaw so the
policy can pan when prone). Vision PPO, encoder warm-started from Stage A, 800K/2M. Clean eval (40 eps):
- contact **47.5%** (pixels ablated **35.0%**) → vision gap **+12.5 pts**.
- mean_toward **+0.073 m** with vision vs **−0.036 m** ablated (vision flips net motion from AWAY to TOWARD).
- substrate: tip 5%, disp 0.650 m (searching+crawling). Video frame-verified: turns toward off-cone
  ball and crawls to it.
FIRST behaviorally-necessary vision win in the project (contrast ±22° where vision was net-zero because
forward-crawl solved it). Modest magnitude (partial from-scratch learning; ablated 35% inflated by
undirected search-wander). Next: more training + search-shaping curriculum (narrow→wide cone).
Files: train_head_search.py, mimo_crawler_pos_wide_hs.xml (act:head_yaw), diag_confound.py.

### HEAD-SEARCH CURRICULUM RESULT (2026-07-04) — vision win sharpened
Search-shaping curriculum (cone ±22→±34→±45→±68) beat the flat run on the ±68 target: contact
**65.0%** vs 47.5%, pixels-ablated 42.5% vs 35.0%, **vision gap +22.5 pts** (vs +12.5), mean_toward
+0.106→−0.092 (stronger sign-flip). Reached 202±1 reward (near-perfect) on ±45/±68; carried
competence through every widening without collapse. Substrate: tip 10% (up from 5%; moves more
aggressively). Video frame-verified. Curriculum made vision MORE behaviorally load-bearing, not just
a better searcher. Files: train_head_search.py --curriculum (CurriculumConeCallback).

### PRISM PRECHECK (2026-07-04) — vision = arousal/presence, not direction; prism blocked (reframe)
Built prism-ghost env (hidden solid real + visible non-physical ghost at real-bearing+offset; head-cam
sees only ghost — verified). Precheck: base head-search policy under prism offset 30/45/60/90/120 keeps
reaching the REAL ball (67-93%) and moving toward the REAL bearing (80%) — reach is INVARIANT to visual
displacement. Reconciles with the +22.5 ablation gap: zeroing pixels hurts (arousal lost), displacing
doesn't (direction not vision-set). So vision on ±68 is load-bearing as a PRESENCE/ENGAGEMENT cue that
triggers a broad proprio/touch search sweep; the sweep touch-homes on the only solid object. The policy
does NOT use vision for reach DIRECTION (Stage A's R²=0.84 latent is present but unused for direction).
Prism aftereffect needs a vision-DIRECTION-dependent reach first (remove the touch-search escape). Not
launched. Infra committed: mimo_crawler_pos_wide_prism.xml, train_prism.py, eval_prism.py, env prism_offset_deg.

### HEAD-SEARCH SEED REPLICATION, seed 1 (2026-07-05) — vision gap replicates in sign, not size; search is lateralized
Same recipe as the curriculum run (cone ±22→±68, warm-started encoder, PPO), seed 1, 2M steps.
Clean eval (40 eps, ±68): contact **70.0%** (ablated **57.5%**) → vision gap **+12.5 pts**;
mean_toward +0.093 vs +0.008 ablated (weaker sign-flip than seed 0's +0.106/−0.092); tip **2.5%**
(best substrate yet — seed 0 was 10%). Gap is positive both seeds, magnitude varies 12.5–22.5:
seed 1's BLIND baseline is far stronger (57.5% vs 42.5%), which mechanically shrinks the gap.
Bearing diagnostic (40 eps, per-spawn bearing logged): misses are NOT noise — **far-left spawns
(< −45°) hit 1/9 (11%) vs far-right (> +45°) 7/8 (88%)**, center ~90%. Seed 1 learned a
RIGHT-LATERALIZED search sweep; the left extreme of the cone is a systematic blind wedge (its one
far-left hit took 700 steps of wander). Video note: first render (seed+7) drew 3 straight
full-length misses and looked like total failure — per-episode logging + frame check on eval seed
confirmed genuine fast reaches (67–237 steps). Don't judge a 70%-contact policy on a 3-episode video.
Seed 2 (1.3M) + posture run (tilt-cost 1.0) queued in chain. Files: eval_vision_policy.py (unchanged).

### HEAD-SEARCH SEED REPLICATION, seed 2 (2026-07-05) — gap +7.5; three-seed verdict: sign replicates, size doesn't
Seed 2, 1.3M steps, same curriculum (cone widened 195K/455K/780K, rode every widening to eval
reward 202±1 — the "near-perfect" training signature). Clean eval (40 eps, ±68): contact **45.0%**
(ablated **37.5%**) → gap **+7.5 pts**; mean_toward **+0.200 vs +0.000** (largest toward-effect of
any seed); tip 2.5%. Bearing diagnostic: NOT lateralized — weak at BOTH extremes (far-left 2/9,
far-right 2/8, center 15/23): a narrow-symmetric search phenotype, despite the perfect-looking
training eval (few-episode EvalCallback draws miss the cone extremes — same "aggregate hides the
tails" trap again). Video frame-verified (approach + head-on-ball contact).
**THREE-SEED VERDICT (task: multi-seed confirm +22.5):** gap is positive in 3/3 seeds — s0 +22.5,
s1 +12.5, s2 +7.5 (mean ~+14) — but magnitude is seed-dependent and the +22.5 headline was the best
of three, not typical. With 40-ep samples each gap alone is ~1σ; 3/3 positive + the toward-effect
make the modest-real reading. The MOST seed-consistent vision signal is mean_toward: sighted ≥+0.09
in all seeds, ablated ≤+0.01 in all seeds (vision reliably converts wander into net approach even
when contact-gap is small). Search phenotype varies wildly by seed: broad (s0), right-lateralized
(s1), narrow-symmetric (s2) — search strategy, and hence the blind baseline, is the main
between-seed variance source; the vision contribution on top is steadier than the contact-gap
suggests. Posture run (tilt-cost 1.0) still training.

### POSTURE-TERM RESULT (2026-07-05) — tilt-cost fails its job AND erases the vision gap
Seed 0, tilt-cost 1.0, same curriculum, 1.3M steps. Clean eval (40 eps, ±68):
- tip rate **12.5%** vs 10% baseline → the posture term did NOT cut tipping (its one job). FAILED.
- contact 62.5% (baseline 65.0%) — overall competence unchanged.
- vision gap **−7.5 pts** (62.5% sighted vs **70.0% ablated** — the best BLIND performance of any
  run) and mean_toward identical sighted/ablated (+0.152/+0.150) → vision is NOT load-bearing at
  all in this policy. Same seed, same curriculum as the +22.5 run; the only change is the tilt term.
Reading: the tilt penalty prices out the aggressive vision-triggered maneuvers (fast turns/lunges
that risk tilting) and training converges on a conservative, broad BLIND sweep instead. The vision
gap is FRAGILE TO REWARD SHAPING: an auxiliary term that leaves aggregate success intact can
silently delete vision dependence. (Ironically the opposite of this branch's name — here the vision
phase was destroyed, not reinforced, by shaping.) Rule going forward: any reward change in a vision
phase must re-measure the ablation gap, not just success/tip rates. Do not use tilt-cost 1.0 in
vision runs; posture control needs a mechanism that doesn't tax search dynamics (or a much smaller
cost, gap-checked). Video frame-verified (healthy prone search, face-on-ball contact).
Chain complete: s1 2M / s2 1.3M / posture 1.3M all evaluated.

### TIME-PRESSURE CALIBRATION (2026-07-05, overnight) — clock cannot close the touch-search escape
Strategist-proposed no-training precheck before committing compute: recorded per-episode touch-step
for s1 + v1 (best seed-1 and +22.5 seed-0 policies), sighted vs ablated, 40 eps each; contact(T)
computed post-hoc for any budget T (valid because the cap only truncates deterministic episodes).
Result: the blind sweep is FAST — ablated contact is 37.5-55% already at T=150-350 — so no clock
setting separates directed reach from blind sweep. Gap(T) for s1 wobbles +5..+15 at every T
(never opens); v1 goes NEGATIVE at tight T (-10 at 150-200: blind beats sighted under pressure).
This is the pre-registered failure signature -> time-pressure-alone REJECTED, moved to the two-ball
decoy-discrimination mechanism. Meta-lesson: the touch-search escape isn't slow groping, it's an
efficient learned sweep; only a WRONG-ANSWER cost (not a time cost) can price it out.
Launched overnight: decoy_v1_s0 (blue decoy, identical physics, wrong-touch -5 + terminate,
min 30-deg separation; curriculum recipe, 1.3M, seed 0). Preflights passed: eye-view check (red vs
blue clearly separable in the actual 32x32 stereo obs), forced-contact semantics test, 50K smoke
(reward climbing, episodes ending by touch). Files: mimo_crawler_env.py (decoy_ball),
train_head_search.py --decoy, eval_vision_policy.py --decoy + WRONG-BALL rate + eye-view panel.

### DECOY RUN 1 (2026-07-06 morning) — task works, but a placement confound gave blind a 63% floor
decoy_v1_s0 (1.3M, curriculum, warm encoder) trained clean (2h13m, ~163 fps). Official eval:
red-contact 57.5% / wrong-ball 32.5% (sighted), 47.5% / 35.0% ablated; raw gap +10; tip 5%.
BUT the per-episode choice diagnostic caught a confound: choice accuracy is IDENTICAL sighted vs
ablated (63.9% vs 62.9%) — and blind picking red at 63% is impossible if the task were symmetric.
Cause: the min-separation rule pushes the DECOY away from center whenever the target spawns
centrally, so red is on average more central and the blind forward-sweep exploits the geometry.
Run 1's "discrimination" was mostly spawn geometry; vision learned ~no discrimination in 1.3M.
Fix (decoy_v2): after computing the bearing pair, randomly assign red/blue to the two bearings —
blind choice is then 50% BY CONSTRUCTION and anything above it is vision. Verified: v1 policy ablated on the fixed env = 55.7% +/- 10.4 (100 eps) — chance restored.
Lesson (again): always give the blind baseline a chance to cheat before crediting vision — the
choice metric needed the same scrutiny as contact rate did in the single-ball task.

### DECOY RUN 2 (2026-07-06) — FIRST ABOVE-CHANCE VISUAL DISCRIMINATION
decoy_v2_s0: exchangeable placement (blind = 50% by construction), fresh 2M, curriculum, warm
encoder. The decisive 100-ep choice diagnostic:
- **sighted choice 63.2% +/- 10.1 (55R/32B) — CI excludes chance (p≈0.017)**
- **ablated choice 49.4% +/- 10.6 (42R/43B) — exactly the designed coin flip**
First time in the project that vision measurably changes WHICH object AB reaches — pixel color is
steering target selection, with no geometric escape available to the blind baseline. Magnitude is
modest (63% vs 50%; pooled with the 40-ep official eval: sighted 60.3% vs ablated 51.3%).
Official eval (40 eps, noisier): contact 45.0% both conditions (raw gap 0.0 — contact rate is no
longer the right metric here; the decoy makes CHOICE the signal), wrong-ball 40.0%/35.0%, tip 10%.
Training-eval reward oscillated 120-202 in the back half — no clean plateau; discrimination may
still grow with steps. Video frame-verified: both balls enter the eye view during approach; the
32x32 blobs are unambiguous. Metric note going forward: report choice accuracy (red /(red+blue))
with n, not contact rate; blind floor is structural 50%.
Next: seed replication + longer training to push discrimination; then the prism displacement test
on a strong discriminator (does AB follow the ghost?) — the original aftereffect experiment.

### GAZE-CHOICE INVERSION (2026-07-06) — AB looks at the ball it does NOT take
User observation (red almost never in the eye panels of the milestone video) → quantified over 40
eps of decoy_v2_s0: in RED-ending episodes, blue is in view 61.5% of steps and red only 13.2%;
in BLUE-ending episodes the mirror (red 40.8%, blue 9.9%). Time-resolved control: the inversion
already holds in the EARLY half of episodes (RED-enders: blue 60% / red 20% early), so it is NOT
the under-the-chin artifact (which only amplifies it late: the approached ball's visibility falls
to 0-7% in the last 50 steps). Reading: the policy FIXATES one ball and approaches the other —
vision acting as a repulsor/monitoring cue rather than red-phototropism. Two mechanisms remain
indistinguishable here: (a) "steer away from the fixated ball" (repulsor control), vs (b) "home on
the chosen ball from memory while keeping the rejected one monitored." Either way, the naive
"approach the red blob you see" story is wrong; the 63% choice asymmetry rides on which ball gets
fixated/rejected. Display upgrades shipped alongside: magenta heading arrow on the overhead panel
(true head +Z from xmat; render-side only) and an occiput "haircut" marker (yellow T on the back of
the skull — brown crown = forward, yellow = rear; obs verified bit-identical, marker massless).

### DECOY EXTENSION RESULT (2026-07-06) — discrimination climbs to 78%: still-forming, not ceiling
decoy_v2_ext_s0: seed-0 continued +1.3M at fixed ±68 (3.3M total; --init-model, no curriculum).
- **choice accuracy 78.2% +/- 8.7 (68R/19B, 100 eps) — up from 63.2% at 2M**; ablated 47.8% +/- 10.2
  (chance, as designed). Official eval agrees exactly: contact 62.5% / wrong-ball 17.5% → 78.1%.
- Confirms the THEORY_LOG longer-training prediction: the color-choice category was still forming;
  +1.3M bought +15 pts. Not yet at the 32x32-signal ceiling.
- Ablated wrong-ball rate 47.5% vs sighted 17.5% — blind grabs whichever ball it meets; sighted
  actively avoids the decoy (consistent with the gaze-choice inversion veto reading).
- Watch-items: tip rate 15% (10% at 2M — climbing as maneuvers sharpen); training eval-reward
  looked mediocre (78-120) while choice soared — REWARD IS A POOR PROXY for discrimination; use
  the choice diagnostic. Video (first with heading arrow + haircut) frame-verified.
Seeds 1 and 2 training next in chain.

### DECOY SEED REPLICATION COMPLETE (2026-07-07) — visual discrimination is seed-ROBUST
All three seeds + extension, 100-ep choice diagnostics (blind floor = structural 50%):
- s0 @2M:   sighted **63.2 ± 10.1**, ablated 47.8 ± 10.2
- s0 @3.3M: sighted **78.2 ± 8.7**,  ablated 47.8 ± 10.2  (extension)
- s1 @2M:   sighted **72.0 ± 9.7**,  ablated 56.7 ± 10.2
- s2 @2M:   sighted **76.7 ± 8.9**,  ablated 45.5 ± 10.4
Sighted CI excludes chance in ALL runs (3/3 seeds; mean @2M ≈ 71%); ablated consistent with 50%
in all (s1 leans high at 56.7 but within CI — no confound tripwire fired). THEORY_LOG predictions
confirmed: sign holds every seed, magnitude varies (63→77), blind floor stays pinned.
Notably UNLIKE the single-ball phase, the discrimination result replicates strongly — closing the
structural escape didn't just create the effect, it stabilized it across seeds.
s2 details: official gap +25.0 (62.5 vs 37.5), blind wrong-ball 55.0% vs sighted 22.5%, tip 0.0%
(best substrate + best gap in the same run; the tip-rate worry from ext_s0 did not replicate).
Videos frame-verified (arrow + haircut instrumentation). NEXT: the flagship prism-displacement
test on a strong discriminator (ext_s0 78% or s2 77%) — THEORY_LOG 2026-07-06 Q4 has the
pre-registered predictions (follow-the-ghost vs arousal-gate vs partial binding).

### PRISM DISPLACEMENT RESULT (2026-07-08 overnight) — AB FOLLOWS THE GHOST. Vision drives target selection.
Whole-field two-ghost displacement (both reals solid+hidden, ghost pair rotated by offset), eval-only,
100 eps/cell, both strong discriminators. Sanity gates passed (off-0 sighted 76.4/77.5 vs known 78/77;
ablated 47.8/45.5 ≈ floor). choice_vs_true by offset:
- ext_s0: 76.4 (0) → 76.7 (15) → 62.0 (30) → **53.2 (45) → 44.7 (60)** → 51.6 (90)
- s2:     77.5 (0) → 65.6 (15) → 47.6 (30) → **53.3 (45) → 37.8 (60)** → 36.3 (90)
Collapse is monotone-ish and — decisively — goes **BELOW chance at 60-90°** in both checkpoints.
Arousal-gating can only degrade toward 50, never below: systematic mis-selection means the displaced
picture is steering. Wrong-ball rate rises with offset in both (21→47% ext; 20→58% s2).
**Conditional smoking gun** (pooled per-episode, all offsets): when the red GHOST appears nearer the
true-BLUE position, AB touches blue **67.7%** (n=334); when it appears nearer true-red, **34.8%**
(n=563). At off 30/45: 74.5%/71.8% vs ~30%. AB goes where red APPEARS and takes whatever solid
object is there. **VERDICT: REGIME 1 — FOLLOW THE GHOST**, in both checkpoints, by the primary
pre-registered metric (choice ≤58 by 45° ✓✓, CI-separated from baseline) plus the conditional
analysis. Honesty note: the pre-registered displayed_red heading fraction was uninformative as
implemented (start→end bearing necessarily lands on a real ball for committed episodes; threshold
unusable) — the conditional analysis above is its valid replacement and is stronger.
Implication: decoy training produced genuinely DIRECTION-CARRYING vision — the "presence/arousal
only" account is dead for these policies. The prism ADAPTATION + AFTEREFFECT experiment (the
project flagship) is now unblocked and meaningful; strategist's draft awaits human approval.
Files: eval_prism_decoy.py, ghost2 in mimo_crawler_pos_wide_prism.xml, env two-ghost placement.

### PRISM ADAPTATION + AFTEREFFECT (2026-07-08 overnight) — NEGATIVE AFTEREFFECT CONFIRMED.
### AB genuinely recalibrated vision-to-action. The experiment the project is named for.
Phase B: ext_s0 continued 1M steps under fixed +30° whole-field prism (decoy task, weights
unfrozen, no reward changes). Phase C: prism removed, 100-ep evals on the adapted model.
- **Adaptation curve (+30°, choice-vs-true):** 62.0 (pre) → 67.0 (100K) → 58.1 (250K) → 56.2
  (500K) → 48.8 (750K) → 54.8 (1M). NO clean aggregate recovery — adaptation was partial and
  unstable (train reward oscillated 35–140). RL is a blunt, slow adapter compared to the classical
  paradigm's minutes.
- **BUT the aftereffect is unambiguous.** Prism-off aggregate fell to 54.7% (baseline 76.4) — and
  the degradation is not uniform, it is LAWFUL: **P(take blue | blue on the MINUS-30° side of red)
  = 92.1% vs 8.3% on the plus side.** The reach now aims ~30° OPPOSITE the trained displacement.
- **Controls:** (1) pre-adaptation the same conditional is symmetric (26.2% vs 21.3%) — the
  asymmetry is created by adaptation; (2) the capture is RED-ANCHORED, not a color-blind lateral
  habit: P(take blue) is 77–100% within 45° of the aftereffect bearing (red−30°) and ~33% beyond —
  a color-blind CCW rule predicts flat. (3) Same-direction bias persists under −30 eval (87.5/13.6).
**Scoring the pre-registered outcomes (THEORY_LOG 2026-07-08):** relearning REFUTED (it predicts
no aftereffect; we have a strong one). Arousal REFUTED (everything changed). RECALIBRATION
CONFIRMED in its decisive signature — the direction-specific, red-anchored, training-created
negative aftereffect — with the honest caveat that Phase-B aggregate recovery was weak/partial
(recalibration-in-progress, not completed; the aftereffect demonstrates the re-mapping exists).
The chain now reads: decoy task made vision direction-carrying (2026-07-06/07) → displacement
showed vision drives selection (2026-07-08) → adaptation re-mapped the vision-action link and
misfires lawfully when the world snaps back — the classical prism-adaptation phenomenon,
reproduced end-to-end in a learned sensorimotor system built from pixels, proprioception and touch.
Files: prism_adapt_s0 run + curve/aftereffect JSONs (prism_battery_aftereffect_*.json).

> **Theory Monitor Note — 2026-07-08**
>
> Behavioral Prediction Framework (recalibration vs. relearning vs. arousal): **CONFIRMED, via the
> Phase-C fork specifically** — the negative aftereffect (54.7% vs 76.4% baseline, lawful
> 92.1%/8.3% conditional capture, controlled against a symmetric pre-adaptation baseline) is a
> pattern only a genuine recalibration can produce; both relearning and arousal predicted an
> immediate snap-back to baseline the instant the prism came off, and neither happened. The
> Phase-B recovery curve itself (62→67→58→56→49→55, ending below where it started) did NOT meet
> its own pre-registered bar (≥65-70% recovery) — read that as the metric being too noisy to trust
> on its own, not as evidence against recalibration, since Phase C is the part of the pre-registration
> built to settle exactly this and it settles it cleanly.
>
> Pattern Learning Framework (is the visual code a map or a reflex): **still UNRESOLVED, but
> narrowed** — the aftereffect proves *some* persistent, carried-over state sits between the camera
> and the reach (there is no displaced picture left to react to in Phase C, yet the miss pattern is
> still bearing-specific), which rules out a pure "react only to what's on screen right now" account.
> It does not yet prove a true spatial map — a single learned "subtract 30 degrees everywhere" bias
> explains the same numbers.
>
> **The most important thing we don't know yet:** whether the after-prism bias is one global
> correction applied the same way at every angle, or a structured remapping that differs by bearing
> — answerable from data already collected (see recommended diagnostic).
>
> **Recommended diagnostic** (not a training run — just a measurement): re-bin the aftereffect
> episodes already on disk by how far the ball is from the trained -30 degree offset direction, and
> check whether the "captures the wrong ball" rate changes smoothly with that distance (a structured
> remap) or is a flat step on/off either side of a fixed line (a single global bias). No new compute.

## 2026-07-08 — Mismatched-shape decoy: vision's color discrimination is SHAPE-INVARIANT (first visual object-agnosticism)

**Question.** "Object-agnostic reach" was established only for PROPRIO (Phase XV R47/R48: held-out ellipsoid/capsule reached like spheres), where agnosticism is nearly by-construction — a touch-driven reach cannot perceive shape. Every VISION result to date used red vs blue **spheres**, so whether the *visual* channel is object-agnostic was untested. This tests it directly: does the decoy policy's color choice survive when the objects are no longer the spheres it trained on?

**Method (zero-shot, no retraining).** Override the two decoy balls' MuJoCo geom primitive at eval time (`eval_decoy_shape.py`; the env loads the model once and resets use `mj_resetData`, so a geom_type/size override persists). Sizes match the cart-env `_SHAPE_MAP` that produced clean proprio shape-transfer. Ran on `decoy_v2_ext_s0` (the clean symmetric seed) and `decoy_v2_s2`, 200 ep/cell, offset 0, exchangeable placement (blind floor 50%). **Confound caught & fixed:** at offset 0 the prism `ghost` bodies are not repositioned/hidden by the env, leaving a fixed RED-sphere ghost in view — a red-sphere reference that would defeat a shape test. Made both ghosts invisible (alpha=0) in ALL conditions incl. control. Verified via sanity render (overhead shows exactly 2 objects; agent left-eye at step 60 clearly sees the red box).

**Result — color-driven, shape-invariant. EXT_S0 choice_vs_true (→RED), 200 ep:**

| cond | red | blue | P(reach red) | 95% CI |
|------|-----|------|-------------|--------|
| A (control) | sphere | sphere | 77.2% | [70.6, 82.7] |
| B | **box** | sphere | 81.4% | [75.2, 86.4] |
| C | sphere | **box** | 80.7% | [74.2, 85.8] |
| D | **box** | **box** | 79.0% | [72.4, 84.3] |
| E | **capsule** | sphere | 76.4% | [69.7, 82.0] |
| F | sphere | **capsule** | 79.1% | [72.5, 84.4] |
| A′ ablated | sphere | sphere | 52.7% | [45.5, 59.8] |
| B′ ablated | **box** | sphere | 52.2% | [45.0, 59.3] |

All six sighted conditions are statistically identical (76–81%). The sphere-preference hypothesis (predicting B≪50%, C high) is refuted: B and C match the control, and **condition D holds at 79% with NEITHER object the trained sphere.** S2 replicates exactly (A 78.9 / B 77.0 / C 78.3 / D 79.1) — its earlier lateralization wedge does not touch shape-invariance.

**Full shape vocabulary (EXT_S0, 200 ep each) — matching the proprio Phase-XV set:** red=ellipsoid/blue=sphere 79.4%; red=cylinder/blue=sphere 76.4%; both-ellipsoid 78.0%; both-capsule 74.6%; **red=box/blue=capsule (two different novel shapes, neither trained) 84.4%** — the hardest case and the cleanest. Invariance is complete across sphere/box/cylinder/ellipsoid/capsule, the exact vocabulary in which proprio shape-transfer was shown (R47/R48).

**Critical control (ablated floors).** A′ and B′ both sit at ~52% (chance), and crucially B′ (red=box) equals A′ (red=sphere): **the box introduces no non-visual touch/physics/placement asymmetry**, so the sighted invariance is genuinely a property of the visual channel, not an artifact of how a cube contacts vs a rolling sphere. The by-bearing structure (central ~90–96%, edges toward chance) is unchanged across all shapes — the same field-of-view limit found in the per-bearing symmetry check, not a shape effect.

**Interpretation.** The decoy policy keys on **color identity, not object geometry.** A policy trained only on spheres discriminates red-from-blue at full strength on boxes and capsules, zero-shot. This is the project's **first demonstration of visual object-agnosticism** — and it is a *stronger* claim than the proprio version, because vision CAN perceive shape (unlike the touch-driven reach) and still ignores it in favor of color. So the corrected, complete statement is: the proprioceptive reach program is shape-invariant largely by construction; the visual discrimination channel is shape-invariant *by learning* — it had shape information available and did not bind to it.

**Files:** `crawler/eval_decoy_shape.py` (new), `crawler/_run_shape_battery.sh`, `crawler/_run_shape_ext.sh`, `crawler/_analyze_shape.py`; results `results/decoy_shape_*.json`.

**Caveats / open:** (1) tested red-vs-blue only — whether the binding is "approach red" vs "avoid blue" is a separable follow-up (recolor test). (2) 32×32 stereo — shape is a coarse silhouette cue at this resolution, which is part of *why* color dominates; a higher-res camera might let shape compete. (3) all offset 0 (no prism); shape × displacement interaction untested.
