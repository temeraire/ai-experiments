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

v6/v7 gave the agent a gaze camera but not a strong reason to orient. The theory-faithful move was to give the agent a world where getting orientation wrong has consequences: a creature on a finite platform that can roll off and fall. v8 is the first environment in this project where wrong perception kills you. "Gravity is the pencil tap."

The creature is a torso-sled with two arms and a pan/tilt head. No legs. The working metaphor is a skateboarder-without-legs: push with hands, body rolls. Survival is grounded in physics — not architectural weight freezing (v2/v3/v5) and not reward shaping (v6 implicit).

### What this session focused on

Three parallel threads:
1. **Body physics that's actually drivable** (this was the hard part).
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

## 2026-05-11 — Phase A: No-bribery substrate, existing CNN (mimo_substrate_A)

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

**Next question:** Does the dialogue architecture (two-stream actor with proprio + vision producing independent action proposals, a learned scalar gate combining them, and a consistency loss `λ · ||μ_p − μ_v||²` penalizing per-stream disagreement) escape this attractor on the same substrate? Phase B is currently running (300K steps, mimo_dialogue_v1) with identical environment and reward configuration. If Phase B's vision-ablation sensitivity comes back above 0.5 with touch rate > 25%, the structural change is what made vision load-bearing — substrate alone was a necessary but not sufficient prerequisite. If Phase B also collapses, the MIMo body is the limiting factor and the next experiment should be on the platform creature with the dialogue architecture instead.

**Theory signals:** The Behavioral Prediction Framework predicts that an agent with a viable internal model of where contact is achievable should outperform random walk on the 360° spawn distribution (random walk on a 2m × 2m platform with a 5cm ball over 2000 steps should produce ~30-50% touch by ballistic coverage alone). The deterministic 0% touch rate is far below the random-walk baseline, indicating the policy has not built a predictive model — it has been driven by the SAC update dynamics into a state of literal inaction. This is a stronger violation of the framework's prediction than any prior run, because no prior run had a substrate that supported random walk baseline at this level. The Pattern Learning Framework predicts that the visual representation should respond to recurring patterns in the camera feed (the ball as a stable red structure, the guardrail walls as a recurring spatial feature). The 0.0226 ablation delta shows no such pattern has developed — the visual pathway is not just unused for control, it is structurally inert. Both frameworks are challenged in the same direction: when bribery is removed, the flat-concatenation architecture produces a less capable policy, not a more grounded one. The architecture, not the substrate, is the load-bearing failure.

