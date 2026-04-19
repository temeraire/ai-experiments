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
