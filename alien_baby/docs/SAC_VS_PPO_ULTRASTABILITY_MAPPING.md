# What Taylor's Ultrastability Theory Requires of an RL Algorithm

A chapter-by-chapter mapping of *The Behavioral Basis of Perception* (James G. Taylor,
1962, Ch. 1–11) onto SAC and PPO, to decide which better tests the theory's claims in a
simulated embodied agent (the Alien Baby MuJoCo infant body: continuous action space,
RND intrinsic curiosity, liveness gate).

Source: `Behavioral Basis of Perception.binarized.docx` (project root), Ch. 1–11.
Candidates: SAC (off-policy, replay buffer, auto entropy-tuning) vs. PPO (on-policy,
no replay buffer).

## Verdict

**SAC** — and the project already made the right call. Across 11 chapters, the theory's
own computational demands (a value landscape over simultaneously-active response
"readiness," backward-chained credit that extends across many trials, and above all the
Chapter 9 finding that old sensorimotor mappings are *retained alongside* new ones rather
than overwritten) line up with what an off-policy, replay-buffer, auto-entropy algorithm
gives natively. PPO isn't disqualified — it's a fine, lighter-weight choice for
short-horizon, single-context correction tasks — but no chapter argues for it over SAC,
and Chapter 9 argues strongly against it.

## Chapter-by-chapter requirements

| Chapter | Core mechanism | Computational requirement | Fit |
|---|---|---|---|
| **1. Introduction** | Ashby's ultrastability: a "critical state" (need/drive) triggers random search; success locks in permanently. Multistable = many such subsystems, serially or parallel dependent. §1.5–1.7 | Drive-gated (not constant) exploration; discrete retention events; subsystems decomposable with explicit dependency order. | **SAC** — auto entropy-tuning collapses toward exploitation as competence rises; RND spikes already act as a "critical state" trigger. |
| **2. Beginnings of Adaptation** | ~10s eligibility-trace conditioning of infant reaching; irradiation/generalization to neighboring states; secondary reinforcement chains sub-movements backward. §2.13–2.20 | Local/short-horizon credit composed into longer chains; smooth generalization across nearby states; entropy that anneals as competence grows. | **SAC** — off-policy TD + replay composes short transitions across many updates; PPO's GAE(λ) chains only within one discarded rollout. |
| **3. Beginnings of Space Perception** | Learned response approaches a near-deterministic function F(state); Π = many simultaneously-active "readiness" states, not one sampled action. §3.3–3.9 | Entropy should collapse post-training; representation should be a landscape over candidate actions, not a single sampled stream. | **SAC** — Q(s,a) is literally a value landscape over the action space; PPO's critic is V(s)-only, with no action-conditioned analogue. |
| **4. Further Development of Space Perception** | New DOFs (trunk) adapt atop a frozen, already-adapted subsystem; compound multi-effector movements get one joint credit event; backward credit propagates through the trace window. §4.5–4.9 | Curriculum that protects prior skills; trajectory-level joint credit; a replay-style mechanism for re-crediting recent transitions once an outcome is known. | **SAC** — the replay buffer *is* the backward-crediting mechanism §4.7 describes; PPO has nothing to revisit once a batch is consumed. |
| **5. Gravity** | Drive-triggered postural correction conditioned to stimuli just before a perturbation; staged curriculum (head→trunk→limbs), each stage frozen once stable; movement-opportunity frequency is rate-limiting. §5.2–5.15 | Short-latency credit is sufficient here — no long replay strictly required; staged curriculum; zero signal without movement. | **Either** — short-horizon postural TD is equally native to both; the first chapter where PPO isn't at a disadvantage. |
| **6. Expanding the Visual Field** | Distance perception bootstrapped from locomotor cycles; secondary reinforcement chains "extend backward almost indefinitely"; re-adaptation expected after scale/embodiment change. §6.5–6.14 | Reward must propagate backward across a bounded-but-extendable multi-step sequence — more than 1-step TD. | **SAC** — off-policy multi-step bootstrapping over a replay buffer matches "almost indefinite" chaining far more cheaply than enlarging PPO's rollout window. |
| **7. Parallax** | Depth perception literally doesn't exist without self-generated viewpoint motion; multi-cue redundant fusion; relational depth learned from co-present objects; drive-modulated attention gate. §7.2–7.15 | Active-sensing action dimensions; multi-cue encoder; episodes with multiple simultaneously visible objects. | **Neutral** — an environment/architecture requirement, not an algorithm one; both SAC and PPO support it equally. |
| **8. Experimental Evidence** (+ Papert appendix) | Re-adaptation to transformed vision occurred *only* in behaviorally-active subjects; passive subjects showed zero correction; gradual exponential learning curve over many trials; subsystems (oculomotor/manual/verbal) adapt at different rates and can conflict. §8.5–8.13 | Validates the liveness gate as theory, not just pragmatics; many trials needed — sample efficiency matters; separate heads per effector/modality. | **SAC** — off-policy sample efficiency is the practical difference between hours and days of simulated wall-clock for a many-trial curve. |
| **9. Three Experiments** (prism/lens) | Adaptation is per-subsystem; the old engram is **not overwritten** — it persists alongside the new one, switching cleanly on a context cue (spectacles on/off) with no interference; Ashby's rule: decomposed subsystems adapt faster than all-variables-active systems. §9.6–9.22 | Retain (don't overwrite) pre-perturbation experience; condition the policy/value function on an explicit context signal; scope adaptation to the smallest active subsystem. | **SAC** — the decisive chapter. A replay buffer *is* a retained memory of both contexts; PPO discards pre-perturbation data by construction. |
| **10. The Perception of Color** | Same conditioning framework on wavelength; color constancy needs an invariant operator pooling object + surround signal; re-adaptation to distorted color mapping is gradual and strictly task/drive-gated. §10.14–10.29 | Shared latent pooling local + contextual signal; learning gated by an active task/drive error, not passive drift. | **SAC** — restates Chapter 9's coexisting-engrams pattern; same replay/context-conditioning argument applies. |
| **11. The Modalities** | Modalities are differentiated by which behaviors get conditioned to which receptor-origin stimuli, not by separate channels; cross-modal integration via a shared spatial reference frame ("interpenetration"); one raw signal can support multiple non-interfering behaviors by context. §11.14–11.16 | Modular heads adapting semi-independently, integrated through a shared latent/spatial code; context-conditioned rather than overwritten mappings. | **Slight SAC** — architecture-agnostic in principle, but SAC's actor/critic/target-network split is a natural home for shared-trunk, multi-head, auxiliary-objective designs. |

## Why Chapter 9 carries the most weight

Most chapters favor SAC on efficiency or representational grounds — real advantages, but
ones a sufficiently patient PPO run could partly compensate for with more environment
steps. Chapter 9's prism/lens experiments are different: they describe a *structural*
claim that PPO cannot approximate no matter how long it trains.

Taylor's subjects didn't relearn a single sensorimotor mapping when the spectacles went
on — they built a **second** mapping that coexisted with the first, cleanly switched by
an incidental context cue (the pressure of the spectacles on the nose), with no
interference given intermittent exposure (§9.18–9.22). That is a description of
retained, context-keyed memory of two policies at once — exactly what a replay buffer
(optionally partitioned or context-conditioned) gives an off-policy learner for free.
PPO's on-policy update discards each rollout after use; once training shifts to
post-perturbation data, there is no mechanism left to keep bootstrapping the
pre-perturbation mapping without external scaffolding (separate policies, EWC-style
regularization) that PPO doesn't provide natively.

If the project ever runs a "prism experiment" analog — suddenly perturbing an
observation or actuator mapping mid-training — this is the chapter that predicts what a
faithful RL implementation needs, and it needs SAC's replay buffer specifically, not
just an off-policy algorithm in the abstract.

## Practical implications for Alien Baby

1. **Context-condition the critic for perturbation experiments.** When testing a
   Ch.9-style mapping flip (inverted actuator, shifted camera, etc.), add an explicit
   context flag to the observation and keep both pre- and post-perturbation transitions
   in the replay buffer — or twin buffers keyed by context — rather than only training
   on fresh post-perturbation data. This directly tests the coexisting-engrams claim
   instead of just re-running standard SAC.

2. **Tie RND spikes to entropy temperature, not just reward.** Chapters 1 and 3 describe
   exploration as intermittent and drive-triggered, collapsing to near-determinism
   between critical states. The project's existing RND novelty signal already
   approximates a "critical state" detector — consider using it to transiently raise the
   target entropy / α on novelty spikes, rather than relying solely on SAC's flat
   auto-entropy target.

3. **Keep freezing subsystems when adding DOFs or modalities.** Chapters 4, 5, and 11 all
   describe new capabilities built atop frozen, already-stable subsystems. This matches
   the project's existing practice of warm-starting follow-on stages from a checkpoint —
   the theory's addition is a reason to try literally freezing early encoder/policy
   weights for a few thousand steps before unfreezing, rather than fine-tuning everything
   from step one.

4. **PPO is a legitimate lighter-weight choice only for short-horizon, single-context
   sub-experiments.** Chapter 5's postural correction is the one case where the theory's
   short-latency credit assignment doesn't need a replay buffer. If a narrowly-scoped
   sub-experiment is ever spun off that doesn't involve context-switching or multi-step
   chaining, PPO's simplicity (no target-network drift, no Q-overestimation bias) is a
   reasonable trade — but no chapter makes PPO the *better-fitting* choice, only an
   adequate one.

**Caveat:** this mapping is about theoretical fit, not training stability. PPO's
practical robustness (harder to diverge, no replay-buffer distribution-shift bugs) is a
real engineering advantage the theory doesn't speak to either way — worth remembering if
a specific SAC run turns out unstable, independent of what the theory predicts.

---
*Companion artifact (visual/table version): see chat history for the rendered page.*
