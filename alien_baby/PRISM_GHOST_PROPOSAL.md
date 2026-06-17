# Prism-Ghost Adaptation — experiment proposal (future / down the road)

*Idea: the user, 2026-06-16. Captured for when vision is directionally load-bearing.*
*Status: NOT YET RUNNABLE — gated on a hard prerequisite (see below). This is the payoff
experiment that the current locomotion/affordance work is building toward.*

## The idea

Simulate the classic **left-right reversing-spectacles** experiment (Stratton 1897; the prism-
adaptation literature) in-silico, using the project's existing **"ghost"** mechanism — a geom that
is **visible but has no physical presence** (`contype=0 conaffinity=0`, already used for the cart
marker).

Setup: a real, solid, touchable target (e.g. a bottle) sits on the subject's **right**. Under a
simulated reversing prism, the head-camera renders the target as a **ghost on the left** (and
suppresses the real target's pixels). The subject:
1. sees the (ghost) target on the left,
2. reaches left — and its hand passes **straight through** the ghost (no mass, no contact, no
   reward),
3. gropes with proprioception, finds the **real** target on the right, and touches it (reward).

Proprioception is ground truth; vision is the modality forced to yield. Over repeated trials the
prediction is that the policy/representation **remaps** so the reversed visual input comes to
drive a reach to the *true* (proprioceptive) side — the in-silico analog of "the visual field
switches back to where the object really is."

## What we measure (objective — not the subjective percept)

1. **Reach-direction flip:** fraction of reaches going to the ghost side vs the true side, over
   training. Does it migrate from visual-left to proprio-right?
2. **Latent remapping (probe):** does the vision latent learn to decode the *true* target
   position from the *reversed* input? This is "the object now appears where it really is,"
   operationalized as a decodable representation.
3. **Negative aftereffect (the clincher):** after adaptation, **remove the prism** (ghost = true
   position). Does the agent now reach *wrong*, to the side it adapted toward? Humans show exactly
   this aftereffect. If AB does too, it is decisive evidence of genuine **recalibration**
   (vision's representation restructured by proprio's learning history = *interpenetration*),
   not mere relearning — and it is something **coordination-only** (Numenta-style voting) could
   never produce.

## Faithful protocol (adaptation, not learn-from-scratch)

- **Phase A — Baseline:** train normal reaching (no prism) until vision drives directional reaches.
- **Phase B — Adaptation:** flip the prism ON; measure how the reach direction + latent remap over
  trials.
- **Phase C — Aftereffect:** flip the prism OFF; measure whether AB reaches wrong (the aftereffect).

Distinguishing A→B→C (true adaptation of an established vision policy) from training-under-reversal
(an agent that never had normal vision) is what makes this the real prism experiment rather than a
relabeled reaching task.

## Why it matters

- **Cleanest in-silico test of proprio-primacy + interpenetration vs coordination.** It is the
  discriminating experiment from the Numenta discussion, made concrete: coordination predicts
  permanent vision-vs-proprio conflict; interpenetration predicts recalibration + aftereffect.
- **Novel.** The crawl-reuse scout found nothing like an RL agent exhibiting prism-adaptation-style
  visual remapping with a negative aftereffect. This would be a genuinely new result.
- It is the **payoff** that justifies the locomotion → affordance → vision-recruited program: it
  gives the crawling work a destination.

## Winnability (consistent with the governing rule)

Every episode is winnable: the agent can always succeed by groping to the real, reachable object.
The ghost is the **perturbation**, not an impossible target — the reach-through-the-ghost "failure"
is the *teaching signal*, while the real object is always reachable. So this respects the rule that
no episode may be unwinnable.

## Hard prerequisite (why it is "down the road")

Requires **vision to be directionally load-bearing** — vision must actually drive the reach, or
there is nothing to remap. It currently does not (representation failure R²≈0.08; likely an
affordance failure because AB cannot pursue). So this is gated on the
locomotion → affordance → vision-recruited chain working first.

## Implementation sketch (when the gate is cleared)

- Env flag `--prism-reversal` (generalize to `--visual-offset <dx>` for an arbitrary vision↔proprio
  displacement, not only a mirror flip — lets us titrate the conflict).
- **Ghost geom:** visible, `contype=0 conaffinity=0` (existing pattern), placed at the mirrored x.
- **Real target:** solid + touchable at true x, but suppressed in the head-cam render (alpha→0 to
  the camera, or a render mask) so vision shows only the ghost.
- **Reward:** contact with the REAL solid object only (the ghost yields nothing).
- **Measures:** per-episode reach-direction log; vision-latent probe vs *true* position; a dedicated
  aftereffect eval block (prism off after adaptation).

## Risks / subtleties

- "Object visually switches back" is the *subjective* analog; our signal is reach-direction +
  latent-remap + aftereffect. Be explicit we measure the mechanism, not a percept.
- Held & Hein: adaptation requires **active, self-produced movement** — which RL reaching provides
  for free (the agent acts, it doesn't passively view).
- The aftereffect test (Phase C) is what separates genuine recalibration from relearning; don't
  skip it.
