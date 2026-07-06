# FUTURE PHASE PROPOSAL — Touch as Information (cross-modal object localization)

*Proposed 2026-07-06, during the decoy-discrimination phase. Origin: DW's observation that a
human bumping a hip on a table corner, or a head on the underside of a table, is not "success"
at anything — it is incidental tactile* information *that updates a spatial model of the world,
which then guides behavior seconds later. AB currently has no way to have that experience.*

## Status quo: touch is a success signal, not an information channel

Today, contact with a ball (any body part — hand, hip, head, foot all count identically) does
exactly two things: delivers reward (+200 red / −5 blue decoy) and **terminates the episode**.
Three consequences:

1. A bump cannot inform ongoing behavior — the episode ends at the moment of first contact.
2. The policy has no tactile observations at all. Contact reaches the agent only through
   reward and termination; its inputs are joints + pixels. (MIMo ships full-body touch
   sensors; we have never wired them into the obs.)
3. The policy is stateless — even a felt bump would be forgotten one step later
   (`memory_obs` exists but is off in this phase).

This design was deliberate: making any-contact terminal+rewarded is what kept early phases
winnable, and making wrong-contact terminal+penalized is what closed the touch-search escape
and produced the first clean vision result (choice accuracy 63.2% at 2M → 78.2% at 3.3M,
blind pinned at the structural 50%).

## The proposal: make touch informative

A phase in which incidental contact teaches object position, the way it does for a crawling
infant:

- **Tactile observations ON.** Wire MIMo's touch sensors (or a minimal per-body-region
  contact-flag vector) into the observation alongside proprio + pixels.
- **Wrong-touch non-terminal.** Bumping the decoy no longer ends the episode (keep a small
  penalty or make it neutral); bumping it becomes *evidence* — "the blue one is HERE."
- **Memory ON.** At minimum the existing `memory_obs` contact flags; better, a small recurrent
  policy, so a bump can influence behavior beyond the next step.
- Optionally: multiple objects / occluded objects, so tactile position knowledge has real work
  to do that vision cannot trivially shortcut.

**The scientific question:** do touch-derived and vision-derived object positions converge on
a shared spatial representation (probe: decode object location from the latent in
touch-only, vision-only, and both conditions), or does the policy maintain two disjoint
strategies? This is the Π-interpenetration thesis in its most concrete form yet.

## Why NOT now (sequencing)

Non-terminal informative touch **partially reopens the touch-search escape** at the tactile
level: a blind policy can adopt "bump a ball; if the episode continues it was blue; go find
the other one" — a legitimate tactile discrimination strategy that solves the decoy task with
zero vision. Realistic (infants absolutely do this), but it would destroy the clean
"above-chance choice = vision" metric exactly while it is finally producing results, and would
muddy the prism-displacement experiment that the discrimination line exists to unblock.

**Prerequisite:** run this phase only after the prism aftereffect question is answered on a
strong visual discriminator.

## Metrics to design in from day one (lessons of this week baked in)

- Give the blind baseline every chance to cheat FIRST: measure a touch-only (pixels-ablated)
  policy's performance ceiling on the task before crediting vision or fusion. Expect it to be
  high — that is the point of the phase, not a confound, but it must be quantified.
- Choice/localization accuracy, not contact rate or training reward (reward was a poor proxy
  twice this week).
- Per-episode logging of which modality supplied the decisive information (e.g., did a decoy
  bump precede a correct red approach?) — the tactile analogue of the gaze-choice diagnostics.
- Latent probes for object position under modality ablations (the representation-failure vs
  policy-failure fork, applied to touch).

## Winnability / liveness notes

Unchanged from current phase (same body, arena, spawn envelope). Non-terminal wrong-touch
makes episodes strictly easier to survive, not harder. Liveness gate stays on.
