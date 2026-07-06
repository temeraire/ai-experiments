# Naming — telling this project as a story

The work we're doing is, at heart, the story of a developing body which is given the tools  to develop its own mind based on that body's interaction with its environment. This means gravity, objects, what is near and  what is far, moving towards and away from things. By learning what it is to have a body in a world, it learns how think and how, eventually, to write out its thoughts via language. The engineering names for what we're
doing — "roll out the policy," "deterministic eval," "ent_coef=0.5" — are accurate, but they hide the thing that's actually interesting: that we are watching something learn what it means to reach out and touch something. It is a process of discrimination as well as generalization: what makes that easy chair different from that stool? And also that they are both chairs. 

This document is the running glossary for the *story* names. When we write a log entry, a session summary, a paper, or a script, we can reach for these instead of the engineering shorthand — or use both, side by side, so the reader sees the machinery and the meaning at once.

The protagonist is **AB** — the alien baby. The world is small and repeatable, made by us, designed to teach. Everything else is something AB does, something we do to AB, or something the world does in response.

---

## The seed

> Engineering: "We roll out the policy and see what it does."
>
> Story: "We see if this alien creature can successfully move, extend an arm, and reach
> out and touch something. We are focused on what AB is learning, and that has a keenly deterministic angle to it: by reaching out and bumping and touching and falling, it learns logic. You can't reach out and touch something if it's far away. You can't lift a car because it's too heavy. You learn about 'can' and 'cannot' each of which is an abstraction from multiple instances -- as is everything else, such as weight, shape, color, texture, smoothness, fragility, etc.

---

## The actors and the world

| Engineering term | Story name | Notes |
|---|---|---|
| agent | AB, the alien baby | The small mind we are coaxing toward life. |
| policy | AB's habits | Its accumulated answer to "when I feel this, what do I do?" |
| environment / sim | the world we built for AB | Small, repeatable, designed to teach. |
| body / MuJoCo model | the body we gave AB | What it has to move with, what it cannot escape. |
| observation | what AB can know in this instant | Body-sense, sometimes vision, sometimes more. |
| action | what AB chooses to do | A pull on each muscle, a twitch of each joint. |
| reward | the world's whisper of yes | The small signal that says "that — that mattered." |
| penalty / negative reward | the world's no | "Whatever you just did, lean away from it." |
| episode | one life | One chance to act before the world resets. |
| timestep | one heartbeat | The smallest unit of AB's experienced time. |

## What we do to AB

| Engineering term | Story name |
|---|---|
| train | rehearse AB — give it many lives so it can begin to find what works |
| roll out the policy | give AB a moment in the world and watch what it does |
| evaluate (deterministic) | take AB's measure — turn off the dice and ask it to act on what it knows |
| checkpoint | a snapshot of AB's mind, frozen so we can come back to it later |
| save / load model | bottle AB's mind / pour it back out |
| warm-start | hand AB a body that already half-knows what to do, instead of starting from nothing |
| ablation (e.g. vision ablation) | take something away from AB and ask: did you ever need it? |
| seed | the hidden coin-flip that decides how AB's first life unfolds |
| curriculum | ease AB into difficulty — start with the world arranged in its favor, then quietly move the prize farther out of reach |
| hyperparameters | the knobs we set before AB is born — the rules of the world it cannot question |

## What AB does

| Engineering term | Story name |
|---|---|
| explore | wander, try things AB has not tried yet |
| exploit | repeat what already worked |
| converge | settle — find one thing that works and stop looking for others |
| plateau | hit a ceiling AB cannot push through with its current strategy |
| collapse / forget | the careful behavior unravels; AB falls back into stillness or noise |
| floor / floor episode | AB gives up — collapses to the ground and lies there as the clock runs out |
| local optimum / attractor | a trap that looks like a destination — a strategy AB falls into and won't leave even though something better exists |
| breakthrough | AB finds something genuinely new — a behavior the old strategy never reached |

## What's inside AB

| Engineering term | Story name |
|---|---|
| neural network / weights | AB's mind |
| CNN / visual encoder | AB's visual cortex — the part of its mind that might learn to see |
| proprioception | AB's body-sense — knowing where its limbs are without needing to look |
| replay buffer | AB's memory of past lives — the experiences it draws on to learn what was worth doing |
| gradient descent / optimizer | the slow tightening of AB's reflexes toward the patterns that have paid off |
| loss | the gap between what AB did and what would have been better |
| entropy coefficient (ent_coef) | AB's restlessness — how much it is paid to keep surprising itself |
| exploration bonus | the reward for fidgeting instead of freezing |
| learning rate | how willing AB is to revise itself each time it learns something new |

## Things that have shown up in this project specifically

| Engineering term | Story name |
|---|---|
| contact reward | the small "yes" when AB's hand brushes a ball |
| offset / ball offset | how far out of easy reach we have placed the prize |
| reach radius | how far AB's body knows how to extend itself |
| cart substrate | the trolley we set AB on so its legs don't have to know how to walk yet |
| vision is not load-bearing | AB's eyes are open but its mind is not looking through them |
| gaze-gated reward | a world where AB only gets the prize if it has actually been looking at it |

---

## Phrase rewrites

A few common engineering sentences and their story-rhythm versions, to show
the pattern in motion.

- "We ran SAC with ent_coef=0.5 and the policy converged on an arm-extension
  strategy."
  → "We turned AB's restlessness up high. After enough lives, it stopped
  freezing and started reaching."

- "Vision was not load-bearing in this substrate."
  → "AB's eyes were open the whole time, but its mind never learned to look
  through them. When we covered the camera, nothing about its behavior
  changed."

- "Warm-starting from a checkpoint with a curriculum target of 0.20 failed
  because the off-policy gradients pulled in opposite directions."
  → "We tried to hand the older, wiser AB a harder world. Its old memories
  told it one thing, the new world told it the opposite, and it ended up
  unable to commit to either."

- "The breakthrough reproduces across three seeds."
  → "We re-rolled the dice that decide AB's first life three times. Each
  time, it found its way to the same answer."

- "Late-training regression: the policy peaks at t=15K and drifts afterward."
  → "AB is brilliant for a moment, then forgets. The peak is real, but it
  doesn't hold."

- "Curriculum target 0.25 is beyond the body's trainable boundary."
  → "We placed the prize too far. AB never bumped into it by accident, so
  it never learned that reaching for it was a thing worth doing."

---

## How to use this document

This is a living glossary. When we coin a new term in the work, add it
here. When we write up a session, reach for these names where they fit
— and where they don't, write the missing ones into this file rather
than reaching back for the engineering term.

Two soft rules:

1. **The story names are not replacements** — they live alongside the
   engineering terms. A log entry might say "ent_coef=0.5 (AB's restlessness
   turned high)." Both are right. The engineering name is auditable; the
   story name is intelligible.

2. **AB is always the protagonist.** When in doubt about how to rename
   something, ask: what is this from AB's point of view? "The optimizer
   updated the weights" becomes "AB's reflexes shifted a little, toward
   whatever just paid off." Keep AB at the center and the rest writes
   itself.
