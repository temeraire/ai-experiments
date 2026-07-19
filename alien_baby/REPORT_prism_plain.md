# What we found about the prism experiment — in plain language

*Written 2026-07-11. A jargon-free companion to the FINDINGS / THEORY_LOG entries.*

---

## The one-sentence version

Our creature's earlier "look at the red ball and go to it" ability is **real** — but the
headline claim that it then *re-learned its vision* the way a human does under prism glasses
does **not** hold up. What actually happened during adaptation is that the creature **stopped
using its eyes to steer** and fell back on a fixed body habit. Those are two very different
things, and this report explains the difference and how we know.

---

## First, the words you asked about

**Prism / "offset."** A prism lens shifts where things *appear* without moving where they
*are*. The ball is straight ahead, but through the prism it looks like it's off to one side.
**"Offset" is just the size of that shift, measured in degrees.** Offset 0 = no prism, normal
vision. Offset 30 = everything looks shifted 30 degrees to the side. Offset 60 = shifted 60
degrees. In our simulation we create this by showing the creature a fake "ghost" ball at the
shifted position while the real, touchable ball stays where it is. (That ghost is the red blob
you saw riding at the creature's head that never stopped the episode — it's a visual-only
placeholder with no physical body, so it can't be touched.)

**"Sweep" and "per-offset."** A sweep just means we test the creature at a whole range of
offsets — 0, 15, 30, 45, 60 degrees — and look at how it behaves at *each* one. "Per-offset"
means "broken down by offset value" rather than lumped together. The result is a little curve:
behavior on the vertical axis, prism strength (offset) on the horizontal axis.

**"Flat sweep" — and why it's the smoking gun.** Here's the intuition. If a creature is
*actually using its eyes*, then shifting what it sees should shift what it does. A normally-sighted
creature, as you crank the prism up, gets fooled more and more — it goes toward where the ball
*appears*, so its "picked the right ball" score falls steadily as the offset grows. That's a
**sloped** curve, and it's the signature of vision being in charge. A **flat** curve — the same
behavior at offset 0 and offset 60, no change — means the visual shift made no difference at all,
which means **the eyes weren't driving the behavior**. Our adapted creature's curve was flat
(~51–55% at every offset). The un-adapted one's was steeply sloped. That contrast is one of our
three pieces of evidence.

---

## The heart of it: two stories that produce the same headline number

You described the real prism experiment correctly, so let me build on your description.

**Story A — "genuine sensorimotor recalibration" (what we originally claimed, the human result).**
A person puts on prism glasses. The ball looks shifted left, so they reach left and miss. Over a
few minutes their brain **re-tunes the link between what they see and how they move** — it learns
"when the ball looks *there*, actually reach *here*." Soon they're hitting the ball again, *still
using their eyes*, just with a corrected aim. The proof this really happened is the **after-effect**:
take the glasses off, and now they miss the *other* way for a while, because the correction is
baked in and keeps applying when it's no longer needed. Crucially, **vision stayed in charge the
whole time** — the person never stopped aiming by sight; they re-aimed by sight. That's the
impressive, human-like result.

**Story B — "adaptation abolished vision's directional steering" (what actually happened).**
Instead of re-tuning its vision, our creature did something cruder: it **gave up on its eyes and
started moving by rote.** Before the prism, it genuinely steered toward the red ball by sight.
During prism adaptation it stopped steering by sight altogether and settled into a fixed body
motion — a habitual sweep/roll — that ignores where the ball visually is. It still touches a ball
and sometimes the "right" one, because its habitual motion happens to land it there, but it is
**no longer aiming by sight at all**. Vision dropped out of the loop rather than being re-mapped
inside it.

**A dartboard analogy for the difference:**
- *Recalibration (Story A):* a dart player given left-shifting glasses learns to throw a little
  to the right to compensate — still aiming by eye, just adjusted. Glasses off, they throw too
  far right for a while (the after-effect).
- *Abolition (Story B):* a dart player, fed up with the glasses, stops aiming and just throws the
  same rote motion at the board every time, no longer looking. They still hit occasionally,
  because the board's usually in roughly that spot — but they've stopped using their eyes.

Our creature became the second dart player.

**Why this matters, and why the two look alike at first.** Both stories can produce an
"after-effect" number — the creature mis-reaching one way once the prism is removed. That number
is what we originally celebrated. But in Story A the after-effect is *leftover re-mapping* (real
visual learning); in Story B it's mostly just the *fixed body habit showing through* once vision
stops correcting it. They're the same headline, produced by opposite mechanisms — one where the
eyes learned, one where the eyes switched off. Telling them apart is the whole job, and it's why
we can't leave the strong claim standing.

---

## How we know it's Story B — three independent checks that agree

We didn't rely on one measurement. Three different looks at the data all point the same way, which
is what makes the conclusion solid rather than a hunch.

1. **The body habit was there all along (the "population bias" check).** We measured the
   creature's overall lopsidedness — which side it tends to grab from — across hundreds of trials.
   Then we measured the *same thing with its eyes turned off entirely*. The blind version showed
   about three-quarters of the same lopsidedness. So most of what we'd been reading as a
   vision-driven "after-effect" was actually a pre-existing motor habit that has nothing to do
   with vision.

2. **The flat sweep (explained above).** A sighted creature's behavior changes as we increase the
   prism shift; the adapted creature's didn't change at all across shifts from 0 to 60 degrees.
   Its behavior wasn't being driven by what it saw.

3. **Its movement stopped tracking the ball (the most direct check).** For each trial we compared
   the direction the creature actually moved against where the red ball actually was. **Before
   adaptation, these tracked each other clearly** — when the ball was more to the left, the
   creature moved more to the left (a correlation of about 0.38, where 0 means "no relationship").
   The blind control showed no such tracking (about 0.00), exactly as it should. **After
   adaptation, the sighted creature's tracking dropped to 0.00 too** — identical to being blind.
   In plain terms: the eyes were steering the body before, and after adaptation they weren't,
   at all.

An independent reviewer (our "theory-monitor" check) re-did this from the raw data by a different
method and got the same collapse, and confirmed the numbers are statistically solid (the
before-vs-after difference is far too large to be chance).

---

## The good news, stated plainly

This is a downgrade of one claim, not a collapse of the project. The **underlying ability is
real**: before any prism, the creature genuinely saw the red ball and steered toward it — its
movement tracked the ball's true position, while a blind version did not. So "the creature uses
vision to choose and pursue a target" stands, and it's a genuine result. What we're correcting is
specifically the *interpretation of the adaptation phase*: the creature did not re-learn its vision
the way a human does under prisms — it set vision aside and leaned on a body habit.

---

## What would sharpen this further (one clean next test)

At a 30-degree prism shift, the fake "ghost" ball and the real ball are still fairly close
together (only about 0.4 meters apart), so "went to the ghost it saw" and "went to the real ball"
can't be cleanly told apart. At a **60-degree** shift they separate enough (~0.8 meters) that the
two stories make clearly different predictions. Running that one condition — no new training
needed, about five minutes — would be the cleanest possible confirmation.

---

## Bottom line

- The creature really does use its eyes to choose and chase a target. (Confirmed.)
- It did **not** re-tune its vision under the prism the way humans do. (Corrected.)
- Under adaptation it **stopped steering by sight** and fell back on a fixed body motion — a
  different and less remarkable mechanism than genuine recalibration. (New finding, checked three
  independent ways and by an independent reviewer.)
