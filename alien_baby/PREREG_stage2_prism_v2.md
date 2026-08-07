# Stage-2 prism recalibration — v2 (fixed rig) + proof that AB registers the "prism-on" cue

**Status:** DRAFT protocol, 2026-07-19. Supersedes the v1 overnight run (`s2v2_*`), which did not
pass: the teaching signal never switched on and the after-effect ruler was broken (see FINDINGS
2026-07-19). v2 fixes the equipment first, then adds the required "prism-on" cue and a battery that
proves AB actually feels it. Every result here to be reported in plain English (per CLAUDE.md).

---

## Plain-English recap of why v1 failed

Two pieces of our own equipment let us down before AB got a fair test:
1. **The teaching signal never turned on.** The "how wrong is the eye's guess?" score (the mismatch
   aux loss) stayed flat the whole run — the eye was never actually pushed to learn the correction.
2. **The after-effect ruler is broken.** Our tool for detecting mis-reaching after the lens comes off
   fails its own self-check, so its numbers are noise.

So v2 is: fix the ruler, fix the engine, and only then ask whether AB re-aims — this time with the
"prism-on" cue he is owed, and a test that proves he registers it.

---

## Fix 1 — repair the after-effect ruler FIRST (free, no training)

`eval_aftereffect_inview` must pass its own self-check before any after-effect number is trusted:
base correlation ≈ +0.5 (not −0.26) and base signed bias ≈ 0° at zero offset (not −109°). The data is
already on disk, so this is a debugging job, not a compute job. Until the ruler reads a known-zero
situation as zero, it cannot be believed on anything else. This blocks everything downstream.

## Fix 2 — make the engine turn (two candidate drives)

The eye only re-aims if something pushes it to. Two candidate pushes, to be tried in order:

**Candidate A — the "prism-on" cue + interleaved practice (Taylor's schedule).** Give AB the on/off
flag (below) and train with the lens **interleaved**: on for some episodes, off for others, the flag
tracking it. In plain terms, this is Taylor wearing the glasses mornings and taking them off after
lunch — each state gets its own practice, so the flag actually means something and AB can learn a
separate calibration for each. This is the first thing to try, because it is also what the governing
rule now requires.

**Candidate B — the "dizziness that fades" drive (fallback, David's idea, grounded in biology).**
When a person first puts on displacing lenses, the clash between what they see and what their balance/
body sense says makes them dizzy and sometimes nauseous; as they train, that clash shrinks and the
dizziness fades. That felt sickness is the *conflict itself* being experienced as something the body
wants to be rid of. Translate it: make the visual-vs-touch mismatch an **aversive cost** — a per-step
discomfort AB is motivated to reduce — rather than only a quiet supervised target. Where the v1 aux
loss trained the eye off to the side and never coupled to what AB was trying to achieve, this puts the
mismatch *into* what AB is trying to maximize, giving him a reason to change. As he adapts (the eye
learns to subtract the 30°), the mismatch shrinks and the discomfort fades — the model of the nausea
going away.
- **Why it might succeed where v1 failed:** it converts a passive teaching target into an active
  motive; the creature is driven to make the bad feeling stop.
- **Failure mode to guard against (important):** real nausea makes you want to STOP moving and look
  away — and AB could "escape" the discomfort by closing/averting his eyes or freezing, which kills
  the very movement adaptation needs. The reward must be built so the ONLY way to reduce the discomfort
  is to ADAPT, not to disengage — the discomfort cannot be dodged by doing nothing (ties to the
  realm-of-possibility rule: "do nothing" must never be the smart move).

---

## The "prism-on" cue (required by governing rule, 2026-07-19)

A single **binary flag appended to AB's body senses (proprioception): 1 every step the lens/ghosts are
active, 0 otherwise.** On/off only — never the offset angle, which would hand AB the answer instead of
making him learn it. This is the translation of Taylor's "glasses-on" sensation (frames on the nose,
narrowed field), which he modeled as a simple on/off afferent. Implementation: one extra observation
dimension in `mimo_crawler_env.py`, set from the prism state each step.

---

## The proof that AB actually registers the cue (the awareness battery)

Adding a bit to AB's senses does not mean he uses it. A signal is only "felt" if his behavior changes
when it changes. **Precondition: this battery is only meaningful on a policy TRAINED under a varying
flag** (Candidate A) — a flag that never varied during learning carries no information and AB will
rightly ignore it. Tests, cheap to conclusive:

1. **Flip sensitivity (cheap).** Freeze a moment, flip only the flag (1→0), change nothing else,
   measure how much AB's action changes. Zero = ignored; nonzero = registered. (The cousin of our
   vision-ablation-sensitivity check.)
2. **Same-scene counterfactual (the human situation itself).** Show AB the *identical* picture — same
   ball, same everything — once with the flag "on" and once "off," and check his AIM differs, in the
   direction that matches correcting for the lens. Same eyes, different reach, purely because he knows
   the lens is on. **This is the core proof.**
3. **False-flag / lie to him (causal proof).** Put the lens on but set the flag to "off." If he now
   mis-reaches as though there were no lens, he was leaning on the flag, not just the pixels. Do the
   reverse too. A creature that can be fooled by a false cue is a creature that was using the cue.
4. **Instant switch = full dual adaptation (the complete demonstration).** He reaches correctly BOTH
   with the lens on and off, and flips the instant the flag flips. That is the entire human phenomenon
   in one behavior: glasses on → immediately account for them; off → immediately account for that.

**Secondary (footnote, not the bar):** linearly decode the flag from AB's hidden activations to confirm
the information is represented inside him. We have been burned by signals that are represented yet do
not drive behavior, so the *behavior* tests above are the real proof; decoding is corroboration only.

**Plain-English pass criteria.** He registers the cue if: flipping the flag changes his aim (Test 1);
on the same picture his aim shifts toward "corrected for the lens" when the flag says on (Test 2); a
false flag makes him mis-reach in the predicted direction (Test 3); and — the strongest — he holds both
calibrations and switches instantly (Test 4).

---

## What Taylor predicts (plain English)

- **Cue present + both states practiced → dual adaptation:** AB holds the lens-on and lens-off
  calibrations at once and switches instantly, with **no after-effect** when the lens comes off. This
  becomes the target, and Test 4 is its proof.
- **Cue withheld on purpose → the classic result:** a single overwritten calibration and a real
  after-effect (mis-reaching the other way) on removal. Valid to run, but only by deliberately leaving
  the cue out and saying so.
- **The flag only means anything if both states were practiced** — Taylor's intermittent wear. A lens
  that is always on (flag always 1) teaches AB nothing about the flag.

---

## Order of operations

1. Repair + re-validate the after-effect ruler (free). Gate: it reads a zero situation as zero.
2. Add the binary prism-on flag to the observation (one dimension).
3. Train Candidate A (cue + interleaved lens on/off). If the engine turns (mismatch drops, aim
   improves under the lens), run the awareness battery + the dual-adaptation switch test.
4. If Candidate A's engine still stalls, add Candidate B (the dizziness/aversive-mismatch drive),
   with the anti-escape guard, and repeat.
5. Report every number in plain English, tied to Taylor's predictions.
