# Alien Baby

**Teaching AI the difference between the *word* "ball" and the thing you'd actually catch.**

AI agents hallucinate because their concepts were never anchored in anything. "Ball" to a language model is a token that co-occurs with other tokens. To a toddler, "ball" is what rolls, what you catch, what you failed to catch and had to chase under the couch. The word is the last layer, not the first.

Alien Baby is a perception-grounding experiment. A simulated creature learns the concept *before* the word — through action, consequence, and survival — and we measure whether the representation it ends up with is the kind of thing language could later attach to.

## The setup

- A creature (torso + two arms + pan/tilt head with camera) lives on a 2m × 2m elevated platform.
- It's hungry. It has to find food — partly by feel, eventually by sight. Falling off the edge kills it.
- It learns in stages: **proprioception first** (groping in the dark, body before world), then **vision added** (seeing the world it already knew by touch).
- Gravity is the pencil tap. Hunger is the steady tap. The physical world corrects its mistakes — no reward shaping, no human labels.

## If you only click one thing

→ **[FINDINGS.md](FINDINGS.md)** — the full experimental arc (v1–v8) with real numbers, failures, and what we learned from each.

## Headline result (v5)

When we add vision to the already-trained proprioceptive agent — with a consistency loss keeping vision on proprioception's manifold — we measure **CKA = 0.998** between Stage 1 and the vision-extended agent on proprioceptive inputs. In plain English: adding vision didn't *displace* the earlier sense. It was built *on top of* it. The representational fabric stayed intact while new information got woven in.

That's the property language will eventually need to attach to.

## What's in this directory

- `envs/` — MuJoCo environments: tabletop arm (v1–v7), platform creature (v8)
- `agents/` — training scripts, staged developmental RL (v1 through v8)
- `FINDINGS.md` — the experimental record
- `GLOSSARY.md` — quick definitions for the terms in FINDINGS
- `tests/` — evaluation scripts, including CKA and neighbor-consistency probes
- `visualization/` — render scripts that produce split-screen video (overhead + head cam)

## Status

Research experiment, not a library. Actively training. v8 (platform creature with survival stakes) is running as of the submission date. Results will land in FINDINGS as they come.

## Who this is for

Builders who've hit the hallucination wall and suspect that *more training on text* isn't going to fix it. This is an early piece of a grounding substrate — small-scale and obviously so — that agents could eventually hook into.
