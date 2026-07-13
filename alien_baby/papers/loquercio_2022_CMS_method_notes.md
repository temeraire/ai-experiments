# Loquercio, Kumar & Malik (2022) — Learning Visual Locomotion with Cross-Modal Supervision (CMS)

Method notes read directly from the full PDF (`loquercio_2022_cross_modal_visual_locomotion.pdf`,
arXiv:2211.03785). Project + videos: https://antonilo.github.io/vision_locomotion/.
Relevance to AB: this is the field's working version of **"use the body to teach the
eye,"** AND it runs an explicit **prism-adaptation test** on the robot — so it is
close prior art for BOTH our mismatch signal and our prism direction.

## The problem they solve
A quadruped (Unitree A1) walks over hard terrain (stairs to 19 cm, curbs to 20 cm,
35° slopes) using **only a monocular RGB camera + proprioception**. Simulating RGB
photorealistically is hard, so they refuse to train vision in sim. Instead: train
the *action* policy in simulation (where privileged terrain info is free), and train
the *vision* module **in the real world** from the robot's own sensors — no labels,
no motion capture.

## The core trick — Cross-Modal Supervision (the part to steal)
**Proprioception is a delayed, accurate teacher for vision.** "I see a point A ahead
of me whose height is currently unknown. But when my feet get to A, its height can be
inferred from my joint angles." So the terrain the camera saw a moment ago is
measured, later, by the body walking over it. Train the vision module to predict the
**future proprioceptive estimate** of the terrain from the **current** egocentric
image, by minimizing the **CMS error** (vision's prediction vs. the later
proprioceptive measurement). This is a **time-shifted** self-supervision signal — no
external ground truth.

**Bootstrapping (chicken-and-egg):** to get a vision system you must first walk to
collect data — but a *blind* policy can already walk (clumsily) using proprioception
to feel the terrain under its feet. So start blind, collect data, train vision, and
the vision-based policy then collects *better* data → lifelong improvement loop.

## Architecture / training pipeline
1. **Two policies trained in simulation** (model-free RL):
   - `π_blind(x_t, z_t, γ_t)` — no lookahead; walks by feeling terrain under the feet.
   - `π(x_t, z_t, γ_t, γ_{t+Δt})` — adds a **15 cm lookahead** terrain estimate.
   - `x_t` = state `[joint pos q_t, joint vel q'_t, prev action a_{t-1}, z_t, γ_t]`.
   - `z_t = μ(e_t)` — 8-dim **extrinsics** latent (payload, motor strength, binarized
     foot contacts, linear velocity, friction).
   - `γ_t = δ(h_t)` — latent of ground-truth terrain height under the feet;
     `γ_{t+Δt} = δ(h_{t+Δt})` — terrain 15 cm ahead.
   - Encoders `μ, δ` and `π_blind` trained jointly end-to-end with RL; then **frozen**,
     and `π` trained on top. Policies are small MLPs (hidden [256,128] / [128,128]).
   - Actions = target joint angles at 100 Hz → torques via PD controller.
   - Sim = RaiSim; RMA-style domain randomization for sim-to-real of the blind policy.
2. **Deploy `π_blind` in the real world**, collect `D = ([I_t, x_t], γ_t)` —
   egocentric frames `I_t`, proprio state `x_t`, proprioceptive geometry estimate `γ_t`.
3. **Train the vision estimator** `g_i` (supervised, on `D`, by CMS error) to predict
   `z_t` and the **future** geometry `γ_{t+Δt}` from vision:
   - Input = latest **3 grayscale frames** (camera 15 Hz) concatenated.
   - Backbone = **ShuffleNet-V2**; take the last layer *before* global-average pooling
     to keep spatial info; project to a 2-channel space with 1-D convs; MLP [128,64].
   - Plus a **50-step history of IMU (roll, pitch)** + desired velocity commands →
     128-dim embedding, concatenated with the ShuffleNet features → final MLP.
4. **Lifelong learning:** keep collecting `D` during execution; keep training the
   visual lookahead predictor. Vision-collected data is higher quality than
   blind-collected (fewer stumbles), so it compounds. Result: 4 stair setups go from
   ~40–60% (blind) to **100%** success with **<30 min** of real data over ~4 days.

## The prism-adaptation test (their §V "Visual Plasticity") — close prior art for AB
They perform "the widely-known prism test" on the robot. Instead of prism wedges they
**rotate the camera ~30° about its yaw axis**, making the terrain ahead nearly
unrecognizable in the image. Classic four-phase protocol:
- **Pre-test:** normal camera, walks well.
- **Exposure:** camera rotated → policy stumbles and drifts while climbing.
- **Adaptation:** fine-tune **only the last 3 layers of the γ predictor for 10 epochs**
  by minimizing CMS error; after ~3 trials (~**80 seconds** of data) the vision module
  accounts for the systematic bias and the robot walks straight again.
- **Post-test:** camera restored to nominal → re-adjusts back over ~2 trials.

**How this differs from AB's prism work (why AB still has room):**
- It is a **camera yaw rotation**, not a lateral **displacement** of the visual field.
- Recalibration is **supervised fine-tuning of a terrain-height predictor**, not
  end-to-end RL, and only the top layers move.
- They **do not quantify a negative aftereffect** — post-test recovery is noted as
  "~2 trials," not measured as an overshoot.
- They **do not distinguish recalibration from strategy-substitution.** AB's whole
  session was about exactly that distinction (heading-tracks-ball vs blind control;
  residual slope across an offset sweep). That measurement is not in this paper.

## What to actually borrow for AB
1. **The time-shift.** Our mismatch signal currently uses contact at the moment the
   body reaches the ball. Loquercio's insight is that *the image from a moment earlier*
   is what should be supervised by *the measurement that arrives later* — align the
   teaching signal to the frame that predicted it, not the frame at contact.
2. **Bootstrapping from the blind policy** is legitimate and expected — mirrors our
   Stage-1 "learn to move by feel first."
3. **Freeze the perception encoders, train the policy on top** (their δ, μ frozen
   before π) — the same separation-of-optimizers discipline as our `MismatchPPO`.
4. **Framing:** cite this as the established form of "use the body to teach the eye,"
   and position AB's contribution as the recalibration-vs-habit distinction +
   quantified aftereffect + RL-from-pixels, not the cross-modal signal itself.
