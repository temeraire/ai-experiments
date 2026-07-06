# Vision Phase Proposal (experiment-strategist, 2026-07-02)

**Context:** Directional crawling to a target is SOLVED using a PRIVILEGED ball-direction
observation (`target_obs`). PPO on a 360° spawn reaches the ball 58% (seed 0) / 78% (seed 1),
and ablating the bearing collapses it to 12%/32% with the creature crawling AWAY — the signal
is proven load-bearing. **The core project question, now cleanly well-posed for the first time:
can the creature read the ball's direction from its CAMERAS and drive the same turn-and-crawl,
replacing the privileged bearing?** Benchmark to fill: 58–78% (privileged) vs 12–32% (blind).

## Recommendations

**1. Architecture — PPO + small CNN, mono camera.** PPO is the only algorithm that gave stable
directed crawling; SAC (which the old MICOA vision path requires) was abandoned as unstable on
this task. Reuse `StereoCrawlerCNN` (`crawler_cnn_extractor.py`, mono/stereo auto-detect,
PIXEL_LATENT_DIM=64) as a PPO `features_extractor`. Mono = single forward cam, halves render cost.
**Gotcha to handle:** PPO's VecNormalize `norm_obs=True` would corrupt the pixel block — for the
vision run use a Dict obs + MultiInputPolicy (clean) or `norm_obs=False` (quick). One contained
code change to `train_crawler_ppo.py`: a `--vision/--mono` path.

**2. Curriculum — straight-to-vision on a 90° cone (= camera FOV).** NOT 360° (balls behind aren't
visible → unwinnable without a learned search-turn → too many things to learn at once). NOT 180°
(forward-crawl alone solves ~47% → masks whether vision does anything → the exact confound that
made the 180° result untrustworthy). Camera fovy=90 (~±45°), so a **90° spawn cone keeps every
ball in frame (winnable, no search) while placing balls off-center enough that crawling straight
forward misses the lateral ones** → lateral steering (reading left-vs-right from pixels) is the
only way to catch them. Smallest winnable task that still forces vision to be load-bearing.
Deliberately NOT proposing aux-decode-loss (failed in R49, lateral R²=0.010) or privileged→vision
distillation yet — straight-to-vision tests the actual open question: is the new behavioral payoff
alone enough to make vision develop.

**3. Winnability** — the 90°=FOV choice guarantees the ball is always in the forward-down camera,
so every episode is winnable without a search action. (Head-swivel/tilt are policy-controlled
motors, so a later 360° "must-search" phase is possible — the hard follow-on, not the first test.)

**4. Success metric** — NOT contact rate alone (forward-crawl gets some). Require BOTH:
- **Vision-ablation gap ≥25 pts** (zero the pixel columns at eval → contacts collapse), mirroring
  the 58→12 `target_obs` ablation.
- **Ball lateral-bearing decode probe from the CNN latent, R² > 0.30** (the bar R49 set and missed).
Both together = vision genuinely reads and uses direction, for the first time in the project.

## The runs

**Preflight (minutes, MANDATORY):** render the untrained mono head-cam for ~5 ball placements
across the 90° cone; confirm the ball is a clear colored blob in-frame from the prone pose at
center AND both edges. If not in-frame (camera tilts ~15° down; a ball at 0.7 m may sit below the
frame) — STOP and fix camera geometry before any training.

**Smoke run (short-first):**
```
python -m alien_baby.crawler.train_crawler_ppo --vision --mono \
  --spawn-cone-deg 90 --spawn-radius 0.70 0.80 --max-steps 1000 \
  --steps 500000 --n-envs 16 --seed 0 --run-tag crawl_ppo_vision_cone90_smoke
```
(Requires the `--vision/--mono` code path first.) Success = no crash, obs/CNN forward pass work,
liveness passes at 10K, ep_rew_mean rising by 500K. Real success signal (follow-on longer run) =
the vision-ablation gap + decode probe above. Est. 3–6 h for 500K (render-bound; why smoke first).

## Biggest risk
Vision failing to become load-bearing — the outcome of EVERY prior vision attempt (Phases IX–XVI:
"integrated but inert," "non-directional"). **Why this time is different:** the cart era had no
directional action, so even a perfect encoder had zero behavioral payoff and the RL gradient had
no reason to build directional visual features. Now there IS a proven directional action with a
measured payoff gap (58 vs 12). For the first time, "read the ball's bearing from pixels" pays off,
so the gradient finally has a reason to shape vision. If it STILL fails under that incentive, that
is a far more informative result than the earlier nulls — it isolates the failure to
representation/optimization rather than "vision had no job."

## Files
- `train_crawler_ppo.py` (MlpPolicy-only; needs `--vision/--mono` CNN path)
- `crawler_cnn_extractor.py` (StereoCrawlerCNN — reuse)
- `mimo_crawler_env.py` (vision/stereo/target_obs/spawn_cone; pixels packed flat, /255)
- `mimo_crawler_pos_wide.xml` (left_eye/right_eye fovy=90, ~15° down-tilt; head_swivel/tilt motors)
- `train_crawler.py` (SAC/MICOA vision path — reference only)
</content>

---

## PREFLIGHT RESULT (2026-07-02) — WINNABILITY BLOCKER, do not launch as-is

Ran the mandatory head-cam visibility check (mono left_eye, 64×64, settled arms_fwd crawl pose,
ball at 0.75 m across the cone). **The ball is only reliably visible within ~±15° of forward, NOT
the ±45° the fovy=90 spec implied:**

| Ball bearing | Red pixels in left_eye | In frame? |
|---|---|---|
| −45° | 0 | NO |
| −22° | 1 (edge) | barely |
| 0° | 59 (at horiz 0.61, right-of-centre) | yes |
| +22° | 0 | NO |
| +45° | 0 | NO |

**Conclusion:** the proposed 90° spawn cone is **NOT winnable** — for most bearings the ball is out
of frame, so vision cannot possibly solve it (governing winnability rule). The effective horizontal
visible cone is ~±15°, and it's off-centre (left_eye offset). **The vision phase needs a
camera/task fix before any training.** Options (a human design call):
1. **Widen the head-cam FOV** (XML: raise fovy / widen the lens on left_eye/right_eye) so a ±45°
   cone is actually visible — keeps the task hard (real lateral steering) and winnable.
2. **Narrow the spawn cone to ~±15° (30°)** to match the current camera — cheapest, but a 30° cone
   is close to where forward-crawl starts to work, reviving the confound risk (needs the
   vision-ablation gap to disambiguate).
3. **Head-search:** let/require the policy to use head_swivel to bring off-axis balls into view —
   adds a search sub-task (harder, but the most "infant-like").
4. Re-check camera tilt/height: the 0° ball rendered right-of-centre and only 59 px — the down-tilt
   and left_eye offset may also need centring so a forward ball sits mid-frame.

Recommended: **option 1 (widen FOV) + re-run this preflight** until the ball is a clear centred blob
across ±45°, THEN the strategist's 90° cone smoke run becomes valid. Do NOT run vision training
until the preflight passes.
