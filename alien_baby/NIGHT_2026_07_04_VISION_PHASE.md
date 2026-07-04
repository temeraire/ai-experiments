# Overnight session 2026-07-04 — Vision-as-reinforcement (Stages A/B/C)

Goal: demonstrate that the proprioceptively-learned crawl gait is **reinforced, not
destroyed**, when vision is introduced — vision supplies the ball bearing, the motor
substrate is preserved. Three architectures, all behind one camera fix.

## DONE

### P0 — Camera FOV fix (winnability gate) ✅
- 2026-07-02 preflight blocked the phase: head-cam saw only ±15° of the spawn cone.
- Built `visualization/cam_visibility_preflight.py` — measures red-ball pixels at the true
  32×32 CNN resolution across a bearing sweep; saves an inspectable montage.
- Findings: horizontal exit + arm self-occlusion cap static visibility; tilt (pitch) does
  not change horizontal span; even fovy=150 can't see ±45° (occlusion), and the ball shrinks
  to 1–3 px. The ball sits on the far horizon at the 0.75 m spawn ring.
- **Resolution: fovy 90→120 in `mimo_crawler_pos_wide.xml`, matched spawn cone ±22°.** Ball
  is a clean 4–14 px blob across ±22° in both eyes, cx monotonic with bearing (0.09→0.95).
  Visually confirmed. Wider static cones need camera reposition / body change (future).

### Blind base (Stage C substrate) ✅
- `crawl_ppo_blind_base` — PPO, 1.5M steps, no target_obs (undirected crawl). best_model saved.
- Eval on ±22° cone (`crawler/eval_crawler.py`, also fulfills P1 substrate/capability logging):
  contact 20%, mean_disp 0.219 m, tip_rate 2.5%, mean_toward +0.173 m.
- Video review: Gemini MISREAD it as "fallen on its back / motionless"; direct frame
  inspection shows prone belly-crawl, posture-stable, translating toward the ball. Stats
  trustworthy. **Competent locomotor substrate confirmed** (the lit-scout precondition).

### Literature scout ✅ (findings folded into plan)
- Plan C == "No More Blind Spots" (Duan 2025, arXiv:2508.11929): frozen competent blind base
  + zero-init residual perception head, action = base+residual — DEMONSTRATED to reinforce
  not degrade. Precondition: base must already be a competent locomotor (met).
- Plan A == "Learning by Cheating" (Chen 2019) + DAgger (relabel, not one-shot BC).
- Shape/pose variety STRENGTHENS reach generalization (GraspXL etc. + our Phase XV). Grasp→
  visual-representation emergence is OPEN/risky (don't assume it fixes an inert encoder).

## KEY REFRAME — the confound and the honest floor
On the ±22° cone the teacher gets 77.5% WITH the true bearing AND 77.5% fed ZERO bearing —
but the zero-bearing number is an OOD artifact (the capable teacher does a covering/search
motion). The **honest floor is the blind base's 20%.** So bearing is worth ~57 pts (20→77.5)
on this cone → vision genuinely has a job and B/C have a real reward incentive. Stage A's
verdict rests on **decode-R² (primary)**, not contact rate (confounded on a narrow cone).

## HEADLINE RESULT (Stage A) — REPRESENTATION FAILURE FIXED
**Vision reads ball direction from pixels AND generalizes.** Lateral decode-R²:
- in-sample 0.945/0.952 (BC/after DAgger)
- **teacher-driven (in-distribution) held-out = 0.841** (fwd 0.648, z 0.779) ← the clean number
- student-driven held-out = −0.011 ← NOT a representation failure; pure DAgger distribution
  shift (student drives to states the CNN never trained on; only 1 DAgger round run). Fix =
  more DAgger rounds.

Versus the project's years-long R²≈0.01–0.08 → **0.84**, a ~10× jump. With the winnable camera
(fovy120, ±22°) + supervised pressure, pixels DO encode ball direction and the CNN generalizes
it on-distribution. Contact rate is confounded on ±22° (teacher-zero is an OOD 73%); R² is the
clean metric; honest floor is the blind base's 20%, ceiling 73–77.5%.

**Why this de-risks Stage C specifically:** PPO is on-policy → the residual CNN trains on the
states the policy itself visits, so the DAgger distribution-shift gap that hurt Stage A's
student-driven number does NOT apply. Stage C's residual sees its own on-distribution states.
Caveat to watch: no image augmentation (DrQ random shift) yet — add if C overfits.

## STAGE C RESULT (residual on blind base, 300K, reward-only) — NOT DESTROYED, not yet REINFORCED
Clean eval (±22°, 30 eps): contact **20.0%** (= blind floor) | pixels-ablated 3.3% | ablation
gap +16.7 pts | substrate PRESERVED (disp 0.235 vs blind 0.219, tip 3.3% vs 2.5%, speed intact).
Reading: the gait is **not destroyed** (substrate held), and vision is **integrated** (ablation
gap), but **inert** — no contact lift over the 20% blind floor. Because Stage A already proved the
representation exists (R²=0.84), this isolates the failure as **policy-gradient, not
representation**: reward alone on ±22° (forward-crawl already gets 20%) is too weak a signal to
grow the steering, even though the direction is decodable in the pixels. The project's chronic
"integrated but inert" is now explained: representation was never the blocker under a winnable
camera; connecting it to behavior by reward alone is.

## STAGE B RESULT (slot-fill on frozen teacher, WARM-STARTED from Stage A CNN) — WORKING
distill-then-RL: vision head initialized from Stage A's bearing CNN (R2=0.84), so vision reads
direction at init; frozen teacher (gait + steering preserved) gets a good bearing immediately.
**CLEAN EVAL (50K best_model, ±22°, 30 eps) — THE DEMONSTRATION:**
- CAPABILITY: **contact 63.3%** (pixels-ablated **20.0%** = exactly the blind floor). Vision
  load-bearing gap **+43.3 pts**. mean_toward +0.218. Vision lifts 20→63% toward the 73–77.5%
  ceiling; removing vision degrades GRACEFULLY to the preserved proprio baseline, never below.
- SUBSTRATE: tip 6.7% (upright ~93%, low), and travel is ENHANCED — disp 0.504 m (blind 0.219),
  speed 2.14 mm/step (blind 0.30): the creature crawls purposefully to the ball.
- GENERALIZATION battery: radius 0.55–0.65 → 80%; 0.85–0.95 → 50% (graceful w/ distance);
  1.0–1.1 → 20% (floor, far beyond training; one broken-episode outlier in mean_toward);
  cone ±15° → 75%. Generalizes across distance and cone width.

**CORRECTION (later same night): the Stage B behavioral win was CONFOUNDED — walked back.**
A polished 300K eval went net-negative (55% pixels vs 75% ablated). Diagnostic (diag_confound.py):
FULL ±22° true 82.5% vs zero 75.0% (bearing worth +7.5 pts); LATERAL band ±15–22° true=zero=65%.
The forward-crawl reach envelope (~0.28 m) covers the whole camera-visible ±22° cone, so the bearing
is behaviorally nearly irrelevant wherever vision can see. The "+43 pt gap" was an unstable-g(0)-
baseline artifact; the fair no-vision control (teacher+zero bearing) is 75%, above Stage B's 55–63%
→ vision did NOT reinforce behavior; it slightly hurt (noise). RL fine-tuning also degraded the
distilled encoder. **What stands: Stage A representation (R²=0.84) — real, unconfounded.** Behavioral
reinforcement is UNPROVABLE on ±22° (winnable cone ⊄ vision-necessary cone). Fix = HEAD-SEARCH phase
for a winnable >±30° spawn forward-crawl can't solve. See THEORY_LOG/FINDINGS corrections.

## RUNNING / QUEUED
- Stage A (`crawler/stage_a_distill.py`): distill bearing from pixels into frozen teacher,
  DAgger; primary metric lateral decode-R² (>0.30 = representation failure FIXED).
  BC R²=0.945 — PASS. DAgger + final held-out eval RUNNING.
- Stage C (`crawler/train_stage_c.py` + `residual_vision_policy.py`): zero-init residual on
  frozen blind base, PPO reward, ±22° cone, 1M steps. RUNNING (smoke passed).
- Stage B (`crawler/train_stage_b.py` + `slotfill_vision_policy.py`): reward-grow vision into
  frozen teacher's bearing slot. CODE READY, queued (launch when resources free).

## Files added this session
- visualization/cam_visibility_preflight.py
- crawler/eval_crawler.py
- crawler/stage_a_distill.py
- crawler/residual_vision_policy.py + train_stage_c.py
- crawler/slotfill_vision_policy.py + train_stage_b.py
- XML: mimo_crawler_pos_wide.xml (left_eye/right_eye fovy 90→120)

## Next (if a stage succeeds): generalization — move target (kept winnable), vary shape
(torus/cone/column), add grasp DOF. Then the prism-ghost aftereffect (PRISM_GHOST_PROPOSAL.md).

## HEAD-SEARCH PHASE — set up + launched (the fix for the ±22° confound)
Prechecks (both mandatory, both passed):
- HS-1 (diag_confound.py): bearing becomes NECESSARY at ±68° — teacher true 72.5% vs zero 52.5%
  (+20 pts), vs only +7.5 at ±22°. Forward-crawl reach is finally exceeded. Target cone = ±68°.
- HS-2 (cam_visibility_preflight --head-swivel-deg): GOTCHA — when PRONE, head_swivel ROLLS
  (its axis points forward), not yaws; head_tilt_side is the pan/yaw joint. ±60° head-yaw brings
  ±75° of world into view → wide cone is winnable by turning the head. BUT head_tilt_side had NO
  actuator → added `act:head_yaw` (nu 25→26, appended last so crawl-pose indices unchanged).
Setup: `train_head_search.py` — vision PPO on ±68° cone (`mimo_crawler_pos_wide_hs.xml`),
StereoCrawlerCNN encoder WARM-STARTED from Stage A (R²=0.84), entropy 0.01 for search
exploration, arms_fwd pose, approach+contact reward. Smoke passed; untrained preflight frames
confirm prone body + ball spawning off-cone (~−68°) on the platform (reachable) + sane cameras.
Launched 2M steps (`head_search_v1`). HONEST SCOPE: this is the hard follow-on — gait + search +
steering are learned together (action side from scratch; only the vision encoder is warm-started),
so a first run is exploratory. Monitoring via checkpoints; success = contacts above the zero-bearing
baseline WITH vision, i.e. the first behaviorally-necessary vision win.

### HEAD-SEARCH progress (head_search_v1, 400K/2M) — LIFT-OFF
Deterministic eval reward: ~0 through 275K (from-scratch gait+search learning phase), then
300K→20, 350K→61, 375K→**142**, 400K→100 (7 New-best events, still improving). That 142 matches
Stage B's rewards on the EASY ±22° cone — but achieved from scratch on the HARD ±68° cone where the
ball spawns off-view. Bimodal (±90 std) = some episodes solved, some not; mean rising 0→140 is real
learning: the creature is starting to search, turn toward off-cone balls, and reach them. Liveness
PASSED (body_motion 1.61). CRUCIAL TEST PENDING: clean contact + pixel-ablation eval at ~800K — if
contacts collapse when pixels are zeroed, that is the project's FIRST behaviorally-necessary vision
win (vision load-bearing where forward-crawl provably fails).
