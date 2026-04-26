## Bug Fix: Compare Mode Not Running All Selected Models

### Issue
When in compare mode with 6 models selected, pressing Enter only runs one model (claude-sonnet-4-5-20250929) instead of comparing all 6 models.

### Root Cause
The `handleKeyPress` function in frontend.py (line 474-477) always calls `sendMessage()` when Enter is pressed, regardless of whether the user is in compare mode. It should check the `compareMode` state and call `compareModels()` instead when in compare mode.

**Current code:**
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();  // Always calls single model
  }
};
```

### Fix
Update `handleKeyPress` to check `compareMode` and call the appropriate function:
- If `compareMode === true`: call `compareModels()`
- If `compareMode === false`: call `sendMessage()`

### Implementation Plan
- [x] Read and understand the code
- [x] Identify the bug location (frontend.py:474-477)
- [x] Create plan in TODO.md
- [x] Commit and push current changes
- [x] Create new git branch for this bug fix (fix/compare-mode-enter-key)
- [x] Update `handleKeyPress` function to check compareMode
- [x] Flask app restarted successfully
- [ ] Manual testing required (user to test in browser)
- [x] Update TODO.md review section
- [ ] Create pull request

### Changes to Make
**File**: `frontend.py` (lines 474-477)

Change from:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();
  }
};
```

To:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    if (compareMode) {
      compareModels();
    } else {
      sendMessage();
    }
  }
};
```

### Testing
- [ ] Start app and enter compare mode
- [ ] Select 6 models
- [ ] Type a prompt and press Enter
- [ ] Verify all 6 models run in parallel
- [ ] Exit compare mode and verify single model still works

---

## REVIEW: Compare Mode Enter Key Fix (2025-12-05)

### Issue
When in compare mode with multiple models selected, pressing Enter would only run a single model (the default model) instead of running all selected models in parallel comparison.

### Root Cause
The `handleKeyPress` function in frontend.py did not check the `compareMode` state. It always called `sendMessage()` which runs a single model, even when the user was in compare mode and had selected multiple models.

### Changes Made
**File**: `frontend.py` (lines 474-483)

**Before**:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();
  }
};
```

**After**:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    if (compareMode) {
      compareModels();
    } else {
      sendMessage();
    }
  }
};
```

### Impact
- **Minimal code change**: Only modified the `handleKeyPress` function to add a conditional check
- **No features removed**: All existing functionality preserved
- **Better UX**: Compare mode now works as expected when pressing Enter
- **No breaking changes**: Single model mode continues to work as before

### Testing Required
User should test:
1. Enter compare mode and select 6 models
2. Type a prompt and press Enter
3. Verify all 6 models run in parallel (not just one)
4. Exit compare mode
5. Verify single model still works when pressing Enter

---

## Enhancement: Persistent Error Messages

### Issue
- Snackbar messages auto-hide after 3 seconds (`autoHideDuration: 3000`)
- Error messages disappear before user can read them
- User walked away and returned to just "error" with no details visible

### Plan
- [ ] Add `snackSeverity` state to track error vs info messages
- [ ] Modify setSnack calls to include severity
- [ ] Make Snackbar persist for errors (no auto-hide), auto-hide for success
- [ ] Add Alert component with close button for better error visibility

### Files to modify
- `frontend.py` - Snackbar component and setSnack calls

---

## Round: Orchestrator Re-skin + Music Tab

### Goal
Rebrand to "Orchestrator" with light editorial theme, section navigation, and a working Music tab (prompt-to-ABC score via abcjs). All existing features preserved.

### Plan
- [x] 1. Create git branch `feature/orchestrator-reskin`
- [x] 2. `frontend.py`: Wrap existing `App` in MUI `ThemeProvider` + `AppBar` ("Orchestrator" / "the boss") + permanent `Drawer` with sections: Conversation, Logbook, Music, Visuals
- [x] 3. Light editorial theme: EB Garamond display font, Inter body, warm off-white `#faf7f1`, ink primary, single accent
- [x] 4. Existing `App` function becomes `ConversationSection` — zero changes to its logic/controls
- [x] 5. Logbook + Visuals = "coming soon" placeholders
- [x] 6. Music section: prompt + key/meter/tempo/bars controls, abcjs CDN for score render + MIDI playback, download .abc
- [x] 7. `routes.py`: Add `POST /music/generate` — calls Ollama via `call_llm()` with ABC-only system prompt, extracts ABC
- [x] 8. Test: server boots, HTML verified to contain all existing features + new elements
- [ ] 9. Manual browser testing (user)

### What is NOT touched
- `config.py`, `models.py`, `llm_client.py`, `storage.py`, `main.py`
- All existing routes in `routes.py` (conversation/*, models/*, upload, export)
- Multi-model comparison, SSE streaming, file uploads, cost tracking, conversation browser

### Review

**Files changed:**
- `frontend.py` — added Orchestrator shell (ThemeProvider, AppBar, Drawer), renamed `App` to `ConversationSection`, added `MusicSection`, `Placeholder`, `Shell` components. Updated title/header styling to editorial theme. All existing conversation logic, compare mode, file uploads, cost tracking, SSE streaming unchanged.
- `routes.py` — added `POST /music/generate` route using existing `call_llm()`. No existing routes modified.

**Files NOT changed:** `config.py`, `models.py`, `llm_client.py`, `storage.py`, `main.py`

**Verification:**
- Python syntax: both files parse clean
- Server boots successfully on :5005
- HTML contains all 10 new markers (Orchestrator, Shell, ThemeProvider, etc.)
- HTML contains all 10 existing feature markers (compareModels, fileInputRef, tokenStats, etc.)
- JS brace count balanced: 460/460

---

## Imported from `local_llm_logger/ToDo.md` (2026-04-11, before deleting that directory)

1. First, I want to change this thing's name. I don't like the term "logger" as it doesn't really say what this thing's purpose is. Got any ideas?

2. We need to update the models. I want to use Kimi K2.5, and Qwen 3.5. I'm getting a 2TB very fast external hard drive, so I'll want to download the new models. 

3. I want the ability to select n number of models, and have them all run simultaneously. THen I want a means to compare their output. I imagine this will involve my own taste.  Sometimes the answer from a small model might be cursory. Sometimes that's a good thing, but more often it's probably not. I will want to come up with criteria  for shaping the models' responses.

4. Can you explain to me the naming convention, and the abilities, of the files and folders we're creating? As I recall, we had a timestamp and model names included in the names. I'm vague on how/when these things get saved. Do you have to hit the save button?
5. I also want to create  a visual representation of these  answer sequences. One thing about answer sequences is that sometimes, prompted by two different models, we end up with two sequences that lead to entirely different places. 
6. Then I'll be looking for musical structures that help express what these things are doing. I know this is a bit abstract now, but the fundamental idea is that music has a structure; it has a melody and harmonies and rhythms, and there is something satisfying about them. The 'harmony' created by an LLM might be able to be represented in going back and forth between modalities. 

---

## Round 1 — Orchestrator shell + theme + prompt-to-score

- [ ] 1. Wrap UI in MUI ThemeProvider + AppBar + Drawer; light editorial theme (EB Garamond display + Inter body, warm off-white bg, single ink accent). Title "Orchestrator", tagline "the boss".
- [ ] 2. Componentize inline React into `<Conversation/>`, `<Logbook/>`, `<Music/>`, `<Visuals/>`. Still inline in `app.py`, no build step.
- [ ] 3. Move all existing UI into `<Conversation/>` with zero feature loss.
- [ ] 4. Logbook + Visuals = "coming soon" placeholders.
- [ ] 5. Backend: `POST /music/generate` — calls Ollama with ABC-only system prompt, extracts ABC, saves via existing `save_artifacts()` plus `result.abc`.
- [ ] 6. Frontend Music tab: prompt + key/tempo/length/style selects, abcjs render via CDN, built-in play, download `.abc`.
- [ ] 7. Test: Conversation unchanged, Music renders+plays, music turn appears in `sessions/`.
- [ ] 8. Review section below.

### Review
_(to be filled in after implementation)_

---

## v5: consistency-loss follow-on

### Goal
Train a follow-on agent whose vision contribution is pulled toward proprio's manifold via an MSE consistency loss on hidden1 activations. Protects proprio, rewards *confirmation* not *sharpening*.

### Plan
- [x] Create `alien_baby/agents/train_v5.py`:
  - Subclass `SAC` → `ConsistencySAC`; override `train()` (copy SB3's, add `λ·MSE(h_full, h_blind)` to actor_loss).
  - `h = latent_pi[1](latent_pi[0](features))`; `obs_blind` zeros pixel cols.
  - Log `train/consistency_loss` to monitor.
- [x] `train_followon_v5()`: mirror `train_followon_v3` but use `ConsistencySAC` with `lambda_consistency=0.1`, `proprio_dim=7`.
- [x] Reuse frozen-actor + gradient-mask pattern from v3 (proprio cols → 0 gradient; drift assertion).
- [x] Train 200k steps; evaluate task success (clean + 100% noise).
- [x] Compute CKA(v5.hidden1, stage1_v3.hidden1) on matched states; expect higher than v3 follow-on.
- [x] Compute neighbor-consistency equivalence-class ratio; expect ≤ v3 follow-on's 0.27.
- [x] Append v5 section to `alien_baby/FINDINGS.md`.

### Review
- `train_v5.py`: `ConsistencySAC` subclass overrides `train()` and adds `λ·MSE(h_full, h_blind)` to actor loss. λ=0.1, 200k steps, proprio-drift = 0.0 verified.
- `tests/evaluate_v5.py`: success (clean + 100% noise), CKA vs stage1_v3, neighbor-consistency ratio.
- **Headline numbers (v5 vs v3):** 100% vs 95% clean success; 80% vs 55% at 100% vision noise; CKA 0.998 vs 0.905; neighbor ratio 0.272 vs 0.250.
- v5 is the first architecture where vision *additively* extends proprio's manifold (CKA ≈ 1.0) rather than displacing it. Full FINDINGS.md section appended.

---

## v6 — Head-mounted gaze camera ("the eye in the head")

### Motivation
Up through v5, the agent has a fixed overhead camera — it sees everything all the time. That's not how vision develops. In the theory's account (confirmed by Chs 6-7 reading), the child has to *orient* to look at what's happening, and depth perception emerges from motion — parallax during locomotion, expansion of retinal images as objects are approached, pair-wise `<, =, >` judgments conditioned to terminal-manipulation responses.

The agent needs two things it currently lacks:
1. **Gaze** — limited field of view that must be pointed
2. **Motion-dependent vision** — as the arm moves toward the target, the visual scene must change in ways that encode depth (parallax + size expansion)

### What Chs 6-7 specifically add

**Ch 6:** The "expanding world" is a function of the moving observer. Each motor cycle produces a predictable expansion of the retinal image. Size/shape constancy is inherited from invariant terminal manipulation — a ball is "round" because the grasping response is the same from every angle.

**Ch 7:** Motion parallax and binocular parallax are *redundant* "channels of communication." The system is overdetermined — multiple depth cues stabilize perception. Critically, **binocular fusion is response-conditioned, not structural**: two retinal images fuse because they both trigger the same singular reaching response, not because their retinal points match geometrically.

### What changes (revised)

**Option A — Single pan/tilt head camera (simpler):**
- One camera on a head body with pan/tilt joints
- Agent gets 2 new action dims (head gaze)
- Motion parallax emerges automatically as the arm-mounted scene (held object + background) shifts under head rotation

**Option B — Stereo head cameras (theory-faithful):**
- Two cameras ~30mm apart on the head
- Agent sees two small images per step
- Binocular parallax is available as an additional depth cue
- Consistency loss extended to action level: *both eyes' views should evoke the same reaching action* (the theory's response-conditioned binocular fusion)

Recommended path: start with A, graduate to B once A works. A is a strict addition to v5; B requires extending the consistency loss.

### The developmental story this tells
1. Hand gropes, finds the ball by touch (Stage 1 — proprio, as before)
2. Eyes learn to look toward where the hand is (Stage 2 — gaze emerges)
3. The agent's motion during reach produces motion parallax in the head camera — depth cue available before stereo
4. (Option B) Two cameras + action-consistency loss: binocular fusion emerges from shared reaching response, not geometric matching
5. Vision stays on proprio's manifold (v5 consistency loss) while encoding depth via motion

### Visualization
Split-screen video:
- Overhead view (experimenter's perspective) — what's happening in the world
- Head camera view (agent's perspective) — what the agent sees
- (Option B) Both head cameras side-by-side — disparity visible as depth

You watch the ball appear/disappear in the agent's camera as it learns to track. Early episodes: head flails, ball rarely in view. Late episodes: head tracks hand, ball consistently in frame during approach, parallax visible as background shifts against held object.

### Plan
- [x] 1. **MuJoCo XML**: `alien_baby/envs/tabletop_v6.xml` adds a head body at (0,-0.28,0.15) with `head_pan` (z-axis hinge) + `head_tilt` (x-axis hinge) joints. `head_cam` on head, FOV 45°, default tilted 30° downward via xyaxes so neutral head sees table center. Overhead kept for rendering.
- [x] 2. **Environment**: `alien_baby/envs/tabletop_gaze_env.py` — `TabletopGazeEnv`. Action 5D (3 arm torque + 2 head position, rescaled [-1,1] → joint ranges). Proprio 9D (adds head pan/tilt angles). Vision from head cam. Reward unchanged from v5 (no reward for looking).
- [x] 3. **Stage 1**: `train_stage1_v6()` — proprio-only SAC on 9D obs + 5D action. Smoke test: 2k steps → 30% success on blind-proprio task. Full run pending.
- [x] 4. **Follow-on**: `train_followon_v6()` — v5 `ConsistencySAC` with `proprio_dim=9`, `lambda=0.1`. Verified end-to-end: obs dim 777 (9 + 16×16×3 pixels), Stage 1 weights transfer, proprio-column drift = 0.0, consistency loss computed. Smoke test: 1k steps → 30% success.
- [x] 5. **Full training runs**: 200k stage1 + 200k followon. Stage 1 = 90% success. Follow-on final checkpoint regressed to 20%; **best EvalCallback checkpoint = 95% clean / 35% full-noise.**
- [x] 6. **Visualization**: `alien_baby/visualization/render_v6.py`. Split-screen (overhead + head cam) solo + 2x2 stage1-vs-followon comparison videos. Rendered at seeds 0/1/2; best-checkpoint solo episodes succeed in 17-36 steps.
- [x] 7. **Evaluation**: `alien_baby/tests/evaluate_v6.py`. Success, CKA, neighbor-consistency, plus gaze metrics (in_view_frac_near vs in_view_frac_far).
- [x] 8. **FINDINGS.md** updated with v6 section — honest mixed result.
- [ ] 9. **(Parked)** Option B (stereo cameras + binocular action-consistency): on hold pending better gaze behavior in monocular case.

### v6 headline
- **Task: 95% clean / 35% under full vision noise.** Vision is more load-bearing than v5 (bigger drop).
- **Interpenetration: CKA = 0.992** — gaze vision still on proprio's manifold.
- **Gaze-following: 31% near vs 27% far.** Barely emerged. Root cause: 45° FOV + 30° default tilt already covers most of the reachable area, so there's no pressure to track.
- **Late-training regression: real.** Final checkpoint worse than mid-training best. Evaluation uses best checkpoint.

### What v6 teaches us
Environmental pressure, not architectural capability, is what forces gaze behavior to emerge. We gave the agent the means to gaze but not the need. the source theory is about conditioning under conditions that make a response *necessary*.

### Follow-up candidates
- Narrower FOV (~20°) to force tracking
- Lower λ or entropy annealing to fix the late-training regression
- Move to Path 4 (LLM-in-loop) now, since vision grounding works even without gaze behavior

---

## v7 — Moving targets + narrow FOV + developmental checkpointing (in progress)

### Motivation
v6 result: gaze behavior barely emerged (31% near vs 27% far) because 45 deg FOV + 30 deg default tilt already saw most of the table. The environment didn't demand tracking. Also: videos looked "jittery and random" rather than showing progressive developmental competence.

v7 aims to fix both problems at once:
1. **Narrow FOV (22 deg)** — target falls out of frame most of the time; agent must pan/tilt to find and follow it.
2. **Moving targets** — objects drift at reset with random initial velocity (0.15-0.35 m/s). Over a 200-step episode they move ~10-30 cm, plus get bumped by the arm. Static "sits there" targets don't require gaze tracking; drifting ones do.
3. **Periodic checkpoints** every 50k steps + a developmental timelapse video script — we can literally watch the agent's behavior improve from random flailing to coordinated gaze-reach-track.

This is **Option 2** from our discussion: make the environment demand the behavior, don't reward-shape it directly. theory-faithful.

### Architecture
- `tabletop_v7.xml`: same head/arm/objects as v6 but FOV=22, object damping=0.05 (low, so they drift but don't bounce forever).
- `TabletopMovingGazeEnv`: subclass of `TabletopGazeEnv` — overrides reset to load v7 XML and give each object a random initial velocity in the 0.15-0.35 m/s range.
- `train_v7.py`: stage 1 + follow-on, uses `CheckpointCallback` to save every 50k steps + `EvalCallback` for best-model tracking.
- `develop_v7.py`: loads every saved checkpoint, renders one episode per checkpoint at the same seed, concatenates into one MP4 timelapse (split-screen overhead + head cam). Plus per-checkpoint solo videos.
- `evaluate_v7.py`: v6 metrics + new developmental metrics:
  - **`saw_before_contact_frac`** — of successful episodes, how often did the target enter the head cam FOV *before* fingertip contact? Tests "gaze precedes reach."
  - **`mean_tracking_error_deg`** — average angular distance between head cam view direction and direction to target. Lower = better tracking.
  - **`mean_time_to_contact`** — efficiency measure; does the agent get faster?
  - Plus full dev arc: all of the above across every checkpoint.

### Running
- Training: 300k stage1 + 1M followon, checkpoint every 50k = ~26 checkpoints on the follow-on. ~2 hours wall time.
- Watcher task auto-runs `develop_v7.py --solos` and `evaluate_v7.py` when training completes. Results in `/tmp/v7_postproc.log`.

### What to look for in results
- If the plan works: `saw_before_contact_frac` → near 100%, `mean_tracking_error_deg` → small (near half-FOV = 11 deg), `mean_time_to_contact` drops across training, `in_view_frac_near` >> `in_view_frac_far`, and the timelapse video shows the agent progressively getting smoother, faster, and more gaze-coordinated.
- If the plan fails (e.g. agent never learns to track, or regresses): the metrics will show it flat across checkpoints, and we'll know the FOV was still too forgiving or the task too hard.

### Status
- [x] XML + env written + smoke-tested (targets move, narrow FOV confirmed by rendered frame)
- [x] Training script + watcher running in background
- [x] Developmental timelapse + eval scripts written
- [ ] Training complete (in progress — ETA ~2h from start)
- [ ] FINDINGS.md v7 section (pending results)

### Open questions
- Head camera FOV: 30° narrow (forces precise gaze) vs 60° wide (forgiving). Start with 45°.
- Head position: centered above arm base, elevated ~20cm. Not physically attached to the arm (would couple head motion to arm motion in confounding ways).
- Gaze control: position-controlled is simpler. Start there; torque control is more biological but harder to learn.
- Stereo baseline (Option B): 30mm mimics infant interocular distance. The small baseline means disparity is small except for close objects — matches how the theory's toddler-scale parallax works.

---

## v8 — Creature on a Platform (survival-grounded perception)

### Motivation

v1–v7 proved the mechanics of interpenetration (staged development, consistency loss, CKA
measurement, frozen proprio). But the arm lives in a consequence-free world. Proprio is
primary only because we declared it so architecturally (freezing weights).

the theory's deeper point — and David's reframe — is that proprio is primary because **the
physical world kills you when you get it wrong.** The truck doesn't care what your visual
system thinks. The pencil tap in the reversing-spectacles experiment wasn't just a training signal — it
was a survival signal: "your body is about to do something the physical world will punish."

v8 creates a world with physical stakes. A creature on a finite platform can roll off the
edge and fall. Gravity is the pencil tap.

### The Creature ("skateboard dude")

A torso on a sled with two arms and a head. No legs. Pushes itself around by pressing
hands against the platform surface.

**Body spec (MuJoCo XML):**
- Box torso on a free joint (low-friction underside)
- Two arms: shoulder (2-DOF: fwd/back + up/down) + elbow (1-DOF) each → 6 arm joints
- Two hands with touch sensors (sphere geoms)
- Head: pan + tilt joints (2-DOF), camera mounted on head
- Touch sensor on torso underside (feels the platform surface)

**Proprio observation (~25 dims):**
- 6 arm joint angles + 6 arm joint velocities = 12
- 2 head joint angles = 2
- Torso orientation (3 euler) = 3
- Torso linear velocity (3) = 3
- Torso angular velocity (3) = 3
- 2 hand touch sensors + 1 torso touch sensor = 3
- Total: ~26

**Action space (8 dims):**
- 6 arm joint torques (gentle gear — no baseball-bat arms)
- 2 head position commands

### The World

- Finite platform: ~3m × 3m, elevated ~2m above ground
- Real edges — no walls, no rails
- Target object placed randomly on the platform
- Below: ground. Falling = impact.

### Reward / Health System

- **Fall off edge**: fixed severe penalty (-500), episode terminates immediately.
  Platform is at a fixed 2m height — every fall is the same. No graduated damage
  for now; if we want ledges/stairs later, that's a future environment.
- **Reaching target**: positive reward for hand touching target
- **Survival bonus**: small per-step reward for staying on platform
- **Edge proximity**: small negative reward when near edge (proprio warning signal)

### Developmental Stages

**Stage 1 — Blind crawling (proprio only)**
- No vision. Move on platform, find target by groping, avoid edges by feel.
- Creature learns: "my hand reaches into void = danger" / "torso tilting = near edge"
- Physical consequences establish proprio as ground truth.
- ~300K–500K steps

**Stage 2 — Vision added**
- Head camera on. Creature can see edges and target.
- Try WITHOUT freezing proprio first. If physical consequences are strong enough,
  proprio should maintain primacy naturally. If not, add freeze back.
- Consistency loss (from v5) keeps vision on proprio's manifold.
- Key prediction: gaze should emerge here because looking prevents dying.
- ~500K–1M steps

**Stage 3 — Spectacles**
- Flip visual input left-right (reversing spectacles)
- Add context bit (spectacles on/off) to observation
- Sharp penalty for reaching in wrong direction based on flipped vision
- Prediction: vision reorganizes to align with proprio, because trusting flipped
  vision = falling off the edge = death.
- Measure CKA between Stage 2 and Stage 3 representations.
- ~500K–1M steps

### What We Measure

- [ ] Stage 1: survival rate, edge-avoidance behavior, target-finding success
- [ ] Stage 2: vision improvement (faster target finding, fewer falls)
- [ ] Stage 2: gaze emergence (does it look before moving?)
- [ ] Stage 2: CKA(stage1, stage2)
- [ ] Stage 2 (no freeze): does proprio primacy emerge naturally from consequence?
- [ ] Stage 3: vision reorganization under spectacles (CKA)
- [ ] Stage 3: survival with spectacles vs naive agent

### Implementation Plan

- [ ] 1. Build MuJoCo XML: creature + platform (`alien_baby/envs/platform_creature.xml`)
- [ ] 2. Build Gymnasium env (`alien_baby/envs/platform_creature_env.py`)
  - [ ] Proprio-only and vision modes
  - [ ] Health system + fall damage
  - [ ] Reward structure
- [ ] 3. Smoke test: creature loads, physics works, can move, can fall off
- [ ] 4. Stage 1 training script + run
- [ ] 5. Stage 1 evaluation
- [ ] 6. Stage 2 training script + run
- [ ] 7. Stage 2 evaluation (including gaze + proprio primacy tests)
- [ ] 8. Stage 3 spectacles experiment
- [ ] 9. Analysis + FINDINGS.md update

### Review

(to be filled in after implementation)

---

## Long-range: Binding an LLM to the theory-grounded agent

### The question
If v5 (and v6) produce hidden-layer representations with genuine theory-faithful properties — equivalence classes, interpenetration, size/shape constancy derived from invariant manipulation — how do those representations come to anchor an LLM's words?

This is the eventual payoff of the whole project. The agent alone is a toy arm that reaches for balls. The LLM alone speaks fluently about everything and is grounded in nothing. The target is a system where the LLM's words are anchored to the agent's behaviorally-formed equivalence classes the way an infant's first words attach to classes it already has from pre-verbal experience.

### Four possible paths (ranked by theory-faithfulness)

1. **Shared embedding space** — Align LLM token embeddings with agent hidden1 activations via a paired (episode, caption) dataset. Correlational, not conditional. LLM's concepts were formed from text, not action; alignment papers over that. Probably insufficient.

2. **Frozen-feature fine-tuning** — Treat hidden1 as a frozen feature extractor, condition LLM output on it. Better, but the LLM is still a module reading features, not a system that *has* perception.

3. **Words as responses (purest theory)** — One transformer, output vocabulary includes both language tokens and action tokens, input includes perception. Saying "ball" and grasping the ball are siblings: both are responses conditioned to the same equivalence class. Requires starting language acquisition *after* embodied grounding (the infant pathway). Hardest to implement; most faithful to the theory.

4. **Embodied RL with language in the loop (planned next)** — Put a pre-trained LLM into the training loop as the policy, with perception input and action output. LLM's existing word-based concepts get reshaped by theory-style conditioning as it has to use them to succeed at embodied tasks. Keeps the LLM's knowledge but submits it to an embodied curriculum.

### Path 4 — rough sketch

**Core idea:** The LLM is the policy. At each step, it receives a textual description of the observation (or ideally, the raw hidden1 vector from a v6-trained perception module), plus any language input from the user, and emits a next action. Over training, the LLM learns which words/plans actually correspond to states where actions succeed.

**Concrete milestones (not committed, just a map):**

- [ ] **Phase 1: Text-mediated bridge.** Caption each agent state with natural language ("hand near red ball, head pointed slightly left"). Pre-trained LLM observes captions, emits action tokens. Agent executes. Reward from v6 env. Confirm that an LLM can solve the reach task purely from linguistic descriptions of perception. If this works, language already suffices as an intermediate format.
- [ ] **Phase 2: Latent-mediated bridge.** Replace captions with hidden1 projections (trained adapter). LLM sees a grounded vector instead of text. Test whether concept-words in the LLM reorganize to attach to the agent's classes. Probe: after Phase 2, does the LLM's embedding of "ball" become more similar to hidden1-"ball" than it was?
- [ ] **Phase 3: Action-consistency constraint.** Extend v5's consistency principle to the LLM: require that the LLM's action predictions under the raw perception vector match its predictions under a textual description of the same state. This is the Path-3 idea (words as responses) applied as a constraint on a Path-4 system. Closes the gap between "LLM reads perception" and "LLM's concepts are perceptual."
- [ ] **Phase 4: Verbal response generation.** The LLM is asked to describe the scene while acting. Its descriptions must be predictive of future success — you can't say "ball to my right" if there's no ball to your right, because the resulting plan fails. This is the adult version of infant babble being shaped by consequences.

### What needs to be true before we start Path 4
- [x] v5 shows hidden1 can live on proprio's manifold (CKA ≈ 1.0)
- [ ] v6 shows hidden1 encodes depth from motion (parallax sensitivity probe)
- [ ] v6 shows size/shape constancy from invariant manipulation (Source Ch 6.19 probe)
- [ ] An evaluation protocol that can distinguish "LLM reading hidden1" from "LLM whose concepts are anchored to hidden1"

### Why not Path 3 yet
Path 3 requires training a model from scratch with no language during the embodied phase, then introducing language as additional response tokens. This is closer to what the source theory actually says should happen, and may be the right long-term direction — but it gives up everything an LLM already knows. Path 4 keeps the LLM's text-scale knowledge and submits it to an embodied curriculum. If Path 4 works partially, it tells us how much of the theory's prediction is achievable without a full ground-up retraining. If Path 4 fails in specific ways (e.g., the LLM keeps hallucinating despite grounded perception), that failure mode itself is informative about whether Path 3 is actually necessary.

---

## v8 process changes: always-watch-first + cameras + FPV smoothing + blur-overlay

### Motivation
Two process problems surfaced watching v8 runs:
1. We kick off long (1M-step) training runs without first verifying that cameras, physics, and reward plumbing are sane. Wasted wall-clock.
2. The head_cam is rigidly bolted to the torso, so every flail shakes the camera. From the video we can't tell whether the creature is upright, falling, or what it's "looking at."

Plus one expressive idea: narrate the agent's perceptual development by blurring the head-cam video early in training and sharpening it as training progresses. **Display-side only** — the agent's actual observation is not modified (that would be the experimenter imposing a curriculum, which conflicts with the "survival grounding / drive not comfort" principle).

### Process rule (new)
Before launching any training run longer than a smoke test: render one episode from whatever starting-point model we're about to train from (random init for stage 1, stage-1 checkpoint for follow-on). If the video looks broken — cameras wrong, creature spawning off-platform, physics exploding — fix it before training.

### Training-length rule (new)
Default follow-on length drops from 1M → 250K steps. After 250K, render from `followon_v8_best/best_model.zip` and decide whether to extend. `CheckpointCallback` continues to save every 50K so we can also render from 100K and 200K mid-run if we want to abort earlier. SAC caveat: the first ~50–100K steps of any SAC run typically look like random flailing even when configured correctly (replay buffer filling, entropy high) — don't judge before ~150K.

### Plan

- [ ] 1. Create git branch `feature/v8-cameras-and-smoothing`.
- [ ] 2. **Pre-training sanity render** — add `alien_baby/visualization/sanity_render.py` that takes `--vision {true,false}` and renders one 150-step episode from a freshly-initialized SAC (no loaded weights). Used to inspect cameras/physics before any long run.
- [ ] 3. **Ringside camera** — add one `<camera name="ringside" .../>` to `alien_baby/envs/platform_creature.xml`, low (~0.4m), just outside the platform edge, angled slightly up so the creature's posture reads against the horizon. Test: render a frame, confirm upright vs. tipped is visually unambiguous.
- [ ] 4. **Head-cam temporal smoothing** — in `render_v8.py`, low-pass filter the head-cam frames experimenter-side (EMA over last N frames, e.g. α=0.3). Does not touch the environment or the agent's observation. Tunable alpha via CLI flag.
- [ ] 5. **Blur-to-sharpen overlay** — in `render_v8.py`, apply a Gaussian blur to the head-cam panel whose σ decreases as the creature's behavior becomes more vision-dependent. Signal: **vision-ablation sensitivity** (not raw task success). At each checkpoint, measure how much the policy's action changes when the pixel columns of the observation are zeroed out — averaged over a fixed set of eval states. High sensitivity = vision is load-bearing → sharp. Low sensitivity = creature is still groping by proprio → blurry. Rationale: success alone is a bad proxy because the creature can solve the task by proprio-grope alone while vision does nothing; we'd then render a sharp-looking video of an agent that isn't actually using its eyes. Mapping details: linear from sensitivity to sharpness, smoothed over the last 3 checkpoint measurements to absorb eval noise, **non-monotonic** (if the creature regresses, the video un-sharpens — that's the phenomenon, not a bug). Display only; environment and observation unchanged.
- [ ] 6. **Three-panel composite** — update `render_v8_episode()` to emit overhead | ringside | head_cam (with smoothing + blur applied to the head_cam panel) side-by-side. Keep existing two-panel rendering available via a flag for back-compat.
- [ ] 7. **Lower default follow-on length** — change `train_v8.py` default `--followon-steps` from 1_000_000 to 250_000. Keep 1M reachable via flag for when a short run looks promising.
- [ ] 8. **Smoke-test everything before any full run:** sanity render (no vision), sanity render (vision), render from an existing checkpoint with the three-panel + blur overlay. Verify ringside reads posture; verify head_cam is noticeably smoother; verify blur fades across checkpoints rendered at 50K, 150K, 250K.
- [ ] 9. Review section.

### What is NOT touched
- `platform_creature_env.py` — the environment, observation space, and reward are unchanged. All three new features are display-side.
- `train_v8.py` hyperparameters other than the default total_timesteps.
- Existing v1–v7 code.

### Resolved design decisions
- Blur signal: vision-ablation sensitivity (Π-style — does zeroing pixels change the action?). Chosen over raw task success because success can rise while vision does nothing.
- Mapping: linear, smoothed over last 3 checkpoints, non-monotonic.

### Review (2026-04-18)

**Branch:** `feature/v8-cameras-and-smoothing`

**Files changed:**
- `alien_baby/envs/platform_creature.xml` — added `<camera name="ringside" pos="0 -1.8 2.15" xyaxes="1 0 0 0 0.028 0.9996" fovy="50"/>`. Low south-side viewpoint that cleanly reads creature posture against platform edge + ground. Smoke render at `/tmp/ringside_test.png` confirmed.
- `alien_baby/visualization/render_v8.py` — rewrote to support: (1) three-panel composite (overhead | ringside | head_cam) with `--panels 2|3`; (2) head-cam temporal EMA smoothing with `--smooth-alpha` (default 0.3); (3) blur-to-sharpen overlay with `--sharpness <float>` OR `--auto-sharpness`; (4) `measure_vision_ablation_sensitivity(model)` helper that returns mean L2 action difference under full vs pixel-zeroed observations.
- `alien_baby/visualization/sanity_render.py` — new script. Renders one 150-step episode from a freshly-initialized SAC (`learning_starts=10**9` so it never updates) to verify cameras + physics before any long training run. `--vision` flag toggles observation mode.
- `alien_baby/agents/train_v8.py` — default `--followon-steps` lowered from 1_000_000 → 250_000.

**Files NOT changed:**
- `platform_creature_env.py` — environment, observation, reward untouched. All new features are display-side.
- Any v1–v7 code.

**Verification:**
- XML loads; MuJoCo lists all four cameras (`overhead`, `side`, `ringside`, `head_cam`).
- Sanity render (blind, 120 steps): ✓ wrote `v8_sanity_blind_seed0.mp4`.
- Sanity render (vision, 60 steps): ✓ wrote `v8_sanity_vision_seed0.mp4`.
- `render_v8.py --auto-sharpness` on `followon_v8_best/best_model.zip`: computed sensitivity = 1.3075 → sharpness 1.00 (trained policy strongly depends on vision, so displayed sharp — correct behavior).
- Manual `--sharpness 0.3` and `--sharpness 1.0` renders produced visibly distinct head-cam panels; ringside panel made upright-vs-tipped instantly readable.

**Open follow-ups (not in this PR):**
- Timelapse script that renders across a sequence of checkpoints with per-checkpoint ablation-sensitivity → smoothed sharpness. The primitive (`measure_vision_ablation_sensitivity`) is in place; wiring it into a `develop_v8.py` analogue is the next step.
- Calibrate `--ablation-max`: 0.5 is a placeholder. Once we have a range of checkpoints with sensitivities from ~0 (early) to ~1+ (late), pick a value that makes the visual progression expressive.

---

## v9 reward reshape — kill ATTRACT bribery (Variant 3: last-mile only)

### Motivation
Overnight 1M-step run (`v9_yawpin_scratch_unfrozen`, 420K before MPS crash) sat flat at
`ep_rew_mean ≈ 61`, `ep_len_mean = 300.0` across all 420 eval episodes. Zero touches, zero
falls, zero ball-losses. Classic drive-reduction collapse: the reward landscape paid ~+0.175
per step for proprio-grope (ATTRACT +0.225 − HUNGER −0.05 at ball-at-0.5m center-platform).
ATTRACT is the bribery vehicle — proximity without contact is a positive reinforcer.

### Fix (Variant 3: last-mile only)
Gate ATTRACT to fire only inside hand-reach (<0.15m), so it provides fine-motor guidance in
the approach but cannot be harvested from a stable grope pose. Keep V9_BALL_INITIAL_SPEED=0
to isolate the reward reshape from dynamics changes. Keep entropy pinned at 0.1.

### Plan
- [ ] 1. Create git branch `feature/v9-reward-last-mile`.
- [ ] 2. **`platform_creature_env.py`**: change `ATTRACT_MAX_DIST = 2.0` → `ATTRACT_MAX_DIST = 0.15`. One-line diff. At d=0 reward is 0.3; at d=0.15 it's 0; outside 0.15 it's 0. ATTRACT_SCALE and HUNGER_PENALTY unchanged.
- [ ] 3. **Component logging in env**: add per-episode `hunger_sum`, `attract_sum`, `touch` to the info dict returned on terminal/truncated step. Accumulators reset in `reset()`, updated in `step()`. Terminal paths (fall/touch/ball-lost/truncate) all emit the sums.
- [ ] 4. **Run-config dump in trainer**: `train_v8.py` — at start of `train_followon_v8`, write all args to `results/followon_v8_<tag>_config.json` so next time we don't have to grep logs for CLI flags.
- [ ] 5. **Sanity check 1 — random policy**: script `alien_baby/tests/random_policy_probe_v9.py`. Roll a random policy for 100 episodes under the new reward. Log per-episode: `hunger_sum`, `attract_sum`, `touch` (0/1), `fell` (0/1), `ball_lost` (0/1), `ep_len`, `total_reward`. Report: fraction with touch, distribution of total_reward, confirm net reward is negative when idle at 0.5m.
- [ ] 6. **Decision gate**: if touch fraction in (5) is 0%, invoke curriculum safeguard — temporarily set `_target_radius_lo = _target_radius_hi = 0.25` for the first 50K training steps, then anneal back to 0.5. Only if needed. (Implementation: add a `target_radius_override` arg to env; trainer callback schedules the anneal.)
- [ ] 7. **Sanity render**: `sanity_render.py --vision` from a freshly-initialized model under the new env. Confirm cameras + spawn + physics still sane (per CLAUDE.md workflow rule).
- [ ] 8. **Smoke train**: 50K-step run to confirm plumbing (reward components reach training logs, ep_lens start varying). Do not judge policy yet.
- [ ] 9. **Real run**: 250K steps (not 1M — per CLAUDE.md workflow). ent_coef pinned at 0.1. Render from `followon_v9_*_best/best_model.zip`. Extend to 1M only if 250K video is promising.
- [ ] 10. Review section.

### Entropy strategy
- This run: pinned at 0.1 (unchanged from last run). Don't change two things at once — the reward reshape is the experimental variable.
- Next run (conditional on this one learning): once left/right symmetry emerges post-reward-fix, anneal entropy to 0.01 to crystallize engrams.

### What is NOT touched
- `platform_creature_v9.xml` — no physics changes.
- v9 ball motion: `V9_BALL_INITIAL_SPEED = 0.0` stays. Static ball this run.
- SAC hyperparameters: lr, buffer, batch, γ, τ, train_freq, gradient_steps unchanged.
- `ATTRACT_SCALE = 0.3`, `HUNGER_PENALTY = -0.05`, `CONTACT_REWARD = 200`, `FALL_PENALTY = -500` — constants unchanged; only the ATTRACT distance window shrinks.
- Stage 1 training path.

### Decisions locked (per David, 2026-04-22)
- Variant 3 (last-mile only). Variant 1 risks sparse-reward abyss; Variant 2 adds implementation surface area without clarity benefit.
- Ball stays static. Fix reward landscape first; add dynamics later.
- Component logging required — verify ATTRACT fires only in <0.15m and net-idle reward is negative.
- Curriculum safeguard (radius 0.25 → 0.5) is conditional on Sanity Check 1 showing 0% touch rate, not upfront.
- Entropy pinned at 0.1 through this run; anneal in a subsequent run once learning is visible.

### Review
_(to be filled in after implementation)_

---

## Mirror Wrapper (stage-1 lateralization-attractor breaker)

### Motivation
Stage-1 runs collapse into a one-sided turret mode (α=0.3 → left, α=0.5 → right). Env is mirror-symmetric, so a per-episode L/R coin-flip wrapper should let the policy generalize across bilateral symmetry and break the attractor.

### Spec deviations from `Session Resume.md`
The resume spec was a best-guess; audit of `platform_creature_env.py` + v9 XML surfaced three errors, all confirmed with David and corrected:
1. **Obs indices.** Quat is 4 elements at `[14:18]`; angvel at `[21:24]` (spec said `[14:17]` / `[20:23]`).
2. **Quat transform.** Sagittal mirror (x=0 plane) requires `(w,x,y,z) → (w,x,-y,-z)`. Spec's `(w,-x,y,-z)` is a *y*-plane reflection — inconsistent with the angvel rule it paired with.
3. **Shoulder roll.** Both arms share `axis="1 0 0"` in the XML, so L/R mirror is swap-only — no sign flip.

### Architecture
`MirrorWrapper(gym.Wrapper)`, per-episode coin flip (default p=0.5). On heads: incoming action un-mirrored → env; env obs mirrored → policy. `info["mirrored"]` populated so Monitor/RewardComponentCallback can log `rollout/mirrored_frac`. `info["spawn_angle"]` / `info["spawn_left"]` flipped to perceived hemisphere so hemisphere-balance telemetry reflects what the policy saw.

**Env state is NOT modified.** Combining state mirror + obs mirror cancels (obs is equivariant under state mirror, so `mirror_obs(obs(mirror(s))) == obs(s)`), defeating the augmentation. Obs/action mirror alone suffices because (a) env dynamics are mirror-symmetric and (b) v9 ball-spawn distribution is already mirror-invariant (angle ∈ `[0, π)` is closed under `θ → π-θ`).

### Files changed
- **New** `alien_baby/envs/mirror_wrapper.py` — module-level `mirror_action`, `mirror_obs`, and `MirrorWrapper` class.
- **New** `alien_baby/tests/test_mirror_wrapper.py` — 10 tests: involution for action & obs (with and without vision), specific-value transforms, pixel column flip, shape preservation, force-mirror test hook, and a physics-level rollout-symmetry test.
- `alien_baby/envs/platform_creature_env.py` — added `"mirrored": False` default to `_info()` so Monitor's info_keywords lookup works whether or not the wrapper is in the chain.
- `alien_baby/agents/train_v8.py` — imports `MirrorWrapper`, adds `--mirror-augmentation` CLI flag, plumbs `mirror_augmentation` through `train_stage1_v8`, wraps the training env (eval env stays unwrapped so `best_model.zip` reflects raw-task performance), adds `"mirrored"` to `FOLLOWON_INFO_KEYWORDS` and `RewardComponentCallback`'s frac group.

### Verification
- 10/10 unit tests pass. `test_mirrored_rollout_matches_mirrored_plain_rollout` is the load-bearing one: stepping the wrapped env from the same seed with zero actions produces obs equal to `mirror_obs(plain obs)` at every step, proving the env truly respects the symmetry and the wrapper is a faithful symmetry transform.
- Smoke test: `Monitor(MirrorWrapper(env))` chain steps cleanly over 12 random episodes; mirror rate 5/12, `mirrored` info key propagates through.

### Launch command
```
python -m alien_baby.agents.train_v8 --stage stage1 --v9 \
  --ent-coef 0.1 --stage1-radius 0.30,0.70 --pbrs-alpha 0.3 \
  --mirror-augmentation
```

### Success gate (from Session Resume)
`dist_reduction > 0.05` at 50K with bilateral symmetry (`rollout/touched_left_frac` and `rollout/touched_right_frac` both > 5% at the 25K gate).
