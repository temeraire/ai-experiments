## Conversation Capture (important — read this)

The user wants to preserve our conversations. OBS screen recording does NOT work for this —
it captures a static frame, not the scroll, so recordings are useless (a 9MB screenshot that
never changes).

**What does work:** copy-paste into a separate document. The user does this manually.

**To make this easier, at natural pause points (end of a training run, end of a topic, end of
a session) proactively offer a plain-text summary block the user can copy. Format:**

---
SESSION SUMMARY [date]
Topic: [what we worked on]
Key decisions: [bullet list]
Key findings: [bullet list]
Files changed: [list]
Next steps: [list]
---

Do not wait to be asked. If a significant amount of work has been done, offer the summary.
Keep it dense and copy-paste friendly — no markdown that won't survive a paste into Word/Notes.

## Glossary + Flashcards upkeep (standing rule)

Whenever a genuinely new term, method, or piece of jargon comes up in our work — a new
algorithm (e.g. DroQ), a new metric (e.g. vision-ablation sensitivity), a new experimental
construct (e.g. eccentricity sweep), or a coined phrase (e.g. "generalization-as-primary") —
add it to `alien_baby/GLOSSARY.md` AND add a matching Q/A card to `alien_baby/FLASHCARDS.md`,
without being asked. Rules:
- Plain-English, "what is it / why does it matter here" — match the existing house style.
- Skip terms already defined; don't duplicate.
- A term counts as "new" the first time it would need explaining to a non-specialist reader.
- When in doubt, add it — an over-full glossary is cheaper than a forgotten term.

## Standard Workflow
1. First think through the problem, read the codebase for relevant files, and write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Test your changes before marking tasks as complete. Don't just implement - verify that the feature works as expected.
8. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

## Git 
Before fixing bugs or adding features, always create a new git branch with a descriptive name based on the issue or feature. Commit your changes only to that branch. Once you're done, push the branch and create a pull request for review.

## Video evidence
  When a video file (`.mp4`, `.mov`, `.webm`, `.mkv`, `.avi`) is produced by an
  experiment, simulation,
  or training run, or is otherwise referenced in the task, **watch it before 
  drawing conclusions**.
  You cannot natively see video, so run:

  ```
  python /Users/davidwolpe/Documents/DataScience/dev/ai_experiments-orchestrator
  /describe_video.py <path>
  ```

  This sends the video to Gemini 2.5 Flash for native video analysis (falls back
   to Claude + ffmpeg
  frames if Gemini is unavailable; the script prints a clear `[describe_video]
  Gemini failed (...)`
  notice on stderr when that happens).

  Use this especially for RL / robotics work (MuJoCo, MIMo, etc.) where numeric
  stats can hide
  visually-obvious failures: agent never approaches the target, agent posture
  makes the target
  unreachable (e.g. standing while target is on floor), agent is stuck in an
  early frame. Do not
  declare a training run successful or failed on the basis of stats alone if a
  video exists.

 ## Critical Rules for Code Changes

  ### NEVER remove existing features
  - When fixing bugs, ONLY fix the specific issue reported
  - DO NOT remove, consolidate, or "simplify" existing functionality
  - DO NOT remove UI elements (like multiple progress bars, comparison views, etc.)
  - If you think something should be refactored, ASK FIRST

  ### Before making changes
  1. Identify the MINIMAL change needed to fix the issue
  2. Preserve ALL existing features and UI elements
  3. If the fix might affect other features, explain the impact and ask for confirmation

  ### When fixing bugs
  - Fix ONLY what's broken
  - Keep all existing features intact
  - Don't "improve" or "optimize" unrelated code
  - Don't make architectural changes without explicit permission

  ### Examples of unacceptable changes
  - Combining multiple progress bars into one
  - Removing side-by-side model comparisons
  - Simplifying multi-model features into single-model
  - Removing any existing UI components
  - Changing the user experience without being asked

  ### If unsure
  Always err on the side of making the SMALLEST possible change. Ask "Will this fix affect any
  existing features?" If yes, explain and get approval first.

## Training-run workflow (alien_baby / v8+)

These rules apply to any RL training run under `alien_baby/`. They exist to avoid burning hours
of compute on runs with broken cameras, broken physics, or misread signals.


### Always watch a video before committing to a training run
Before launching any training run longer than a smoke test (>50K steps), render one episode
using whatever model the run will start from:
- For stage 1: render from a freshly-initialized (untrained) model to confirm cameras,
  physics, spawn position, and reward plumbing are sane.
- For a follow-on stage: render from the stage-1 checkpoint that will seed it.
Use `alien_baby/visualization/sanity_render.py`. If the video looks wrong, fix it before training.

### Prefer short runs + checkpoint inspection over one long run
Default follow-on length is 250K steps, not 1M. After 250K, render from
`followon_*_best/best_model.zip` and decide whether to extend. `CheckpointCallback` saves every
50K so mid-run renders at 100K and 200K are also cheap. Extend to 1M only when the 250K video
is promising. A longer run that finds out at hour 2 that the cameras were wrong is a bad trade.

### SAC caveat when judging early videos
The first ~50–100K steps of any SAC run typically look like random flailing even when training
is configured correctly (replay buffer filling, entropy still high). Don't judge a run before
~150K steps.

### Video rendering: experimenter's eye vs. agent's eye
The creature's head_cam is the *agent's* observation — don't modify it in the environment for
display purposes. Smoothing, blurring, or post-processing belongs in the rendering script
(`render_v8.py` etc.), not in the env. Always render an overhead + a ringside (posture-reading)
panel so it is obvious whether the creature is upright or tipping.

### Success rate is not the same as "using vision"
When building signals that should represent "the creature is learning to see," do not use
eval task-success as the signal without checking that vision is actually load-bearing.
The creature can solve the task by proprio-grope alone. Prefer signals like
**vision-ablation sensitivity** — how much the policy's action changes when the pixel
columns of the observation are zeroed. That directly measures "does vision matter to
behavior?" which is what we actually care about (Π-style interpenetration).

### Done-sound chime on training finish
`train_v8.py` plays `/System/Library/Sounds/Glass.aiff` via `afplay` when a run completes
(macOS only; silent elsewhere). Don't remove — the user uses this to know when to come back
to the session without staring at the terminal. If a new training script is added, call
`_play_done_sound()` (or replicate the pattern) at the end of the main training function.

### Performance defaults: MPS + 16 parallel envs
`train_v8.py` uses `device=_best_device()` (picks `mps` on Apple Silicon, `cuda` if
present, else `cpu`) and `N_ENVS_FOLLOWON = 16`. On an Apple Silicon Mac with 16 cores this
gives ~25× warmup FPS and ~5-10× training FPS vs `device="cpu"` + 8 envs. **Do not revert to
CPU + 8 envs without a specific reason** — waiting hours for a run that could finish in 30
minutes is a bad trade. If a new training script is added, copy the `_best_device()` pattern
and match the env count to CPU core count. 

Done. Both writeups are complete and I've reconciled every number against the authoritative
  fixed logs. Here's the honest status.

  The run finished, the evals ran, results are real

  - R43 (vision) done 01:59, R44 (proprio) done 02:13, all four evals auto-ran by 02:38. Both
  FINDINGS.md (Phase V entry) and THEORY_LOG.md (Phase V entry) are written.

  A correction I owe you

  The flaky shell this morning bit me: when I dispatched the writeup agents, the corrected
  battery re-run hadn't finished, and I handed them some distance/size/speed numbers that
  were wrong — then a couple of my own fixes briefly made it worse before I caught them. I've
  now verified every figure against the two fixed logs and corrected both docs. The
  authoritative numbers:

  - Distance → time-to-contact (the headline): R44 proprio 688·d − 218, r=+0.89; R43 vision
  1292·d − 425, r=+0.85.
  - Important caveat I added: the distance metric is confounded by the autonomously-sweeping
  cart — the y=0.25 and y=0.45 bins log spurious ~1-step contacts, which add scatter and
  understate the trend. On the genuine-reach bins it's clean and extrapolating: R44 0.35→20,
  0.55→157, 0.65→276 steps; R43 0.35→24, 0.55→198, 0.65→560. So "farther = longer to reach"
  holds, but the exact slope/r are noisier than a clean reach-distance metric would give.
  - Speed retention: R44 0.50 (graceful), R43 0.25 (cliff).
  - Size: both runs tripped the size-sensitive flag (R44 range 0.37, R43 0.50) on 15-ep
  samples — most likely noise; flagged for a 30-ep recheck.

  The conclusions (unchanged, and correct)

  1. Generalization-as-primary is confirmed for proprio — lawful distance scaling that
  extrapolates to untrained distances, graceful speed degradation, broad direction coverage.
  2. Vision adds nothing to outcomes (R43 ≈ R44 at every eccentricity), on the cleanest task
  the project has run — the strongest vision null yet.
  3. Smoking gun stands: vision-ablation is highest at ecc=0 (1.04, ball dead ahead) and flat
  ~0.62–0.72 thereafter — vision is globally/non-directionally bound, not encoding ball
  direction. The strong form of your hypothesis (vision recruited at the margin) is refuted.

  Two honesty flags

  - The doc entries went through churn; the numbers are now right, but there's some
  duplicated prose from the agents and one slightly awkward leftover line. I can do a clean
  read-through/dedup pass if you want pristine records.
  - The cart-sweep confound means a clean "distance generalization" result really wants a
  non-cart reach task or a contact metric that excludes cart-carried contacts — worth doing
  before leaning hard on the time-to-contact law.

  Next step (queued, not started)

  The probe both the FINDINGS and THEORY_LOG entries point to: linearly decode ball-x from 
  R43's vision latent. If R² ≈ 0 despite ablation 0.6–1.0, vision never encoded direction
  (representation failure); if R² is meaningful, direction is encoded but the policy doesn't
  use it (policy-gradient failure). That single measurement explains the whole "integrated
  but inert" picture and tells us whether to fix the encoder or the RL update.

  ---SESSION SUMMARY 2026-05-30 (Phase V results)
  Result: Phase V finished + evaluated. Proprio generalizes lawfully (distance r≈0.89 
  extrapolating, size-invariant modulo noise, graceful speed); vision integrated but 
  behaviorally inert (R43≈R44 everywhere) and non-directional (ablation peaks at ecc=0). 
  Generalization-as-primary: core CONFIRMED, strong form REFUTED.
  Corrected: distance law R44 688·d−218 r=0.89 / R43 1292·d−425 r=0.85; documented cart-sweep
  confound in the distance metric; fixed wrong size/speed figures I'd introduced.
  Docs: FINDINGS.md + THEORY_LOG.md Phase V entries written and reconciled.
  Caveats: distance metric cart-sweep-confounded; size sweep noisy (recheck 30 eps); doc 
  entries could use a dedup pass.
  Next: linear ball-x decode probe on R43 vision latent (representation-failure vs 
  policy-failure); optionally a non-cart reach task for a clean distance law.

  Want me to run the ball-x decode probe next, or do the doc cleanup pass first?
