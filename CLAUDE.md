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