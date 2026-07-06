# How to start the next session

Copy and paste the block below as the first message to Claude in a fresh terminal session. It's written so Claude can pick up cold without you having to brief in real-time.

---

## Copy-paste prompt for next session

```
We're resuming the Alien Baby / MIMo crawler project. Please read these documents in order before doing anything else:

1. alien_baby/PROJECT_STATUS.md — current best numbers and last-run timestamp
2. alien_baby/FINDINGS.md — full experimental history (most recent entries: Phase D-HER, Phase E / E2)
3. alien_baby/THEORY_LOG.md — hypothesis tracking with refutations
4. alien_baby/NEXT_SESSION.md — this file's "Open at session start" section below

After reading, confirm you understand:
- The current canonical failure mode ("one move then freeze")
- Which interventions have been tested and refuted (A, B, C, C2, D, E, E2, F, F-2k)
- The four candidate framings for the next experiment (F-curiosity, F-sustained, F-tipped-init, F-info-velocity)
- The agent workflow we're using: experiment-strategist proposes, I approve, training-engineer launches, results-analyst writes FINDINGS, theory-monitor updates THEORY_LOG

Two tasks I want done before any new experiment is launched:

1. Phase F-2k (just completed last session) needs a FINDINGS.md writeup and a THEORY_LOG.md hypothesis-status update. Dispatch results-analyst and theory-monitor (in parallel) to do those.

2. Then dispatch experiment-strategist to formally propose what should be Phase G — the leading candidate from last session's discussion was F-tipped-init (randomize spawn into already-tipped postures), based on F-2k's "Mixed" outcome (eval std 99.5 = some episodes near the ball, most not, depending on random orientation luck). The strategist should consider the other three candidates too and explain its choice.

When the strategist proposes, present the proposal to me for approval before launching.

Two standing instructions:

A. Per alien_baby/CLAUDE.md's Conversation Capture section, regenerate alien_baby/CONVERSATION_LOG.md at every natural pause point this session. The command is:
   python -m alien_baby.utils.export_conversation
   (Always reads the most recently modified JSONL for this project. Overwrites the markdown.)

B. Per CLAUDE.md, at natural pause points (end of phase, end of topic, end of session) PROACTIVELY offer a plain-text SESSION SUMMARY block I can copy-paste — do not wait to be asked.
```

---

## Open at session start

These are the specific operational items pending from the prior session:

**Pending writeups (do these first, in parallel):**
- Phase F-2k FINDINGS.md entry — covers both Phase F (max-steps=600 confound) and F-2k (rerun with max-steps=2000)
- Phase F-2k THEORY_LOG.md hypothesis update

**Phase F-2k summary numbers** (for the writeup agents):
- Run tag: mimo_phase_f_2k_nearball_outside_threshold
- Eval at 250K: mean_reward -1989.35 +/- 99.48, ep_len_mean 1960
- ent_coef pinned at 0.2 (held), critic_loss ~1
- Compared to E2 (ball at 0.10m): per-step reward WORSE (-0.99 vs -0.95)
- But eval std 99.5 vs E2's 0.30 — huge per-episode variance, meaning some eval rollouts land near the ball, most do not
- Disambiguation result: Phase E2's signal was largely geometric refuge (near ball at 0.10m inside goal threshold from spawn). When ball moved outside threshold (0.30m), the per-step floor returns, BUT the random-orientation lottery produces occasional near-goal trajectories. Falls on the "Mixed" branch of the strategist's Phase F decision tree.

**Phase G candidates** (the four framings from last session's discussion):
- **F-tipped-init** (leading): Randomize spawn quaternion across multiple postures (prone, supine, side-left, side-right). Removes the "freeze on side" refuge — every initial posture would need active control to maintain. Single-file env change.
- **F-curiosity**: Add Random Network Distillation (RND) intrinsic reward so freezing has opportunity cost. SB3-Contrib has RND. Moderate complexity.
- **F-sustained**: Reward shaping change: bonus = scale * |hip_velocity| * (1 if action_was_nonzero_last_step else 0.5). Punish action-stalling, not just stillness. Single-file env change.
- **F-info-velocity**: Bake hip_speed into env info dict and modify her_wrapper.compute_reward to read it back. Fixes HER's 4/5 relabel dilution of the velocity bonus. Two-file change.

**Bugs / methodological gotchas to be aware of:**
- her_wrapper.py composes HER's sparse signal with inner_reward, but HER's relabel mechanism re-runs compute_reward on 4/5 of sampled transitions — those see only the sparse term. The velocity bonus only flows through 1/5 of policy gradients. See multi-line comment in her_wrapper.py around line 135.
- train_crawler.py's default --max-steps is 600. If you want 2000-step episodes (the E2-comparable regime), specify it explicitly.
- Phase F initially missed this and ran at max-steps=600, confounding comparison. F-2k was the corrected rerun.
- Hip-as-goal limitation in her_wrapper.py is known and intentionally kept in place (user decision). Don't redesign achieved_goal without explicit user approval.

**Files changed across the prior session:**
- alien_baby/crawler/train_crawler.py: faulthandler.enable(); try/except + os._exit(0) at end of __main__; progress_bar=False (these fix the Python 3.13 + MuJoCo finalization race that crashed the original Phase D-HER launch)
- alien_baby/crawler/her_wrapper.py: reward composes HER sparse + inner_reward (with caveat comment about HER relabel dilution)
- alien_baby/visualization/render_her.py: NEW. Renders HER-trained policies (Dict obs + MultiInputPolicy)
- alien_baby/utils/export_conversation.py: NEW. Generates alien_baby/CONVERSATION_LOG.md from the session JSONL
- alien_baby/utils/__init__.py: NEW (empty, makes utils a Python package)
- alien_baby/CONVERSATION_LOG.md: NEW. Markdown transcript of user messages + Claude's text replies
- alien_baby/FINDINGS.md: appended Phase D-HER entry; appended Phase E / E2 entry
- alien_baby/THEORY_LOG.md: appended Phase D-HER hypothesis update; appended Phase E / E2 hypothesis update
- alien_baby/PROJECT_STATUS.md: best numbers + wrapper-bug note + last-run timestamp updated

**Established failure modes (the project's growing taxonomy):**
- "No motion at all" — Phases A, B, C, C2 (auto-entropy collapse to zero-action deterministic policy)
- "One move then freeze" — Phases D, E2, F, F-2k (commits to one initial action then freezes; not entropy collapse since ent_coef pin holds)
- The dependency chain (body → motion → encounter → memory → map) still breaks at link 2.

**Central question, restated:**
Is "one move then freeze" a rational SAC actor equilibrium (no further action is judged better than holding still), or a geometric refuge artifact (sit still and collect free reward where the goal zone overlaps spawn)? F-2k's eval std 99.5 says: partly geometric (some rollouts land near ball, depending on random orientation), partly equilibrium. The next experiment should target the equilibrium side — make stillness genuinely costly across all initial conditions.
