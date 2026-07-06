# Phases — canonical ledger

The single source of truth for phase names. Earlier work used a mix of letters
(A–H) and Roman numerals (I–V), then letters again (W, X). As of 2026-06-15 we
use **one consistent scheme**:

- **Integer Roman numeral** = a new driving question / intervention (`Phase XV`).
- **`.2` / `.3` suffix** = a re-run or single-knob variant of the *same* question
  (the old `C2` / `E2` / `F-2k` semantics, generalized). E.g. `Phase V.2`.
- **Title + axis tag** carry the "why"; the number stays clean and sortable.

Axes: **INFRA** (plumbing), **SUBSTRATE** (the world/body), **ARCH** (network
structure), **TASK** (what's being asked), **REWARD** (shaping/algorithm),
**DIAGNOSTIC** (isolating a confound), **MODALITY** (is vision *productive*?),
**GENERALIZATION** (transfer to unseen objects).

> Git history, filenames (`launch_phase_iv.sh`, `PHASE_G_PROPOSAL.md`), and raw
> conversation transcripts retain the old names — use this crosswalk to read them.

| New | Old | Axis | Driving question / intervention | Why we moved here | Runs |
|---|---|---|---|---|---|
| **0** | 0 | INFRA | Plumbing: VecNormalize, MirrorWrapper, tilt-termination, buffer 500K | groundwork | — |
| **I** | A | SUBSTRATE | No-bribery substrate + existing CNN | first honest test: does an honest substrate make vision matter? → **silent** | — |
| **II** | B | ARCH | Dialogue architecture (two-stream + gate + consistency loss) | substrate failed → try an architectural fix → degenerated to a monologue | — |
| **III** | C | TASK | Fixed-ball reach, blind proprio | architecture failed → simplify the task to find any floor → zero-motion collapse | — |
| **III.2** | C2 | TASK | III + memory_obs (touch flags) | give the stateless policy memory of its own contacts | — |
| **IV** | D | REWARD | HER + memory + small velocity bonus | still collapsed → bring in the literature standard for sparse goals | — |
| **V** | E | REWARD | Caregiver scaffold + 5× velocity bonus | seed accidental contact with a near ball | — |
| **V.2** | E2 | REWARD | V with `her_wrapper` bug fixed | velocity bonus had been silently zeroed → first valid test; first non-zero eval std | — |
| **VI** | F | DIAGNOSTIC | Near ball *outside* goal threshold | was the reward just geometric overlap? | — |
| **VI.2** | F-2k | DIAGNOSTIC | VI with `--max-steps 2000` | max-steps confound → rerun → "bet on the lottery" | — |
| **VII** | G | SUBSTRATE | Cart substrate, ent=0.5 + vel bonus | HER line exhausted → new substrate → **first 20/20** | R3, R19 |
| **VIII** | H | SUBSTRATE | Moving balls | vision still inert across 6 substrates → last substrate hypothesis → **REFUTED** | R32–R35 |
| **IX** | I | ARCH | MICOA (PoE + symmetric KL agreement) | substrate avenue closed → architectural integration → vision inert | R36, R37 |
| **X** | II | ARCH | MICOA + temporal predictive KL | symmetric inert → predictive → **first ablation-barrier break** | R38 |
| **XI** | III | ARCH | σ-clamp tuning (single-horizon t+1) | active-not-productive → tighten σ → vision load-bearing **AND task solved** | R39, R40 |
| **XII** | IV | MODALITY | MICOA+vision vs proprio control, moving balls | is load-bearing vision *productive*? → **no** (R41 < proprio R42) | R41, R42 |
| **XIII** | V | GENERALIZATION | Eccentricity sweep, static reachable | clean static task → **strongest vision null; proprio generalizes lawfully** | R43, R44 |
| **XIV** | W | INFRA | DroQ critic validation | adopt sample-efficient critic; **methodological correction** (valid metrics) | R45 |
| **XV** | X | GENERALIZATION | Object variety (size / shape / combined) | **zero-shot transfer confirmed**; reproducible 0.075 anomaly | R46, R47, R48 |

**Next phase: XVI.**
