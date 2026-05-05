"""
Overnight sweep runner — 2026-05-05.

Hybrid plan from todo.md (greenlit by David):
  0. Sanity render Phase 0 best checkpoint (skipped here — done separately).
  1. Phase 0 baseline validation:  plumbing fully on, MlpPolicy. 250K.
  2. CNN encoder (Idea G):         GAP_ANALYSIS Rank 1 fix. 250K.
  3. Hand-only + spawn cone (A):   shrink the equivalence class. 250K.
  4. XOR color random (B):         pure visual category. 250K.

Plus: Idea D (proprio-noise eval on Phase 0 best) interleaved before training.

Each experiment runs `train_v8.py` as a subprocess. Stdout streams to the
experiment's log; we don't parse it for live kill criterion (keeping the
runner minimal). We DO post-process each experiment's eval log to record
the best touch_rate and final touch_rate. Best checkpoints are auto-rendered
afterwards.

This runner is intentionally serial (one experiment at a time) so each can
use the full SubprocVecEnv 16-env parallelism without contention. Total
wall clock budget: ~3-4 hours of training + ~30 min of analysis/render.
"""

import argparse
import csv
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # ai_experiments/
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
SWEEP_TAG = "overnight_2026_05_05"
SWEEP_DIR = RESULTS_DIR / f"sweep_{SWEEP_TAG}"

# Default: phase-0 plumbing on (vec_normalize + mirror), 250K steps,
# warm-start from stage1_v8_checkpoint, v9 disabled (stationary ball, matches
# v8 default). Each experiment overrides only what's specific to it.
COMMON_FLAGS = [
    "--stage", "followon",
    "--followon-steps", "250000",
    "--checkpoint-interval", "50000",
    "--vec-normalize",
    "--mirror-augmentation",
    "--seed", "42",
]

EXPERIMENTS = [
    {
        "id": "1_phase0_baseline",
        "description": "Phase 0 plumbing only (Mlp, vec_normalize, mirror, "
                       "tilt term, ctrl cost, 500K buffer). 250K. Reference.",
        "flags": ["--run-tag", f"sweep_{SWEEP_TAG}_1_phase0_baseline"],
    },
    {
        "id": "2_cnn_encoder",
        "description": "Phase 0 + DrQ-v2 4-conv encoder (50d pixel latent). "
                       "GAP_ANALYSIS Rank 1.",
        "flags": ["--run-tag", f"sweep_{SWEEP_TAG}_2_cnn_encoder",
                  "--cnn-encoder"],
    },
    {
        "id": "3_hand_only_cone",
        "description": "CNN + hand-only contact + ±45° spawn cone (Idea A). "
                       "Shrinks the equivalence class — must aim the hand.",
        "flags": ["--run-tag", f"sweep_{SWEEP_TAG}_3_hand_only_cone",
                  "--cnn-encoder",
                  "--hand-only-contact",
                  "--spawn-cone-deg", "90"],
    },
    {
        "id": "4_xor_color",
        "description": "CNN + XOR color random (Idea B). 50/50 red-good "
                       "blue-bad — vision is the sole discriminator.",
        "flags": ["--run-tag", f"sweep_{SWEEP_TAG}_4_xor_color",
                  "--cnn-encoder",
                  "--xor-color-random"],
    },
]


def run_subprocess(cmd, log_path, env=None, timeout=14400):
    """Stream stdout to log file, return (returncode, wall_seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = -1
    return rc, time.time() - t0


def parse_final_eval(train_log_path):
    """Read 'Follow-on v8: X/20 touches, Y/20 falls' from end of train log."""
    if not train_log_path.exists():
        return None
    text = train_log_path.read_text(errors="ignore")
    import re
    m = re.search(r"Follow-on v8.*?:\s*(\d+)/20 touches,\s*(\d+)/20 falls", text)
    if m:
        return {"final_touches": int(m.group(1)), "final_falls": int(m.group(2))}
    return None


def find_best_checkpoint(run_tag):
    """Locate followon_v8_<run_tag>_best/best_model.zip. Returns Path or None."""
    candidate = RESULTS_DIR / f"followon_v8_{run_tag}_best" / "best_model.zip"
    return candidate if candidate.exists() else None


def render_best(run_tag, out_dir):
    """Render the best checkpoint with three-panel + auto-sharpness overlay."""
    ckpt = find_best_checkpoint(run_tag)
    if ckpt is None:
        return None, "checkpoint missing"
    out_path = out_dir / f"render_{run_tag}.mp4"
    cmd = [
        sys.executable, "-m", "alien_baby.visualization.render_v8",
        "--checkpoint", str(ckpt),
        "--out", str(out_path),
        "--panels", "3",
        "--auto-sharpness",
        "--episodes", "3",
    ]
    log_path = out_dir / f"render_{run_tag}.log"
    rc, wall = run_subprocess(cmd, log_path, timeout=900)
    return out_path if rc == 0 else None, ("ok" if rc == 0 else f"rc={rc}")


def proprio_noise_eval(checkpoint_path, out_dir):
    out_path = out_dir / "proprio_noise_eval_phase0.json"
    log_path = out_dir / "proprio_noise_eval_phase0.log"
    cmd = [
        sys.executable, "-m", "alien_baby.sweeps.proprio_noise_eval",
        "--checkpoint", str(checkpoint_path),
        "--episodes", "50",
        "--noise", "1.5",
        "--out-json", str(out_path),
    ]
    rc, wall = run_subprocess(cmd, log_path, timeout=600)
    if rc == 0 and out_path.exists():
        return json.loads(out_path.read_text()), wall
    return None, wall


def write_summary(rows, sweep_dir, started_at):
    md = [
        "# Overnight Sweep Results — 2026-05-05",
        "",
        f"Sweep tag: `{SWEEP_TAG}`",
        f"Started: {started_at.isoformat(timespec='seconds')}",
        f"Finished: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Per-experiment outcomes",
        "",
        "| ID | Final touches /20 | Final falls /20 | Wall time (s) | Status |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['id']} | "
            f"{r.get('final_touches', '–')} | "
            f"{r.get('final_falls', '–')} | "
            f"{r['wall_seconds']:.0f} | "
            f"{r['status']} |"
        )
    md.append("")
    md.append("## Recommended next step")
    md.append("")
    # naive heuristic: the experiment with the highest final_touches wins
    completed = [r for r in rows if r.get("final_touches") is not None]
    if completed:
        best = max(completed, key=lambda r: r["final_touches"])
        md.append(f"Best of the night: **{best['id']}** with "
                  f"{best['final_touches']}/20 touches.")
    md.append("")
    md.append("## CSV")
    md.append("")
    md.append(f"See `{(sweep_dir / 'results.csv').name}` for raw data and "
              f"per-experiment train logs / videos in this folder.")
    (sweep_dir / "OVERNIGHT_RESULTS.md").write_text("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="",
                        help="Comma-separated experiment IDs to skip "
                             "(e.g. '4_xor_color').")
    parser.add_argument("--phase0-checkpoint",
                        default=str(RESULTS_DIR
                                    / "followon_v8_phase0_validation_best"
                                    / "best_model.zip"),
                        help="Path used for Idea D proprio-noise eval.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip end-of-sweep video rendering.")
    parser.add_argument("--no-noise-eval", action="store_true",
                        help="Skip Idea D proprio-noise eval.")
    args = parser.parse_args()

    started_at = datetime.datetime.now()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}

    rows = []
    csv_path = SWEEP_DIR / "results.csv"
    csv_fields = ["id", "wall_seconds", "returncode", "status",
                  "final_touches", "final_falls", "run_tag"]

    # Idea D — proprio-noise eval on existing Phase 0 best (5 min, no training).
    if not args.no_noise_eval:
        print("=" * 60)
        print("[noise-eval] Idea D — proprio-noise eval on Phase 0 best")
        print("=" * 60)
        ckpt_path = pathlib.Path(args.phase0_checkpoint)
        if ckpt_path.exists():
            res, wall = proprio_noise_eval(ckpt_path, SWEEP_DIR)
            print(f"  wall = {wall:.1f}s")
            if res:
                print(f"  verdict = {res.get('verdict', '?')}")
        else:
            print(f"  SKIPPED — checkpoint not found at {ckpt_path}")

    # Run each experiment in sequence
    for exp in EXPERIMENTS:
        if exp["id"] in skip_ids:
            print(f"\n[skip] {exp['id']}")
            continue
        print("\n" + "=" * 60)
        print(f"[exp] {exp['id']}: {exp['description']}")
        print("=" * 60)
        log_path = SWEEP_DIR / f"{exp['id']}.train.log"
        cmd = [sys.executable, "-m", "alien_baby.agents.train_v8"] + \
              COMMON_FLAGS + exp["flags"]
        # Force unbuffered for live tailing
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        rc, wall = run_subprocess(cmd, log_path, env=env, timeout=14400)
        run_tag = next((v for f, v in zip(exp["flags"], exp["flags"][1:])
                        if f == "--run-tag"), "")
        eval_info = parse_final_eval(log_path) or {}
        status = "ok" if rc == 0 else f"rc={rc}"
        row = {
            "id": exp["id"],
            "wall_seconds": wall,
            "returncode": rc,
            "status": status,
            "final_touches": eval_info.get("final_touches"),
            "final_falls": eval_info.get("final_falls"),
            "run_tag": run_tag,
        }
        rows.append(row)
        # Append to CSV after each experiment so we don't lose data on crash
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[exp] {exp['id']}: status={status} wall={wall:.0f}s "
              f"final_touches={row['final_touches']} "
              f"final_falls={row['final_falls']}")

    # Render best checkpoints (skip on --no-render)
    if not args.no_render:
        print("\n" + "=" * 60)
        print("[render] best-checkpoint videos")
        print("=" * 60)
        for r in rows:
            if r["status"] != "ok":
                continue
            if not r["run_tag"]:
                continue
            out_path, status = render_best(r["run_tag"], SWEEP_DIR)
            print(f"  {r['id']}: {status} -> {out_path}")

    write_summary(rows, SWEEP_DIR, started_at)
    print("\n" + "=" * 60)
    print(f"OVERNIGHT SWEEP COMPLETE — see {SWEEP_DIR}/OVERNIGHT_RESULTS.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
