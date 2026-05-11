"""
analyze_run.py — produce the standard end-of-run report for a crawler training run.

Reads evaluations.npz to get touch-rate-over-time, renders N seed episodes,
extracts diagnostic frames, runs vision-ablation, and prints a verdict.

Usage:
    python -m alien_baby.visualization.analyze_run \\
        --run-dir alien_baby/results/mimo_substrate_A \\
        --seeds 0 1 2 3 4 \\
        --spawn-cone-deg 360 --max-steps 2000
"""

import argparse
import pathlib
import subprocess
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, PROPRIO_DIM
from alien_baby.visualization.vision_ablation import measure_ablation


def summarize_evals(run_dir: pathlib.Path):
    eval_path = run_dir / "evaluations.npz"
    if not eval_path.exists():
        return None
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    rewards   = data["results"]      # shape: (n_evals, n_eval_episodes)
    lengths   = data["ep_lengths"]   # shape: (n_evals, n_eval_episodes)
    mean_rew = rewards.mean(axis=1)
    mean_len = lengths.mean(axis=1)
    # Estimate touch rate from episode lengths: ep_len < max_steps suggests termination = contact
    return {
        "timesteps": timesteps.tolist(),
        "mean_reward": mean_rew.tolist(),
        "mean_ep_length": mean_len.tolist(),
        "ep_lengths_per_eval": lengths.tolist(),
    }


def render_episodes(run_dir: pathlib.Path, best_dir: pathlib.Path, seeds, steps,
                     vision_flag=True):
    ckpt = best_dir / "best_model.zip"
    if not ckpt.exists():
        ckpt = run_dir / "final_model.zip"
    cmd = [
        "python", "-m", "alien_baby.visualization.render_crawler",
        "--checkpoint", str(ckpt),
        "--seeds", *[str(s) for s in seeds],
        "--steps", str(steps),
        "--label", run_dir.name,
    ]
    if vision_flag:
        cmd.append("--vision")
    print(f"Rendering: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-800:])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-400:]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=list(range(15)))
    p.add_argument("--steps", type=int, default=400, help="Steps per rendered episode")
    p.add_argument("--ablation-steps", type=int, default=200)
    p.add_argument("--spawn-cone-deg", type=float, default=360.0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--frames-outdir", default="/tmp/ab_run_frames")
    p.add_argument("--no-frames", action="store_true",
                   help="Skip frame extraction (still renders videos)")
    args = p.parse_args()

    run_dir  = pathlib.Path(args.run_dir)
    best_dir = pathlib.Path(str(run_dir) + "_best")
    name     = run_dir.name

    print("=" * 70)
    print(f"ANALYSIS: {name}")
    print("=" * 70)

    # 1) Eval history from evaluations.npz
    evals = summarize_evals(run_dir)
    if evals:
        print("\n[eval history]")
        print(f"{'timestep':>10}  {'mean_rew':>10}  {'mean_ep_len':>11}  {'touch_est':>9}")
        for t, r, L, eps in zip(evals["timesteps"], evals["mean_reward"],
                                  evals["mean_ep_length"], evals["ep_lengths_per_eval"]):
            touched = sum(1 for x in eps if x < args.max_steps)
            print(f"{t:>10d}  {r:>+10.2f}  {L:>11.1f}  {touched:>4d}/{len(eps):<4d}")
    else:
        print("\n[eval history] no evaluations.npz yet")

    # 2) Render videos
    print(f"\n[rendering {len(args.seeds)} episodes from best checkpoint]")
    render_episodes(run_dir, best_dir, args.seeds, args.steps, vision_flag=True)

    # 3) Extract frames for a representative seed
    if not args.no_frames:
        first_seed = args.seeds[0]
        video = pathlib.Path(__file__).parent.parent / "results" / "videos" / f"crawler_{name}_seed{first_seed}.mp4"
        if video.exists():
            frames_dir = pathlib.Path(args.frames_outdir)
            print(f"\n[extracting frames from {video.name}]")
            subprocess.run([
                "python", "-m", "alien_baby.visualization.extract_frames",
                str(video), "--even", "8", "--outdir", str(frames_dir),
            ], check=False)
            print(f"  frames in: {frames_dir}")

    # 4) Vision ablation
    print(f"\n[vision ablation: {len(args.ablation_seeds)} seeds, "
          f"{args.ablation_steps} steps/ep]")
    ckpt = best_dir / "best_model.zip"
    if not ckpt.exists():
        ckpt = run_dir / "final_model.zip"
    vn = run_dir / "vec_normalize.pkl"
    try:
        result = measure_ablation(
            str(ckpt), str(vn) if vn.exists() else None,
            args.ablation_seeds, args.ablation_steps,
            spawn_cone_deg=args.spawn_cone_deg, max_steps=args.max_steps,
        )
        print(f"  mean delta   : {result['mean_delta']:.4f}")
        print(f"  median delta : {result['median_delta']:.4f}")
        print(f"  touch rate   : {result['touch_rate']:.0%}  "
              f"({len(result['touched_seeds'])}/{result['n_seeds']})")
        if result["mean_delta"] < 0.1:
            verdict = "VISION SILENT"
        elif result["mean_delta"] < 0.5:
            verdict = "VISION WEAKLY INFLUENCES"
        elif result["mean_delta"] < 1.5:
            verdict = "VISION LOAD-BEARING"
        else:
            verdict = "VISION DOMINATES"
        print(f"\n  VERDICT: {verdict}")
    except Exception as e:
        print(f"  ablation failed: {e}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print(f"  videos: alien_baby/results/videos/crawler_{name}_seed*.mp4")
    if not args.no_frames:
        print(f"  frames: {args.frames_outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
