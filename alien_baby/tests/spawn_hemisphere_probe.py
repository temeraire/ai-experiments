"""Spawn-hemisphere diagnostic probe.

Loads a trained SAC checkpoint, runs N episodes at a fixed spawn radius with
deterministic action selection, and bins per-episode touch success by spawn
hemisphere (left vs right, body-frame).

Answers the question: is the policy's failure mode a directional bias (right
works, left fails) or spatial / FOV (both hemispheres fail similarly at
longer distances)?

Usage:
    python -m alien_baby.tests.spawn_hemisphere_probe \\
        --checkpoint alien_baby/results/followon_v8_v9_anneal_030to050_checkpoints/followon_v8_v9_anneal_030to050_50000_steps.zip \\
        --radius 0.40 --episodes 1000
"""

import argparse
import numpy as np
from stable_baselines3 import SAC

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv


def run(checkpoint, radius, n_episodes, seed_base=10000, deterministic=True,
        vision=True, v9=True, stage=1):
    env = PlatformCreatureEnv(
        vision=vision, v9=v9, stage=stage,
        target_radius_override=(radius, radius),
    )
    model = SAC.load(checkpoint, env=env, device="cpu")

    rows = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        # Capture initial hand-to-ball distance BEFORE first step.
        initial_dist = env._nearest_hand_dist()
        info = {}
        final_dist = initial_dist
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            final_dist = env._nearest_hand_dist()
            if term or trunc:
                break
        rows.append({
            "ep": i,
            "spawn_angle": info["spawn_angle"],
            "spawn_left": info["spawn_left"],
            "touched": bool(info["touched"]),
            "fell": bool(info["fell"]),
            "ball_lost": bool(info["ball_lost"]),
            "ep_len": info["ep_len"],
            "mean_dist": info["mean_dist"],
            "initial_dist": initial_dist,
            "final_dist": final_dist,
            "dist_reduction": initial_dist - final_dist,
        })
    env.close()
    return rows


def summarize(rows, radius):
    n = len(rows)
    touched = np.array([r["touched"] for r in rows])
    left = np.array([r["spawn_left"] for r in rows])
    angles = np.array([r["spawn_angle"] for r in rows])
    ep_lens = np.array([r["ep_len"] for r in rows])
    fell = np.array([r["fell"] for r in rows])
    lost = np.array([r["ball_lost"] for r in rows])

    init_dists = np.array([r["initial_dist"] for r in rows])
    fin_dists = np.array([r["final_dist"] for r in rows])
    dist_red = np.array([r["dist_reduction"] for r in rows])

    print(f"\n=== Spawn hemisphere probe @ r={radius} ===")
    print(f"Episodes: {n}")
    print(f"Overall touched_frac: {touched.mean():.3f}")
    print(f"Falls: {fell.sum()}   Ball-losts: {lost.sum()}")
    print(f"Mean ep_len: {ep_lens.mean():.1f}")
    print()
    print("Locomotion signal (did the creature close distance?):")
    print(f"  initial_dist  mean: {init_dists.mean():.3f}")
    print(f"  final_dist    mean: {fin_dists.mean():.3f}")
    print(f"  dist_reduction  mean: {dist_red.mean():+.3f}   "
          f"(positive = creature got closer)")
    # Among timed-out episodes only (no touch, no fall, no ball-lost), did
    # distance still shrink? This is the purest locomotion signal — touches
    # don't inflate it.
    stalled = ~touched & ~fell & ~lost
    if stalled.sum() > 0:
        print(f"  dist_reduction  timed-out only (n={stalled.sum()}): "
              f"{dist_red[stalled].mean():+.3f}")
    print()

    # Left/right hemisphere breakdown (body frame: angle in [0, π])
    mask_r = ~left  # angle <= π/2 → right hemisphere
    mask_l = left   # angle >  π/2 → left hemisphere
    print(f"RIGHT hemisphere (angle < π/2, {mask_r.sum()} episodes):")
    print(f"  touched_frac: {touched[mask_r].mean():.3f}"
          f"   mean_ep_len: {ep_lens[mask_r].mean():.1f}")
    print(f"LEFT  hemisphere (angle > π/2, {mask_l.sum()} episodes):")
    print(f"  touched_frac: {touched[mask_l].mean():.3f}"
          f"   mean_ep_len: {ep_lens[mask_l].mean():.1f}")
    print()

    # Finer-grained: 30° bins. Auto-detect half-circle (v9) vs full-circle (v8).
    max_angle = angles.max()
    if max_angle > np.pi + 0.1:
        edges = np.linspace(0, 2 * np.pi, 13)  # full-circle v8: 12 bins × 30°
        print("Per-bin (30° slices, full circle — v8 stage 1):")
    else:
        edges = np.linspace(0, np.pi, 7)  # half-circle v9: 6 bins × 30°
        print("Per-bin (30° slices from right to left — v9 front hemisphere):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (angles >= lo) & (angles < hi + 1e-6)
        if in_bin.sum() == 0:
            continue
        lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
        print(f"  [{lo_deg:5.1f}°, {hi_deg:5.1f}°]  "
              f"n={in_bin.sum():3d}  "
              f"touched_frac={touched[in_bin].mean():.3f}  "
              f"mean_ep_len={ep_lens[in_bin].mean():.1f}")

    print()
    print("Decision rule:")
    print("  Left ≈ 0%, Right ≈ 100%  → directional bias, implement mirror aug")
    print("  Left ≈ Right ≈ 30-40%   → temporal / FOV bottleneck; widen FOV or extend max_steps")
    print("  Symmetric U-shape (worst at 0° and 180°, best at 90°) → head-turning / FOV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--radius", type=float, default=0.40)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (deterministic=False). "
                             "Default is deterministic for clean diagnosis.")
    parser.add_argument("--no-vision", action="store_true",
                        help="Use blind-proprio env (matches stage-1 checkpoints).")
    parser.add_argument("--no-v9", action="store_true",
                        help="Use v8 env (static ball, slide joints) instead of v9.")
    parser.add_argument("--stage", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    rows = run(
        args.checkpoint, args.radius, args.episodes,
        seed_base=args.seed_base, deterministic=not args.stochastic,
        vision=not args.no_vision, v9=not args.no_v9, stage=args.stage,
    )
    summarize(rows, args.radius)
