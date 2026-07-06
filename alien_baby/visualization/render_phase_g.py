"""
render_phase_g.py — pre-flight sanity render for Phase G cart substrate.

Renders ONE episode (default 2000 steps) from a freshly-initialized
(untrained / random-action) policy on MimoCrawlerCartEnv.

Per CLAUDE.md: "Before launching any training run longer than a smoke test
(>50K steps), render one episode using whatever model the run will start
from. For stage 1: render from a freshly-initialized (untrained) model to
confirm cameras, physics, spawn position, and reward plumbing are sane."

What this render confirms:
  - Cart traverses platform smoothly
  - Cart bounces at all four edges (at least once)
  - AB stays rigidly mounted on the cart (does not fall off)
  - Two fixed balls are visible at (0.7, 0) and (0, 0.7)
  - Hunger cost accrues quadratically (logged to console at key steps)
  - Stereo cameras show the scene from AB's POV

Output: overhead + ringside two-panel composite at 25fps
        printed path: alien_baby/results/videos/phase_g_preflight_v1.mp4

Usage:
    python -m alien_baby.visualization.render_phase_g
    python -m alien_baby.visualization.render_phase_g --steps 2000 --seed 42
"""

import argparse
import pathlib
import os
import faulthandler
faulthandler.enable()

import numpy as np
import mujoco
import imageio

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR   = RESULTS_DIR / "videos"
RENDER_H = 480
RENDER_W = 480


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000,
                   help="Number of env steps to render (2000 = full episode).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cart-speed", type=float, default=0.15)
    p.add_argument("--hunger-base", type=float, default=0.05)
    p.add_argument("--hunger-rate", type=float, default=0.20)
    p.add_argument("--hunger-scale", type=float, default=500.0)
    p.add_argument("--out", default=None,
                   help="Output video path. Defaults to results/videos/phase_g_preflight_v1.mp4")
    args = p.parse_args()

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.out) if args.out else VIDEO_DIR / "phase_g_preflight_v1.mp4"

    env = MimoCrawlerCartEnv(
        vision=True,   # enable cameras so we confirm stereo FOV
        fixed_ball_positions=[(0.7, 0.0), (0.0, 0.7)],
        memory_obs=True,
        hip_actuation=False,
        cart_speed=args.cart_speed,
        hunger_base=args.hunger_base,
        hunger_rate=args.hunger_rate,
        hunger_scale=args.hunger_scale,
        max_steps=args.steps,
    )

    # Render renderer (overhead + ringside)
    renderer = mujoco.Renderer(env.model, RENDER_H, RENDER_W)

    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)
    last_frame = None

    bounce_count = 0
    prev_vx, prev_vy = None, None
    contact_steps = []
    cost_log = []   # (step, steps_since_contact, hunger_cost)

    print(f"Phase G pre-flight render: {args.steps} steps, seed={args.seed}")
    print(f"Cart speed: {args.cart_speed} m/s | hunger: base={args.hunger_base} "
          f"rate={args.hunger_rate} scale={args.hunger_scale}")
    print(f"Output: {out_path}")
    print()
    print(f"{'Step':>6} | {'cart_x':>7} | {'cart_y':>7} | {'cart_vx':>8} | "
          f"{'cart_vy':>8} | {'s_since_c':>9} | {'hunger_cost':>11} | event")
    print("-" * 90)

    for step in range(args.steps):
        # Render overhead + ringside panels
        panels = []
        for cam, lbl in [("overhead", "OVERHEAD"), ("ringside", "RINGSIDE")]:
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render().copy()
            panels.append(_add_label(frame, lbl))
        composite = np.concatenate(panels, axis=1)
        writer.append_data(composite)
        last_frame = composite

        # Random action (untrained policy)
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # Track bounces
        vx, vy = info["cart_vx"], info["cart_vy"]
        event = ""
        if prev_vx is not None:
            if vx * prev_vx < 0:
                bounce_count += 1
                event += f"BOUNCE_X(#{bounce_count})"
            if vy * prev_vy < 0:
                bounce_count += 1
                event += f"BOUNCE_Y(#{bounce_count})"
        prev_vx, prev_vy = vx, vy

        # Track contacts
        if info.get("touched_ball1") and (not contact_steps or contact_steps[-1] != step):
            contact_steps.append(step)
            event += " CONTACT_B1"
        if info.get("touched_ball2") and (not contact_steps or contact_steps[-1] != step):
            contact_steps.append(step)
            event += " CONTACT_B2"

        # Log every 200 steps and any bounce/contact
        if step % 200 == 0 or event:
            s = info["steps_since_contact"]
            hc = info["hunger_cost"]
            cost_log.append((step + 1, s, hc))
            print(f"{step+1:>6} | {info['cart_x']:>7.4f} | {info['cart_y']:>7.4f} | "
                  f"{vx:>8.4f} | {vy:>8.4f} | {s:>9} | {hc:>11.5f} | {event}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}: "
                  f"terminated={terminated} truncated={truncated}")
            break

    # Hold last frame for 1 second
    if last_frame is not None:
        for _ in range(25):
            writer.append_data(last_frame)
    writer.close()
    renderer.close()
    env.close()

    # Summary
    print()
    print("=" * 60)
    print("PRE-FLIGHT RENDER SUMMARY")
    print("=" * 60)
    print(f"Total steps rendered: {step + 1}")
    print(f"Wall bounces detected: {bounce_count}")
    print(f"Ball contacts: {len(contact_steps)} at steps {contact_steps}")
    print()
    print("Hunger cost spot-check:")
    print(f"  {'step':>6} | {'s_since_c':>9} | {'hunger_cost':>11}")
    for s_step, s_since, s_cost in cost_log[:12]:
        print(f"  {s_step:>6} | {s_since:>9} | {s_cost:>11.5f}")
    print()
    print(f"Video saved to: {out_path}")
    print()
    print("CHECKLIST (review video before greenlighting 250K run):")
    print(f"  [ ] Cart moves smoothly across platform")
    print(f"  [ ] Cart bounced {bounce_count} times (expect >= 2 in 2000 steps)")
    print(f"  [ ] AB stays rigidly mounted on cart (no detachment)")
    print(f"  [ ] Both balls visible at fixed positions")
    print(f"  [ ] Hunger cost is non-zero and increasing quadratically")

    os._exit(0)


if __name__ == "__main__":
    main()
