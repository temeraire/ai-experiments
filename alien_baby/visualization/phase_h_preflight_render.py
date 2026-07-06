"""
Phase H pre-flight render — sanity-render one untrained episode at ball-speed=0.08.

Renders 2000 steps with random actions, producing an overhead + ringside + head_cam
three-panel composite video. Used to confirm:
  - Balls move at the requested speed
  - Balls bounce off all four platform edges
  - Balls are visible from head_cam as they pass
  - Balls remain on the platform for the full episode (zero touches from random actions)

Usage:
    python -m alien_baby.visualization.phase_h_preflight_render \
        --ball-speed 0.08 --steps 2000 --seed 0
"""

import pathlib
import argparse
import numpy as np
import mujoco
import imageio

from alien_baby.crawler.mimo_crawler_cart_env import (
    MimoCrawlerCartEnv, PLATFORM_TOP_Z
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_W = 480
RENDER_H = 480


def _add_label(img, text):
    """Add a text label to the top-left of an image (no PIL dependency)."""
    # Just return the image — label skipped if no PIL available
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.text((4, 4), text, fill=(255, 255, 255))
        return np.array(pil)
    except Exception:
        return img


def render_preflight(ball_speed=0.08, steps=2000, seed=0, fixed_offset=0.15):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"phase_h_preflight_speed{ball_speed:.2f}_seed{seed}.mp4"

    env = MimoCrawlerCartEnv(
        vision=True,  # need head_cam
        max_steps=steps + 10,
        strength_scale=1.0,
        n_substeps=4,
        fixed_ball_positions=[(fixed_offset, 0.35), (-fixed_offset, -0.35)],
        memory_obs=True,
        stereo=True,
        cart_speed=0.15,
        hip_actuation=False,
        ball_speed=ball_speed,
    )

    renderer_main = mujoco.Renderer(env.model, RENDER_H, RENDER_W)
    renderer_head = mujoco.Renderer(env.model, RENDER_H, RENDER_W)

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    ball1_positions = []
    ball2_positions = []
    touches = 0
    bounces_x = 0
    bounces_y = 0
    prev_vel = env._ball_vel.copy()

    for step in range(steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)

        # Track bounces (velocity sign flip)
        new_vel = env._ball_vel.copy()
        if ball_speed > 0.0:
            for i in range(2):
                if new_vel[i, 0] * prev_vel[i, 0] < 0:
                    bounces_x += 1
                if new_vel[i, 1] * prev_vel[i, 1] < 0:
                    bounces_y += 1
        prev_vel = new_vel.copy()

        b1x = float(env.data.qpos[env._tgt1_qadr])
        b1y = float(env.data.qpos[env._tgt1_qadr + 1])
        b2x = float(env.data.qpos[env._tgt2_qadr])
        b2y = float(env.data.qpos[env._tgt2_qadr + 1])
        ball1_positions.append((b1x, b1y))
        ball2_positions.append((b2x, b2y))

        if info.get("touched_ball1") or info.get("touched_ball2"):
            touches += 1

        # Render panels
        renderer_main.update_scene(env.data, camera="overhead")
        overhead = _add_label(renderer_main.render().copy(), f"OVERHEAD step={step}")
        renderer_main.update_scene(env.data, camera="ringside")
        ringside = _add_label(renderer_main.render().copy(), "RINGSIDE")
        renderer_head.update_scene(env.data, camera="left_eye")
        head = _add_label(renderer_head.render().copy(), "HEAD_CAM")

        composite = np.concatenate([overhead, ringside, head], axis=1)
        writer.append_data(composite)

        if terminated or truncated:
            break

    writer.close()
    renderer_main.close()
    renderer_head.close()
    env.close()

    # Summary stats
    b1_xs = [p[0] for p in ball1_positions]
    b1_ys = [p[1] for p in ball1_positions]
    b2_xs = [p[0] for p in ball2_positions]
    b2_ys = [p[1] for p in ball2_positions]

    print(f"\nPhase H pre-flight render complete:")
    print(f"  Video: {out_path}")
    print(f"  Steps rendered: {step + 1}")
    print(f"  Touches (should be 0 for random actions): {touches}")
    print(f"  Ball speed: {ball_speed} m/s")
    print(f"  Ball1 X range: [{min(b1_xs):.3f}, {max(b1_xs):.3f}]")
    print(f"  Ball1 Y range: [{min(b1_ys):.3f}, {max(b1_ys):.3f}]")
    print(f"  Ball2 X range: [{min(b2_xs):.3f}, {max(b2_xs):.3f}]")
    print(f"  Ball2 Y range: [{min(b2_ys):.3f}, {max(b2_ys):.3f}]")
    print(f"  X-axis bounces detected: {bounces_x}")
    print(f"  Y-axis bounces detected: {bounces_y}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball-speed", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-offset", type=float, default=0.15)
    args = parser.parse_args()
    render_preflight(args.ball_speed, args.steps, args.seed, args.fixed_offset)
