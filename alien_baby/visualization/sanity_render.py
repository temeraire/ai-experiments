"""
Pre-training sanity render — watch this BEFORE kicking off a long training run.

Renders one short episode from a freshly-initialized (untrained) SAC policy,
using the three-panel v8 composite (overhead + ringside + head_cam).

Purpose: confirm cameras, physics, spawn, and reward plumbing are sane
before burning hours of compute. If the creature spawns off-platform, the
physics explodes, or a camera is wrong, you want to know now.

Usage:
    python -m alien_baby.visualization.sanity_render
    python -m alien_baby.visualization.sanity_render --vision
    python -m alien_baby.visualization.sanity_render --seed 7 --steps 200
"""

import pathlib
import argparse
import mujoco
import imageio
import numpy as np
from stable_baselines3 import SAC

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_HEIGHT = 480
RENDER_WIDTH = 480


def sanity_render(vision=False, seed=0, steps=150):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tag = "vision" if vision else "blind"
    out_path = VIDEO_DIR / f"v8_sanity_{tag}_seed{seed}.mp4"

    env = PlatformCreatureEnv(vision=vision)
    model = SAC("MlpPolicy", env, learning_starts=10**9, verbose=0, device="cpu", seed=seed)
    renderer = mujoco.Renderer(env.model, RENDER_HEIGHT, RENDER_WIDTH)

    obs, _ = env.reset(seed=seed)
    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    camera_names = ["overhead", "ringside", "head_cam"]
    labels = ["OVERHEAD", "RINGSIDE", "HEAD CAM"]

    last = None
    fell = False
    touched = False
    steps_taken = 0
    for step in range(steps):
        panels = []
        for cam, lbl in zip(camera_names, labels):
            renderer.update_scene(env.data, camera=cam)
            panels.append(_add_label(renderer.render().copy(), lbl))
        composite = np.concatenate(panels, axis=1)
        writer.append_data(composite)
        last = composite

        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)
        steps_taken = step + 1
        if terminated:
            fell = info.get("fell", False)
            touched = info.get("touched", False)
        if terminated or truncated:
            break

    if last is not None:
        for _ in range(25):
            writer.append_data(last)

    writer.close()
    renderer.close()
    env.close()

    outcome = "TOUCHED" if touched else ("FELL" if fell else "TIMEOUT")
    print(f"Sanity video: {out_path}")
    print(f"  vision={vision}  seed={seed}  steps={steps_taken}  outcome={outcome}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", action="store_true",
                        help="Include head-cam pixels in the observation (vision mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    sanity_render(vision=args.vision, seed=args.seed, steps=args.steps)
