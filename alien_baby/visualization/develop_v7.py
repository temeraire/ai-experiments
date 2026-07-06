"""
Render the developmental arc of v7 training.

Loads every saved checkpoint from alien_baby/results/followon_v7_checkpoints/,
rolls out one episode per checkpoint using the same seed, and concatenates
all of them into a single timelapse MP4. Each segment is labeled with its
training step count so you can literally watch the "creature" get better.

Each frame is a split-screen: overhead view + head camera view. Also writes
per-checkpoint solo videos for detailed inspection.
"""

import pathlib
import re
import numpy as np
import mujoco
import imageio

from alien_baby.envs import TabletopMovingGazeEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_v5 import ConsistencySAC
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
CHECKPOINT_DIR = RESULTS_DIR / "followon_v7_checkpoints"
RENDER_WIDTH = 480
RENDER_HEIGHT = 480


def _render_overhead_and_head(env, renderer):
    data = env.unwrapped.data
    renderer.update_scene(data, camera="overhead")
    overhead = renderer.render().copy()
    renderer.update_scene(data, camera="head_cam")
    head = renderer.render().copy()
    return overhead, head


def _list_checkpoints():
    """Return list of (step_count, path) sorted by step_count."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"{CHECKPOINT_DIR} not found — has training run?")
    paths = sorted(CHECKPOINT_DIR.glob("followon_v7_*_steps.zip"))
    out = []
    for p in paths:
        m = re.search(r"(\d+)_steps\.zip$", p.name)
        if m:
            out.append((int(m.group(1)), str(p).replace(".zip", "")))
    return sorted(out)


def render_episode(model, env, renderer, writer, max_steps=200, label=None, hold_last_frames=10):
    """Render one episode into an already-open writer. Returns (steps_taken, succeeded)."""
    obs, _ = env.reset(seed=0)  # same seed across all checkpoints for apples-to-apples
    last = None
    succeeded = False
    steps = 0
    for step in range(max_steps):
        overhead, head = _render_overhead_and_head(env, renderer)
        overhead = _add_label(overhead, "OVERHEAD")
        head = _add_label(head, "HEAD CAM")
        comp = np.concatenate([overhead, head], axis=1)
        if label is not None:
            comp = _add_label(comp, label)
        writer.append_data(comp)
        last = comp

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps = step + 1
        if terminated:
            succeeded = True
            break
        if truncated:
            break
    if last is not None:
        for _ in range(hold_last_frames):
            writer.append_data(last)
    return steps, succeeded


def render_developmental_timelapse():
    """Render one MP4 that concatenates episodes from every checkpoint."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = _list_checkpoints()
    if not checkpoints:
        raise RuntimeError("No checkpoints found to render.")

    print(f"Found {len(checkpoints)} checkpoints:")
    for steps, _ in checkpoints:
        print(f"  step {steps:>8}")

    out_path = VIDEO_DIR / "v7_developmental_timelapse.mp4"

    base_env = TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True)
    env = FlattenVisionWrapper(base_env)
    renderer = mujoco.Renderer(base_env.model, RENDER_HEIGHT, RENDER_WIDTH)

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    for steps, ckpt_path in checkpoints:
        label = f"STEP {steps}"
        print(f"  Rendering {label}...")
        model = ConsistencySAC.load(ckpt_path)
        n_steps, ok = render_episode(model, env, renderer, writer, label=label)
        status = "REACHED" if ok else "TIMEOUT"
        print(f"    {n_steps} steps, {status}")

    writer.close()
    renderer.close()
    env.close()
    print(f"\nTimelapse saved: {out_path}")
    return str(out_path)


def render_per_checkpoint_solos():
    """Also save one MP4 per checkpoint for detailed inspection."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = _list_checkpoints()
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    base_env = TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True)
    env = FlattenVisionWrapper(base_env)
    renderer = mujoco.Renderer(base_env.model, RENDER_HEIGHT, RENDER_WIDTH)

    for steps, ckpt_path in checkpoints:
        out = VIDEO_DIR / f"v7_step{steps:08d}.mp4"
        writer = imageio.get_writer(str(out), fps=25, quality=8)
        model = ConsistencySAC.load(ckpt_path)
        n_steps, ok = render_episode(model, env, renderer, writer, label=f"STEP {steps}")
        writer.close()
        print(f"  {out.name}: {n_steps} steps, {'REACHED' if ok else 'TIMEOUT'}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solos", action="store_true", help="Also render one MP4 per checkpoint")
    args = parser.parse_args()

    render_developmental_timelapse()
    if args.solos:
        render_per_checkpoint_solos()
