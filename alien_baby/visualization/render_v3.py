"""
Render v3 comparison video: blind-proprio Stage 1 alone vs. blind-proprio + vision.

Shows what the agent looks like when it cannot sense the target via proprioception
(must grope around to find it) versus when vision is available to localize it.

Also renders a v2-vs-v3 side-by-side where useful.
"""

import pathlib
import numpy as np
import mujoco
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopReachEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.visualization.render_episodes import _render_frame, _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_WIDTH = 480
RENDER_HEIGHT = 480


def _make_env(vision: bool, hide_target: bool):
    env = Monitor(TabletopReachEnv(vision=vision, target_object=0,
                                    hide_target_offset=hide_target))
    if vision:
        env = FlattenVisionWrapper(env)
    return env


def render_v3_comparison(seed=42, max_steps=200, panels_spec=None):
    """
    panels_spec: list of (label, model, vision_on, hide_target) tuples
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"comparison_v3_seed{seed}.mp4"

    envs, renderers, observations, done = {}, {}, {}, {}
    for label, model, vision, hide in panels_spec:
        env = _make_env(vision, hide)
        envs[label] = env
        renderers[label] = mujoco.Renderer(env.unwrapped.model, RENDER_HEIGHT, RENDER_WIDTH)
        obs, _ = env.reset(seed=seed)
        observations[label] = obs
        done[label] = False

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    for step in range(max_steps):
        panels = []
        for label, model, vision, hide in panels_spec:
            frame = _render_frame(envs[label], renderers[label])
            frame = _add_label(frame, label)

            if not done[label]:
                action, _ = model.predict(observations[label], deterministic=True)
                obs, reward, terminated, truncated, info = envs[label].step(action)
                observations[label] = obs
                if terminated or truncated:
                    done[label] = True

            panels.append(frame)

        composite = np.concatenate(panels, axis=1)
        writer.append_data(composite)

        if all(done.values()):
            for _ in range(25):
                writer.append_data(composite)
            break

    writer.close()
    for r in renderers.values():
        r.close()
    for e in envs.values():
        e.close()

    print(f"  Saved: {out_path}  ({step + 1} steps)")
    return str(out_path)


def main():
    stage1_v3_path = str(RESULTS_DIR / "stage1_v3_checkpoint")
    followon_v3_path = str(RESULTS_DIR / "followon_v3_checkpoint")

    for path in (stage1_v3_path, followon_v3_path):
        if not pathlib.Path(path + ".zip").exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}.zip")

    stage1_v3 = SAC.load(stage1_v3_path)
    followon_v3 = SAC.load(followon_v3_path)

    # Label, model, vision_on, hide_target_offset
    panels = [
        ("STAGE1 BLIND", stage1_v3, False, True),
        ("FOLLOWON VISION", followon_v3, True, True),
    ]

    print("Rendering v3 comparison videos...")
    for seed in [0, 1, 2]:
        render_v3_comparison(seed=seed, panels_spec=panels)


if __name__ == "__main__":
    main()
