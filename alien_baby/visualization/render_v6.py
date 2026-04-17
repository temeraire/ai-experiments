"""
Render v6 split-screen: overhead view + head camera view.

Left panel: overhead of the full arena (experimenter's perspective).
Right panel: what the agent sees through its head camera.

You can literally watch the target appear/disappear in the agent's view as
it learns to orient its head — the visual confirmation of gaze behavior.
"""

import pathlib
import numpy as np
import mujoco
import imageio
from stable_baselines3 import SAC

from alien_baby.envs import TabletopGazeEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_v5 import ConsistencySAC
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_WIDTH = 480
RENDER_HEIGHT = 480


def _render_overhead_and_head(env, renderer):
    """Return (overhead_frame, head_cam_frame), both RENDER_HEIGHT x RENDER_WIDTH."""
    data = env.unwrapped.data
    renderer.update_scene(data, camera="overhead")
    overhead = renderer.render().copy()
    renderer.update_scene(data, camera="head_cam")
    head = renderer.render().copy()
    return overhead, head


def render_v6_episode(model, seed=0, max_steps=200, label=None, wrap_vision=True):
    """Render one episode split-screen (overhead + head cam) and save as MP4."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"v6_{label or 'episode'}_seed{seed}.mp4"
    out_path = VIDEO_DIR / out_name

    base_env = TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True)
    env = FlattenVisionWrapper(base_env) if wrap_vision else base_env

    renderer = mujoco.Renderer(base_env.model, RENDER_HEIGHT, RENDER_WIDTH)

    obs, _ = env.reset(seed=seed)
    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    last_composite = None
    steps_taken = 0
    for step in range(max_steps):
        overhead, head = _render_overhead_and_head(env, renderer)
        overhead = _add_label(overhead, "OVERHEAD")
        head = _add_label(head, "HEAD CAM")
        composite = np.concatenate([overhead, head], axis=1)
        writer.append_data(composite)
        last_composite = composite

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps_taken = step + 1
        if terminated or truncated:
            break

    # Hold last frame briefly
    if last_composite is not None:
        for _ in range(25):
            writer.append_data(last_composite)

    writer.close()
    renderer.close()
    env.close()
    print(f"  Saved: {out_path}  ({steps_taken} steps)")
    return str(out_path)


def render_v6_comparison(stage1, followon, seed=0, max_steps=200):
    """
    Stage 1 (blind, no vision) vs follow-on (with gaze camera), same seed.
    2x2 grid: [S1 overhead, S1 head-cam-placeholder] / [FO overhead, FO head-cam].
    Stage 1 doesn't really "see" — we still render its head cam for visual parity,
    but the stage-1 agent's head is uncontrolled (just whatever stage 1 learned,
    which is probably nothing useful).
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"v6_comparison_seed{seed}.mp4"

    s1_env = TabletopGazeEnv(vision=False, target_object=0, hide_target_offset=True)
    fo_base = TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True)
    fo_env = FlattenVisionWrapper(fo_base)

    s1_renderer = mujoco.Renderer(s1_env.model, RENDER_HEIGHT, RENDER_WIDTH)
    fo_renderer = mujoco.Renderer(fo_base.model, RENDER_HEIGHT, RENDER_WIDTH)

    s1_obs, _ = s1_env.reset(seed=seed)
    fo_obs, _ = fo_env.reset(seed=seed)

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    s1_done = False
    fo_done = False
    last_comp = None
    steps_taken = 0

    for step in range(max_steps):
        s1_over, s1_head = _render_overhead_and_head(s1_env, s1_renderer)
        fo_over, fo_head = _render_overhead_and_head(fo_env, fo_renderer)

        s1_over = _add_label(s1_over, "STAGE1 OVERHEAD")
        s1_head = _add_label(s1_head, "STAGE1 HEAD")
        fo_over = _add_label(fo_over, "FOLLOWON OVERHEAD")
        fo_head = _add_label(fo_head, "FOLLOWON HEAD")

        top = np.concatenate([s1_over, s1_head], axis=1)
        bot = np.concatenate([fo_over, fo_head], axis=1)
        comp = np.concatenate([top, bot], axis=0)
        writer.append_data(comp)
        last_comp = comp

        if not s1_done:
            a, _ = stage1.predict(s1_obs, deterministic=True)
            s1_obs, _, term, trunc, _ = s1_env.step(a)
            if term or trunc:
                s1_done = True
        if not fo_done:
            a, _ = followon.predict(fo_obs, deterministic=True)
            fo_obs, _, term, trunc, _ = fo_env.step(a)
            if term or trunc:
                fo_done = True
        steps_taken = step + 1
        if s1_done and fo_done:
            break

    if last_comp is not None:
        for _ in range(25):
            writer.append_data(last_comp)

    writer.close()
    s1_renderer.close()
    fo_renderer.close()
    s1_env.close()
    fo_env.close()
    print(f"  Saved: {out_path}  ({steps_taken} steps)")
    return str(out_path)


def main():
    stage1_path = str(RESULTS_DIR / "stage1_v6_checkpoint")
    followon_path = str(RESULTS_DIR / "followon_v6_checkpoint")

    for path in (stage1_path, followon_path):
        if not pathlib.Path(path + ".zip").exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}.zip")

    stage1 = SAC.load(stage1_path)
    followon = ConsistencySAC.load(followon_path)

    print("Rendering v6 solo videos (follow-on only)...")
    for seed in [0, 1, 2]:
        render_v6_episode(followon, seed=seed, label="followon")

    print("\nRendering v6 stage1-vs-followon comparisons...")
    for seed in [0, 1, 2]:
        render_v6_comparison(stage1, followon, seed=seed)


if __name__ == "__main__":
    main()
