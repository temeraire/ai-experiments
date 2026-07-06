"""
Render agent episodes as MP4 videos.

Produces:
- Individual agent videos (one per agent)
- Side-by-side comparison video (all agents on the same seed)
"""

import pathlib
import numpy as np
import mujoco
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopReachEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_WIDTH = 480
RENDER_HEIGHT = 480


def _render_frame(env, renderer):
    """Render a high-res frame from the environment."""
    renderer.update_scene(env.unwrapped.data, camera="overhead")
    return renderer.render().copy()


def _add_label(frame, text):
    """Burn a simple text label into the top-left of the frame."""
    # Simple approach: white rectangle with dark text using pixel drawing
    # Since we don't want to add PIL as a dependency, we'll use a colored bar
    h, w = frame.shape[:2]
    bar_h = 30
    frame[:bar_h, :, :] = frame[:bar_h, :, :] // 3  # darken top bar

    # Encode text as simple block letters (good enough for labels)
    # Each char is 5 pixels wide
    x_offset = 10
    y_offset = 8
    for ch in text:
        if x_offset > w - 20:
            break
        _draw_char(frame, ch, x_offset, y_offset)
        x_offset += 7
    return frame


# Minimal 5x5 pixel font for labeling
_FONT = {
    'S': [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
    'T': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
    'A': [[0,1,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
    'G': [[1,1,1],[1,0,0],[1,0,1],[1,0,1],[1,1,1]],
    'E': [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,1,1]],
    'D': [[1,1,0],[1,0,1],[1,0,1],[1,0,1],[1,1,0]],
    'L': [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1]],
    '-': [[0,0,0],[0,0,0],[1,1,1],[0,0,0],[0,0,0]],
    'O': [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
    'N': [[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1]],
    'C': [[1,1,1],[1,0,0],[1,0,0],[1,0,0],[1,1,1]],
    'F': [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,0,0]],
    'U': [[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
    'I': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
    'R': [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,0,1]],
    'B': [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,1,0]],
    'W': [[1,0,1],[1,0,1],[1,0,1],[1,1,1],[1,0,1]],
    'V': [[1,0,1],[1,0,1],[1,0,1],[1,0,1],[0,1,0]],
    'H': [[1,0,1],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
    'M': [[1,0,1],[1,1,1],[1,0,1],[1,0,1],[1,0,1]],
    'P': [[1,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0]],
    'Y': [[1,0,1],[1,0,1],[0,1,0],[0,1,0],[0,1,0]],
    'K': [[1,0,1],[1,1,0],[1,0,0],[1,1,0],[1,0,1]],
    '0': [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
    '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
    '2': [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
    '3': [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],
    ' ': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
}


def _draw_char(frame, ch, x, y, scale=2):
    ch = ch.upper()
    glyph = _FONT.get(ch)
    if glyph is None:
        return
    for row_i, row in enumerate(glyph):
        for col_i, val in enumerate(row):
            if val:
                py = y + row_i * scale
                px = x + col_i * scale
                frame[py:py+scale, px:px+scale, :] = 255


def render_single_agent(model, name, seed=42, max_steps=200):
    """Render one episode of an agent and save as MP4."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{name}.mp4"

    is_vision = name in ("staged", "allatonce", "fusion")
    env = Monitor(TabletopReachEnv(vision=is_vision, target_object=0))
    if is_vision:
        env = FlattenVisionWrapper(env)

    # High-res renderer for video
    mj_model = env.unwrapped.model
    renderer = mujoco.Renderer(mj_model, RENDER_HEIGHT, RENDER_WIDTH)

    obs, _ = env.reset(seed=seed)
    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    for step in range(max_steps):
        frame = _render_frame(env, renderer)
        frame = _add_label(frame, name)
        writer.append_data(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # Hold last frame for a beat
            for _ in range(15):
                writer.append_data(frame)
            break

    writer.close()
    renderer.close()
    env.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


def render_comparison(agents_dict, seed=42, max_steps=200, label=None):
    """
    Render all agents side-by-side on the same seed.
    agents_dict: {"name": model, ...}
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"comparison_{label}.mp4" if label else "comparison.mp4"
    out_path = VIDEO_DIR / fname
    n_agents = len(agents_dict)

    # Set up environments and renderers for each agent
    envs = {}
    renderers = {}
    observations = {}

    for name, model in agents_dict.items():
        is_vision = name in ("staged", "allatonce", "fusion")
        env = Monitor(TabletopReachEnv(vision=is_vision, target_object=0))
        if is_vision:
            env = FlattenVisionWrapper(env)
        envs[name] = env
        renderers[name] = mujoco.Renderer(env.unwrapped.model, RENDER_HEIGHT, RENDER_WIDTH)
        obs, _ = env.reset(seed=seed)
        observations[name] = obs

    # Composite frame: agents side by side
    comp_w = RENDER_WIDTH * n_agents
    writer = imageio.get_writer(str(out_path), fps=25, quality=8)
    done = {name: False for name in agents_dict}

    for step in range(max_steps):
        panels = []
        for name, model in agents_dict.items():
            frame = _render_frame(envs[name], renderers[name])
            frame = _add_label(frame, name)

            if not done[name]:
                action, _ = model.predict(observations[name], deterministic=True)
                obs, reward, terminated, truncated, info = envs[name].step(action)
                observations[name] = obs
                if terminated or truncated:
                    done[name] = True

            panels.append(frame)

        composite = np.concatenate(panels, axis=1)
        writer.append_data(composite)

        if all(done.values()):
            # Hold final frame
            for _ in range(25):
                writer.append_data(composite)
            break

    writer.close()
    for r in renderers.values():
        r.close()
    for e in envs.values():
        e.close()

    print(f"  Saved comparison: {out_path}")
    return str(out_path)


def render_all(n_seeds=3):
    """Render individual and comparison videos for all agents."""
    print("\n" + "=" * 60)
    print("RENDERING AGENT VIDEOS")
    print("=" * 60)

    agents = {}
    for name, path in [
        ("staged", str(RESULTS_DIR / "stage2_checkpoint")),
        ("allatonce", str(RESULTS_DIR / "allatonce_checkpoint")),
        ("fusion", str(RESULTS_DIR / "fusion_checkpoint")),
    ]:
        if pathlib.Path(path + ".zip").exists():
            agents[name] = SAC.load(path)
            print(f"  Loaded {name}")

    # Also load Stage 1 for comparison
    s1_path = str(RESULTS_DIR / "stage1_checkpoint")
    if pathlib.Path(s1_path + ".zip").exists():
        agents["stage1-proprio"] = SAC.load(s1_path)
        print(f"  Loaded stage1-proprio")

    # Individual agent videos
    for name, model in agents.items():
        render_single_agent(model, name, seed=42)

    # Side-by-side comparison (use the 3 main agents)
    comparison_agents = {k: v for k, v in agents.items() if k != "stage1-proprio"}
    for seed in range(n_seeds):
        print(f"\n  Comparison seed {seed}:")
        render_comparison(comparison_agents, seed=seed, label=f"seed{seed}")

    print(f"\n  All videos saved to: {VIDEO_DIR}")
    return str(VIDEO_DIR)


if __name__ == "__main__":
    render_all()
