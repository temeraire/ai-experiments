"""
Hand-drive test — bypass the policy and send scripted commands to the body.

Question this answers: is the v8 body *physically capable* of locomotion?
Or is it stuck-by-design no matter what controller drives it?

The XML comments say locomotion is supposed to come from both arms paddling
in unison (the `shoulder_roll` motors swing the arm forward-down-back like
a rowboat oar). So we send a sinusoidal synchronous paddle command on those
two channels, hold everything else at zero, and watch if the chassis
translates.

Outcomes:
- Chassis translates >0.3m in 8s → body is locomotion-capable, RL just
  hasn't found the policy. Move on to a locomotion-only training stage.
- Chassis stays put / twitches → body design (mass, friction, gear,
  geometry) is the bottleneck. Fix the body before training anything.

Usage:
    python -m alien_baby.visualization.hand_drive_test
    python -m alien_baby.visualization.hand_drive_test --pattern alternating
    python -m alien_baby.visualization.hand_drive_test --pattern hold
"""

import argparse
import pathlib

import imageio
import mujoco
import numpy as np

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"

RENDER_HEIGHT = 480
RENDER_WIDTH = 480
RENDER_FPS = 30

DURATION_SEC = 8.0
PADDLE_FREQ_HZ = 1.5
PADDLE_AMPLITUDE = 1.0


# Action layout (from PlatformCreatureEnv / XML actuator order):
#   0  left_shoulder_pitch    (motor)
#   1  left_shoulder_roll     (motor)  ← paddle DOF
#   2  left_elbow             (motor)
#   3  right_shoulder_pitch   (motor)
#   4  right_shoulder_roll    (motor)  ← paddle DOF
#   5  right_elbow            (motor)
#   6  head_pan               (position)
#   7  head_tilt              (position)
LEFT_PADDLE = 1
RIGHT_PADDLE = 4


def make_action(t: float, pattern: str) -> np.ndarray:
    a = np.zeros(8, dtype=np.float32)
    omega = 2 * np.pi * PADDLE_FREQ_HZ
    if pattern == "synchronous":
        s = PADDLE_AMPLITUDE * np.sin(omega * t)
        a[LEFT_PADDLE] = s
        a[RIGHT_PADDLE] = s
    elif pattern == "alternating":
        a[LEFT_PADDLE] = PADDLE_AMPLITUDE * np.sin(omega * t)
        a[RIGHT_PADDLE] = PADDLE_AMPLITUDE * np.sin(omega * t + np.pi)
    elif pattern == "hold":
        # Pin paddle at full forward swing — tests whether sustained
        # arm-on-platform contact alone produces any push.
        a[LEFT_PADDLE] = PADDLE_AMPLITUDE
        a[RIGHT_PADDLE] = PADDLE_AMPLITUDE
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return a


def hand_drive(pattern: str = "synchronous", seed: int = 0):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"hand_drive_{pattern}_seed{seed}.mp4"

    env = PlatformCreatureEnv(vision=False, stage=1)
    env.reset(seed=seed)

    timestep = float(env.model.opt.timestep)
    n_steps = int(DURATION_SEC / timestep)
    frame_skip = max(1, int(round(1.0 / (RENDER_FPS * timestep))))

    renderer = mujoco.Renderer(env.model, RENDER_HEIGHT, RENDER_WIDTH)
    writer = imageio.get_writer(str(out_path), fps=RENDER_FPS, quality=8)

    torso_id = env._torso_id
    start_pos = env.data.xpos[torso_id].copy()
    max_horiz_disp = 0.0
    trajectory = [start_pos.copy()]

    cameras = ["overhead", "ringside"]
    labels = ["OVERHEAD", "RINGSIDE"]

    for step in range(n_steps):
        t = step * timestep
        env.data.ctrl[:] = make_action(t, pattern)
        mujoco.mj_step(env.model, env.data)

        cur_pos = env.data.xpos[torso_id].copy()
        horiz = float(np.linalg.norm(cur_pos[:2] - start_pos[:2]))
        max_horiz_disp = max(max_horiz_disp, horiz)

        if step % frame_skip == 0:
            panels = []
            for cam, lbl in zip(cameras, labels):
                renderer.update_scene(env.data, camera=cam)
                panels.append(_add_label(renderer.render().copy(), lbl))
            writer.append_data(np.concatenate(panels, axis=1))
            trajectory.append(cur_pos.copy())

    writer.close()

    end_pos = env.data.xpos[torso_id].copy()
    disp = end_pos - start_pos
    horiz = float(np.linalg.norm(disp[:2]))

    print(f"Pattern:           {pattern}")
    print(f"Duration:          {DURATION_SEC:.1f}s ({n_steps} steps)")
    print(f"Start torso xy:    ({start_pos[0]:+.3f}, {start_pos[1]:+.3f})")
    print(f"End torso xy:      ({end_pos[0]:+.3f}, {end_pos[1]:+.3f})")
    print(f"Net displacement:  ({disp[0]:+.3f}, {disp[1]:+.3f})  |xy|={horiz:.3f}m")
    print(f"Max excursion:     {max_horiz_disp:.3f}m")
    print(f"Video:             {out_path}")
    return horiz, max_horiz_disp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="synchronous",
                        choices=["synchronous", "alternating", "hold", "all"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.pattern == "all":
        for p in ("synchronous", "alternating", "hold"):
            print(f"\n=== {p} ===")
            hand_drive(p, args.seed)
    else:
        hand_drive(args.pattern, args.seed)


if __name__ == "__main__":
    main()
