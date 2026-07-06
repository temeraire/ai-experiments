#!/usr/bin/env python3
"""Render the Phase XV 0.075 dead-band failure so we can SEE the mechanism.

The diagnostic (diag_0075_anomaly.py) showed a REAL, seed-independent trough in
both-touched success across ball radii ~[0.070, 0.080] (~0.45-0.55 vs ~0.95 at
0.053). The failure signature is "touches ONE ball, then stalls before the second"
(large `one` count, hugely negative reward from the hunger penalty on long
non-terminating episodes). This renders that, using the EXACT eval env config
(build_env_and_model: ecc=0.10, ball_y=0.35, fixed radius), with an overhead +
ringside panel per the project's posture-reading rule. Renders a 0.053 control too.

Usage:
    python -m alien_baby.visualization.render_0075_diag \
        --run-tag phase_x_R46_droq_proprio_sizevariety \
        --radii 0.075 0.053 --seeds 20000 20001 20002 --steps 600
"""
import argparse
import pathlib
import numpy as np
import imageio.v2 as imageio
import mujoco

from alien_baby.visualization.eval_phase_v import build_env_and_model

VIDEO_DIR = pathlib.Path(__file__).parent.parent / "results" / "videos"
RENDER_W = RENDER_H = 480
ECC = 0.10
BALL_Y = 0.35


def render_radius(run_tag, radius, seeds, steps, label):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEO_DIR / f"{label}_r{radius:.3f}_diag.mp4"
    writer = imageio.get_writer(str(out), fps=20, quality=8)

    vec_env, model, _ = build_env_and_model(
        run_tag, eccentricity=ECC, ball_speed=0.0, ball_radius=radius, ball_y=BALL_Y)
    raw = vec_env.venv.envs[0]
    renderer = mujoco.Renderer(raw.model, RENDER_H, RENDER_W)

    for seed in seeds:
        obs, _ = raw.reset(seed=seed)
        obs = vec_env.normalize_obs(obs.reshape(1, -1))
        b1 = b2 = False
        first_touch_step = None
        total_r = 0.0
        for step in range(steps):
            renderer.update_scene(raw.data, camera="overhead")
            overhead = renderer.render().copy()
            renderer.update_scene(raw.data, camera="ringside")
            ringside = renderer.render().copy()
            writer.append_data(np.concatenate([overhead, ringside], axis=1))

            action, _ = model.predict(obs, deterministic=True)
            obs_raw, rew, term, trunc, info = raw.step(action[0])
            obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
            total_r += float(rew)
            nb1, nb2 = info.get("touched_ball1", False), info.get("touched_ball2", False)
            if (nb1 or nb2) and first_touch_step is None and step > 0:
                first_touch_step = step
            b1 = b1 or nb1
            b2 = b2 or nb2
            if term or trunc:
                break
        outcome = ("BOTH" if b1 and b2 else
                   "ONE-STUCK" if (b1 ^ b2) else "NEITHER")
        print(f"  r={radius:.3f} seed={seed} steps={step+1} reward={total_r:+.1f} "
              f"b1={'T' if b1 else '-'} b2={'T' if b2 else '-'} "
              f"first_touch={first_touch_step} -> {outcome}")

    renderer.close()
    vec_env.close()
    writer.close()
    print(f"Video: {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="phase_x_R46_droq_proprio_sizevariety")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.075, 0.053])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20000, 20001, 20002])
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    label = args.run_tag.split("_")[1] if "_" in args.run_tag else args.run_tag
    print(f"Rendering {args.run_tag}  radii={args.radii} seeds={args.seeds}")
    for r in args.radii:
        render_radius(args.run_tag, r, args.seeds, args.steps, label)


if __name__ == "__main__":
    import os
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)
