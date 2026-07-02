"""
Hand-drive AFFORDANCE test for the CURRENT crawler body (MimoCrawlerEnv,
action_mode="position_offset"). Diagnostic only — no RL, no learning, and it
does NOT modify the env / XML / training code.

Question it answers (the fork left open by rnd_propulsion_400k, verdict B):
  Can this body TRANSLATE its center of mass across the floor AT ALL when the
  limbs are driven, by hand, at full amplitude in sensible gait patterns?
    - YES (net CoM displacement > ~0.05 m under some pattern) -> the gait is
      physically possible; RL simply never sampled it -> cause A -> imitation
      seeding is the indicated next step.
    - NO  (CoM only rocks/twitches in place) -> the 0.4 rad offset clamps +
      prone default pose physically block propulsion -> cause B -> widen the
      clamps / change the default pose before any further RL or imitation.

We bypass the policy: send scripted position-offset targets (a in [-1,1],
mapped by the env to ctrl = act_mid + a*act_half, so a=+/-1 hits the clamp
extremes) and step the physics directly. The body is allowed to settle into
its prone equilibrium first, so the initial spawn-drop is not miscounted as
translation.

Actuator index layout (from mimo_crawler_pos.xml <actuator> order):
   0 hip_bend  1 hip_twist  2 hip_lean  3 chest_twist  4 chest_lean
   5 head_swivel  6 head_tilt
   7 r_sh_horiz  8 r_sh_adab  9 r_sh_rot  10 r_elbow
  11 l_sh_horiz 12 l_sh_adab 13 l_sh_rot  14 l_elbow
  15 r_hip_flex 16 r_hip_ab  17 r_hip_rot 18 r_knee 19 r_ankle
  20 l_hip_flex 21 l_hip_ab  22 l_hip_rot 23 l_knee 24 l_ankle

Usage (from repo root, PYTHONPATH=<repo root>):
    python -m alien_baby.visualization.hand_drive_crawler --pattern all
"""

import argparse
import pathlib

import imageio
import mujoco
import numpy as np

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"

RENDER_H, RENDER_W, RENDER_FPS = 480, 640, 30
SETTLE_SEC = 1.0
DRIVE_SEC = 8.0
FREQ_HZ = 1.2
AMP = 1.0  # full amplitude -> clamp extremes; "can it move under the BEST commands?"

# Limb actuator indices (arms + legs). Spine/head (0-6) held neutral.
R_ARM = [7, 8, 9, 10]
L_ARM = [11, 12, 13, 14]
R_LEG = [15, 16, 17, 18, 19]
L_LEG = [20, 21, 22, 23, 24]
ALL_LIMB = R_ARM + L_ARM + R_LEG + L_LEG


def make_action(t: float, pattern: str, nu: int) -> np.ndarray:
    a = np.zeros(nu, dtype=np.float32)
    w = 2 * np.pi * FREQ_HZ
    s = AMP * np.sin(w * t)
    if pattern == "synchronous":
        # every limb paddles in unison
        for i in ALL_LIMB:
            a[i] = s
    elif pattern == "alternating":
        # diagonal crawl: (right arm + left leg) antiphase to (left arm + right leg)
        for i in R_ARM + L_LEG:
            a[i] = s
        for i in L_ARM + R_LEG:
            a[i] = -s
    elif pattern == "belly_crawl":
        # arms reach forward & pull (shoulder horiz/adab + elbow) while legs push,
        # legs a quarter-cycle behind the arms so pull and push cooperate.
        arm = AMP * np.sin(w * t)
        leg = AMP * np.sin(w * t - np.pi / 2)
        for i in R_ARM + L_ARM:
            a[i] = arm
        for i in R_LEG + L_LEG:
            a[i] = leg
    elif pattern == "hold":
        # limbs pinned at the +1 clamp extreme: static push test
        for i in ALL_LIMB:
            a[i] = AMP
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return np.clip(a, -1.0, 1.0)


def hand_drive(pattern: str, seed: int = 0, render: bool = True, xml_path=None):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_wide" if xml_path else ""
    out_path = VIDEO_DIR / f"hand_drive_crawler{tag}_{pattern}_seed{seed}.mp4"

    env = MimoCrawlerEnv(vision=False, action_mode="position_offset",
                         spawn_radius=(0.70, 0.80), xml_path=xml_path)
    env.reset(seed=seed)
    model, data = env.model, env.data
    dt = float(model.opt.timestep)
    nu = model.nu

    ra = model.jnt_qposadr[env._root_joint_id]

    def root_xy():
        return data.qpos[ra:ra + 2].copy()

    def root_xyz():
        return data.qpos[ra:ra + 3].copy()

    # 1) settle into prone equilibrium at neutral targets (a=0 -> ctrl=act_mid)
    neutral_ctrl = env._act_mid.copy()
    for _ in range(int(SETTLE_SEC / dt)):
        data.ctrl[:] = neutral_ctrl
        mujoco.mj_step(model, data)

    start_xy = root_xy()
    start_xyz = root_xyz()

    renderer = writer = None
    if render:
        renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
        writer = imageio.get_writer(str(out_path), fps=RENDER_FPS, quality=8)
    frame_skip = max(1, int(round(1.0 / (RENDER_FPS * dt))))

    n_steps = int(DRIVE_SEC / dt)
    max_excursion = 0.0
    for step in range(n_steps):
        t = step * dt
        a = make_action(t, pattern, nu)
        data.ctrl[:] = env._act_mid + a * env._act_half
        mujoco.mj_step(model, data)

        excursion = float(np.linalg.norm(root_xy() - start_xy))
        max_excursion = max(max_excursion, excursion)

        if render and step % frame_skip == 0:
            panels = []
            for cam in ("overhead", "ringside"):
                renderer.update_scene(data, camera=cam)
                panels.append(renderer.render().copy())
            writer.append_data(np.concatenate(panels, axis=1))

    if render:
        writer.close()

    end_xy = root_xy()
    end_xyz = root_xyz()
    net = float(np.linalg.norm(end_xy - start_xy))
    dz = float(end_xyz[2] - start_xyz[2])

    print(f"--- pattern: {pattern} (seed {seed}) ---")
    print(f"  start xy   : ({start_xy[0]:+.3f}, {start_xy[1]:+.3f})  z={start_xyz[2]:+.3f}")
    print(f"  end   xy   : ({end_xy[0]:+.3f}, {end_xy[1]:+.3f})  z={end_xyz[2]:+.3f}")
    print(f"  NET |dxy|  : {net:.3f} m   (dz={dz:+.3f})")
    print(f"  max excursn: {max_excursion:.3f} m")
    if render:
        print(f"  video      : {out_path}")
    return {"pattern": pattern, "net": net, "max_excursion": max_excursion,
            "dz": dz, "video": str(out_path) if render else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="all",
                    choices=["synchronous", "alternating", "belly_crawl", "hold", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--xml", default=None, help="override body XML (e.g. widened-clamp variant)")
    args = ap.parse_args()

    patterns = (["synchronous", "alternating", "belly_crawl", "hold"]
                if args.pattern == "all" else [args.pattern])
    results = [hand_drive(p, args.seed, render=not args.no_render, xml_path=args.xml)
               for p in patterns]

    print("\n==== AFFORDANCE SUMMARY ====")
    best = max(results, key=lambda r: r["net"])
    for r in results:
        print(f"  {r['pattern']:<12} net={r['net']:.3f} m  max_excursion={r['max_excursion']:.3f} m")
    print(f"\n  BEST net CoM translation: {best['net']:.3f} m  (pattern={best['pattern']})")
    if best["net"] > 0.05:
        print("  VERDICT: TRANSLATION ACHIEVABLE (>0.05 m) -> cause A (gait-discovery); imitation indicated.")
    else:
        print("  VERDICT: TRANSLATION BLOCKED (<0.05 m, in-place only) -> cause B (clamps/pose); widen clamps / change pose.")
    print(f"  Best-pattern video: {best['video']}")


if __name__ == "__main__":
    main()
