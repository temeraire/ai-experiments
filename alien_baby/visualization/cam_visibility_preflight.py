"""
cam_visibility_preflight.py — WINNABILITY GATE for the vision phase.

Measures whether the ball (bright red) is actually visible in the head cameras
across a bearing sweep, at the TRUE training resolution (32x32) the CNN sees.
The 2026-07-02 preflight found only +/-15 deg visible with fovy=90; this tool
lets us iterate camera geometry until +/-45 deg is a clear blob in >=1 eye.

For each bearing it settles the creature into the prone arms_fwd crawl pose,
places the ball at that bearing on the floor at a fixed radius, renders both
eyes at 32x32, and counts red pixels + centroid. Also saves an upscaled montage
(rows = left/right eye, cols = bearings) for visual inspection.

Usage:
  PYTHONPATH=<repo> python -m alien_baby.visualization.cam_visibility_preflight \
    --xml alien_baby/crawler/mimo_crawler_pos_wide.xml --fovy 90 --radius 0.75
"""
import argparse
import pathlib
import numpy as np
import mujoco
import imageio

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, CAM_H, CAM_W

OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "cam_preflight"


def red_mask(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r > 120) & (g < 90) & (b < 90)


def _tilt_quat(model, cid, alpha_deg):
    """Set a fixed head camera to look `alpha_deg` below horizontal (prone frame).
    Derivation: camera-right = world+X = head(0,-1,0); camera-up = head(-cosa,0,sina);
    camera looks down -Z_cam. R columns [X_cam, Y_cam, Z_cam] in head frame."""
    a = np.deg2rad(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [0.0, -ca, -sa],
        [-1.0, 0.0, 0.0],
        [0.0,  sa, -ca],
    ], dtype=np.float64)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    model.cam_quat[cid] = q


def measure(xml, fovy, radius, bearings, settle_steps, seed, tag, tilt_deg=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = MimoCrawlerEnv(
        vision=True, stereo=True, crawl_pose=CRAWL_POSES["arms_fwd"],
        target_obs=True, action_mode="position_offset", xml_path=xml,
        random_start_orientation=False, spawn_radius=(radius, radius),
        max_steps=5000,
    )
    model, data = env.model, env.data

    # Optional FOV / tilt overrides (applied to both eyes) without editing the XML.
    for name in ("left_eye", "right_eye"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if fovy is not None:
            model.cam_fovy[cid] = fovy
        if tilt_deg is not None:
            _tilt_quat(model, cid, tilt_deg)

    env.reset(seed=seed)
    # Settle the body into the crawl pose with neutral (pose-holding) action.
    for _ in range(settle_steps):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    settled_qpos = data.qpos.copy()

    root_qadr = model.jnt_qposadr[env._root_joint_id]
    root_xy = settled_qpos[root_qadr:root_qadr + 2].copy()
    tgt_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    tgt_qadr = model.jnt_qposadr[tgt_jid]
    floor_z = float(settled_qpos[tgt_qadr + 2])  # keep the spawn height

    rend = mujoco.Renderer(model, CAM_H, CAM_W)
    rows = {"left_eye": [], "right_eye": []}
    print(f"\n=== cam visibility [{tag}] xml={pathlib.Path(xml).name} fovy={fovy} "
          f"radius={radius} res={CAM_H}x{CAM_W} ===")
    print(f"{'bearing':>8} | {'L_red':>6} {'L_cx':>5} {'L_cy':>5} | {'R_red':>6} {'R_cx':>5} {'R_cy':>5} | in-frame")
    results = []
    for th in bearings:
        data.qpos[:] = settled_qpos
        # Forward = world +Y, lateral(right) = world +X (prone spine along +Y).
        rad = np.deg2rad(th)
        bx = root_xy[0] + radius * np.sin(rad)
        by = root_xy[1] + radius * np.cos(rad)
        data.qpos[tgt_qadr:tgt_qadr + 3] = [bx, by, floor_z]
        data.qpos[tgt_qadr + 3:tgt_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        line = {}
        for eye in ("left_eye", "right_eye"):
            rend.update_scene(data, camera=eye)
            img = rend.render().copy()
            m = red_mask(img)
            n = int(m.sum())
            if n > 0:
                ys, xs = np.nonzero(m)
                cx, cy = xs.mean() / CAM_W, ys.mean() / CAM_H
            else:
                cx, cy = -1, -1
            line[eye] = (n, cx, cy)
            rows[eye].append(img)
        (ln, lcx, lcy), (rn, rcx, rcy) = line["left_eye"], line["right_eye"]
        seen = "YES" if max(ln, rn) >= 2 else ("edge" if max(ln, rn) == 1 else "NO")
        print(f"{th:>8.0f} | {ln:>6d} {lcx:>5.2f} {lcy:>5.2f} | "
              f"{rn:>6d} {rcx:>5.2f} {rcy:>5.2f} | {seen}")
        results.append((th, ln, rn, seen))

    # Montage: rows = eyes, cols = bearings, upscaled 6x for visibility.
    scale = 6
    def up(img):
        return np.repeat(np.repeat(img, scale, 0), scale, 1)
    montage_rows = []
    for eye in ("left_eye", "right_eye"):
        montage_rows.append(np.concatenate([up(i) for i in rows[eye]], axis=1))
    montage = np.concatenate(montage_rows, axis=0)
    out = OUT_DIR / f"cam_preflight_{tag}.png"
    imageio.imwrite(str(out), montage)
    env.close()

    n_ok = sum(1 for _, _, _, s in results if s == "YES")
    span_ok = [th for th, _, _, s in results if s == "YES"]
    print(f"  montage: {out}")
    print(f"  bearings clearly visible (>=2 px): {n_ok}/{len(bearings)}  span={span_ok}")
    passed = all(s == "YES" for th, _, _, s in results if abs(th) <= 45)
    print(f"  PASS (all |bearing|<=45 visible): {passed}")
    return out, results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="alien_baby/crawler/mimo_crawler_pos_wide.xml")
    p.add_argument("--fovy", type=float, default=None, help="override both eyes' fovy")
    p.add_argument("--radius", type=float, default=0.75)
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--tilt-deg", type=float, default=None,
                   help="degrees below horizontal for both eyes (15=current XML)")
    p.add_argument("--bearings", type=float, nargs="+",
                   default=[-45, -30, -22, -15, -7, 0, 7, 15, 22, 30, 45])
    args = p.parse_args()
    measure(args.xml, args.fovy, args.radius, args.bearings,
            args.settle_steps, args.seed, args.tag, args.tilt_deg)
