"""
Pose search for a crawl-READY default pose (minimal-change track, 2026-07-02).

Literature (scout 2026-07-01): position-offset control only works when the offsets
are taken around a default pose already close to the target behavior ("A Walk in the
Park"). Our default is flat prone -> offsets never cross into a propulsive stance.
This tool searches candidate default poses by RE-CENTERING each limb actuator's
ctrlrange on a candidate joint angle (in-memory, no XML churn), then scores the pose
by the SAME objective that matters: net hand-driven CoM translation across the 4 crawl
gaits. Best pose wins; only the winner gets rendered.

No env/XML/training files are modified. Diagnostic only.

Run:  PYTHONPATH=<repo root> python -m alien_baby.visualization.pose_search_crawler
"""
import pathlib
import mujoco
import numpy as np

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv
from alien_baby.visualization.hand_drive_crawler import (
    make_action, R_ARM, L_ARM, R_LEG, L_LEG, ALL_LIMB,
    SETTLE_SEC, DRIVE_SEC, VIDEO_DIR,
)

WIDE_XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"

# Candidate default poses: {actuator_index: center_angle_rad}. Unlisted -> keep 0
# (symmetric-range midpoint). Signs are guesses; the translation score corrects them.
POSES = {
    "neutral_prone": {},  # baseline: prone, ranges centered at 0 (== current wide body)
    "belly_prop": {
        7: 0.6, 11: 0.6,      # shoulders swung forward (arms ahead on ground)
        8: -0.4, 12: -0.4,    # shoulders add/ab in toward midline
        10: -0.6, 14: -0.6,   # elbows bent to prop
        15: -0.6, 20: -0.6,   # hips flexed (knees drawn forward under body)
        18: -0.9, 23: -0.9,   # knees bent
        19: -0.3, 24: -0.3,   # ankles
    },
    "belly_prop_neg": {       # same but flip the ambiguous-sign joints
        7: -0.6, 11: -0.6,
        8: 0.4, 12: 0.4,
        10: -0.6, 14: -0.6,
        15: -0.6, 20: -0.6,
        18: -0.9, 23: -0.9,
        19: 0.3, 24: 0.3,
    },
    "arms_fwd_prone": {       # legs neutral, arms reaching forward for a commando drag
        7: 0.7, 11: 0.7,
        10: -0.4, 14: -0.4,
    },
    "arms_fwd_neg": {
        7: -0.7, 11: -0.7,
        10: -0.4, 14: -0.4,
    },
    "low_crouch": {           # deeper flexion to lift belly toward hands-and-knees
        8: -0.5, 12: -0.5,
        10: -0.8, 14: -0.8,
        15: -0.9, 20: -0.9,
        18: -1.2, 23: -1.2,
        19: -0.4, 24: -0.4,
    },
    # lower / wider base (belly-down, arms out for stability) — less tip-prone
    "belly_wide": {
        7: 0.5, 11: 0.5,      # arms forward
        8: 0.5, 12: 0.5,      # arms OUT (abduct) for a wide base
        10: -0.5, 14: -0.5,   # elbows slightly bent
        15: -0.5, 20: -0.5,   # hips mild flex
        18: -0.7, 23: -0.7,   # knees mild bend
        16: 0.4, 21: 0.4,     # hips abduct (legs out = wide base)
    },
    "belly_wide_negarm": {
        7: -0.5, 11: -0.5,
        8: 0.5, 12: 0.5,
        10: -0.5, 14: -0.5,
        15: -0.5, 20: -0.5,
        18: -0.7, 23: -0.7,
        16: 0.4, 21: 0.4,
    },
    "shallow_crouch": {       # low_crouch but less knee flex (lower CoM, wider base)
        8: -0.4, 12: -0.4,
        10: -0.6, 14: -0.6,
        15: -0.6, 20: -0.6,
        18: -0.8, 23: -0.8,
        16: 0.3, 21: 0.3,     # slight hip abduction for base width
        19: -0.3, 24: -0.3,
    },
}


def build_env_with_pose(centers):
    """Load wide body, re-center limb actuator ranges on `centers`, keep half-widths."""
    env = MimoCrawlerEnv(vision=False, action_mode="position_offset",
                         xml_path=WIDE_XML, spawn_radius=(0.70, 0.80))
    m = env.model
    lo = m.actuator_ctrlrange[:, 0].copy()
    hi = m.actuator_ctrlrange[:, 1].copy()
    half = (hi - lo) / 2.0
    # joint hard limits (to clamp the recentred range)
    jlo = np.full(m.nu, -np.inf); jhi = np.full(m.nu, np.inf)
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        if m.jnt_limited[jid]:
            jlo[i], jhi[i] = m.jnt_range[jid]
    new_lo, new_hi = lo.copy(), hi.copy()
    for i, c in centers.items():
        h = min(half[i], c - jlo[i], jhi[i] - c) if np.isfinite(jlo[i]) else half[i]
        h = max(h, 0.05)
        new_lo[i] = c - h
        new_hi[i] = c + h
    m.actuator_ctrlrange[:, 0] = new_lo
    m.actuator_ctrlrange[:, 1] = new_hi
    env._act_lo = new_lo
    env._act_hi = new_hi
    env._act_mid = (new_lo + new_hi) / 2.0
    env._act_half = (new_hi - new_lo) / 2.0
    return env


def _up_tilt(R_start, R_end):
    """Angle (deg) the body's dorsal(up) axis tilts from world-up, ignoring yaw.
    up_local = R_start^T @ world_up (the body-frame vector pointing up when settled);
    tilt = angle between R_end @ up_local and world_up."""
    up_local = R_start.T @ np.array([0.0, 0.0, 1.0])
    up_end = R_end @ up_local
    c = np.clip(up_end[2] / (np.linalg.norm(up_end) + 1e-9), -1, 1)
    return float(np.degrees(np.arccos(c)))


def _settle_pose(env, centers, seed=0):
    m, d = env.model, env.data
    dt = float(m.opt.timestep)
    env.reset(seed=seed)
    for i, c in centers.items():
        d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] = c
    mujoco.mj_forward(m, d)
    for _ in range(int(SETTLE_SEC / dt)):
        d.ctrl[:] = env._act_mid
        mujoco.mj_step(m, d)


def score_pose(name, centers, seed=0):
    """Score = axial forward translation |dy| while belly-down. Posture-aware:
    disqualify poses that tip (up-axis tilt > 35 deg) or that drift while static."""
    env = build_env_with_pose(centers)
    m, d = env.model, env.data
    dt = float(m.opt.timestep)
    ra = m.jnt_qposadr[env._root_joint_id]
    bid = m.jnt_bodyid[env._root_joint_id]
    nu = m.nu

    # --- static-stability gate: settle then hold neutral 3s, must barely move / not tip
    _settle_pose(env, centers, seed)
    s_xy = d.qpos[ra:ra + 2].copy(); R0 = d.xmat[bid].reshape(3, 3).copy()
    for _ in range(int(3.0 / dt)):
        d.ctrl[:] = env._act_mid; mujoco.mj_step(m, d)
    static_drift = float(np.linalg.norm(d.qpos[ra:ra + 2] - s_xy))
    static_tilt = _up_tilt(R0, d.xmat[bid].reshape(3, 3))
    static_ok = static_drift < 0.08 and static_tilt < 20 and d.qpos[ra + 2] > 1.6

    results = {}
    best_fwd = 0.0
    for pattern in ("synchronous", "alternating", "belly_crawl", "hold"):
        _settle_pose(env, centers, seed)
        start = d.qpos[ra:ra + 2].copy(); Rs = d.xmat[bid].reshape(3, 3).copy()
        for step in range(int(DRIVE_SEC / dt)):
            a = make_action(step * dt, pattern, nu)
            d.ctrl[:] = env._act_mid + a * env._act_half
            mujoco.mj_step(m, d)
        dxy = d.qpos[ra:ra + 2] - start
        dy = float(dxy[1]); dx = float(dxy[0])          # forward = +/-Y (spine axis)
        tilt = _up_tilt(Rs, d.xmat[bid].reshape(3, 3))
        zf = float(d.qpos[ra + 2])
        tipped = tilt > 35 or not np.isfinite(dy) or zf < 1.6
        fwd = 0.0 if tipped else abs(dy)
        results[pattern] = {"fwd": fwd, "dy": dy, "dx": dx, "tilt": tilt, "zf": zf,
                            "tipped": tipped}
        best_fwd = max(best_fwd, fwd)

    bp = max(results, key=lambda p: results[p]["fwd"])
    r = results[bp]
    flags = ("" if static_ok else "STATIC-UNSTABLE ") + ("" if best_fwd > 0 else "ALL-TIPPED")
    print(f"  {name:16s} fwd={best_fwd:.3f}m via {bp:11s} "
          f"(dy={r['dy']:+.3f} dx={r['dx']:+.3f} tilt={r['tilt']:.0f}deg) "
          f"static[drift={static_drift:.3f} tilt={static_tilt:.0f}] {flags}")
    return {"name": name, "best": best_fwd, "static_ok": static_ok,
            "best_pattern": bp, "by_pattern": results, "centers": centers}


def main():
    print("POSE SEARCH — axial forward translation (belly-down) per candidate default pose")
    scored = [score_pose(n, c) for n, c in POSES.items()]
    # prefer statically-stable poses that actually translate forward
    ok = [s for s in scored if s["static_ok"] and s["best"] > 0]
    winner = max(ok or scored, key=lambda s: s["best"])
    print(f"\nWINNER: {winner['name']}  forward={winner['best']:.3f} m  "
          f"(static_ok={winner['static_ok']})")
    print("centers:", winner["centers"])
    best_pat = winner["best_pattern"]
    env = build_env_with_pose(winner["centers"])
    m, d = env.model, env.data
    dt = float(m.opt.timestep); ra = m.jnt_qposadr[env._root_joint_id]
    import imageio
    from alien_baby.visualization.hand_drive_crawler import RENDER_H, RENDER_W, RENDER_FPS
    env.reset(seed=0)
    for i, c in winner["centers"].items():
        d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] = c
    mujoco.mj_forward(m, d)
    for _ in range(int(SETTLE_SEC / dt)):
        d.ctrl[:] = env._act_mid; mujoco.mj_step(m, d)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEO_DIR / f"pose_search_winner_{winner['name']}_{best_pat}.mp4"
    r = mujoco.Renderer(m, RENDER_H, RENDER_W)
    w = imageio.get_writer(str(out), fps=RENDER_FPS, quality=8)
    fs = max(1, int(round(1.0 / (RENDER_FPS * dt))))
    for step in range(int(DRIVE_SEC / dt)):
        a = make_action(step * dt, best_pat, m.nu)
        d.ctrl[:] = env._act_mid + a * env._act_half
        mujoco.mj_step(m, d)
        if step % fs == 0:
            panels = []
            for cam in ("overhead", "ringside"):
                r.update_scene(d, camera=cam); panels.append(r.render().copy())
            w.append_data(np.concatenate(panels, axis=1))
    w.close()
    print(f"winner video: {out}")


if __name__ == "__main__":
    main()
