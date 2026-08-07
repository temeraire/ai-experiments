"""
eval_decoy_shape.py — VISUAL object-agnosticism test on the two-ball decoy task.

The decoy policies were trained on red-SPHERE vs blue-SPHERE. This asks, zero-shot,
whether the visual choice keys on COLOR (shape-invariant — genuine visual
object-agnosticism) or is partly a SHAPE preference (it locks onto the round
trained shape). We override the two balls' geom primitive at eval time and read
choice_vs_true (fraction that reach the RED ball == target_geom).

Disentangling 2x2 (all offset=0, camera sees the real balls):
  A  red=sphere  blue=sphere   -> control, reproduce ~76-79%
  B  red=box     blue=sphere   -> if color-driven: stays high; if sphere-preferring: DROPS
  C  red=sphere  blue=box      -> if color-driven: stays high; if sphere-preferring: stays/rises
  D  red=box     blue=box      -> color discrimination when NEITHER object is the trained shape
Reading: color-driven  => B ~= C ~= A (always prefers red).
         shape-preferring(sphere) => B low, C high (always prefers the sphere).

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_decoy_shape \
    --model alien_baby/results/decoy_v2_ext_s0_best/best_model.zip \
    --red box --blue sphere --eval-eps 200 --run-tag ext_s0_Rbox_Bsph
  # sanity frames only:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_decoy_shape \
    --model ... --red box --blue sphere --sanity-frames /path/prefix
"""
import argparse
import json
import pathlib

import numpy as np
import mujoco
from PIL import Image

from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES
# reuse the exact per-episode logic the prism battery uses
from alien_baby.crawler.eval_prism_decoy import run_cell, XML

RESULTS = pathlib.Path(__file__).parent.parent / "results"

# geom primitives (type, size[3]) sized comparably to the trained sphere r=0.053.
# Matches the cart-env _SHAPE_MAP that produced clean proprio shape-transfer.
SHAPE_MAP = {
    "sphere":    (int(mujoco.mjtGeom.mjGEOM_SPHERE),    [0.053, 0.0,   0.0]),
    "box":       (int(mujoco.mjtGeom.mjGEOM_BOX),       [0.045, 0.045, 0.045]),
    "capsule":   (int(mujoco.mjtGeom.mjGEOM_CAPSULE),   [0.038, 0.038, 0.0]),
    "cylinder":  (int(mujoco.mjtGeom.mjGEOM_CYLINDER),  [0.045, 0.030, 0.0]),
    "ellipsoid": (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), [0.060, 0.045, 0.038]),
}


# named colors for the recolor / approach-vs-avoid mechanism test (rgba)
COLOR_MAP = {
    "red":   [1.0, 0.12, 0.12, 1.0],
    "blue":  [0.12, 0.30, 1.0, 1.0],
    "green": [0.12, 0.80, 0.20, 1.0],
    "yellow":[0.95, 0.85, 0.10, 1.0],
}


def make_env(red_shape, blue_shape, cone_deg, radius, max_steps, seed,
             target_color=None, decoy_color=None, offset=0.0):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=True,
        prism_offset_deg=offset,
    )

    def _set_shape(gid, shape):
        # persists: model loaded once, resets use mj_resetData which does not rewrite
        # geom_type/geom_size.
        gtype, gsize = SHAPE_MAP[shape]
        env.model.geom_type[gid] = gtype
        env.model.geom_size[gid, :] = gsize

    ghost_ids = {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in ("ghost_geom", "ghost2_geom")}

    if offset == 0.0:
        # No prism: the two REAL balls are both SEEN and touched -> override their
        # shapes. The prism ghosts are unused; hide them (alpha=0) so a stray fixed
        # red-sphere ghost cannot confound the shape test.
        _set_shape(env._target_geom_id, red_shape)
        _set_shape(env._target2_geom_id, blue_shape)
        for gid in ghost_ids.values():
            if gid >= 0:
                env.model.geom_rgba[gid, 3] = 0.0
        seen_red, seen_blue = env._target_geom_id, env._target2_geom_id
    else:
        # Prism: the head-cam sees the GHOSTS (reals are hidden+solid, only touched).
        # Override the GHOST shapes (the seen picture) and leave the real balls as the
        # trained sphere so contact physics is held constant -> isolates "does
        # follow-the-ghost depend on the SEEN shape?". Ghosts stay visible.
        _set_shape(ghost_ids["ghost_geom"], red_shape)
        _set_shape(ghost_ids["ghost2_geom"], blue_shape)
        seen_red, seen_blue = ghost_ids["ghost_geom"], ghost_ids["ghost2_geom"]

    # optional recolor (mechanism test: is the binding "approach red" or "avoid blue"?)
    # NOTE: reward is tied to target_geom regardless of its color; changing the color
    # tells us what visual cue the policy actually uses. Recolor whichever geoms are SEEN.
    if target_color is not None:
        env.model.geom_rgba[seen_red, :] = COLOR_MAP[target_color]
    if decoy_color is not None:
        env.model.geom_rgba[seen_blue, :] = COLOR_MAP[decoy_color]
    env.reset(seed=seed)
    return env


def sanity_frames(env, prefix, model=None, step_to=0):
    """Render overhead + ringside + agent left-eye. If model+step_to given, drive
    the policy step_to steps first so the eye view is a real mid-search frame."""
    obs, _ = env.reset(seed=123)
    if model is not None and step_to > 0:
        for _ in range(step_to):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    for cam, res in (("overhead", (480, 480)), ("ringside", (360, 640)),
                     ("left_eye", (200, 200))):
        r = mujoco.Renderer(env.model, res[0], res[1])
        r.update_scene(env.data, camera=cam)
        Image.fromarray(r.render()).save(f"{prefix}_{cam}.png")
        r.close()
    print(f"  wrote sanity frames {prefix}_*.png (stepped {step_to})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--red", default="sphere", choices=list(SHAPE_MAP))
    p.add_argument("--blue", default="sphere", choices=list(SHAPE_MAP))
    p.add_argument("--target-color", default=None, choices=list(COLOR_MAP),
                   help="recolor the rewarded target_geom (mechanism test)")
    p.add_argument("--decoy-color", default=None, choices=list(COLOR_MAP),
                   help="recolor the decoy target2_geom (mechanism test)")
    p.add_argument("--eval-eps", type=int, default=200)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--offset", type=float, default=0.0,
                   help="whole-field prism offset (deg). 0 = no prism (override REAL "
                        "ball shapes); nonzero = override the displaced GHOST shapes "
                        "(shape x displacement test).")
    p.add_argument("--sanity-frames", default=None,
                   help="path prefix; render frames of the scene and exit")
    p.add_argument("--run-tag", default="decoy_shape")
    args = p.parse_args()

    env = make_env(args.red, args.blue, args.cone_deg, args.radius,
                   args.max_steps, args.seed,
                   target_color=args.target_color, decoy_color=args.decoy_color,
                   offset=args.offset)
    if args.sanity_frames:
        model = PPO.load(args.model, device="cpu")
        sanity_frames(env, args.sanity_frames, model=model, step_to=60)
        return

    model = PPO.load(args.model, device="cpu")
    r = run_cell(model, env, args.eval_eps, args.max_steps,
                 np.deg2rad(args.offset), ablate=args.ablate)
    cvt = r["choice_vs_true"]
    se = (np.sqrt(cvt * (1 - cvt) / (r["red"] + r["blue"])) if cvt is not None else 0)
    print(f"\n=== {args.run_tag}  red={args.red} blue={args.blue}"
          f"  {'ABLATED' if args.ablate else 'sighted'} ===")
    print(f"  choice_vs_true (->RED): {cvt*100:.1f}% +/- {se*196:.1f}"
          f"  ({r['red']}R/{r['blue']}B, {r['neither']} neither)")
    print(f"  wrong_ball (->BLUE)   : {r['wrong_ball']*100:.1f}%")
    out = RESULTS / f"decoy_shape_{args.run_tag}.json"
    out.write_text(json.dumps({"args": vars(args), **r}, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
