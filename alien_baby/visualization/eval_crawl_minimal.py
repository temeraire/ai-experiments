"""
eval_crawl_minimal.py — quantitative evaluation for crawl_minimal_400k.

Answers three questions:
  1. Contact rate: fraction of episodes that touched the ball vs timed out vs tip-terminated.
  2. Is it crawling? CoM net displacement and ball distance change per episode.
  3. Video: render 3 deterministic episodes (overhead + ringside).

Usage:
    python -m alien_baby.visualization.eval_crawl_minimal
"""

import pathlib
import numpy as np
import mujoco
import imageio

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES
from alien_baby.crawler.rnd_wrapper import RNDRewardWrapper

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR   = RESULTS_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

CKPT = RESULTS_DIR / "crawl_minimal_400k_best" / "best_model.zip"
VN_PATH = RESULTS_DIR / "crawl_minimal_400k" / "vec_normalize.pkl"

# Match train_crawler args exactly:
XML_PATH      = "alien_baby/crawler/mimo_crawler_pos_wide.xml"
CRAWL_POSE    = CRAWL_POSES["arms_fwd"]           # {7: 0.7, 11: 0.7, 10: -0.4, 14: -0.4}
TERM_TILT     = 50.0
TIP_PENALTY   = -5.0
SPAWN_RADIUS  = (0.70, 0.80)
ACTION_MODE   = "position_offset"
STRENGTH      = 0.7
MAX_STEPS     = 600
APPROACH_SCALE = 10.0
VELOCITY_BONUS = 0.0
STEP_COST     = 0.0

N_EVAL_EPS    = 30
RENDER_SEEDS  = [0, 1, 2]
RENDER_STEPS  = 600


def make_eval_env():
    """Reconstruct the eval env exactly as train_crawler builds it."""
    def _init():
        env = MimoCrawlerEnv(
            vision=False,
            strength_scale=STRENGTH,
            spawn_cone_deg=180,
            max_steps=MAX_STEPS,
            n_substeps=4,
            approach_reward_scale=APPROACH_SCALE,
            velocity_bonus_scale=VELOCITY_BONUS,
            random_start_orientation=False,
            memory_obs=False,
            stereo=True,
            action_mode=ACTION_MODE,
            spawn_radius=SPAWN_RADIUS,
            step_cost=STEP_COST,
            xml_path=XML_PATH,
            crawl_pose=CRAWL_POSE,
            terminate_tilt_deg=TERM_TILT,
            tip_penalty=TIP_PENALTY,
        )
        return env
    return _init


def load_model():
    """Load best_model.zip with matching VecNormalize stats."""
    vec_env = DummyVecEnv([make_eval_env()])
    if VN_PATH.exists():
        vec_env = VecNormalize.load(str(VN_PATH), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  VecNormalize stats loaded from: {VN_PATH}")
    else:
        print("  WARNING: no vec_normalize.pkl found — running without obs normalization")

    # RND is pass-through on eval (augment=False) — must match train wrapper depth
    vec_env = RNDRewardWrapper(vec_env, augment=False, verbose=0)

    model = SAC.load(str(CKPT), env=vec_env, device="cpu")
    print(f"  Model loaded from: {CKPT}")
    return model, vec_env


# --------------------------------------------------------------------------
# QUANTITATIVE EVAL
# --------------------------------------------------------------------------

def run_eval(model, vec_env, n_eps=N_EVAL_EPS, seed_offset=0):
    """
    Run n_eps deterministic episodes. For each, record:
      - touched (bool)
      - tip_terminated (bool)
      - timed_out (bool)
      - start_dist: ball distance at episode start
      - end_dist:   ball distance at episode end
      - com_displacement: net XY CoM translation over episode
    """
    results = []
    # We need an INNER env to track physics state directly (start pos, ball pos)
    inner_env = MimoCrawlerEnv(
        vision=False,
        strength_scale=STRENGTH,
        spawn_cone_deg=180,
        max_steps=MAX_STEPS,
        n_substeps=4,
        approach_reward_scale=APPROACH_SCALE,
        velocity_bonus_scale=VELOCITY_BONUS,
        random_start_orientation=False,
        memory_obs=False,
        stereo=True,
        action_mode=ACTION_MODE,
        spawn_radius=SPAWN_RADIUS,
        step_cost=STEP_COST,
        xml_path=XML_PATH,
        crawl_pose=CRAWL_POSE,
        terminate_tilt_deg=TERM_TILT,
        tip_penalty=TIP_PENALTY,
    )

    for ep in range(n_eps):
        seed = seed_offset + ep * 7  # deterministic but spread out
        obs_raw, _ = inner_env.reset(seed=seed)

        # Record start CoM (root body XY)
        root_qadr = inner_env.model.jnt_qposadr[inner_env._root_joint_id]
        start_xy = inner_env.data.qpos[root_qadr:root_qadr + 2].copy()
        start_dist = inner_env._ball_dist()

        # Normalize obs using VecNormalize stats
        obs_norm = obs_raw.copy()
        if hasattr(vec_env, "obs_rms"):
            # Walk down the wrapper stack to find VecNormalize
            vn = vec_env
            while not isinstance(vn, VecNormalize):
                vn = vn.venv
            obs_norm = np.clip(
                (obs_raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                -vn.clip_obs, vn.clip_obs,
            ).astype(np.float32)

        done = False
        touched = False
        tip_terminated = False
        steps_taken = 0

        while not done:
            action, _ = model.predict(obs_norm, deterministic=True)
            obs_raw, reward, terminated, truncated, info = inner_env.step(action)

            steps_taken += 1
            touched = info.get("touched", False)
            done = terminated or truncated

            # Re-normalize obs
            obs_norm = obs_raw.copy()
            if hasattr(vec_env, "obs_rms"):
                vn = vec_env
                while not isinstance(vn, VecNormalize):
                    vn = vn.venv
                obs_norm = np.clip(
                    (obs_raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                    -vn.clip_obs, vn.clip_obs,
                ).astype(np.float32)

            if terminated and not touched:
                # terminated without touch = tip-terminated
                tip_terminated = True

        end_xy = inner_env.data.qpos[root_qadr:root_qadr + 2].copy()
        end_dist = inner_env._ball_dist()
        com_displacement = float(np.linalg.norm(end_xy - start_xy))
        dist_change = start_dist - end_dist  # positive = moved TOWARD ball

        timed_out = (steps_taken >= MAX_STEPS) and not touched and not tip_terminated

        results.append({
            "ep": ep,
            "seed": seed,
            "touched": touched,
            "tip_terminated": tip_terminated,
            "timed_out": timed_out,
            "steps": steps_taken,
            "start_dist": start_dist,
            "end_dist": end_dist,
            "dist_change": dist_change,
            "com_displacement": com_displacement,
        })

        status = "TOUCH" if touched else ("TIP" if tip_terminated else "TIMEOUT")
        print(f"  ep {ep:2d} [{status:7}] steps={steps_taken:3d}  "
              f"start_d={start_dist:.3f}  end_d={end_dist:.3f}  "
              f"Δd={dist_change:+.3f}  CoM_disp={com_displacement:.3f}")

    inner_env.close()
    return results


def print_summary(results):
    n = len(results)
    n_touched = sum(r["touched"] for r in results)
    n_tip     = sum(r["tip_terminated"] for r in results)
    n_timeout = sum(r["timed_out"] for r in results)

    touch_disps    = [r["com_displacement"] for r in results if r["touched"]]
    notouched_disps = [r["com_displacement"] for r in results if not r["touched"]]
    all_disps      = [r["com_displacement"] for r in results]
    dist_changes   = [r["dist_change"] for r in results]
    start_dists    = [r["start_dist"] for r in results]
    end_dists      = [r["end_dist"] for r in results]

    print("\n" + "=" * 70)
    print("CONTACT RATE TABLE")
    print("=" * 70)
    print(f"  N episodes: {n}")
    print(f"  Contacts (touched ball):   {n_touched}/{n}  ({100*n_touched/n:.1f}%)")
    print(f"  Tip-terminated:            {n_tip}/{n}     ({100*n_tip/n:.1f}%)")
    print(f"  Timed-out (600 steps):     {n_timeout}/{n}    ({100*n_timeout/n:.1f}%)")

    print("\nDISPLACEMENT / DISTANCE TABLE")
    print(f"  Mean start distance to ball:   {np.mean(start_dists):.3f} m")
    print(f"  Mean end distance to ball:     {np.mean(end_dists):.3f} m")
    print(f"  Mean dist change (+ = closer): {np.mean(dist_changes):+.3f} m")
    print(f"  Mean CoM displacement ALL:     {np.mean(all_disps):.3f} m")
    if touch_disps:
        print(f"  Mean CoM displacement TOUCH:   {np.mean(touch_disps):.3f} m  (n={len(touch_disps)})")
    if notouched_disps:
        print(f"  Mean CoM displacement NO-TOUCH:{np.mean(notouched_disps):.3f} m  (n={len(notouched_disps)})")

    # Correlation between CoM displacement and contact
    touched_arr = np.array([float(r["touched"]) for r in results])
    disp_arr    = np.array([r["com_displacement"] for r in results])
    if disp_arr.std() > 1e-6:
        corr = float(np.corrcoef(disp_arr, touched_arr)[0, 1])
        print(f"  Correlation(CoM_disp, contact): {corr:+.3f}")
    print("=" * 70)


# --------------------------------------------------------------------------
# VIDEO RENDERING
# --------------------------------------------------------------------------

def render_episodes(model, vec_env, seeds=RENDER_SEEDS, max_steps=RENDER_STEPS,
                    label="crawl_minimal_400k"):
    """Render overhead + ringside for each seed into crawler_<label>_seed<N>.mp4."""
    inner_env = MimoCrawlerEnv(
        vision=False,
        strength_scale=STRENGTH,
        spawn_cone_deg=180,
        max_steps=max_steps,
        n_substeps=4,
        approach_reward_scale=APPROACH_SCALE,
        velocity_bonus_scale=VELOCITY_BONUS,
        random_start_orientation=False,
        memory_obs=False,
        stereo=True,
        action_mode=ACTION_MODE,
        spawn_radius=SPAWN_RADIUS,
        step_cost=STEP_COST,
        xml_path=XML_PATH,
        crawl_pose=CRAWL_POSE,
        terminate_tilt_deg=TERM_TILT,
        tip_penalty=TIP_PENALTY,
    )
    renderer_overhead = mujoco.Renderer(inner_env.model, 480, 480)
    renderer_ringside = mujoco.Renderer(inner_env.model, 480, 480)

    video_paths = []
    for seed in seeds:
        out_path = VIDEO_DIR / f"crawler_{label}_seed{seed}.mp4"
        obs_raw, _ = inner_env.reset(seed=seed)

        obs_norm = obs_raw.copy()
        if hasattr(vec_env, "obs_rms"):
            vn = vec_env
            while not isinstance(vn, VecNormalize):
                vn = vn.venv
            obs_norm = np.clip(
                (obs_raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                -vn.clip_obs, vn.clip_obs,
            ).astype(np.float32)

        writer = imageio.get_writer(str(out_path), fps=25, quality=8)
        done = False
        step_i = 0
        touched = False

        while not done and step_i < max_steps:
            action, _ = model.predict(obs_norm, deterministic=True)
            obs_raw, _, terminated, truncated, info = inner_env.step(action)
            touched = info.get("touched", touched)
            done = terminated or truncated
            step_i += 1

            renderer_overhead.update_scene(inner_env.data, camera="overhead")
            img_oh = renderer_overhead.render().copy()
            renderer_ringside.update_scene(inner_env.data, camera="ringside")
            img_rs = renderer_ringside.render().copy()
            frame = np.concatenate([img_oh, img_rs], axis=1)  # side-by-side
            writer.append_data(frame)

            obs_norm = obs_raw.copy()
            if hasattr(vec_env, "obs_rms"):
                obs_norm = np.clip(
                    (obs_raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                    -vn.clip_obs, vn.clip_obs,
                ).astype(np.float32)

        writer.close()
        status = "TOUCH" if touched else "NO_TOUCH"
        print(f"  Video [{status}] seed={seed}  steps={step_i}  -> {out_path}")
        video_paths.append(str(out_path))

    renderer_overhead.close()
    renderer_ringside.close()
    inner_env.close()
    return video_paths


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== crawl_minimal_400k — Quantitative Eval ===\n")

    print("Loading model...")
    model, vec_env = load_model()

    print(f"\nRunning {N_EVAL_EPS} deterministic eval episodes...")
    results = run_eval(model, vec_env, n_eps=N_EVAL_EPS)
    print_summary(results)

    print(f"\nRendering {len(RENDER_SEEDS)} episodes for visual inspection...")
    video_paths = render_episodes(model, vec_env, seeds=RENDER_SEEDS)

    print("\nDone.")
    print(f"Video paths: {video_paths}")
