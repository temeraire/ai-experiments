"""
eval_phase_h.py — Touch-counting deterministic eval for Phase H moving-ball runs.

Runs 20 deterministic seeds per policy/offset/speed combination.
Reports: mean reward, both-touched count, only-one count, neither count.

Also runs vision-ablation (pixels zeroed) for the two vision policies.

Usage:
    python -m alien_baby.visualization.eval_phase_h

All paths are hard-coded for the four Phase H best_model.zip checkpoints.
"""

import faulthandler
import os
import pathlib
import numpy as np

faulthandler.enable()

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv, PROPRIO_DIM

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

N_SEEDS       = 20
MAX_STEPS     = 2000
ABLATION_STEPS = 200   # steps per seed for single-step ablation delta

RUNS = {
    "R32_proprio_speed08": {
        "checkpoint":    RESULTS_DIR / "mimo_phase_h_R32_proprio_speed08_best" / "best_model.zip",
        "vec_normalize": RESULTS_DIR / "mimo_phase_h_R32_proprio_speed08" / "vec_normalize.pkl",
        "vision":       False,
        "train_speed":  0.08,
    },
    "R33_vision_speed08": {
        "checkpoint":    RESULTS_DIR / "mimo_phase_h_R33_vision_speed08_best" / "best_model.zip",
        "vec_normalize": RESULTS_DIR / "mimo_phase_h_R33_vision_speed08" / "vec_normalize.pkl",
        "vision":       True,
        "train_speed":  0.08,
    },
    "R34_proprio_speed05": {
        "checkpoint":    RESULTS_DIR / "mimo_phase_h_R34_proprio_speed05_best" / "best_model.zip",
        "vec_normalize": RESULTS_DIR / "mimo_phase_h_R34_proprio_speed05" / "vec_normalize.pkl",
        "vision":       False,
        "train_speed":  0.05,
    },
    "R35_vision_speed05": {
        "checkpoint":    RESULTS_DIR / "mimo_phase_h_R35_vision_speed05_best" / "best_model.zip",
        "vec_normalize": RESULTS_DIR / "mimo_phase_h_R35_vision_speed05" / "vec_normalize.pkl",
        "vision":       True,
        "train_speed":  0.05,
    },
}


def _build_env_and_model(run_cfg, ball_speed, offset):
    """Build a fresh vec-normalized env + loaded model. Returns (vec_env, model)."""
    def _env_fn():
        env = MimoCrawlerCartEnv(
            vision=run_cfg["vision"],
            strength_scale=1.0,
            max_steps=MAX_STEPS,
            cart_speed=0.15,
            ball_speed=ball_speed,
            hip_actuation=False,
            memory_obs=True,
            hunger_base=0.05,
            hunger_rate=0.20,
            hunger_scale=500.0,
            fixed_ball_positions=[(offset, 0.35), (-offset, -0.35)],
        )
        return env

    raw_env   = DummyVecEnv([_env_fn])
    vec_env   = VecNormalize.load(str(run_cfg["vec_normalize"]), raw_env)
    vec_env.training   = False
    vec_env.norm_reward = False

    model = SAC.load(str(run_cfg["checkpoint"]), env=vec_env, device="cpu")
    return vec_env, model


def _run_episode(seed, vec_env, model, blind_pixels=False):
    """
    Run one deterministic episode.
    Returns (episode_reward, touched_ball1, touched_ball2).
    blind_pixels: if True, zero the pixel portion of the obs before predict.
    """
    # Reset with seed
    obs, _ = vec_env.venv.envs[0].reset(seed=seed)
    obs = vec_env.normalize_obs(obs.reshape(1, -1))

    ep_reward = 0.0
    touched_b1 = False
    touched_b2 = False

    for _ in range(MAX_STEPS):
        query_obs = obs.copy()
        if blind_pixels:
            query_obs[:, PROPRIO_DIM:] = 0.0

        action, _ = model.predict(query_obs, deterministic=True)
        obs_raw, rew_raw, terminated, truncated, info = vec_env.venv.envs[0].step(action[0])
        obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
        ep_reward += float(rew_raw)

        if info.get("touched_ball1", False):
            touched_b1 = True
        if info.get("touched_ball2", False):
            touched_b2 = True

        if terminated or truncated:
            break

    return ep_reward, touched_b1, touched_b2


def run_touch_eval(run_name, run_cfg, ball_speed, offset):
    """Run deterministic touch-counting eval for one (policy, speed, offset) cell."""
    print(f"  Eval: {run_name} | ball_speed={ball_speed:.2f} | offset={offset:.2f} | {N_SEEDS} seeds")

    vec_env, model = _build_env_and_model(run_cfg, ball_speed, offset)

    rewards      = []
    both_touched = 0
    only_one     = 0
    neither      = 0

    for seed in range(N_SEEDS):
        ep_rew, tb1, tb2 = _run_episode(seed, vec_env, model, blind_pixels=False)
        rewards.append(ep_rew)
        if tb1 and tb2:
            both_touched += 1
        elif tb1 or tb2:
            only_one += 1
        else:
            neither += 1

    vec_env.close()

    mean_reward = float(np.mean(rewards))
    std_reward  = float(np.std(rewards))

    print(f"    mean={mean_reward:.1f} ± {std_reward:.1f} | "
          f"both={both_touched}/{N_SEEDS} "
          f"one={only_one}/{N_SEEDS} "
          f"neither={neither}/{N_SEEDS}")

    return {
        "mean_reward": mean_reward,
        "std_reward":  std_reward,
        "both_touched": both_touched,
        "only_one":     only_one,
        "neither":      neither,
        "n_seeds":      N_SEEDS,
    }


def run_ablation(run_name, run_cfg, ball_speed, offset):
    """
    Vision ablation for one vision policy.

    (a) Single-step action delta over ABLATION_STEPS steps x 10 seeds.
    (b) Episode-level: 20 episodes with pixels normal vs 20 with pixels zeroed.
    """
    if not run_cfg["vision"]:
        print(f"  Ablation: {run_name} — proprio only, skipping")
        return None

    print(f"  Ablation: {run_name} | ball_speed={ball_speed:.2f} | offset={offset:.2f}")

    # (a) Single-step delta
    vec_env, model = _build_env_and_model(run_cfg, ball_speed, offset)
    all_deltas = []
    for seed in range(10):
        obs, _ = vec_env.venv.envs[0].reset(seed=seed)
        obs = vec_env.normalize_obs(obs.reshape(1, -1))
        step_count = 0
        for _ in range(ABLATION_STEPS):
            full_action, _ = model.predict(obs, deterministic=True)
            blind_obs = obs.copy()
            blind_obs[:, PROPRIO_DIM:] = 0.0
            blind_action, _ = model.predict(blind_obs, deterministic=True)
            delta = float(np.linalg.norm(full_action - blind_action))
            all_deltas.append(delta)

            obs_raw, _, terminated, truncated, _ = vec_env.venv.envs[0].step(full_action[0])
            obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
            step_count += 1
            if terminated or truncated:
                # re-seed and keep going
                obs, _ = vec_env.venv.envs[0].reset(seed=seed + 200)
                obs = vec_env.normalize_obs(obs.reshape(1, -1))

    mean_delta   = float(np.mean(all_deltas))
    median_delta = float(np.median(all_deltas))
    print(f"    Single-step delta: mean={mean_delta:.4f}, median={median_delta:.4f}")
    vec_env.close()

    # (b) Episode-level
    both_normal = 0
    both_zeroed = 0

    for use_blind, label in [(False, "normal"), (True, "zeroed")]:
        vec_env, model = _build_env_and_model(run_cfg, ball_speed, offset)
        ep_both = 0
        for seed in range(N_SEEDS):
            _, tb1, tb2 = _run_episode(seed, vec_env, model, blind_pixels=use_blind)
            if tb1 and tb2:
                ep_both += 1
        vec_env.close()
        print(f"    Episode pixels_{label}: both-touched={ep_both}/{N_SEEDS}")
        if label == "normal":
            both_normal = ep_both
        else:
            both_zeroed = ep_both

    return {
        "mean_delta":          mean_delta,
        "median_delta":        median_delta,
        "both_touched_normal": both_normal,
        "both_touched_zeroed": both_zeroed,
        "n_seeds":             N_SEEDS,
    }


def main():
    print("\n" + "=" * 70)
    print("PHASE H EVAL: Touch-counting + Vision Ablation")
    print("=" * 70)

    all_results     = {}
    ablation_results = {}

    # -----------------------------------------------------------------------
    # Touch-counting eval matrix
    # -----------------------------------------------------------------------
    # For each run:
    #   1. Primary: training speed, offset=0.15
    #   2. Static ball: speed=0.0, offset=0.15
    #   3. Reach curve: training speed at offset=0.10
    #   4. Reach curve: training speed at offset=0.05

    print("\n--- TOUCH-COUNTING EVAL ---")
    for run_name, run_cfg in RUNS.items():
        ts = run_cfg["train_speed"]
        for ball_speed, offset, label in [
            (ts,   0.15, "primary"),
            (0.0,  0.15, "static"),
            (ts,   0.10, "reach_0.10"),
            (ts,   0.05, "reach_0.05"),
        ]:
            key = (run_name, ball_speed, offset)
            result = run_touch_eval(run_name, run_cfg, ball_speed, offset)
            all_results[key] = result

    # -----------------------------------------------------------------------
    # Vision ablation (R33 and R35 only)
    # -----------------------------------------------------------------------
    print("\n--- VISION ABLATION ---")
    for run_name in ["R33_vision_speed08", "R35_vision_speed05"]:
        run_cfg = RUNS[run_name]
        ts = run_cfg["train_speed"]
        abl = run_ablation(run_name, run_cfg, ball_speed=ts, offset=0.15)
        ablation_results[run_name] = abl

    # -----------------------------------------------------------------------
    # Print summary tables
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("TABLE 1: Primary — training speed, offset=0.15")
    print("=" * 70)
    header = f"{'Run':<30} {'mean_rew':>10} {'std':>8} {'both':>6} {'one':>6} {'neither':>8}"
    print(header)
    for run_name, run_cfg in RUNS.items():
        ts = run_cfg["train_speed"]
        key = (run_name, ts, 0.15)
        r = all_results.get(key, {})
        if r:
            print(f"{run_name:<30} {r['mean_reward']:>10.1f} {r['std_reward']:>8.1f} "
                  f"{r['both_touched']:>4}/{N_SEEDS:<2} "
                  f"{r['only_one']:>4}/{N_SEEDS:<2} "
                  f"{r['neither']:>6}/{N_SEEDS}")

    print("\n" + "=" * 70)
    print("TABLE 2: Static ball (speed=0.0), offset=0.15")
    print("=" * 70)
    print(header)
    for run_name, run_cfg in RUNS.items():
        key = (run_name, 0.0, 0.15)
        r = all_results.get(key, {})
        if r:
            print(f"{run_name:<30} {r['mean_reward']:>10.1f} {r['std_reward']:>8.1f} "
                  f"{r['both_touched']:>4}/{N_SEEDS:<2} "
                  f"{r['only_one']:>4}/{N_SEEDS:<2} "
                  f"{r['neither']:>6}/{N_SEEDS}")

    print("\n" + "=" * 70)
    print("TABLE 3: Reach curve — training speed, offsets 0.05 & 0.10")
    print("=" * 70)
    print(f"{'Run':<30} {'offset':>8} {'mean_rew':>10} {'both':>10}")
    for run_name, run_cfg in RUNS.items():
        ts = run_cfg["train_speed"]
        for offset in [0.05, 0.10]:
            key = (run_name, ts, offset)
            r = all_results.get(key, {})
            if r:
                print(f"{run_name:<30} {offset:>8.2f} {r['mean_reward']:>10.1f} "
                      f"{r['both_touched']:>4}/{N_SEEDS}")

    print("\n" + "=" * 70)
    print("TABLE 4: Vision Ablation (R33 and R35)")
    print("=" * 70)
    for run_name, abl in ablation_results.items():
        if abl:
            print(f"\n{run_name}:")
            print(f"  (a) Single-step L2 delta (200 steps x 10 seeds):")
            print(f"      mean={abl['mean_delta']:.4f}  median={abl['median_delta']:.4f}")
            print(f"  (b) Episode-level (20 seeds each):")
            print(f"      pixels NORMAL:  both-touched = {abl['both_touched_normal']}/{N_SEEDS}")
            print(f"      pixels ZEROED:  both-touched = {abl['both_touched_zeroed']}/{N_SEEDS}")
            delta_n = abl['both_touched_zeroed'] - abl['both_touched_normal']
            print(f"      zeroed vs normal: {delta_n:+d}  "
                  f"({delta_n/max(1,abl['both_touched_normal'])*100:+.0f}% relative)")

    print("\n" + "=" * 70)
    print("PHASE H EVAL COMPLETE")
    print("=" * 70)

    os._exit(0)


if __name__ == "__main__":
    main()
