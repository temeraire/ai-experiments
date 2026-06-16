"""
eval_phase_i.py — Touch-counting + vision-ablation eval for Phase I MICOA runs.

Adapted from eval_phase_h.py. The two MICOA-specific differences:
  1. Loads the checkpoint via MICOASAC.load(...) so the MICOA agreement-loss
     attributes survive (purely for compatibility — eval is deterministic and
     does not call train()).
  2. Reports a third metric beyond reward and both-touched: sigma_combined
     at episode start and contact moments (read from the extractor's stored
     attributes after each forward pass).

Usage:
    python -m alien_baby.visualization.eval_phase_i --run-tag mimo_phase_i_R36_micoa_beta0.1
"""

import argparse
import faulthandler
import os
import pathlib
import numpy as np
import torch

faulthandler.enable()

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv, PROPRIO_DIM
from alien_baby.agents.micoa_architecture import MICOASAC, MICOAExtractor

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

N_SEEDS        = 20
MAX_STEPS      = 2000
ABLATION_STEPS = 200


def build_env_and_model(run_dir, ball_speed, offset):
    """Build a fresh vec-normalized env + load the MICOASAC model."""
    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=True,
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

    raw_env = DummyVecEnv([_env_fn])
    vn_path = RESULTS_DIR / run_dir / "vec_normalize.pkl"
    vec_env = VecNormalize.load(str(vn_path), raw_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    ckpt_path = RESULTS_DIR / run_dir / "final_model.zip"
    if not ckpt_path.exists():
        ckpt_path = RESULTS_DIR / f"{run_dir}_best" / "best_model.zip"
    print(f"  Loading checkpoint: {ckpt_path}", flush=True)
    model = MICOASAC.load(str(ckpt_path), env=vec_env, device="cpu")
    return vec_env, model


def run_episode(seed, vec_env, model, blind_pixels=False, record_sigma=False):
    """Run one deterministic episode. Returns dict with reward, touches, optional sigma trace."""
    obs, _ = vec_env.venv.envs[0].reset(seed=seed)
    obs = vec_env.normalize_obs(obs.reshape(1, -1))

    ep_reward = 0.0
    touched_b1 = False
    touched_b2 = False
    hand_b1 = False
    hand_b2 = False
    sigma_trace = []
    contact_step = None

    for step in range(MAX_STEPS):
        query_obs = obs.copy()
        if blind_pixels:
            query_obs[:, PROPRIO_DIM + 2:] = 0.0  # +2 for memory_obs flags

        action, _ = model.predict(query_obs, deterministic=True)

        if record_sigma:
            extractor = model.policy.features_extractor
            if isinstance(extractor, MICOAExtractor):
                sigma_trace.append(extractor.last_sigma_combined)

        obs_raw, rew_raw, terminated, truncated, info = vec_env.venv.envs[0].step(action[0])
        obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
        ep_reward += float(rew_raw)

        b1 = info.get("touched_ball1", False)
        b2 = info.get("touched_ball2", False)
        if (b1 or b2) and contact_step is None:
            contact_step = step
        if b1: touched_b1 = True
        if b2: touched_b2 = True
        if info.get("hand_touched_ball1", False): hand_b1 = True
        if info.get("hand_touched_ball2", False): hand_b2 = True

        if terminated or truncated:
            break

    return {
        "reward":       ep_reward,
        "touched_b1":   touched_b1,
        "touched_b2":   touched_b2,
        "hand_b1":      hand_b1,
        "hand_b2":      hand_b2,
        "sigma_trace":  sigma_trace,
        "contact_step": contact_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True,
                        help="Run tag — looks under alien_baby/results/<tag> and <tag>_best")
    parser.add_argument("--ball-speed", type=float, default=0.0)
    parser.add_argument("--offset",     type=float, default=0.15)
    args = parser.parse_args()

    print(f"\n{'='*70}\nPHASE I EVAL: {args.run_tag}\n{'='*70}\n")

    # --- Touch-counting eval (normal pixels) -------------------------------
    vec_env, model = build_env_and_model(args.run_tag, args.ball_speed, args.offset)
    rewards, both, one, neither = [], 0, 0, 0
    hand_both, hand_one, hand_neither = 0, 0, 0
    sigma_avg_at_contact   = []
    sigma_avg_pre_contact  = []
    for seed in range(N_SEEDS):
        r = run_episode(seed, vec_env, model, blind_pixels=False, record_sigma=True)
        rewards.append(r["reward"])
        # Broad metric (any body part)
        if r["touched_b1"] and r["touched_b2"]: both += 1
        elif r["touched_b1"] or r["touched_b2"]: one += 1
        else: neither += 1
        # Strict hand-only metric
        if r["hand_b1"] and r["hand_b2"]: hand_both += 1
        elif r["hand_b1"] or r["hand_b2"]: hand_one += 1
        else: hand_neither += 1
        st = r["sigma_trace"]
        cs = r["contact_step"]
        if cs is not None and cs > 0 and len(st) > cs:
            sigma_avg_pre_contact.append(float(np.mean(st[:cs])))
            sigma_avg_at_contact.append(float(st[cs]))
    vec_env.close()

    mean_reward = float(np.mean(rewards))
    std_reward  = float(np.std(rewards))
    print(f"  Touch eval ({N_SEEDS} seeds): mean={mean_reward:.1f} ± {std_reward:.1f}")
    print(f"    ANY body part: both={both}/{N_SEEDS}  one={one}/{N_SEEDS}  neither={neither}/{N_SEEDS}")
    print(f"    HAND-only    : both={hand_both}/{N_SEEDS}  one={hand_one}/{N_SEEDS}  neither={hand_neither}/{N_SEEDS}")
    if sigma_avg_at_contact:
        print(f"  sigma_combined: pre-contact mean={np.mean(sigma_avg_pre_contact):.4f}  "
              f"at-contact mean={np.mean(sigma_avg_at_contact):.4f}")
        delta_sigma = float(np.mean(sigma_avg_pre_contact) - np.mean(sigma_avg_at_contact))
        print(f"  sigma drop at contact: {delta_sigma:+.4f}  "
              f"(positive = corner forming at contact)")

    # --- Vision ablation: single-step action delta -------------------------
    vec_env, model = build_env_and_model(args.run_tag, args.ball_speed, args.offset)
    all_deltas = []
    for seed in range(10):
        obs, _ = vec_env.venv.envs[0].reset(seed=seed)
        obs = vec_env.normalize_obs(obs.reshape(1, -1))
        for _ in range(ABLATION_STEPS):
            full_a, _  = model.predict(obs, deterministic=True)
            blind_obs  = obs.copy()
            blind_obs[:, PROPRIO_DIM + 2:] = 0.0
            blind_a, _ = model.predict(blind_obs, deterministic=True)
            all_deltas.append(float(np.linalg.norm(full_a - blind_a)))
            obs_raw, _, term, trunc, _ = vec_env.venv.envs[0].step(full_a[0])
            obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
            if term or trunc:
                obs, _ = vec_env.venv.envs[0].reset(seed=seed + 200)
                obs = vec_env.normalize_obs(obs.reshape(1, -1))
    mean_delta   = float(np.mean(all_deltas))
    median_delta = float(np.median(all_deltas))
    print(f"\n  Vision-ablation single-step action delta: "
          f"mean={mean_delta:.4f}  median={median_delta:.4f}")
    vec_env.close()

    # --- Vision ablation: episode-level ------------------------------------
    vec_env, model = build_env_and_model(args.run_tag, args.ball_speed, args.offset)
    both_zeroed = 0
    hand_both_zeroed = 0
    for seed in range(N_SEEDS):
        r = run_episode(seed, vec_env, model, blind_pixels=True)
        if r["touched_b1"] and r["touched_b2"]:
            both_zeroed += 1
        if r["hand_b1"] and r["hand_b2"]:
            hand_both_zeroed += 1
    print(f"  Episode-level (ANY body): NORMAL both={both}/{N_SEEDS}  "
          f"ZEROED both={both_zeroed}/{N_SEEDS}  "
          f"delta={(both-both_zeroed):+d}")
    print(f"  Episode-level (HAND-only): NORMAL both={hand_both}/{N_SEEDS}  "
          f"ZEROED both={hand_both_zeroed}/{N_SEEDS}  "
          f"delta={(hand_both-hand_both_zeroed):+d}")
    vec_env.close()

    print(f"\n{'='*70}\nPHASE I EVAL COMPLETE\n{'='*70}\n")

    # Print structured summary for the heartbeat to parse.
    print("PHASE_I_EVAL_JSON_BEGIN")
    import json
    print(json.dumps({
        "run_tag":              args.run_tag,
        "ball_speed":           args.ball_speed,
        "offset":               args.offset,
        "n_seeds":              N_SEEDS,
        "mean_reward":          mean_reward,
        "std_reward":           std_reward,
        "both_touched_normal":  both,
        "only_one":             one,
        "neither":              neither,
        "both_touched_zeroed":  both_zeroed,
        "hand_both_normal":     hand_both,
        "hand_one":             hand_one,
        "hand_neither":         hand_neither,
        "hand_both_zeroed":     hand_both_zeroed,
        "ablation_delta_mean":  mean_delta,
        "ablation_delta_median": median_delta,
        "sigma_pre_contact":    float(np.mean(sigma_avg_pre_contact)) if sigma_avg_pre_contact else None,
        "sigma_at_contact":     float(np.mean(sigma_avg_at_contact)) if sigma_avg_at_contact else None,
    }, indent=2))
    print("PHASE_I_EVAL_JSON_END")

    os._exit(0)


if __name__ == "__main__":
    main()
