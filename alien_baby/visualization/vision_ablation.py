"""
vision_ablation.py — measure whether a policy's actions actually depend on vision.

For each step of N eval episodes, compute:
  action_full  = policy.predict(obs)
  action_blind = policy.predict(obs_with_pixels_zeroed)
  delta        = L2(action_full − action_blind)

Mean delta = "vision-ablation sensitivity":
  ≈ 0.0   policy ignores vision (or vision and proprio produce identical actions)
  > 0.5   policy meaningfully changes its action based on pixels
  > 1.0   vision dominantly steers behavior

Run:
    python -m alien_baby.visualization.vision_ablation \\
        --checkpoint alien_baby/results/mimo_substrate_A_best/best_model.zip \\
        --vec-normalize alien_baby/results/mimo_substrate_A/vec_normalize.pkl \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --steps 200
"""
import argparse
import pathlib
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, PROPRIO_DIM


def measure_ablation(checkpoint, vec_normalize_path, seeds, steps_per_episode,
                     spawn_cone_deg=360, max_steps=2000, strength_scale=0.7,
                     velocity_bonus_scale=0.0, approach_reward_scale=0.0):
    # Build env (vision=True so obs has pixel dims to zero out)
    env_fns = [lambda: MimoCrawlerEnv(
        vision=True, max_steps=max_steps, spawn_cone_deg=spawn_cone_deg,
        strength_scale=strength_scale, velocity_bonus_scale=velocity_bonus_scale,
        approach_reward_scale=approach_reward_scale,
    )]
    vec_env = DummyVecEnv(env_fns)
    if vec_normalize_path is not None and pathlib.Path(vec_normalize_path).exists():
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = SAC.load(checkpoint, env=vec_env, device="cpu")

    per_seed_deltas = []
    per_seed_touched = []
    inner_env = vec_env.unwrapped.envs[0].unwrapped if hasattr(vec_env, "unwrapped") else None

    for seed in seeds:
        # Use the raw inner env so we can manually zero pixel obs at each step
        obs_norm = vec_env.reset()
        # Force seed by stepping the inner env
        if hasattr(vec_env, "envs"):
            vec_env.envs[0].reset(seed=int(seed))
        # Re-read normalized obs after seeded reset
        from gymnasium.utils import seeding as _sd  # noqa
        # Workaround: vec_env doesn't expose seeded reset; just step with seed-init reset above
        # and let normalize_obs handle the rest
        deltas = []
        touched = False
        for _ in range(steps_per_episode):
            full_action, _ = model.predict(obs_norm, deterministic=True)

            # Pixel-zeroed observation
            blind_obs = obs_norm.copy()
            blind_obs[:, PROPRIO_DIM:] = 0.0
            blind_action, _ = model.predict(blind_obs, deterministic=True)

            deltas.append(np.linalg.norm(full_action - blind_action, axis=-1).mean())

            obs_norm, rew, done, info = vec_env.step(full_action)
            if done[0]:
                touched = bool(info[0].get("touched", False))
                break

        per_seed_deltas.append(np.mean(deltas))
        per_seed_touched.append(touched)

    return {
        "mean_delta": float(np.mean(per_seed_deltas)),
        "median_delta": float(np.median(per_seed_deltas)),
        "per_seed_delta": list(map(float, per_seed_deltas)),
        "touch_rate": float(np.mean(per_seed_touched)),
        "touched_seeds": [s for s, t in zip(seeds, per_seed_touched) if t],
        "n_seeds": len(seeds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vec-normalize", default=None,
                   help="Path to vec_normalize.pkl (auto-inferred if omitted)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--steps", type=int, default=200,
                   help="Max steps per ablation episode (caps each rollout)")
    p.add_argument("--spawn-cone-deg", type=float, default=360.0)
    p.add_argument("--max-steps", type=int, default=2000)
    args = p.parse_args()

    if args.vec_normalize is None:
        ckpt = pathlib.Path(args.checkpoint)
        # Try sibling and run-dir
        for candidate in [ckpt.parent / "vec_normalize.pkl",
                          ckpt.parent.parent / ckpt.parent.name.replace("_best", "") / "vec_normalize.pkl"]:
            if candidate.exists():
                args.vec_normalize = str(candidate)
                break

    print(f"checkpoint    : {args.checkpoint}")
    print(f"vec_normalize : {args.vec_normalize}")
    print(f"seeds         : {args.seeds}  steps/ep: {args.steps}")
    print(f"spawn cone    : {args.spawn_cone_deg}°  max_steps: {args.max_steps}\n")

    result = measure_ablation(
        args.checkpoint, args.vec_normalize, args.seeds, args.steps,
        spawn_cone_deg=args.spawn_cone_deg, max_steps=args.max_steps,
    )

    print("Vision-ablation sensitivity (L2 action delta when pixels zeroed):")
    print(f"  mean   : {result['mean_delta']:.4f}")
    print(f"  median : {result['median_delta']:.4f}")
    print(f"  per-seed: {[f'{d:.3f}' for d in result['per_seed_delta']]}")
    print(f"\nTouch rate: {result['touch_rate']:.0%}  ({len(result['touched_seeds'])}/{result['n_seeds']})")
    print(f"Touched seeds: {result['touched_seeds']}")

    # Verdict
    print("\n" + "=" * 60)
    if result["mean_delta"] < 0.1:
        verdict = "VISION IS SILENT — policy ignores pixels"
    elif result["mean_delta"] < 0.5:
        verdict = "VISION INFLUENCES BUT IS WEAK"
    elif result["mean_delta"] < 1.5:
        verdict = "VISION IS LOAD-BEARING — actions meaningfully change with pixels"
    else:
        verdict = "VISION DOMINATES — large pixel→action coupling"
    print(f"Verdict: {verdict}")

    return result


if __name__ == "__main__":
    main()
