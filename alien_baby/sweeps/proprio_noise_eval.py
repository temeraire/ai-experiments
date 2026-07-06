"""
Idea D — proprio-noise stress test (eval-only, no training).

Loads a trained follow-on checkpoint and runs N episodes under three
conditions:
  1. clean       — observation passed through unmodified
  2. proprio_noise — Gaussian σ injected into the first PROPRIO_DIM dims
  3. vision_zero — pixel slice (dims PROPRIO_DIM..end) zeroed

Reports touch_rate per condition. The diagnostic question is: under
proprio noise, does vision compensate? If the policy survives proprio
noise but collapses under vision_zero, vision is load-bearing. If it
survives both, proprio dominates and vision is decorative. If it
survives neither, the policy hasn't really learned the task.

Usage:
    python -m alien_baby.sweeps.proprio_noise_eval \\
        --checkpoint .../followon_v8_phase0_validation_best/best_model.zip \\
        --episodes 50 --noise 1.5
"""

import argparse
import json
import pathlib
import numpy as np

from stable_baselines3 import SAC
from alien_baby.envs.platform_creature_env import PlatformCreatureEnv

PROPRIO_DIM = 29


def evaluate(model, env, episodes=50, mode="clean", noise_sigma=1.5,
             seed_base=10000):
    successes = 0
    falls = 0
    rng = np.random.default_rng(seed_base)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        info = None
        while not done:
            obs_in = obs.copy()
            if mode == "proprio_noise":
                obs_in[:PROPRIO_DIM] = obs_in[:PROPRIO_DIM] + rng.normal(
                    0.0, noise_sigma, size=PROPRIO_DIM
                ).astype(obs_in.dtype)
            elif mode == "vision_zero":
                obs_in[PROPRIO_DIM:] = 0.0
            action, _ = model.predict(obs_in, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("touched"):
            successes += 1
        if info.get("fell"):
            falls += 1
    return {
        "mode": mode,
        "touch_rate": successes / episodes,
        "fall_rate": falls / episodes,
        "touches": successes,
        "falls": falls,
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a SAC.load-able .zip (e.g. "
                             "followon_v8_*_best/best_model.zip).")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--noise", type=float, default=1.5,
                        help="Gaussian σ for proprio-noise injection.")
    parser.add_argument("--v9", action="store_true",
                        help="Use v9 env (matches recent training default).")
    parser.add_argument("--out-json", default=None,
                        help="Optional path to dump results as JSON.")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    env = PlatformCreatureEnv(vision=True, v9=args.v9)
    model = SAC.load(args.checkpoint, env=env, device="cpu")

    results = {}
    for mode in ("clean", "proprio_noise", "vision_zero"):
        print(f"\n--- Mode: {mode} ---")
        r = evaluate(model, env, episodes=args.episodes, mode=mode,
                     noise_sigma=args.noise)
        print(f"  touch_rate = {r['touch_rate']:.3f}  "
              f"({r['touches']}/{r['episodes']})  "
              f"fall_rate = {r['fall_rate']:.3f}")
        results[mode] = r

    print("\n=== Summary ===")
    print(f"clean         : {results['clean']['touch_rate']:.3f}")
    print(f"proprio_noise : {results['proprio_noise']['touch_rate']:.3f}  "
          f"(σ = {args.noise})")
    print(f"vision_zero   : {results['vision_zero']['touch_rate']:.3f}")
    delta_proprio = (results['clean']['touch_rate']
                     - results['proprio_noise']['touch_rate'])
    delta_vision = (results['clean']['touch_rate']
                    - results['vision_zero']['touch_rate'])
    print(f"Δ(proprio_noise) = {delta_proprio:+.3f}  "
          f"Δ(vision_zero) = {delta_vision:+.3f}")
    if delta_vision > 0.10 and delta_proprio < 0.10:
        verdict = "VISION LOAD-BEARING (proprio robust, vision required)"
    elif delta_proprio > 0.10 and delta_vision < 0.10:
        verdict = "PROPRIO DOMINANT (vision decorative)"
    elif delta_proprio < 0.05 and delta_vision < 0.05:
        verdict = "BOTH ROBUST (probably easy task or both useful)"
    else:
        verdict = "MIXED (both modalities contribute)"
    print(f"Verdict: {verdict}")

    if args.out_json:
        out = {
            "checkpoint": args.checkpoint,
            "noise_sigma": args.noise,
            "results": results,
            "delta_proprio": delta_proprio,
            "delta_vision": delta_vision,
            "verdict": verdict,
        }
        pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out_json}")

    env.close()


if __name__ == "__main__":
    main()
