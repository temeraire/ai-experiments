"""Random-policy probe for v9 under the reshaped (last-mile) reward.

Rolls a random policy for N episodes and reports per-episode stats so we can
verify, before burning any training compute:
  1. ATTRACT only fires in the <ATTRACT_MAX_DIST zone (attract_sum is small).
  2. Net reward for "drift and do nothing" is negative (HUNGER dominates).
  3. Touch fraction under random exploration is non-zero (else we're in a
     sparse-reward hole and the curriculum safeguard should be invoked).
"""

import argparse
import numpy as np

from alien_baby.envs.platform_creature_env import (
    PlatformCreatureEnv, ATTRACT_SCALE, ATTRACT_MAX_DIST, HUNGER_PENALTY,
)


def run(n_episodes=100, vision=True, v9=True, seed=0):
    env = PlatformCreatureEnv(vision=vision, v9=v9)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        info = {}
        while True:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        rows.append({
            "ep": i,
            "total_reward": total_reward,
            "hunger_sum": info.get("hunger_sum", 0.0),
            "attract_sum": info.get("attract_sum", 0.0),
            "ep_len": info.get("ep_len", 0),
            "touched": bool(info.get("touched", False)),
            "fell": bool(info.get("fell", False)),
            "ball_lost": bool(info.get("ball_lost", False)),
        })
    env.close()
    return rows


def summarize(rows):
    n = len(rows)
    tot = np.array([r["total_reward"] for r in rows])
    hun = np.array([r["hunger_sum"] for r in rows])
    att = np.array([r["attract_sum"] for r in rows])
    ln = np.array([r["ep_len"] for r in rows])
    touched = np.array([r["touched"] for r in rows])
    fell = np.array([r["fell"] for r in rows])
    lost = np.array([r["ball_lost"] for r in rows])

    print(f"Episodes: {n}")
    print(f"Constants: ATTRACT_SCALE={ATTRACT_SCALE}, "
          f"ATTRACT_MAX_DIST={ATTRACT_MAX_DIST}, "
          f"HUNGER_PENALTY={HUNGER_PENALTY}")
    print()
    print("Per-episode stats (mean / min / max):")
    print(f"  total_reward  : {tot.mean():8.3f}  {tot.min():8.3f}  {tot.max():8.3f}")
    print(f"  hunger_sum    : {hun.mean():8.3f}  {hun.min():8.3f}  {hun.max():8.3f}")
    print(f"  attract_sum   : {att.mean():8.3f}  {att.min():8.3f}  {att.max():8.3f}")
    print(f"  ep_len        : {ln.mean():8.1f}  {ln.min():8.0f}  {ln.max():8.0f}")
    print()
    print("Termination counts:")
    print(f"  touched  : {touched.sum():3d} / {n}  ({100*touched.mean():.1f}%)")
    print(f"  fell     : {fell.sum():3d} / {n}  ({100*fell.mean():.1f}%)")
    print(f"  ball_lost: {lost.sum():3d} / {n}  ({100*lost.mean():.1f}%)")
    print()

    attract_bribery = att.mean() / max(abs(hun.mean()), 1e-9)
    print(f"Sanity: mean(attract_sum) / |mean(hunger_sum)| = {attract_bribery:.3f}")
    print("  If < 1: hunger dominates, net-idle reward is negative (good).")
    print("  If > 1: attract still bribing, reshape didn't work.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-v9", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = run(
        n_episodes=args.episodes,
        vision=not args.no_vision,
        v9=not args.no_v9,
        seed=args.seed,
    )
    summarize(rows)
