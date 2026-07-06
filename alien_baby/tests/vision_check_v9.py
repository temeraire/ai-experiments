"""Vision check: is the ball actually visible in the 32×32 head_cam at 0.5m?

Renders one frame from head_cam at reset for N random seeds, counts red-ball
pixels, and reports the distribution. If the ball is consistently <5 pixels,
the creature's visual cortex has nothing reliable to correlate with hunger
reduction — the curriculum safeguard (shrink spawn radius) is needed before
any training attempt, regardless of the reward landscape being correct.
"""

import argparse
import numpy as np

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv


def count_red_pixels(rgb_flat, h=32, w=32, red_thresh=0.5, other_thresh=0.35):
    """A pixel counts as 'red ball' if R > red_thresh AND G,B < other_thresh.
    Ball material in platform_creature_v9.xml is a pure saturated red."""
    img = rgb_flat.reshape(h, w, 3)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return int(((r > red_thresh) & (g < other_thresh) & (b < other_thresh)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None,
                        help="Optional path to save the first frame as PNG for eyeball inspection.")
    args = parser.parse_args()

    env = PlatformCreatureEnv(vision=True, v9=True)
    counts = []
    first_img = None
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        proprio_dim = env._proprio_dim()
        pixels = obs[proprio_dim:]
        n_red = count_red_pixels(pixels)
        counts.append(n_red)
        if first_img is None:
            first_img = pixels.reshape(32, 32, 3)

    counts = np.array(counts)
    print(f"Ball-pixel counts over {args.episodes} resets (32×32 head_cam, ball at 0.5m):")
    print(f"  mean: {counts.mean():.2f}")
    print(f"  min : {counts.min()}")
    print(f"  max : {counts.max()}")
    print(f"  n_zero (ball not visible at all): {(counts == 0).sum()} / {args.episodes}")
    print(f"  n_lt_5  (ball < 5 px — hard to detect): {(counts < 5).sum()} / {args.episodes}")
    print()
    print("Decision rule:")
    print("  mean ≥ 10 px, n_lt_5 small → visual cortex has enough signal; launch smoke.")
    print("  mean <  5 px OR n_zero > 20% → ball is invisible from spawn angle;")
    print("    need curriculum (shrink radius) OR widen FOV before launching.")

    if args.save:
        try:
            from PIL import Image
            img8 = (np.clip(first_img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img8).save(args.save)
            print(f"Wrote first frame to {args.save}")
        except ImportError:
            print("PIL not installed; skipping --save.")

    env.close()


if __name__ == "__main__":
    main()
