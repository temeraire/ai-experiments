"""eval_vsteer_choice.py — is VISION load-bearing for the walker's driver?

Runs a trained vision-steer policy two ways on the SAME episodes:
  - SIGHTED: normal observation (proprio + stereo eyes)
  - BLIND  : identical, but the pixel columns are zeroed (proprio only)
For each we report, over N episodes:
  - CONTACT%   : reached either ball
  - CHOICE %RED: of the episodes that reached a ball, the fraction that reached the RED target
  - winnability: BOTH balls in view at reset (should be ~100 by construction)

If vision is load-bearing, SIGHTED choice-%red should sit well above the BLIND floor (~50%,
since with pixels zeroed the policy cannot tell red from blue and can only pick by geometry).

Also renders a few SIGHTED episodes (agent eye-view + side view) to video, because a choice
number can't tell a directed reach from an incidental roll-in — we watch to confirm.

  python -m alien_baby.crawler.eval_vsteer_choice --model alien_baby/results/vsteer_s6_best/best_model.zip \
      --episodes 60 --run-tag vsteer_s6
"""
import argparse
import numpy as np, mujoco
import imageio.v2 as iio
from stable_baselines3 import PPO
from alien_baby.crawler.train_vision_steer import VisionSteerEnv, PROP


def _both_in_view(env):
    env.rend.update_scene(env.data, camera="left_eye"); im = env.rend.render()
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    red = int(np.sum((R > 150) & (G < 90) & (B < 90)))
    blue = int(np.sum((B > 150) & (R < 90) & (G < 120)))
    return red >= 2, blue >= 2


def run(model, env, n, blind, seed0):
    contact = red_choice = both_view = 0
    for e in range(n):
        env.rng = np.random.default_rng(seed0 + e)      # SAME episodes for sighted vs blind
        obs, _ = env.reset()
        rv, bv = _both_in_view(env); both_view += (rv and bv)
        done = False
        while not done:
            o = obs.copy()
            if blind:
                o[PROP:] = 0.0                           # zero the pixel columns
            a, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            if info["red"]:
                contact += 1; red_choice += 1; break
            if info["blue"]:
                contact += 1; break
    ch = (100.0 * red_choice / contact) if contact else float("nan")
    return dict(contact=100.0 * contact / n, choice_red=ch,
                both_view=100.0 * both_view / n, n=n)


def render(model, env, path, episodes, seed0):
    frames = []
    big = mujoco.Renderer(env.model, 256, 256)
    eye = mujoco.Renderer(env.model, 256, 256)
    for e in range(episodes):
        env.rng = np.random.default_rng(seed0 + 900 + e)
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            big.update_scene(env.data, camera="side"); side = big.render()
            eye.update_scene(env.data, camera="left_eye"); ev = eye.render()
            frames.append(np.concatenate([side, ev], axis=1))
            done = term or trunc
    iio.mimsave(path, frames, fps=20)   # mimsave (not imwrite) for a multi-frame mp4
    print(f"wrote {path} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--run-tag", default="vsteer")
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = VisionSteerEnv(seed=args.seed)

    sighted = run(model, env, args.episodes, blind=False, seed0=args.seed)
    blind = run(model, env, args.episodes, blind=True, seed0=args.seed)
    print(f"=== {args.run_tag} choice eval (n={args.episodes}) ===")
    print(f"  winnability (BOTH balls in view at reset): {sighted['both_view']:.0f}%")
    print(f"  SIGHTED : contact {sighted['contact']:.0f}%   choice %RED {sighted['choice_red']:.0f}%")
    print(f"  BLIND   : contact {blind['contact']:.0f}%   choice %RED {blind['choice_red']:.0f}%   (~50 = can't tell color)")
    gap = sighted["choice_red"] - blind["choice_red"]
    print(f"  --> vision gap (sighted-blind choice): {gap:+.0f} pts  "
          f"[{'LOAD-BEARING' if gap >= 12 else 'weak/none'}]")
    if not args.no_render:
        render(model, env, f"scratch_render/{args.run_tag}_choice.mp4", episodes=4, seed0=args.seed)


if __name__ == "__main__":
    main()
