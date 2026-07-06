"""One-off touch-counting eval for Phase IV R42 (plain SAC, no MICOA).

Mirrors the eval recipe used for R40/R41 (eval_phase_i.py), but loads via plain SAC.
"""
import os
import pathlib
import sys

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RUN_TAG = "phase_iv_R42_proprio_speed08"
N_SEEDS = 20
MAX_STEPS = 2000


def _build(ball_speed: float, offset: float):
    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=False,
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

    raw = DummyVecEnv([_env_fn])
    vec_norm_path = RESULTS_DIR / RUN_TAG / "vec_normalize.pkl"
    vec = VecNormalize.load(str(vec_norm_path), raw)
    vec.training = False
    vec.norm_reward = False

    ckpt = RESULTS_DIR / f"{RUN_TAG}_best" / "best_model.zip"
    model = SAC.load(str(ckpt), env=vec, device="cpu")
    return vec, model


def _run_episode(seed: int, vec, model):
    vec.seed(seed)
    obs = vec.reset()
    total_r = 0.0
    ball1_touched = False
    ball2_touched = False
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec.step(action)
        total_r += reward[0]
        i0 = info[0]
        ball1_touched = ball1_touched or i0.get("touched_ball1", False)
        ball2_touched = ball2_touched or i0.get("touched_ball2", False)
        if done[0]:
            break
    return total_r, ball1_touched, ball2_touched


def main():
    print(f"\n=== R42 eval: {RUN_TAG} ===")
    for ball_speed, offset, label in [(0.08, 0.15, "moving08"), (0.0, 0.15, "static")]:
        vec, model = _build(ball_speed, offset)
        rewards = []
        both = one = neither = 0
        for s in range(N_SEEDS):
            r, b1, b2 = _run_episode(s, vec, model)
            rewards.append(r)
            if b1 and b2:
                both += 1
            elif b1 or b2:
                one += 1
            else:
                neither += 1
        rewards = np.asarray(rewards)
        print(
            f"\n[{label}] ball_speed={ball_speed:.2f} offset={offset:.2f}"
            f"\n  mean_reward = {rewards.mean():.1f}  std = {rewards.std():.1f}"
            f"\n  both={both}/{N_SEEDS}  one={one}/{N_SEEDS}  neither={neither}/{N_SEEDS}"
        )
        sys.stdout.flush()
        del model, vec
    print("\n=== R42 eval complete ===")


if __name__ == "__main__":
    main()
