"""
Phase A: locomotion-only training.

The ball is irrelevant. Survival stakes are off. The reward is just the
torso's horizontal speed. The body learns that motor output produces
self-motion, before any goal is added.

Per the hand-drive test, both synchronous and alternating shoulder_roll
paddles produce ~5 cm/s of forward motion. RL should at minimum match
this — and with a continuous, dense, direction-agnostic reward, ideally
discover faster coordinations involving shoulder_pitch and elbow.

Stage 0 of the env (huge platform, no fall penalty, no edge warn) is
re-used for the body+platform. The env's own reward (hunger, attract,
contact) is discarded by the LocomotionRewardWrapper.

Usage:
    python -m alien_baby.agents.train_phase_a
    python -m alien_baby.agents.train_phase_a --steps 50000 --run-tag smoke
    python -m alien_baby.agents.train_phase_a --ent-coef 0.05
"""

import argparse
import pathlib

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.agents.train_v8 import _best_device, _play_done_sound

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_ENVS = 16

PHASE_A_INFO_KEYWORDS = ("loco_speed_mean", "loco_displacement", "loco_steps")


class LocomotionRewardWrapper(gym.Wrapper):
    """Replaces the env's reward with the torso's horizontal speed.

    r_t = ||v_xy|| where v_xy is the world-frame horizontal linear
    velocity of the root free joint (qvel[0:2]). Direction-agnostic and
    pattern-agnostic: any motor output that produces self-motion gets
    paid; sitting still and spinning-in-place pay nothing.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._spawn_xy = self.env.unwrapped.data.qpos[0:2].copy()
        self._speed_sum = 0.0
        self._n_steps = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        qvel = self.env.unwrapped.data.qvel
        speed = float(np.hypot(qvel[0], qvel[1]))
        self._speed_sum += speed
        self._n_steps += 1
        if terminated or truncated:
            cur_xy = self.env.unwrapped.data.qpos[0:2]
            info["loco_speed_mean"] = self._speed_sum / max(self._n_steps, 1)
            info["loco_displacement"] = float(
                np.linalg.norm(cur_xy - self._spawn_xy)
            )
            info["loco_steps"] = float(self._n_steps)
        return obs, speed, terminated, truncated, info


class LocoMetricsCallback(BaseCallback):
    """Surface mean per-episode locomotion metrics to SB3's logger so
    they appear in the rollout/ block alongside ep_rew_mean."""

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if not buf:
            return True
        for key in ("loco_speed_mean", "loco_displacement", "loco_steps"):
            vals = [ep[key] for ep in buf if key in ep]
            if vals:
                self.logger.record(f"rollout/{key}", float(np.mean(vals)))
        return True


def _make_env():
    def _make():
        env = PlatformCreatureEnv(vision=False, stage=0)
        env = LocomotionRewardWrapper(env)
        return Monitor(env, info_keywords=PHASE_A_INFO_KEYWORDS)
    return _make


def train_phase_a(total_timesteps, seed, checkpoint_interval, run_tag, ent_coef):
    suffix = f"_{run_tag}" if run_tag else ""
    print("=" * 60)
    print(f"PHASE A: locomotion-only training")
    print(f"  steps={total_timesteps:,}  envs={N_ENVS}  seed={seed}")
    print(f"  reward = ||v_xy|| (torso horizontal speed)")
    print(f"  env = stage=0 (huge platform, no fall penalty), vision=False")
    print(f"  ent_coef = {ent_coef}")
    print("=" * 60)

    env = SubprocVecEnv(
        [_make_env() for _ in range(N_ENVS)], start_method="spawn",
    )
    eval_env = Monitor(
        LocomotionRewardWrapper(PlatformCreatureEnv(vision=False, stage=0)),
        info_keywords=PHASE_A_INFO_KEYWORDS,
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=N_ENVS,
        learning_starts=5000,
        ent_coef=ent_coef,
        verbose=1,
        seed=seed,
        device=_best_device(),
    )

    metrics_cb = LocoMetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / f"phase_a{suffix}_best"),
        log_path=str(RESULTS_DIR / f"phase_a{suffix}_logs"),
        eval_freq=max(10000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(checkpoint_interval // N_ENVS, 1),
        save_path=str(RESULTS_DIR / f"phase_a{suffix}_checkpoints"),
        name_prefix=f"phase_a{suffix}",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([metrics_cb, eval_cb, ckpt_cb]),
    )

    final_path = str(RESULTS_DIR / f"phase_a{suffix}_checkpoint")
    model.save(final_path)
    print(f"\nPhase A saved to {final_path}")

    env.close()
    eval_env.close()
    _play_done_sound()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000,
                        help="Total env steps. 300K default — render the best "
                             "checkpoint and decide whether to extend. Don't "
                             "commit to 1M without seeing motion in the video.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--run-tag", default=None,
                        help="Suffix for output dirs (best, checkpoints, logs).")
    parser.add_argument("--ent-coef", default="auto",
                        help="'auto' (default) or a float. The dense per-step "
                             "speed reward should keep auto-tuned entropy from "
                             "collapsing the way the sparse v8 reward did, but "
                             "pin (e.g. 0.05) if the policy locks in.")
    args = parser.parse_args()

    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = args.ent_coef

    train_phase_a(
        total_timesteps=args.steps,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        run_tag=args.run_tag,
        ent_coef=ent_coef,
    )
