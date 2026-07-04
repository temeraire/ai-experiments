"""
train_stage_b.py — Stage B: reward-grow vision to fill the frozen teacher's bearing slot
(SlotFillVisionPolicy). Same env as Stage C (vision, target_obs=False); teacher frozen.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.train_stage_b \
    --steps 1000000 --run-tag stage_b_v1
"""
import argparse
import pathlib
import pickle
import subprocess

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from alien_baby.crawler.mimo_crawler_env import CRAWL_POSES
from alien_baby.crawler.train_crawler import make_env, LivenessGateCallback, _SafeSaveEvalCallback
from alien_baby.crawler.slotfill_vision_policy import SlotFillVisionPolicy

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"
TEACHER = "crawl_ppo_2M"
TEACHER_BEST = "crawl_ppo_2M_best"


def _chime():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception:
        pass


def load_teacher_norm():
    with open(RESULTS / TEACHER / "vec_normalize.pkl", "rb") as f:
        vn = pickle.load(f)
    mean = vn.obs_rms.mean.astype(np.float32)               # 72-dim
    std = np.sqrt(vn.obs_rms.var + vn.epsilon).astype(np.float32)
    return mean, std, float(vn.clip_obs)


def build_envs(args):
    ck = dict(
        vision=True, stereo=True, target_obs=False, approach_reward_scale=10.0,
        velocity_bonus_scale=0.0, action_mode="position_offset",
        spawn_radius=tuple(args.spawn_radius), step_cost=0.0, xml_path=XML,
        crawl_pose=CRAWL_POSES["arms_fwd"], terminate_tilt_deg=50.0, tip_penalty=-5.0,
    )
    train_env = DummyVecEnv([
        make_env(i, args.seed, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.0,
                            training=False)
    return train_env, eval_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--spawn-cone-deg", type=float, default=44.0)
    p.add_argument("--spawn-radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--run-tag", default="stage_b_v1")
    p.add_argument("--device", default="mps")
    p.add_argument("--warmstart-cnn", default=None,
                   help="path to Stage A bearing_cnn.pt to warm-start the vision head")
    args = p.parse_args()

    tag = args.run_tag
    print(f"\n=== Stage B slot-fill vision training: {tag} ===")
    print(f"  steps={args.steps} n_envs={args.n_envs} cone=+/-{args.spawn_cone_deg/2:.0f} "
          f"device={args.device}")

    tmean, tstd, tclip = load_teacher_norm()
    train_env, eval_env = build_envs(args)

    model = PPO(
        SlotFillVisionPolicy, train_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
        clip_range=0.2, verbose=1, seed=args.seed, device=args.device,
        policy_kwargs=dict(
            teacher_model_path=str(RESULTS / TEACHER_BEST / "best_model.zip"),
            teacher_mean=tmean, teacher_std=tstd, teacher_clip=tclip,
            warmstart_cnn=args.warmstart_cnn,
        ),
    )

    best_dir = str(RESULTS / f"{tag}_best")
    ckpt_dir = str(RESULTS / tag)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 50_000 // args.n_envs), save_path=ckpt_dir,
                           name_prefix="ckpt", save_vecnormalize=True),
        _SafeSaveEvalCallback(eval_env, best_model_save_path=best_dir, log_path=ckpt_dir,
                              eval_freq=max(1, 20_000 // args.n_envs), n_eval_episodes=10,
                              deterministic=True, render=False),
        LivenessGateCallback(gate_step=10_000, min_motion=0.05, out_dir=ckpt_dir),
    ]
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), progress_bar=False)
    model.save(str(RESULTS / f"{tag}_final"))
    train_env.save(str(RESULTS / tag / "vec_normalize.pkl"))
    print(f"=== Stage B done: {tag} ===")
    _chime()


if __name__ == "__main__":
    main()
