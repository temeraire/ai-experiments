"""
train_crawler_ppo.py — PPO trainer for the crawl task (the MIMo-recipe escalation).

Why PPO: the SAC target_obs runs proved DIRECTED crawling emerges (mean toward-ball
translation +0.08..+0.20 m, 7-23% contacts) but never STABILISES — every run shows
burst-to-40/70-then-regress, best_model catches a lucky peak, nothing locks in
(Pattern-Learning "no stable lock-in"). The literature scout (2026-07-01) found MIMo's
only working whole-body skill (supine->prone rolling, ICDL 2026) used PPO + dense
potential shaping + sparse success bonus + metabolic cost — NOT sparse-contact SAC.
PPO is on-policy (no replay-buffer drift), the standard stability fix. This script
reuses the exact crawl env + make_env from train_crawler (crawl-ready pose, widened
body, tip-termination, signed approach reward, target_obs) and only swaps SAC->PPO.

Run (matches the SAC crawl_targetobs config):
  PYTHONPATH=<repo> python alien_baby/crawler/train_crawler_ppo.py \
    --target-obs --max-steps 1000 --steps 2000000 --run-tag crawl_ppo_2M
"""
import argparse
import pathlib
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from alien_baby.crawler.mimo_crawler_env import CRAWL_POSES
from alien_baby.crawler.train_crawler import make_env, LivenessGateCallback, _SafeSaveEvalCallback

RESULTS = pathlib.Path(__file__).parent.parent / "results"


def _chime():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception:
        pass


def build_envs(args, crawl_pose):
    ck = dict(
        vision=False, approach_reward_scale=args.approach_reward_scale,
        velocity_bonus_scale=0.0, action_mode="position_offset",
        spawn_radius=tuple(args.spawn_radius), step_cost=args.step_cost,
        xml_path=args.xml_path, crawl_pose=crawl_pose,
        terminate_tilt_deg=args.terminate_tilt_deg, tip_penalty=args.tip_penalty,
        target_obs=args.target_obs,
    )
    train_env = SubprocVecEnv([
        make_env(i, args.seed, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)
    return train_env, eval_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--spawn-cone-deg", type=float, default=180)
    p.add_argument("--spawn-radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--approach-reward-scale", type=float, default=10.0)
    p.add_argument("--step-cost", type=float, default=0.0)
    p.add_argument("--terminate-tilt-deg", type=float, default=50.0)
    p.add_argument("--tip-penalty", type=float, default=-5.0)
    p.add_argument("--xml-path", default="alien_baby/crawler/mimo_crawler_pos_wide.xml")
    p.add_argument("--crawl-pose", default="arms_fwd", choices=list(CRAWL_POSES.keys()))
    p.add_argument("--target-obs", action="store_true")
    p.add_argument("--run-tag", default="crawl_ppo")
    # PPO hyperparameters (locomotion-typical)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-liveness-gate", action="store_true")
    p.add_argument("--liveness-gate-step", type=int, default=10_000)
    p.add_argument("--liveness-min-motion", type=float, default=0.05)
    args = p.parse_args()

    crawl_pose = CRAWL_POSES.get(args.crawl_pose)
    tag = args.run_tag
    print(f"\n=== PPO crawler training: {tag} ===")
    print(f"  steps={args.steps} n_envs={args.n_envs} max_steps={args.max_steps} "
          f"target_obs={args.target_obs} pose={args.crawl_pose}")

    train_env, eval_env = build_envs(args, crawl_pose)

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef, clip_range=0.2, verbose=1, seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256]), device="cpu",
    )

    best_dir = str(RESULTS / f"{tag}_best")
    ckpt_dir = str(RESULTS / tag)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 25_000 // args.n_envs), save_path=ckpt_dir,
                           name_prefix="ckpt", save_vecnormalize=True),
        _SafeSaveEvalCallback(
            eval_env, best_model_save_path=best_dir, log_path=ckpt_dir,
            eval_freq=max(1, 10_000 // args.n_envs), n_eval_episodes=10,
            deterministic=True, render=False),
    ]
    if not args.no_liveness_gate:
        callbacks.append(LivenessGateCallback(
            gate_step=args.liveness_gate_step, min_motion=args.liveness_min_motion,
            out_dir=ckpt_dir))

    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks),
                progress_bar=False)

    model.save(str(RESULTS / f"{tag}_final"))
    train_env.save(str(RESULTS / tag / "vec_normalize.pkl"))
    print(f"=== PPO training done: {tag} ===")
    _chime()


if __name__ == "__main__":
    main()
