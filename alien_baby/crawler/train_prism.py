"""
train_prism.py — Prism adaptation (Phase B). Fine-tune the head-search vision policy under a
visual displacement: vision sees a GHOST at (real bearing + prism_offset_deg); the real ball is
hidden but solid+rewarding. The policy initially crawls to the ghost (wrong side, no reward) and
must RE-MAP to crawl to the real (proprioceptive) side. Adaptation of an established vision policy
— the in-silico prism experiment (Stratton 1897; PRISM_GHOST_PROPOSAL.md).

Base = head_search_curriculum_v1 (vision drives the crawl). Same obs/action space (the prism XML
adds only a mocap ghost). Checkpoints let eval_prism.py trace reach-direction migration over trials.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.train_prism \
    --base alien_baby/results/head_search_curriculum_v1_best/best_model.zip \
    --prism-offset 30 --steps 800000 --run-tag prism_off30
"""
import argparse
import pathlib
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from alien_baby.crawler.mimo_crawler_env import CRAWL_POSES
from alien_baby.crawler.train_crawler import make_env, LivenessGateCallback, _SafeSaveEvalCallback

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide_prism.xml"


def _chime():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception:
        pass


def build_envs(args):
    ck = dict(
        vision=True, stereo=True, target_obs=False, approach_reward_scale=10.0,
        velocity_bonus_scale=0.0, action_mode="position_offset",
        spawn_radius=tuple(args.spawn_radius), step_cost=0.0, xml_path=XML,
        crawl_pose=CRAWL_POSES["arms_fwd"], terminate_tilt_deg=50.0, tip_penalty=-5.0,
        prism_offset_deg=args.prism_offset,
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
    p.add_argument("--base", required=True, help="head-search policy to adapt")
    p.add_argument("--prism-offset", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=800000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--spawn-cone-deg", type=float, default=136.0)
    p.add_argument("--spawn-radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--run-tag", default="prism_off30")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    tag = args.run_tag
    print(f"\n=== PRISM adaptation: {tag} (offset +{args.prism_offset:.0f}) ===")
    print(f"  base={args.base} steps={args.steps} cone=+/-{args.spawn_cone_deg/2:.0f}")

    train_env, eval_env = build_envs(args)
    model = PPO.load(args.base, env=train_env, device=args.device)
    model.ent_coef = 0.01   # keep some exploration to find the real side

    best_dir = str(RESULTS / f"{tag}_best")
    ckpt_dir = str(RESULTS / tag)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 50_000 // args.n_envs), save_path=ckpt_dir,
                           name_prefix="ckpt", save_vecnormalize=True),
        _SafeSaveEvalCallback(eval_env, best_model_save_path=best_dir, log_path=ckpt_dir,
                              eval_freq=max(1, 25_000 // args.n_envs), n_eval_episodes=10,
                              deterministic=True, render=False),
        LivenessGateCallback(gate_step=10_000, min_motion=0.05, out_dir=ckpt_dir),
    ]
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks),
                progress_bar=False, reset_num_timesteps=True)
    model.save(str(RESULTS / f"{tag}_final"))
    train_env.save(str(RESULTS / tag / "vec_normalize.pkl"))
    print(f"=== prism adaptation done: {tag} ===")
    _chime()


if __name__ == "__main__":
    main()
