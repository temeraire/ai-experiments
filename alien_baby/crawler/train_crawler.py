"""
train_crawler.py — SAC training for the MIMo-based infant crawler.

Curriculum: --strength-scale controls muscle strength (0.0–1.0).
  Start weak (0.2–0.3) so the body settles naturally from standing before
  the policy learns to exploit full torque. Increase in later runs.

Usage:
    python -m alien_baby.crawler.train_crawler
    python -m alien_baby.crawler.train_crawler --strength-scale 0.3 --steps 500000
    python -m alien_baby.crawler.train_crawler --strength-scale 1.0 --init-from results/...
"""

import argparse
import pathlib
import datetime
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN, PIXEL_LATENT_DIM
from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def make_env(rank, seed, strength_scale, spawn_cone_deg, max_steps, n_substeps,
             vision=False, approach_reward_scale=2.0, velocity_bonus_scale=0.05,
             fixed_ball_positions=None, random_start_orientation=False,
             memory_obs=False, stereo=True):
    def _init():
        env = MimoCrawlerEnv(
            vision=vision,
            strength_scale=strength_scale,
            spawn_cone_deg=spawn_cone_deg,
            max_steps=max_steps,
            n_substeps=n_substeps,
            approach_reward_scale=approach_reward_scale,
            velocity_bonus_scale=velocity_bonus_scale,
            fixed_ball_positions=fixed_ball_positions,
            random_start_orientation=random_start_orientation,
            memory_obs=memory_obs,
            stereo=stereo,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    tag = args.run_tag or (
        f"crawler_s{args.strength_scale:.1f}_"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    )
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = RESULTS_DIR / (tag + "_best")
    best_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Crawler training: {tag} ===")
    print(f"  strength_scale={args.strength_scale}  steps={args.steps}")
    print(f"  spawn_cone={args.spawn_cone_deg}°  max_steps={args.max_steps}")
    print(f"  n_envs={args.n_envs}  seed={args.seed}\n")

    # Vision=True: MuJoCo Metal renderer fails in forked subprocesses on macOS.
    # Use DummyVecEnv (single process) for vision; SubprocVecEnv for no-vision.
    VecEnvCls = DummyVecEnv if args.vision else SubprocVecEnv

    # Training envs
    train_env = VecEnvCls([
        make_env(i, args.seed, args.strength_scale, args.spawn_cone_deg,
                 args.max_steps, args.n_substeps, vision=args.vision,
                 approach_reward_scale=args.approach_reward_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 fixed_ball_positions=args.fixed_ball_positions,
                 random_start_orientation=args.random_start_orientation,
                 memory_obs=args.memory_obs,
                 stereo=not args.mono)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env (single, deterministic)
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, args.strength_scale, args.spawn_cone_deg,
                 args.max_steps, args.n_substeps, vision=args.vision,
                 approach_reward_scale=args.approach_reward_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 fixed_ball_positions=args.fixed_ball_positions,
                 random_start_orientation=args.random_start_orientation,
                 memory_obs=args.memory_obs,
                 stereo=not args.mono)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                             training=False)

    if args.init_from:
        print(f"  Loading checkpoint: {args.init_from}")
        model = SAC.load(
            args.init_from,
            env=train_env,
            device="mps" if args.mps else "cpu",
        )
        # Load matching VecNormalize stats if present
        vn_path = pathlib.Path(args.init_from).parent / "vec_normalize.pkl"
        if vn_path.exists():
            train_env = VecNormalize.load(str(vn_path), train_env)
            eval_env  = VecNormalize.load(str(vn_path), eval_env)
            eval_env.training = False
            print(f"  Loaded VecNormalize stats: {vn_path}")
    else:
        if args.vision:
            policy_kwargs = dict(
                features_extractor_class=StereoCrawlerCNN,
                features_extractor_kwargs=dict(
                    proprio_dim=PROPRIO_DIM,
                    pixel_latent_dim=PIXEL_LATENT_DIM,
                ),
                net_arch=[256, 256],
            )
        else:
            policy_kwargs = dict(net_arch=[256, 256])

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_entropy=args.target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="mps" if args.mps else "cpu",
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_interval // args.n_envs, 1),
            save_path=str(out_dir),
            name_prefix="crawler_ckpt",
            save_vecnormalize=True,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(out_dir),
            eval_freq=max(args.checkpoint_interval // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
    ])

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        reset_num_timesteps=(args.init_from is None),
        progress_bar=True,
    )

    # Save final model + VecNormalize
    final_path = out_dir / "final_model"
    model.save(str(final_path))
    train_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\nFinal model: {final_path}.zip")
    print(f"Best model:  {best_dir}/best_model.zip")

    train_env.close()
    eval_env.close()
    return str(final_path) + ".zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-substeps", type=int, default=4,
                        help="Physics steps per policy step. 4 × 5ms = 20ms per action "
                             "(50Hz control). Increases effective episode length without "
                             "adding more policy decisions.")
    parser.add_argument("--strength-scale", type=float, default=0.7,
                        help="Muscle strength curriculum (0.1=newborn, 1.0=18-month). "
                             "Start at 0.3-0.4, increase in later runs.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--spawn-cone-deg", type=float, default=180,
                        help="Half-angle of ball spawn cone (180=full forward hemisphere).")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Episode step limit (~30s at 20Hz).")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--init-from", default=None,
                        help="Path to checkpoint .zip to continue from.")
    parser.add_argument("--run-tag", default=None,
                        help="Output directory name. Auto-generated if omitted.")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple MPS (Metal) GPU acceleration.")
    parser.add_argument("--vision", action="store_true",
                        help="Enable stereo vision (left_eye + right_eye cameras). "
                             "Uses StereoCrawlerCNN feature extractor.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="SAC learning rate. DrQ-v2 recommends 1e-4 for pixel obs "
                             "(not 3e-4 which is for state obs). Higher causes actor to "
                             "over-exploit critic Q-estimates → collapse.")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="SAC replay buffer size. 500K caused cliff collapse when buffer "
                             "filled at ~125K steps; 100K rotates continuously.")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Steps before SAC begins training. Larger = more diverse "
                             "initial data before first update.")
    parser.add_argument("--velocity-bonus-scale", type=float, default=0.05,
                        help="Reward per unit hip horizontal speed. 0.0 = no bribery; "
                             "creature must find its own reason to move.")
    parser.add_argument("--approach-reward-scale", type=float, default=2.0,
                        help="Scale for ball-approach reward (0.0 = off). "
                             "High values cause critic instability with pixel obs.")
    parser.add_argument("--ent-coef", default="auto",
                        help="SAC entropy coefficient. 'auto' = adaptive. "
                             "Fixed float (e.g. 0.2) holds exploration open.")
    parser.add_argument("--target-entropy", default="auto",
                        help="SAC target entropy for auto ent_coef. "
                             "'auto' = -dim(action). Float overrides (e.g. -10.0).")
    parser.add_argument("--fixed-ball-positions", default=None,
                        help="Phase C: comma-separated x,y pairs separated by semicolons. "
                             "Example: '0.7,0.0' for one fixed ball, "
                             "'0.7,0.0;0.0,0.7' for two. Overrides random spawn.")
    parser.add_argument("--random-start-orientation", action="store_true",
                        help="Phase C: rotate prone quaternion around world Z by a random "
                             "angle per reset. Creature spawns facing a random direction.")
    parser.add_argument("--memory-obs", action="store_true",
                        help="Phase C: append 2 binary flags (touched_ball1, touched_ball2) "
                             "to the proprio observation. Lets a stateless policy condition "
                             "on its own past contacts within an episode.")
    parser.add_argument("--mono", action="store_true",
                        help="Use a single forward camera (left_eye) instead of stereo. "
                             "Halves the pixel observation dimension. The XML still shows "
                             "two visible eyes — only the camera input is mono.")
    args = parser.parse_args()
    # Convert numeric strings to float
    if args.ent_coef != "auto":
        args.ent_coef = float(args.ent_coef)
    if args.target_entropy != "auto":
        args.target_entropy = float(args.target_entropy)
    # Parse fixed-ball-positions string into list of (x, y) tuples
    if args.fixed_ball_positions:
        balls = []
        for chunk in args.fixed_ball_positions.split(";"):
            x, y = chunk.split(",")
            balls.append((float(x), float(y)))
        args.fixed_ball_positions = balls
    train(args)
