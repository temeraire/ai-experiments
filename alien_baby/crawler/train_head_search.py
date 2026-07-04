"""
train_head_search.py — HEAD-SEARCH phase: the first task where vision is behaviorally
NECESSARY. Ball spawns on a WIDE cone (±68°, beyond the forward-crawl reach envelope, so
the diagnostic shows the bearing is worth ~+20 pts) and OUTSIDE the static ±22° camera view,
so the policy must actively SEARCH — turn its head (new head_yaw actuator = head_tilt_side,
which yaws when prone) and/or its body — to bring the ball into view, read its direction from
pixels, and crawl to it.

This is the setup the ±22° confound could not provide: a winnable task (ball bringable into
view by a ±60° head-yaw, precheck-confirmed) where forward-crawl FAILS, so vision can finally
convert its R²=0.84 directional read into a real contact gain ("reinforced" testable at last).

Vision encoder warm-started from Stage A (bearing_cnn.pt) so vision isn't learned from scratch;
gait + search + steering ARE learned by reward (the hard part). Entropy drives search
exploration. macOS: vision => DummyVecEnv. Honest expectation: this is the hard follow-on; a
first run is exploratory.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.train_head_search \
    --steps 2000000 --run-tag head_search_v1 --warmstart-cnn alien_baby/results/stage_a_v1/bearing_cnn.pt
"""
import argparse
import pathlib
import subprocess

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from stable_baselines3.common.callbacks import BaseCallback

from alien_baby.crawler.mimo_crawler_env import CRAWL_POSES
from alien_baby.crawler.train_crawler import make_env, LivenessGateCallback, _SafeSaveEvalCallback
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide_hs.xml"   # has the head_yaw actuator

# Search-shaping curriculum: (fraction of total steps, spawn_cone_deg). Start narrow so the
# policy masters gait+approach where forward-crawl works, then widen so the ball moves off the
# static view and SEARCH becomes necessary — build the skills in sequence, not all at once.
CONE_SCHEDULE = [(0.0, 44.0), (0.15, 68.0), (0.35, 90.0), (0.60, 136.0)]


class CurriculumConeCallback(BaseCallback):
    """Widen spawn_cone_deg on train + eval envs as training progresses."""

    def __init__(self, schedule, total_steps, eval_env, verbose=0):
        super().__init__(verbose)
        self.schedule = schedule
        self.total = total_steps
        self.eval_env = eval_env
        self._current = None

    def _target_cone(self):
        frac = self.num_timesteps / max(1, self.total)
        cone = self.schedule[0][1]
        for f, c in self.schedule:
            if frac >= f:
                cone = c
        return cone

    def _on_step(self):
        cone = self._target_cone()
        if cone != self._current:
            self._current = cone
            self.training_env.set_attr("spawn_cone_deg", cone)
            self.eval_env.set_attr("spawn_cone_deg", cone)
            print(f"[curriculum] step {self.num_timesteps}: spawn cone -> +/-{cone/2:.0f}")
        return True


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


def warmstart_encoder(model, path):
    """Load Stage A bearing CNN conv-trunk weights into the policy's StereoCrawlerCNN."""
    sd = torch.load(path, map_location="cpu")
    conv = {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")}
    fe = model.policy.features_extractor
    missing = fe.cnn.load_state_dict(conv, strict=False)
    print(f"[head_search] warm-started encoder conv from {path} ({missing})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--spawn-cone-deg", type=float, default=136.0)   # +/-68, vision-necessary
    p.add_argument("--spawn-radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)   # >0 to drive search exploration
    p.add_argument("--run-tag", default="head_search_v1")
    p.add_argument("--device", default="mps")
    p.add_argument("--warmstart-cnn", default=None)
    p.add_argument("--curriculum", action="store_true",
                   help="search-shaping: widen the spawn cone over training (CONE_SCHEDULE)")
    args = p.parse_args()

    tag = args.run_tag
    # With the curriculum, envs START at the first (narrow) cone; the callback widens them.
    if args.curriculum:
        args.spawn_cone_deg = CONE_SCHEDULE[0][1]
    print(f"\n=== HEAD-SEARCH training: {tag} ===")
    print(f"  steps={args.steps} n_envs={args.n_envs} start_cone=+/-{args.spawn_cone_deg/2:.0f} "
          f"curriculum={args.curriculum} ent={args.ent_coef} device={args.device}")

    train_env, eval_env = build_envs(args)
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
        clip_range=0.2, verbose=1, seed=args.seed, device=args.device,
        policy_kwargs=dict(
            features_extractor_class=StereoCrawlerCNN,
            net_arch=[256, 256],
        ),
    )
    if args.warmstart_cnn:
        warmstart_encoder(model, args.warmstart_cnn)

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
    if args.curriculum:
        callbacks.append(CurriculumConeCallback(CONE_SCHEDULE, args.steps, eval_env))
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), progress_bar=False)
    model.save(str(RESULTS / f"{tag}_final"))
    train_env.save(str(RESULTS / tag / "vec_normalize.pkl"))
    print(f"=== head-search done: {tag} ===")
    _chime()


if __name__ == "__main__":
    main()
