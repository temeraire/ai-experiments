"""train_ant_gait.py — a command-STEERABLE quadruped gait (the low-level 'legs' for eyes-on-walker).

The Ant learns to track a commanded (forward speed, turn rate). Frozen later as AB's legs; the
vision layer will emit that command. Trained on quad_walker.xml (the eyed body) so it transfers
directly; the target ball is parked far away and ignored here (gait only).

  python -m alien_baby.crawler.train_ant_gait --steps 2000000 --run-tag ant_gait_s0
"""
import argparse, os, subprocess
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

XML = "alien_baby/crawler/quad_walker.xml"
STAND = np.array([0, 0, 0.55, 1, 0, 0, 0, 0, 1.0, 0, -1.0, 0, -1.0, 0, 1.0])  # ant standing pose


class AntGaitEnv(gym.Env):
    def __init__(self, max_steps=1000, seed=0):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.n_sub = 5
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (28,), np.float32)
        self.cmd = np.zeros(2, np.float32)

    def _obs(self):
        q, v = self.data.qpos, self.data.qvel
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q[3:7]); R = R.reshape(3, 3)
        up_body = R.T @ np.array([0, 0, 1.0])          # gravity-up in body frame (orientation)
        linv_body = R.T @ v[0:3]                        # torso lin vel in body frame
        return np.concatenate([
            [q[2]], up_body, linv_body, v[3:6], q[7:15], v[6:14], self.cmd]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:15] = STAND + self.rng.uniform(-0.03, 0.03, 15)
        self.data.qpos[15:22] = [50, 50, 0.1, 1, 0, 0, 0]   # park the ball far away
        self.data.qvel[:14] = self.rng.uniform(-0.03, 0.03, 14)
        self.cmd = np.array([self.rng.uniform(0.0, 0.6), self.rng.uniform(-0.8, 0.8)], np.float32)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        q, v = self.data.qpos, self.data.qvel
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q[3:7]); R = R.reshape(3, 3)
        up_z = (R.T @ np.array([0, 0, 1.0]))[2]
        vx_body = (R.T @ v[0:3])[0]
        yaw_rate = v[5]
        r_v = np.exp(-4.0 * (vx_body - self.cmd[0]) ** 2)
        r_y = np.exp(-4.0 * (yaw_rate - self.cmd[1]) ** 2)
        ctrl_cost = 0.005 * np.sum(np.square(action))
        reward = 0.5 + r_v + 0.5 * r_y - ctrl_cost           # 0.5 alive bonus
        fell = q[2] < 0.28 or up_z < 0.4
        term = bool(fell)
        trunc = self.t >= self.max_steps
        return self._obs(), float(reward), term, trunc, {}


def _play_done_sound():
    try: subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception: pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--run-tag", default="ant_gait_s0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs("alien_baby/results", exist_ok=True)
    venv = SubprocVecEnv([(lambda i=i: AntGaitEnv(seed=args.seed + i)) for i in range(args.n_envs)])
    evalenv = SubprocVecEnv([lambda: AntGaitEnv(seed=args.seed + 999)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=args.seed, device="cpu")
    ev = EvalCallback(evalenv, best_model_save_path=f"alien_baby/results/{args.run_tag}_best",
                      eval_freq=20000, n_eval_episodes=5, deterministic=True, verbose=1)
    cp = CheckpointCallback(save_freq=50000, save_path=f"alien_baby/results/{args.run_tag}_ckpt",
                            name_prefix=args.run_tag)
    print(f"=== ANT GAIT training: {args.run_tag}  steps={args.steps} n_envs={args.n_envs} ===")
    model.learn(total_timesteps=args.steps, callback=[ev, cp], progress_bar=False)
    model.save(f"alien_baby/results/{args.run_tag}_final")
    _play_done_sound()
    print(f"=== ant gait done: {args.run_tag} ===")


if __name__ == "__main__":
    main()
