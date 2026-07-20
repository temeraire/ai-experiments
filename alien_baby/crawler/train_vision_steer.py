"""train_vision_steer.py — the DRIVER: eyes steer the frozen-gait quadruped to the ball it sees.

Hierarchical: a high-level policy reads the stereo eye-view (+ a little torso proprio) and outputs a
2D command (forward speed, turn rate). That command drives the FROZEN gait (ant_gait_v3) for k
sub-steps. Reward = getting closer to / reaching the ball. This is where vision becomes load-bearing:
the only way to reach a ball off to the side is to SEE it and turn toward it.

  python -m alien_baby.crawler.train_vision_steer --steps 300000 --run-tag vsteer_s0
"""
import argparse, os, subprocess
import numpy as np, mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN

XML = "alien_baby/crawler/quad_walker.xml"
GAIT = "alien_baby/results/ant_gait_v3_best/best_model.zip"
STAND = np.array([0, 0, 0.55, 1, 0, 0, 0, 0, 1.0, 0, -1.0, 0, -1.0, 0, 1.0])
CAM = 32
PROP = 9   # high-level proprio: torso up-vector(3) + lin vel body(3) + ang vel(3)


class VisionSteerEnv(gym.Env):
    def __init__(self, max_steps=120, k_sub=8, seed=0):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.gait = PPO.load(GAIT, device="cpu")
        self.rend = mujoco.Renderer(self.model, CAM, CAM)
        self.max_steps, self.k = max_steps, k_sub
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)          # [fwd, turn]
        self.observation_space = spaces.Box(-np.inf, np.inf, (PROP + 2 * CAM * CAM * 3,), np.float32)

    def _R(self):
        M = np.zeros(9); mujoco.mju_quat2Mat(M, self.data.qpos[3:7]); return M.reshape(3, 3)

    def _gait_obs(self, cmd):
        q, v, R = self.data.qpos, self.data.qvel, self._R()
        return np.concatenate([[q[2]], R.T @ [0, 0, 1.0], R.T @ v[0:3], v[3:6],
                               q[7:15], v[6:14], cmd]).astype(np.float32)

    def _obs(self):
        v, R = self.data.qvel, self._R()
        prop = np.concatenate([R.T @ [0, 0, 1.0], R.T @ v[0:3], v[3:6]]).astype(np.float32)
        self.rend.update_scene(self.data, camera="left_eye");  L = self.rend.render().astype(np.float32).ravel() / 255.0
        self.rend.update_scene(self.data, camera="right_eye"); Rr = self.rend.render().astype(np.float32).ravel() / 255.0
        return np.concatenate([prop, L, Rr])

    def _dist(self):        # to RED target
        return float(np.linalg.norm(self.data.qpos[15:17] - self.data.qpos[0:2]))

    def _dist_blue(self):   # to BLUE decoy
        return float(np.linalg.norm(self.data.qpos[22:24] - self.data.qpos[0:2]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:15] = STAND + self.rng.uniform(-0.02, 0.02, 15)
        # RED target + BLUE decoy, both in the wide cone, separated so ONLY colour vision picks red.
        # A blind policy can't tell them apart -> goes for whichever's on its path -> eats the blue
        # penalty ~half the time; seeing the colour is the only way to reliably reach red.
        rr = self.rng.uniform(0.5, 1.1); ar = self.rng.uniform(-1.4, 1.4)
        rb = self.rng.uniform(0.5, 1.1)
        ab = ar + self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.5, 1.0)
        self.data.qpos[15:22] = [rr * np.cos(ar), rr * np.sin(ar), 0.1, 1, 0, 0, 0]
        self.data.qpos[22:29] = [rb * np.cos(ab), rb * np.sin(ab), 0.1, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0; self.prev = self._dist()
        return self._obs(), {}

    def step(self, action):
        cmd = np.array([0.25 * (action[0] + 1.0), 0.8 * float(action[1])], np.float32)  # fwd 0..0.5, turn +-0.8
        for _ in range(self.k):                       # k gait-control decisions per high-level command
            ga, _ = self.gait.predict(self._gait_obs(cmd), deterministic=True)
            for _ in range(5):                        # gait was trained at n_sub=5 mj_steps/action
                self.data.ctrl[:] = ga
                mujoco.mj_step(self.model, self.data)
        self.t += 1
        d = self._dist(); db = self._dist_blue()
        up_z = (self._R().T @ np.array([0, 0, 1.0]))[2]
        reached_red = d < 0.55
        reached_blue = db < 0.55
        fell = self.data.qpos[2] < 0.28 or up_z < 0.4
        reward = 3.0 * (self.prev - d) + 0.02 - (5.0 if fell else 0.0)
        if reached_red:  reward += 10.0
        if reached_blue: reward -= 8.0        # decoy penalty: reaching blue is bad
        self.prev = d
        term = bool(reached_red or reached_blue or fell)
        return self._obs(), float(reward), term, self.t >= self.max_steps, {"red": reached_red, "blue": reached_blue}


def _chime():
    try: subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception: pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-tag", default="vsteer_s0")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs("alien_baby/results", exist_ok=True)
    venv = DummyVecEnv([(lambda i=i: VisionSteerEnv(seed=args.seed + i)) for i in range(args.n_envs)])
    evalenv = DummyVecEnv([lambda: VisionSteerEnv(seed=args.seed + 777)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=8, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.005, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                   features_extractor_kwargs=dict(proprio_dim=PROP),
                                   net_arch=[128, 128]),
                verbose=1, seed=args.seed, device=args.device)
    ev = EvalCallback(evalenv, best_model_save_path=f"alien_baby/results/{args.run_tag}_best",
                      eval_freq=10000, n_eval_episodes=10, deterministic=True, verbose=1)
    cp = CheckpointCallback(save_freq=25000, save_path=f"alien_baby/results/{args.run_tag}_ckpt",
                            name_prefix=args.run_tag)
    print(f"=== VISION-STEER training: {args.run_tag} steps={args.steps} n_envs={args.n_envs} dev={args.device} ===")
    model.learn(total_timesteps=args.steps, callback=[ev, cp], progress_bar=False)
    model.save(f"alien_baby/results/{args.run_tag}_final")
    _chime()
    print(f"=== vision-steer done: {args.run_tag} ===")


if __name__ == "__main__":
    main()
