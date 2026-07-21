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
        self.dt = self.n_sub * self.model.opt.timestep     # policy-step duration (for feet-air-time)
        # the 4 ankle tips are the "feet"; track their ground contact for the feet-air-time reward
        self.foot_gids = [self.model.geom(n).id for n in
                          ("left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom")]
        self.floor_gid = self.model.geom("floor").id
        self.air_time = np.zeros(4, np.float32)
        self.last_action = np.zeros(8, np.float32)

    def _foot_contacts(self):
        c = np.zeros(4, bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.floor_gid in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == self.floor_gid else con.geom1
                if other in self.foot_gids:
                    c[self.foot_gids.index(other)] = True
        return c

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
        # Park BOTH target balls far away. (Bug found 2026-07-20: only ball 1 was parked; ball 2 sat
        # 1 m from the creature, and after the balls were enlarged to r=0.5 / ~105 kg for the vision
        # task it spawned OVERLAPPING the creature's foot -- the huge contact force flung it over every
        # reset, which is what made gait v4-v8 fall/never learn. The gait must train on a clear floor.)
        self.data.qpos[15:22] = [50, 50, 0.1, 1, 0, 0, 0]
        self.data.qpos[22:29] = [50, -50, 0.1, 1, 0, 0, 0]
        self.data.qvel[:14] = self.rng.uniform(-0.03, 0.03, 14)
        # v10: ACHIEVABLE speed range (v9 with the bug fixed walked fine but only ~0.1 m/s because the
        # 0.3-1.2 target was always far, so the saturating reward barely paid -> it gave up on speed).
        # A 0.3-0.8 range it can actually hit makes the tracking reward worth chasing. Turn +-0.6.
        self.cmd = np.array([self.rng.uniform(0.3, 0.8), self.rng.uniform(-0.6, 0.6)], np.float32)
        self.air_time = np.zeros(4, np.float32)
        self.last_action = np.zeros(8, np.float32)
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
        up_body = R.T @ np.array([0, 0, 1.0])
        up_z = up_body[2]
        linv_body = R.T @ v[0:3]
        vx_body, vy_body, vz_body = linv_body
        w_body = R.T @ v[3:6]
        yaw_rate = v[5]
        # v8 = the STANDARD velocity-command locomotion recipe (Rudin 2022 / legged_gym / Isaac-Lab),
        # ported into our stack after the lit-scout reuse check. Two SATURATING exp tracking terms
        # (forward-speed + turn-rate, comparably weighted) so once the command is met there is no gain
        # from spinning faster (this is what killed v5's circling), PLUS feet-air-time (rewards real
        # steps) and the upright/smoothness regularizers that make it walk instead of shuffle.
        r_lin = np.exp(-((vx_body - self.cmd[0]) ** 2) / 0.25)   # forward-speed tracking (saturating)
        r_ang = np.exp(-((yaw_rate - self.cmd[1]) ** 2) / 0.25)  # turn-rate tracking (positive reward)
        contact = self._foot_contacts()                          # feet-air-time: reward a ~0.3s swing
        first_contact = (self.air_time > 0.0) & contact          # then a clean touchdown -> real gait
        self.air_time += self.dt
        r_air = float(np.sum((self.air_time - 0.3) * first_contact))
        self.air_time[contact] = 0.0
        p_orient = up_body[0] ** 2 + up_body[1] ** 2             # stay flat/upright
        p_vz = vz_body ** 2                                      # no bouncing
        p_wxy = w_body[0] ** 2 + w_body[1] ** 2                  # no roll/pitch spin
        p_arate = np.sum((action - self.last_action) ** 2)      # smooth actions
        p_drift = vy_body ** 2                                   # no sideways slide (curve/arc)
        ctrl_cost = np.sum(np.square(action))
        # v10 weights: forward tracking up (1.5) to chase real speed; turn tracking up (1.0, ~equal)
        # and drift penalty up (0.2) to hold a straight heading instead of v9's gentle arc.
        reward = (1.5 * r_lin + 1.0 * r_ang + 1.0 * r_air
                  - 2.0 * p_vz - 0.05 * p_wxy - 0.5 * p_orient
                  - 0.01 * p_arate - 0.2 * p_drift - 0.0005 * ctrl_cost)
        self.last_action = np.array(action, np.float32)
        # resample the command within the episode so BOTH turn directions get practised
        if self.t % 120 == 0:
            self.cmd = np.array([self.rng.uniform(0.0, 0.6), self.rng.uniform(-0.8, 0.8)], np.float32)
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
