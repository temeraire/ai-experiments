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
GAIT = "alien_baby/results/ant_gait_v10_best/best_model.zip"   # the first WORKING walker (v3-v9 broken;
# v4-v8 were sabotaged by a 105kg ball spawned on the creature in the gait env -- fixed 2026-07-20)
STAND = np.array([0, 0, 0.55, 1, 0, 0, 0, 0, 1.0, 0, -1.0, 0, -1.0, 0, 1.0])
CAM = 32
PROP = 9   # high-level proprio: torso up-vector(3) + lin vel body(3) + ang vel(3)


class VisionSteerEnv(gym.Env):
    def __init__(self, max_steps=150, k_sub=8, seed=0,
                 single_ball=False, cone=0.45, reach=1.0):
        # single_ball: SPATIAL-BEARING mode (2026-07-20). ONE red ball at a random bearing in
        # +-cone rad of MEASURED gaze-forward; the blue decoy is PARKED far away (same [50,-50]
        # trick the gait env uses). Success then requires turning the correct WAY, which needs
        # the ball's DIRECTION from vision -- not a colour flag. Defaults reproduce the old
        # two-ball colour-choice env byte-for-byte.
        # cone/reach: strategist 2026-07-20 -- at reach=1.0 a straight-ahead motor habit wins
        # whenever |bearing| < asin(1.0/r) = 0.41 rad @2.5m, i.e. most of a +-0.45 cone. Widening
        # to 0.9 (FOV is ~1.05) and tightening reach to 0.75 drops that no-steering floor.
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.gait = PPO.load(GAIT, device="cpu")
        self.rend = mujoco.Renderer(self.model, CAM, CAM)
        self.max_steps, self.k = max_steps, k_sub
        self.single_ball, self.cone, self.reach = bool(single_ball), float(cone), float(reach)
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

    def red_in_view(self):  # is the RED target currently visible in the (left) eye?
        self.rend.update_scene(self.data, camera="left_eye")
        im = self.rend.render()
        R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
        return int(np.sum((R > 150) & (G < 90) & (B < 90))) >= 2

    def facing_deg(self):   # world yaw the creature is facing, degrees
        M = np.zeros(9); mujoco.mju_quat2Mat(M, self.data.qpos[3:7]); M = M.reshape(3, 3)
        return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

    def gaze_bearing(self):
        """Signed bearing (rad) of the RED ball in the CYCLOPEAN EYE frame: +left / -right, 0 = dead
        ahead of where the creature actually LOOKS. Per the CLAUDE.md GAZE rule this is measured from
        the EYE (which sits ~0.37 m forward of the torso on a down-tilted head), against the camera's
        own forward axis -- NOT from the world origin and NOT from the torso, whose axes are ~90 deg
        off the functional forward. This is the ground-truth x for the R2 instrument check."""
        lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")
        eye = 0.5 * (self.data.cam_xpos[lid] + self.data.cam_xpos[rid])
        f = -self.data.cam_xmat[lid].reshape(3, 3)[:, 2]     # camera looks down its own -z
        f = f[:2] / (np.linalg.norm(f[:2]) + 1e-9)           # horizontal gaze direction
        v = self.data.qpos[15:17] - eye[:2]                  # eye -> red ball, horizontal
        v = v / (np.linalg.norm(v) + 1e-9)
        return float(np.arctan2(f[0] * v[1] - f[1] * v[0], f[0] * v[0] + f[1] * v[1]))

    def _dist(self):        # to RED target
        return float(np.linalg.norm(self.data.qpos[15:17] - self.data.qpos[0:2]))

    def _dist_blue(self):   # to BLUE decoy
        return float(np.linalg.norm(self.data.qpos[22:24] - self.data.qpos[0:2]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:15] = STAND + self.rng.uniform(-0.02, 0.02, 15)
        # MEASURE the creature's ACTUAL forward (don't assume +x) and place BOTH balls in front of
        # THAT, within +-30deg of where it truly looks, at a visible range -- so a target is NEVER
        # spawned in a blind spot (winnability). Positions from the SAME distribution, RANDOMLY
        # labelled red/blue, so only COLOUR distinguishes them. (Out-of-view search comes LATER.)
        M = np.zeros(9); mujoco.mju_quat2Mat(M, self.data.qpos[3:7]); M = M.reshape(3, 3)
        fwd = float(np.arctan2(M[1, 0], M[0, 0]))         # world yaw of the body's forward (+x) axis
        # Creature is 1.44 m foot-to-foot (feet reach 0.72 m from center), so a target must sit
        # 2-3 BODY-LENGTHS out to make reaching an actual walk-and-steer, not a one-step touch.
        # Balls are body-sized (r=0.5) so they stay visible to ~6 m; z=0.5 = ball resting on floor.
        # Sample until the two balls are >=1.2 m apart (radius sum is 1.0) -- else two light spheres
        # spawn INTERPENETRATING and MuJoCo's overlap-resolution flings them across the map.
        if self.single_ball:
            # SPATIAL-BEARING mode: one red ball at a random bearing; blue parked far off-world.
            a1 = self.rng.uniform(-self.cone, self.cone)
            r1 = self.rng.uniform(2.5, 4.0)
            self.data.qpos[15:22] = [r1 * np.cos(fwd + a1), r1 * np.sin(fwd + a1), 0.5, 1, 0, 0, 0]
            self.data.qpos[22:29] = [50, -50, 0.5, 1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            self.t = 0; self.prev = self._dist()
            return self._obs(), {}
        for _ in range(50):
            a1 = self.rng.uniform(-0.45, 0.45)
            a2 = float(np.clip(a1 + self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.35, 0.55), -0.55, 0.55))
            r1, r2 = self.rng.uniform(2.5, 4.0), self.rng.uniform(2.5, 4.0)
            p1 = np.array([r1 * np.cos(fwd + a1), r1 * np.sin(fwd + a1)])
            p2 = np.array([r2 * np.cos(fwd + a2), r2 * np.sin(fwd + a2)])
            if np.linalg.norm(p1 - p2) >= 1.2:
                break
        ri = int(self.rng.integers(2))                    # random which is the red target
        rp, bp = (p1, p2) if ri == 0 else (p2, p1)
        self.data.qpos[15:22] = [rp[0], rp[1], 0.5, 1, 0, 0, 0]
        self.data.qpos[22:29] = [bp[0], bp[1], 0.5, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0; self.prev = self._dist()
        return self._obs(), {}

    def step(self, action):
        # map policy action to the gait's SWEET SPOT: v10 walks fastest at cmd 0.6 (0.23 m/s) and
        # slows to 0.09 m/s at 0.8, so cap forward at 0.6, not 0.8. Turn +-0.6 matches training.
        cmd = np.array([0.3 * (action[0] + 1.0), 0.6 * float(action[1])], np.float32)  # fwd 0..0.6, turn +-0.6
        for _ in range(self.k):                       # k gait-control decisions per high-level command
            ga, _ = self.gait.predict(self._gait_obs(cmd), deterministic=True)
            for _ in range(5):                        # gait was trained at n_sub=5 mj_steps/action
                self.data.ctrl[:] = ga
                mujoco.mj_step(self.model, self.data)
        self.t += 1
        d = self._dist(); db = self._dist_blue()
        up_z = (self._R().T @ np.array([0, 0, 1.0]))[2]
        reached_red = d < self.reach  # ball surface (r=0.5) is ~0.5 m in from center; creature reaches 0.72 m
        reached_blue = db < self.reach
        fell = self.data.qpos[2] < 0.28 or up_z < 0.4
        reward = 3.0 * (self.prev - d) + 0.02 - (5.0 if fell else 0.0)
        if reached_red:  reward += 10.0
        if reached_blue: reward -= 8.0        # decoy penalty: reaching blue is bad
        self.prev = d
        term = bool(reached_red or reached_blue or fell)
        return self._obs(), float(reward), term, self.t >= self.max_steps, {"red": reached_red, "blue": reached_blue}


def make_bearing_env(seed=0, cone=0.9, reach=1.0):
    """The SPATIAL-BEARING task env (one ball, random bearing). Single source of truth so the
    preflight, the trainer, the eval and the renderer are all guaranteed to score the same task.

    CALIBRATED 2026-07-20 by measurement, not assumption:
      reach=1.0 -- NOT 0.75. A perfect-vision scripted oracle scores 0% at 0.75 and 100% at 1.0:
        ball radius 0.5 + body radius floors centre-distance at ~0.93 m, and the ball is light
        (density=3) so it gets punted before you can get closer. 0.75 is UNWINNABLE (winnability
        rule) -- it looks like a harder task and is actually an impossible one.
      cone=0.9 -- with the measured numbers below this gives a 70-point window to detect vision in:
        best TARGET-INDEPENDENT policy (best fixed turn, ignores the ball) = 30%
        perfect-vision oracle                                              = 100%
      NB cmd_turn=0 is NOT "straight ahead": the v10 gait drifts +77..+112 deg left over an
      episode, so the motor-habit floor must be measured by sweeping fixed turns, not assumed.
    """
    return VisionSteerEnv(seed=seed, single_ball=True, cone=cone, reach=reach)


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
    p.add_argument("--single-ball", action="store_true",
                   help="SPATIAL-BEARING task: one ball at a random bearing, decoy parked. Success "
                        "requires turning the correct WAY, so vision must supply DIRECTION.")
    p.add_argument("--cone", type=float, default=0.9, help="bearing spread, rad (single-ball)")
    p.add_argument("--reach", type=float, default=1.0, help="contact threshold, m (single-ball)")
    p.add_argument("--init-vision", default=None,
                   help="transplant a crawler vision encoder: load its conv trunk (which already "
                        "learned to see our ball at 32x32) and FREEZE it; retrain only the steering "
                        "head. Reuse, not re-derive.")
    args = p.parse_args()
    os.makedirs("alien_baby/results", exist_ok=True)
    ekw = dict(single_ball=args.single_ball, cone=args.cone, reach=args.reach) if args.single_ball else {}
    venv = DummyVecEnv([(lambda i=i: VisionSteerEnv(seed=args.seed + i, **ekw)) for i in range(args.n_envs)])
    evalenv = DummyVecEnv([lambda: VisionSteerEnv(seed=args.seed + 777, **ekw)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=8, gamma=0.99,
                gae_lambda=0.95, ent_coef=0.005, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                   features_extractor_kwargs=dict(proprio_dim=PROP),
                                   net_arch=[128, 128]),
                verbose=1, seed=args.seed, device=args.device)
    if args.init_vision:
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
        _, params, _ = load_from_zip_file(args.init_vision, device=args.device)
        # copy the pixel pathway (conv trunk + proj + bearing head) from the crawler encoder into all
        # 3 aliased extractor copies; the conv trunk is body-agnostic (it sees pixels, not legs).
        px = {k: v for k, v in params["policy"].items()
              if any(s in k for s in ["cnn.", "proj.", "bearing_head."])}
        res = model.policy.load_state_dict(px, strict=False)
        frozen = 0
        for attr in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
            fe = getattr(model.policy, attr, None)
            if fe is not None:
                for p_ in fe.cnn.parameters():
                    p_.requires_grad = False; frozen += 1
        print(f"[init-vision] transplanted conv/proj/bearing from {args.init_vision} "
              f"(loaded {len(px)} tensors), froze conv trunk ({frozen} params). Retraining head only.")
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
