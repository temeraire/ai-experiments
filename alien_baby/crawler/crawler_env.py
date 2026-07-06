"""
CrawlerEnv — clean Gymnasium environment for the quadruped crawler.

Body: torso + 2 arms (shoulder 2DOF + elbow) + 2 legs (hip 2DOF + knee) + head (pan/tilt).
Vision: optional stereo left_eye + right_eye cameras (40×40 RGB each).
Platform: guardrails on all 4 sides — no fall termination.
Contact: any body part touching the ball counts.

Observation (proprio, always):
  torso pos xy (2) + torso quat (4) + torso linvel (3) + torso angvel (3)
  + 14 joint positions + 14 joint velocities
  = 40 values

Observation (vision, optional):
  left_eye  40×40×3 flattened (4800)
  right_eye 40×40×3 flattened (4800)
  = 9600 values appended after proprio

Action: 14 continuous torques (arms + legs) + 2 position setpoints (head).
"""

import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = pathlib.Path(__file__).parent / "crawler.xml"

# Platform top surface z-coordinate
PLATFORM_TOP_Z = 2.025

# Episode length
DEFAULT_MAX_STEPS = 600

# Ball spawn: uniformly within this radius of torso, in the forward hemisphere
DEFAULT_SPAWN_RADIUS = (0.3, 0.8)   # (min, max) meters from torso center
DEFAULT_SPAWN_CONE_DEG = 180        # forward hemisphere

# Rewards
CONTACT_REWARD = 200.0
STEP_COST = -0.05
VELOCITY_BONUS_SCALE = 0.05

# Camera render resolution
CAM_H = 40
CAM_W = 40

PROPRIO_DIM = 41  # 3 pos + 4 quat + 6 vel + 14 jpos + 14 jvel
VISION_DIM = CAM_H * CAM_W * 3 * 2   # two eyes


class CrawlerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vision=False, max_steps=DEFAULT_MAX_STEPS,
                 spawn_radius=None, spawn_cone_deg=DEFAULT_SPAWN_CONE_DEG,
                 render_mode=None):
        super().__init__()
        self.vision = vision
        self.max_steps = max_steps
        self.spawn_radius = spawn_radius or DEFAULT_SPAWN_RADIUS
        self.spawn_cone_deg = spawn_cone_deg
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        n_obs = PROPRIO_DIM + (VISION_DIM if vision else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        n_act = self.model.nu   # number of actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

        if vision:
            self._renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        else:
            self._renderer = None

        self._step = 0
        self._touched = False
        self._target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        # world body id=0 (platform, rails, ground) — used to filter contact pairs
        self._world_body_id = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Spawn torso at platform center, upright crawl pose
        torso_qpos_start = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]
        self.data.qpos[torso_qpos_start:torso_qpos_start + 3] = [0.0, 0.0, 2.12]
        self.data.qpos[torso_qpos_start + 3:torso_qpos_start + 7] = [1, 0, 0, 0]  # w x y z

        # Place ball in forward hemisphere at random distance
        angle_rad = np.deg2rad(self.spawn_cone_deg / 2)
        theta = self.np_random.uniform(-angle_rad, angle_rad)   # azimuth in cone
        r = self.np_random.uniform(*self.spawn_radius)
        bx = r * np.sin(theta)
        by = r * np.cos(theta)
        bz = PLATFORM_TOP_Z + 0.05   # ball radius above platform

        target_qpos_start = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        ]
        self.data.qpos[target_qpos_start:target_qpos_start + 3] = [bx, by, bz]
        self.data.qpos[target_qpos_start + 3:target_qpos_start + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._touched = False
        self._contact_rewarded = False
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        self._step += 1

        obs = self._get_obs()
        torso_vel = self.data.qvel[0:3]
        torso_speed = float(np.linalg.norm(torso_vel[:2]))  # horizontal speed

        contacted = self._check_contact()
        if contacted:
            self._touched = True

        reward = STEP_COST + VELOCITY_BONUS_SCALE * torso_speed
        terminated = False
        truncated = self._step >= self.max_steps

        info = {"touched": self._touched, "step": self._step}

        if contacted and not getattr(self, "_contact_rewarded", False):
            reward += CONTACT_REWARD
            self._contact_rewarded = True
            terminated = True

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _check_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            # Contact between target geom and any other geom
            if g1 == self._target_geom_id or g2 == self._target_geom_id:
                # Make sure the other geom is part of the crawler (not the platform/rails)
                other_geom = g2 if g1 == self._target_geom_id else g1
                other_body = self.model.geom_bodyid[other_geom]
                if other_body != 0:  # 0 = world body (platform, rails, ground)
                    return True
        return False

    # ------------------------------------------------------------------
    def _get_obs(self):
        # Torso free-joint qpos starts at 0 by convention (it's the first joint)
        root_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]
        root_vel_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]

        torso_pos  = self.data.qpos[root_adr:root_adr + 3].copy()      # 3
        torso_quat = self.data.qpos[root_adr + 3:root_adr + 7].copy()  # 4
        torso_vel  = self.data.qvel[root_vel_adr:root_vel_adr + 6].copy()  # 6: linvel + angvel

        # 14 hinge joint positions (arms + legs + head), 14 velocities
        hinge_joints = [
            "fl_shoulder_pitch", "fl_shoulder_yaw", "fl_elbow",
            "fr_shoulder_pitch", "fr_shoulder_yaw", "fr_elbow",
            "rl_hip_pitch", "rl_hip_yaw", "rl_knee",
            "rr_hip_pitch", "rr_hip_yaw", "rr_knee",
            "head_pan", "head_tilt",
        ]
        jpos = np.array([
            self.data.qpos[self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in hinge_joints
        ], dtype=np.float32)
        jvel = np.array([
            self.data.qvel[self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in hinge_joints
        ], dtype=np.float32)

        proprio = np.concatenate([
            torso_pos.astype(np.float32),    # 3
            torso_quat.astype(np.float32),   # 4 — orientation signal for tipping
            torso_vel.astype(np.float32),    # 6
            jpos,                            # 14
            jvel,                            # 14
        ])  # total: 41... wait let me recount: 3+4+6+14+14 = 41

        if not self.vision:
            return proprio

        self._renderer.update_scene(self.data, camera="left_eye")
        left_img = self._renderer.render().copy().astype(np.float32) / 255.0
        self._renderer.update_scene(self.data, camera="right_eye")
        right_img = self._renderer.render().copy().astype(np.float32) / 255.0

        return np.concatenate([proprio, left_img.ravel(), right_img.ravel()])

    # ------------------------------------------------------------------
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        self._renderer.update_scene(self.data, camera="ringside")
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
