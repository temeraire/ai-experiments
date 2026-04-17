"""
v8 environment: creature on a platform with survival stakes.

A torso-sled with two arms and a pan/tilt head on a finite elevated platform.
Fall off the edge and you die. Reach the target ball to get reward.
Gravity is the pencil tap.
"""

import pathlib
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


XML_PATH = str(pathlib.Path(__file__).parent / "platform_creature.xml")

CAM_HEIGHT = 16
CAM_WIDTH = 16

PLATFORM_HALF = 1.0
PLATFORM_Z = 2.0
FALL_PENALTY = -500.0
HUNGER_PENALTY = -0.05
EDGE_WARN_DIST = 0.3
EDGE_WARN_PENALTY = -0.5
CONTACT_REWARD = 200.0
ATTRACT_SCALE = 0.3
ATTRACT_MAX_DIST = 2.0


class PlatformCreatureEnv(gym.Env):
    """Creature on a platform. Fall off = death. Touch target = reward."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vision=False, render_mode=None):
        super().__init__()
        self.vision = vision
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Cache body/geom/site IDs
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self._left_hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand"
        )
        self._right_hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand"
        )
        self._target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._left_hand_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_geom"
        )
        self._right_hand_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand_geom"
        )
        self._target_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom"
        )
        self._head_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam"
        )
        self._overhead_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead"
        )

        # Target slide joint addresses
        self._target_x_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
        ]
        self._target_y_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
        ]

        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Action: 6 arm torques + 2 head position commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._build_observation_space()

        if self.vision:
            self._renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)
        else:
            self._renderer = None

        self._step_count = 0
        self._max_steps = 300
        self._prev_dist = None
        self._touched = False

    def _build_observation_space(self):
        proprio_dim = self._proprio_dim()
        if self.vision:
            total = proprio_dim + CAM_HEIGHT * CAM_WIDTH * 3
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
            )

    def _proprio_dim(self):
        # 6 arm joint pos + 6 arm joint vel + 2 head joint pos
        # + 4 torso orientation (quat) + 3 torso lin vel + 3 torso ang vel
        # + 2 hand touch + 1 torso touch
        # + 2 torso xy position on platform (distance from center)
        return 6 + 6 + 2 + 4 + 3 + 3 + 3 + 2  # = 29

    def _get_proprio(self):
        sd = self.data.sensordata
        # Arm joint positions (6)
        arm_pos = sd[0:6].copy()
        # Arm joint velocities (6)
        arm_vel = sd[6:12].copy()
        # Head joint positions (2)
        head_pos = sd[12:14].copy()
        # Touch sensors (3)
        touch = sd[14:17].copy()

        # Torso orientation (quaternion, 4)
        torso_quat = self.data.qpos[3:7].copy()
        # Torso linear velocity (3)
        torso_vel = self.data.qvel[0:3].copy()
        # Torso angular velocity (3)
        torso_angvel = self.data.qvel[3:6].copy()

        # Torso xy on platform (how far from center — edge awareness)
        torso_xy = self.data.xpos[self._torso_id][:2].copy()

        return np.concatenate([
            arm_pos, arm_vel, head_pos,
            torso_quat, torso_vel, torso_angvel,
            touch, torso_xy,
        ]).astype(np.float32)

    def _get_pixels(self):
        self._renderer.update_scene(self.data, camera=self._head_cam_id)
        pixels = self._renderer.render()
        return (pixels.astype(np.float32) / 255.0).flatten()

    def _get_obs(self):
        proprio = self._get_proprio()
        if self.vision:
            pixels = self._get_pixels()
            return np.concatenate([proprio, pixels])
        return proprio

    def _check_hand_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            hand_geoms = {self._left_hand_geom, self._right_hand_geom}
            if (g1 in hand_geoms and g2 == self._target_geom) or \
               (g2 in hand_geoms and g1 == self._target_geom):
                return True
        return False

    def _has_fallen(self):
        torso_z = self.data.xpos[self._torso_id][2]
        return torso_z < PLATFORM_Z - 0.2

    def _edge_distance(self):
        """How far the torso center is from the nearest edge."""
        tx, ty = self.data.xpos[self._torso_id][:2]
        dx = PLATFORM_HALF - abs(tx)
        dy = PLATFORM_HALF - abs(ty)
        return min(dx, dy)

    def _nearest_hand_dist(self):
        """Distance from nearest hand to target."""
        target_pos = self.data.xpos[self._target_id]
        left_pos = self.data.xpos[self._left_hand_id]
        right_pos = self.data.xpos[self._right_hand_id]
        d_left = np.linalg.norm(left_pos - target_pos)
        d_right = np.linalg.norm(right_pos - target_pos)
        return min(d_left, d_right)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._touched = False
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        # Randomize target position on platform
        if self.np_random is not None:
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(0.2, 0.7)
            self.data.qpos[self._target_x_addr] = radius * np.cos(angle)
            self.data.qpos[self._target_y_addr] = radius * np.sin(angle)

        mujoco.mj_forward(self.model, self.data)
        self._prev_dist = self._nearest_hand_dist()
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)

        # Step physics (5 substeps for stability)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False

        # Check if fallen off platform
        if self._has_fallen():
            reward = FALL_PENALTY
            terminated = True
            return obs, reward, terminated, truncated, {"fell": True, "touched": False}

        # Hunger: every step without food hurts
        reward += HUNGER_PENALTY

        # Edge proximity warning
        edge_dist = self._edge_distance()
        if edge_dist < EDGE_WARN_DIST:
            reward += EDGE_WARN_PENALTY * (1.0 - edge_dist / EDGE_WARN_DIST)

        # Absolute attract: stronger signal the closer the hand is to target
        curr_dist = self._nearest_hand_dist()
        reward += ATTRACT_SCALE * max(0.0, 1.0 - curr_dist / ATTRACT_MAX_DIST)
        self._prev_dist = curr_dist

        # Contact: eat the target, episode ends
        if self._check_hand_contact():
            reward += CONTACT_REWARD
            self._touched = True
            terminated = True
            return obs, reward, terminated, truncated, {
                "fell": False,
                "touched": True,
                "edge_dist": edge_dist,
            }

        # Truncate if max steps
        if self._step_count >= self._max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {
            "fell": False,
            "touched": self._touched,
            "edge_dist": edge_dist,
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
