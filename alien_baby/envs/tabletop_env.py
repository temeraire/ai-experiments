"""
Tabletop reaching environment with configurable sensory channels.

Based on the proven Gymnasium Reacher design: planar arm with Z-axis hinges.
Supports toggling vision on/off for staged developmental training
per the interpenetration idea theory.
"""

import pathlib
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


XML_PATH = str(pathlib.Path(__file__).parent / "tabletop.xml")

CAM_WIDTH = 64
CAM_HEIGHT = 64


class TabletopReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, vision=False, target_object=0, render_mode=None,
                 hide_target_offset=False, target_size=None):
        super().__init__()
        self.vision = vision
        self.target_object = target_object
        self.render_mode = render_mode
        # When True, proprio drops the 3-dim fingertip-to-target offset,
        # so the agent cannot localize the target via proprio alone.
        self.hide_target_offset = hide_target_offset
        # Optional override for the target object's geom size (sphere radius).
        # When set, shrinks/grows the target ball — tighter size forces the
        # agent to be more precise about contact location.
        self.target_size = target_size

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._fingertip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip"
        )
        self._fingertip_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_geom"
        )
        self._target_body_names = ["object_0", "object_1", "object_2"]
        self._target_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._target_body_names
        ]
        self._target_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj{i}_geom")
            for i in range(3)
        ]
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead"
        )

        # Object slide joint qpos addresses (each object has 2 slide joints: x, y)
        self._obj_qpos_addrs = []
        for i in range(3):
            jx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_x")
            jy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_y")
            addr_x = self.model.jnt_qposadr[jx]
            addr_y = self.model.jnt_qposadr[jy]
            self._obj_qpos_addrs.append((addr_x, addr_y))

        # Optionally resize target geom (applied only to the active target object).
        if self.target_size is not None:
            target_geom = self._target_geom_ids[self.target_object]
            self.model.geom_size[target_geom, 0] = float(self.target_size)

        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Action: 3 joint torques in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._build_observation_space()

        if self.vision:
            self._renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)
        else:
            self._renderer = None

        self._step_count = 0
        self._max_steps = 200

    def _build_observation_space(self):
        # Proprio: 3 joint pos + 3 joint vel + 1 touch (+ 3 rel pos if not hidden)
        proprio_dim = 7 if self.hide_target_offset else 10
        low = np.full(proprio_dim, -np.inf, dtype=np.float32)
        high = np.full(proprio_dim, np.inf, dtype=np.float32)

        if self.vision:
            self.observation_space = spaces.Dict({
                "proprio": spaces.Box(low, high, dtype=np.float32),
                "vision": spaces.Box(
                    0, 255, shape=(CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8
                ),
            })
        else:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def _is_touching_target(self):
        target_geom_id = self._target_geom_ids[self.target_object]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self._fingertip_geom_id and g2 == target_geom_id) or \
               (g2 == self._fingertip_geom_id and g1 == target_geom_id):
                return True
        return False

    def _get_proprio_obs(self):
        # Joint positions (3) from sensors
        joint_pos = self.data.sensordata[1:4].copy()
        # Joint velocities (3) from sensors
        joint_vel = self.data.sensordata[4:7].copy()
        # Touch: binary contact with target
        touch = np.array([1.0 if self._is_touching_target() else 0.0])

        if self.hide_target_offset:
            # Proprio-blind mode: no direct target-location signal.
            return np.concatenate([joint_pos, joint_vel, touch]).astype(np.float32)

        # Relative target position (2D, but we include 3D for consistency)
        fingertip_pos = self.data.site_xpos[self._fingertip_site_id].copy()
        target_body_id = self._target_body_ids[self.target_object]
        target_pos = self.data.xpos[target_body_id].copy()
        rel_pos = (target_pos - fingertip_pos).astype(np.float32)

        return np.concatenate([joint_pos, joint_vel, touch, rel_pos]).astype(np.float32)

    def _get_vision_obs(self):
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render().copy()

    def _get_obs(self):
        proprio = self._get_proprio_obs()
        if self.vision:
            return {"proprio": proprio, "vision": self._get_vision_obs()}
        return proprio

    def _get_fingertip_target_dist(self):
        fingertip_pos = self.data.site_xpos[self._fingertip_site_id]
        target_body_id = self._target_body_ids[self.target_object]
        target_pos = self.data.xpos[target_body_id]
        return float(np.linalg.norm(fingertip_pos - target_pos))

    def _get_reward(self):
        distance = self._get_fingertip_target_dist()
        touching = self._is_touching_target()

        # Dense distance reward + proximity bonuses + contact
        reward = -distance
        if distance < 0.08:
            reward += 0.5
        if distance < 0.04:
            reward += 1.0
        if touching:
            reward += 5.0

        # Small control penalty
        reward -= 0.01 * np.sum(np.square(self.data.ctrl))

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        # Randomize object positions via slide joints
        if self.np_random is not None:
            for addr_x, addr_y in self._obj_qpos_addrs:
                # Slide joints: qpos is offset from ref position
                # Place objects within arm reach (~0.24m from origin)
                angle = self.np_random.uniform(0, 2 * np.pi)
                radius = self.np_random.uniform(0.1, 0.22)
                self.data.qpos[addr_x] = radius * np.cos(angle)
                self.data.qpos[addr_y] = radius * np.sin(angle)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward()
        distance = self._get_fingertip_target_dist()
        touching = self._is_touching_target()

        terminated = touching
        truncated = self._step_count >= self._max_steps

        if terminated:
            reward += 10.0

        return obs, reward, terminated, truncated, {
            "distance": distance, "touching": touching
        }

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)
            self._renderer.update_scene(self.data, camera=self._cam_id)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
