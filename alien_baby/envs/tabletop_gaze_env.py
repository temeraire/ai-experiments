"""
v6: Tabletop env with pan/tilt head camera.

Same arm + 3 objects as v1-v5, but adds:
- A head body (pan + tilt) with its own 45-deg-FOV camera.
- 2 position-controlled head actuators on top of the 3 arm torque actuators.
- Head joint angles included in proprio (so the agent knows where it's looking).

Vision (when enabled) comes from the head camera, not the overhead camera.
The overhead camera is kept in the XML for rendering/visualization only.

Proprio (9 dims, hide_target_offset=True; 12 dims otherwise):
  [3 joint pos, 3 joint vel, 1 touch, 2 head angles]  (+3 target offset if not hidden)

Action (5 dims):
  [3 arm torques, 2 head gaze positions]
"""

import pathlib
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


XML_PATH = str(pathlib.Path(__file__).parent / "tabletop_v6.xml")

CAM_WIDTH = 64
CAM_HEIGHT = 64


class TabletopGazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, vision=False, target_object=0, render_mode=None,
                 hide_target_offset=True):
        super().__init__()
        self.vision = vision
        self.target_object = target_object
        self.render_mode = render_mode
        self.hide_target_offset = hide_target_offset

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
        self._head_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam"
        )
        self._overhead_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead"
        )

        # Object slide joint qpos addresses
        self._obj_qpos_addrs = []
        for i in range(3):
            jx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_x")
            jy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_y")
            addr_x = self.model.jnt_qposadr[jx]
            addr_y = self.model.jnt_qposadr[jy]
            self._obj_qpos_addrs.append((addr_x, addr_y))

        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Action: 3 arm torques + 2 head position commands, all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self._build_observation_space()

        if self.vision:
            self._renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)
        else:
            self._renderer = None

        self._step_count = 0
        self._max_steps = 200

        # Head ctrl ranges from XML (for rescaling [-1, 1] → actual range)
        self._head_pan_range = (-1.2, 1.2)
        self._head_tilt_range = (-0.8, 0.8)

    def _build_observation_space(self):
        # Proprio: 3 jpos + 3 jvel + 1 touch + 2 head angles (+ 3 rel pos if not hidden)
        proprio_dim = 9 if self.hide_target_offset else 12
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
        # sensordata layout (from XML): touch(1), 3 jointpos, 3 jointvel, 2 head pos = 9 values
        # Index 0 = touch sensor; 1-3 = joint0-2 pos; 4-6 = vel; 7-8 = head pan/tilt pos
        touch_sensor = self.data.sensordata[0:1].copy()
        joint_pos = self.data.sensordata[1:4].copy()
        joint_vel = self.data.sensordata[4:7].copy()
        head_pos = self.data.sensordata[7:9].copy()
        # Override touch: use contact detection for target-specific touch (touch sensor
        # fires on any contact including the table)
        touch = np.array([1.0 if self._is_touching_target() else 0.0], dtype=np.float32)

        if self.hide_target_offset:
            return np.concatenate([joint_pos, joint_vel, touch, head_pos]).astype(np.float32)

        fingertip_pos = self.data.site_xpos[self._fingertip_site_id].copy()
        target_body_id = self._target_body_ids[self.target_object]
        target_pos = self.data.xpos[target_body_id].copy()
        rel_pos = (target_pos - fingertip_pos).astype(np.float32)
        return np.concatenate([joint_pos, joint_vel, touch, head_pos, rel_pos]).astype(np.float32)

    def _get_vision_obs(self):
        self._renderer.update_scene(self.data, camera=self._head_cam_id)
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
        # Reward is unchanged from v1-v5 (distance + touch). No reward for looking.
        distance = self._get_fingertip_target_dist()
        touching = self._is_touching_target()
        reward = -distance
        if distance < 0.08:
            reward += 0.5
        if distance < 0.04:
            reward += 1.0
        if touching:
            reward += 5.0
        # Small control penalty on arm only; head control is free
        reward -= 0.01 * np.sum(np.square(self.data.ctrl[:3]))
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        if self.np_random is not None:
            for addr_x, addr_y in self._obj_qpos_addrs:
                angle = self.np_random.uniform(0, 2 * np.pi)
                radius = self.np_random.uniform(0.1, 0.22)
                self.data.qpos[addr_x] = radius * np.cos(angle)
                self.data.qpos[addr_y] = radius * np.sin(angle)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        # Arm actuators 0-2 take torque in [-1, 1] directly.
        # Head actuators 3-4 are position-controlled: rescale [-1, 1] to joint range.
        ctrl = np.zeros(5, dtype=np.float32)
        ctrl[0:3] = np.clip(action[0:3], -1.0, 1.0)
        pan_lo, pan_hi = self._head_pan_range
        tilt_lo, tilt_hi = self._head_tilt_range
        pan_cmd = np.clip(action[3], -1.0, 1.0)
        tilt_cmd = np.clip(action[4], -1.0, 1.0)
        ctrl[3] = 0.5 * (pan_cmd + 1.0) * (pan_hi - pan_lo) + pan_lo
        ctrl[4] = 0.5 * (tilt_cmd + 1.0) * (tilt_hi - tilt_lo) + tilt_lo
        self.data.ctrl[:] = ctrl
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
            self._renderer.update_scene(self.data, camera=self._overhead_cam_id)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
