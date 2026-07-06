"""
v7 env: moving-target gaze camera world.

Same as v6 TabletopGazeEnv but:
1. Uses `tabletop_v7.xml` (narrow 22 deg FOV head camera — forces gaze to track).
2. On reset, all three objects get a small random initial velocity so they
   drift across the table. Targets don't hold still; the agent must visually
   re-localize them continuously.

Damping on object slide joints is 0.05 (low but non-zero) so motion decays
gradually over an episode rather than bouncing forever.
"""

import pathlib
import numpy as np
import mujoco

from alien_baby.envs.tabletop_gaze_env import TabletopGazeEnv, CAM_HEIGHT, CAM_WIDTH


XML_PATH_V7 = str(pathlib.Path(__file__).parent / "tabletop_v7.xml")

# Initial speed range (m/s). Objects drift ~10-30 cm over a 200-step (4 sec)
# episode, bouncing off walls and each other, and can be bumped further by
# the arm. Damping on the slide joints (set in XML to 0.05) gradually slows
# them so they don't fly around forever.
OBJ_INIT_SPEED_MIN = 0.15
OBJ_INIT_SPEED_MAX = 0.35


class TabletopMovingGazeEnv(TabletopGazeEnv):
    """Pan/tilt head cam + narrow FOV + drifting target objects."""

    def __init__(self, vision=False, target_object=0, render_mode=None,
                 hide_target_offset=True):
        # Bypass parent __init__ that loads v6 XML; load v7 XML here.
        import gymnasium as gym
        from gymnasium import spaces
        gym.Env.__init__(self)

        self.vision = vision
        self.target_object = target_object
        self.render_mode = render_mode
        self.hide_target_offset = hide_target_offset

        self.model = mujoco.MjModel.from_xml_path(XML_PATH_V7)
        self.data = mujoco.MjData(self.model)

        # Cache IDs (same as v6)
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

        # Object slide joint qpos / qvel addresses
        self._obj_qpos_addrs = []
        self._obj_qvel_addrs = []
        for i in range(3):
            jx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_x")
            jy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_y")
            self._obj_qpos_addrs.append(
                (self.model.jnt_qposadr[jx], self.model.jnt_qposadr[jy])
            )
            self._obj_qvel_addrs.append(
                (self.model.jnt_dofadr[jx], self.model.jnt_dofadr[jy])
            )

        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

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
        self._head_pan_range = (-1.2, 1.2)
        self._head_tilt_range = (-0.8, 0.8)

    def reset(self, seed=None, options=None):
        # Call the grandparent reset (gym.Env.reset) to get np_random set up
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        self._step_count = 0
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        if self.np_random is not None:
            for (addr_x, addr_y), (vx_addr, vy_addr) in zip(
                self._obj_qpos_addrs, self._obj_qvel_addrs
            ):
                # Random starting position (same distribution as v6)
                angle = self.np_random.uniform(0, 2 * np.pi)
                radius = self.np_random.uniform(0.1, 0.22)
                self.data.qpos[addr_x] = radius * np.cos(angle)
                self.data.qpos[addr_y] = radius * np.sin(angle)

                # Random initial velocity (drift direction, small speed)
                v_angle = self.np_random.uniform(0, 2 * np.pi)
                v_speed = self.np_random.uniform(OBJ_INIT_SPEED_MIN, OBJ_INIT_SPEED_MAX)
                self.data.qvel[vx_addr] = v_speed * np.cos(v_angle)
                self.data.qvel[vy_addr] = v_speed * np.sin(v_angle)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
