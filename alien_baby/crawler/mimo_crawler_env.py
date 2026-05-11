"""
MimoCrawlerEnv — Gymnasium environment wrapping the MIMo-based infant body.

Body: MIMo mitten-hand body (stripped to 25 actuated DOF + passive wrists/fingers/toes).
Platform: 2m x 2m with guardrails — no fall termination.
Vision: optional stereo left_eye + right_eye cameras (40x40 RGB each).

Curriculum: strength_scale (0.0–1.0) multiplies ctrl before physics step.
  0.1 = newborn (barely any torque, body collapses under gravity)
  0.5 = ~6-month infant
  1.0 = full 18-month strength

Observation (proprio):
  root pos xy (2) + root quat (4) + root linvel (3) + root angvel (3)
  + 25 joint positions + 25 joint velocities
  + vestibular acc (3) + vestibular gyro (3)
  = 69 values

Observation (vision, optional):
  left_eye  40x40x3 flattened (4800) + right_eye 40x40x3 (4800) = 9600
  appended after proprio → 9663 total

Action: 25 continuous values in [-1, 1], scaled by strength_scale.
"""

import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = pathlib.Path(__file__).parent / "mimo_crawler.xml"

PLATFORM_TOP_Z  = 2.025
DEFAULT_MAX_STEPS   = 600
DEFAULT_SPAWN_RADIUS = (0.5, 1.2)
DEFAULT_SPAWN_CONE_DEG = 180

CONTACT_REWARD      = 200.0
STEP_COST           = -0.05
VELOCITY_BONUS_SCALE = 0.05
APPROACH_REWARD_SCALE = 2.0   # reward per metre of progress toward the ball

CAM_H = 32
CAM_W = 32

PROPRIO_DIM = 69  # 3 pos + 4 quat + 6 vel + 25 jpos + 25 jvel + 3 acc + 3 gyro
VISION_DIM  = CAM_H * CAM_W * 3 * 2  # stereo: left + right, each (H, W, 3)

# 25 actuated joints in actuator order
ACTUATED_JOINTS = [
    "robot:hip_bend1", "robot:hip_rot1", "robot:hip_lean1",
    "robot:chest_rot", "robot:chest_lean",
    "robot:head_swivel", "robot:head_tilt",
    "robot:right_shoulder_horizontal", "robot:right_shoulder_ad_ab", "robot:right_shoulder_rotation",
    "robot:right_elbow",
    "robot:left_shoulder_horizontal", "robot:left_shoulder_ad_ab", "robot:left_shoulder_rotation",
    "robot:left_elbow",
    "robot:right_hip1", "robot:right_hip2", "robot:right_hip3",
    "robot:right_knee", "robot:right_foot1",
    "robot:left_hip1", "robot:left_hip2", "robot:left_hip3",
    "robot:left_knee", "robot:left_foot1",
]

# Prone quaternion: body lying face-down, spine pointing forward (+Y).
# R maps: MIMo +X (face/belly) → world -Z (belly down)
#         MIMo +Z (spine/head) → world +Y (forward)
#         MIMo +Y (left side)  → world -X
# Derived quaternion [w, x, y, z] = [0.5, -0.5, 0.5, 0.5]
_PRONE_QUAT = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float64)
_PRONE_Z    = 2.15   # hip height above world origin when prone


class MimoCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vision=False, max_steps=DEFAULT_MAX_STEPS,
                 spawn_radius=None, spawn_cone_deg=DEFAULT_SPAWN_CONE_DEG,
                 strength_scale=1.0, n_substeps=4, render_mode=None,
                 approach_reward_scale=APPROACH_REWARD_SCALE,
                 velocity_bonus_scale=VELOCITY_BONUS_SCALE):
        super().__init__()
        self.vision = vision
        self.max_steps = max_steps
        self.spawn_radius = spawn_radius or DEFAULT_SPAWN_RADIUS
        self.spawn_cone_deg = spawn_cone_deg
        self.strength_scale = strength_scale
        self.approach_reward_scale = approach_reward_scale
        self.velocity_bonus_scale = velocity_bonus_scale
        self.n_substeps = n_substeps   # physics steps per policy step
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data  = mujoco.MjData(self.model)

        n_obs = PROPRIO_DIM + (VISION_DIM if vision else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Cache body/geom ids
        self._root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom"
        )
        # MIMo body geom ids (anything not world/platform/rails/target)
        self._mimo_body_ids = self._get_mimo_body_ids()

        # Cache joint qpos/dof addresses for fast obs
        self._jpos_adr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ] for j in ACTUATED_JOINTS
        ], dtype=np.int32)
        self._jvel_adr = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ] for j in ACTUATED_JOINTS
        ], dtype=np.int32)

        # Vestibular sensor indices
        self._vest_acc_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "vestibular_acc")
        self._vest_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "vestibular_gyro")

        if vision:
            self._cam_renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        else:
            self._cam_renderer = None

        self._render_renderer = None
        self._step = 0
        self._contact_rewarded = False

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Place MIMo prone (face down) on platform, spine pointing +Y
        root_qadr = self.model.jnt_qposadr[self._root_joint_id]
        root_dadr = self.model.jnt_dofadr[self._root_joint_id]
        self.data.qpos[root_qadr:root_qadr + 3] = [0.0, 0.0, _PRONE_Z]
        self.data.qpos[root_qadr + 3:root_qadr + 7] = _PRONE_QUAT
        # Small random joint noise so episodes differ
        for adr in self._jpos_adr:
            self.data.qpos[adr] += self.np_random.uniform(-0.05, 0.05)

        # Spawn ball in forward hemisphere
        angle_rad = np.deg2rad(self.spawn_cone_deg / 2)
        theta = self.np_random.uniform(-angle_rad, angle_rad)
        r     = self.np_random.uniform(*self.spawn_radius)
        bx, by = r * np.sin(theta), r * np.cos(theta)

        tgt_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt_qadr = self.model.jnt_qposadr[tgt_jid]
        self.data.qpos[tgt_qadr:tgt_qadr + 3] = [bx, by, PLATFORM_TOP_Z + 0.053]
        self.data.qpos[tgt_qadr + 3:tgt_qadr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._contact_rewarded = False
        self._prev_ball_dist = self._ball_dist()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        # Curriculum: scale torques by strength; advance n_substeps physics steps
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0) * self.strength_scale
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        obs = self._get_obs()

        # Horizontal speed of hip (root body)
        root_dadr = self.model.jnt_dofadr[self._root_joint_id]
        torso_speed = float(np.linalg.norm(
            self.data.qvel[root_dadr:root_dadr + 2]  # vx, vy
        ))

        contacted = self._check_ball_contact()

        curr_dist = self._ball_dist()
        approach  = self._prev_ball_dist - curr_dist   # positive = got closer
        self._prev_ball_dist = curr_dist

        reward = STEP_COST + self.velocity_bonus_scale * torso_speed + self.approach_reward_scale * approach
        terminated = False
        truncated  = self._step >= self.max_steps

        if contacted and not self._contact_rewarded:
            reward += CONTACT_REWARD
            self._contact_rewarded = True
            terminated = True

        return obs, reward, terminated, truncated, {
            "touched": self._contact_rewarded,
            "step": self._step,
            "strength_scale": self.strength_scale,
        }

    # ------------------------------------------------------------------
    def _ball_dist(self):
        """XY distance from root (hip) to ball."""
        root_qadr = self.model.jnt_qposadr[self._root_joint_id]
        tgt_jid   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt_qadr  = self.model.jnt_qposadr[tgt_jid]
        hip_xy  = self.data.qpos[root_qadr:root_qadr + 2]
        ball_xy = self.data.qpos[tgt_qadr:tgt_qadr + 2]
        return float(np.linalg.norm(hip_xy - ball_xy))

    # ------------------------------------------------------------------
    def _check_ball_contact(self):
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._target_geom_id or g2 == self._target_geom_id:
                other = g2 if g1 == self._target_geom_id else g1
                if other in self._mimo_body_ids:
                    return True
        return False

    def _get_mimo_body_ids(self):
        """All geom ids that belong to MIMo (not world/platform/rails/target)."""
        world_geoms = {"ground", "platform", "rail_N", "rail_S", "rail_E", "rail_W", "target_geom"}
        ids = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name not in world_geoms:
                ids.add(i)
        return ids

    # ------------------------------------------------------------------
    def _get_obs(self):
        root_qadr = self.model.jnt_qposadr[self._root_joint_id]
        root_dadr = self.model.jnt_dofadr[self._root_joint_id]

        root_pos  = self.data.qpos[root_qadr:root_qadr + 3].astype(np.float32)
        root_quat = self.data.qpos[root_qadr + 3:root_qadr + 7].astype(np.float32)
        root_vel  = self.data.qvel[root_dadr:root_dadr + 6].astype(np.float32)

        jpos = self.data.qpos[self._jpos_adr].astype(np.float32)
        jvel = self.data.qvel[self._jvel_adr].astype(np.float32)

        # Vestibular: acc + gyro (6 values)
        vest_adr_acc  = self.model.sensor_adr[self._vest_acc_id]
        vest_adr_gyro = self.model.sensor_adr[self._vest_gyro_id]
        vest_acc  = self.data.sensordata[vest_adr_acc:vest_adr_acc + 3].astype(np.float32)
        vest_gyro = self.data.sensordata[vest_adr_gyro:vest_adr_gyro + 3].astype(np.float32)

        proprio = np.concatenate([
            root_pos,   # 3
            root_quat,  # 4
            root_vel,   # 6
            jpos,       # 25
            jvel,       # 25
            vest_acc,   # 3
            vest_gyro,  # 3
        ])  # = 69... wait: 3+4+6+25+25+3+3 = 69, not 63. Let me fix PROPRIO_DIM.

        if not self.vision:
            return proprio

        self._cam_renderer.update_scene(self.data, camera="left_eye")
        left_img  = self._cam_renderer.render().copy().astype(np.float32) / 255.0
        self._cam_renderer.update_scene(self.data, camera="right_eye")
        right_img = self._cam_renderer.render().copy().astype(np.float32) / 255.0

        return np.concatenate([proprio, left_img.ravel(), right_img.ravel()])

    # ------------------------------------------------------------------
    def render(self):
        if self._render_renderer is None:
            self._render_renderer = mujoco.Renderer(self.model, 480, 480)
        self._render_renderer.update_scene(self.data, camera="ringside")
        return self._render_renderer.render().copy()

    def close(self):
        if self._cam_renderer is not None:
            self._cam_renderer.close()
        if self._render_renderer is not None:
            self._render_renderer.close()
