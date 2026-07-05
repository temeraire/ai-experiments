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

XML_PATH     = pathlib.Path(__file__).parent / "mimo_crawler.xml"
XML_PATH_POS = pathlib.Path(__file__).parent / "mimo_crawler_pos.xml"

PLATFORM_TOP_Z  = 2.025
DEFAULT_MAX_STEPS   = 600
DEFAULT_SPAWN_RADIUS = (0.5, 1.2)
DEFAULT_SPAWN_CONE_DEG = 180

CONTACT_REWARD      = 200.0
STEP_COST           = -0.05
VELOCITY_BONUS_SCALE = 0.05
APPROACH_REWARD_SCALE = 2.0   # reward per metre of progress toward the ball

# Crawl-ready default-pose presets (2026-07-02, minimal-change track). Keys are
# actuator indices in the widened body (mimo_crawler_pos_wide.xml); values are the
# joint angle (rad) the position-offset range is re-centred on. "arms_fwd" = the
# commando/belly-crawl pose the pose search found best (0.225 m axial hand-driven
# translation, belly-down): both shoulders swung forward (act 7,11) + elbows slightly
# bent (act 10,14). Everything else stays at its widened neutral.
CRAWL_POSES = {
    "arms_fwd": {7: 0.7, 11: 0.7, 10: -0.4, 14: -0.4},
}

CAM_H = 32
CAM_W = 32

PROPRIO_DIM = 69  # 3 pos + 4 quat + 6 vel + 25 jpos + 25 jvel + 3 acc + 3 gyro
VISION_DIM_STEREO = CAM_H * CAM_W * 3 * 2  # stereo: left + right, each (H, W, 3)
VISION_DIM_MONO   = CAM_H * CAM_W * 3      # single camera, (H, W, 3)
# Legacy alias used by older callers; assumes stereo for backward compat.
VISION_DIM = VISION_DIM_STEREO

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
                 spawn_cone_min_deg=0.0,
                 strength_scale=1.0, n_substeps=4, render_mode=None,
                 approach_reward_scale=APPROACH_REWARD_SCALE,
                 velocity_bonus_scale=VELOCITY_BONUS_SCALE,
                 fixed_ball_positions=None,
                 random_start_orientation=False,
                 memory_obs=False,
                 stereo=True,
                 action_mode="torque",
                 step_cost=STEP_COST,
                 xml_path=None,
                 crawl_pose=None,
                 terminate_tilt_deg=None,
                 tip_penalty=0.0,
                 tilt_cost=0.0,
                 tilt_cost_deg=30.0,
                 prism_offset_deg=0.0,
                 target_obs=False):
        super().__init__()
        self.vision = vision
        self.max_steps = max_steps
        self.spawn_radius = spawn_radius or DEFAULT_SPAWN_RADIUS
        self.spawn_cone_deg = spawn_cone_deg
        # spawn_cone_min_deg: if >0, exclude the near-forward wedge so |bearing| lies in
        # [min/2, max/2] with random sign — a LATERAL-BAND spawn. Makes forward-crawl fail
        # and the ball's direction behaviorally necessary (defeats the +/-22 forward-crawl
        # confound). Default 0.0 keeps every prior run bit-identical.
        self.spawn_cone_min_deg = float(spawn_cone_min_deg)
        self.strength_scale = strength_scale
        self.approach_reward_scale = approach_reward_scale
        self.velocity_bonus_scale = velocity_bonus_scale
        # Per-step penalty ("cost of existing"). Default STEP_COST keeps every
        # prior run bit-identical; set to 0.0 to stop punishing existence so the
        # do-nothing optimum is no longer the smart move (see CLAUDE.md
        # "Realm of possibility").
        self.step_cost = float(step_cost)
        self.n_substeps = n_substeps   # physics steps per policy step
        self.render_mode = render_mode
        # Phase C: fixed ball positions + variable starting orientation
        # fixed_ball_positions: None (random spawn), or list of 1 or 2 (x, y) tuples
        # random_start_orientation: if True, prone quat rotated by random Z-angle
        # memory_obs: if True, append 2 binary flags (touched_ball1, touched_ball2) to proprio.
        #   Lets a stateless policy condition on its own past contacts.
        # stereo: if True (default), use both eye cameras and emit 2× pixel obs.
        #   if False, use only left_eye (mono vision). The visual eye geoms in the
        #   XML still show two eyes — only the camera input is mono.
        self.fixed_ball_positions = fixed_ball_positions
        self.random_start_orientation = random_start_orientation
        self.memory_obs = memory_obs
        self.stereo = stereo
        # Crawl-ready default pose (2026-07-02, minimal-change track). crawl_pose is a
        # dict {actuator_index: center_angle_rad}: the position-offset ranges for those
        # actuators are re-centred on the angle (offsets around a locomotion-adjacent
        # pose, per "A Walk in the Park") and the joints start there. terminate_tilt_deg
        # ends the episode (with tip_penalty) if the body's dorsal axis tilts past that
        # many degrees from world-up — kills the tip-over/roll exploit. All default
        # off/None so prior runs are unchanged.
        self.crawl_pose = crawl_pose or {}
        self.terminate_tilt_deg = terminate_tilt_deg
        self.tip_penalty = float(tip_penalty)
        # Posture term: a small GRADED cost as the body's tilt rises above tilt_cost_deg
        # toward the tip limit — discourages the aggressive near-tips a fast searcher makes,
        # without penalizing normal prone motion (tilt~0). Default 0 = existing runs unchanged.
        self.tilt_cost = float(tilt_cost)
        self.tilt_cost_deg = float(tilt_cost_deg)
        # target_obs: append the ball-1 position in the BODY frame (3 numbers) to the
        # observation. This is the directional signal a go-to-target crawler needs —
        # without it the approach reward has no gradient the policy can act on (the
        # crawl_minimal_400k run crawled but only made *random-walk* contact). This is
        # privileged target info; vision is meant to eventually supply it.
        self.target_obs = target_obs
        # Phase XVI R49: action_mode selects torque vs position-offset control.
        # "torque": original ctrl=action*strength_scale (no change from prior phases).
        # "position_offset": action∈[-1,1] is mapped linearly onto each actuator's
        #   ctrlrange, so ctrl = lo + (action+1)/2*(hi-lo). Requires mimo_crawler_pos.xml.
        self.action_mode = action_mode

        if xml_path is not None:
            _xml = pathlib.Path(xml_path)
        elif action_mode == "position_offset":
            _xml = XML_PATH_POS
        else:
            _xml = XML_PATH

        self.model = mujoco.MjModel.from_xml_path(str(_xml))
        self.data  = mujoco.MjData(self.model)

        # Cache per-actuator ctrlrange for position_offset mapping
        if action_mode == "position_offset":
            self._act_lo  = self.model.actuator_ctrlrange[:, 0].copy()
            self._act_hi  = self.model.actuator_ctrlrange[:, 1].copy()
            # Crawl-ready pose: re-centre selected actuators' ranges on the pose angle,
            # keeping the half-width (clamped to the joint's hard limit).
            if self.crawl_pose:
                half = (self._act_hi - self._act_lo) / 2.0
                for i, c in self.crawl_pose.items():
                    jid = self.model.actuator_trnid[i, 0]
                    if self.model.jnt_limited[jid]:
                        jlo, jhi = self.model.jnt_range[jid]
                        h = min(half[i], c - jlo, jhi - c)
                    else:
                        h = half[i]
                    h = max(float(h), 0.05)
                    self._act_lo[i] = c - h
                    self._act_hi[i] = c + h
            self._act_mid = (self._act_lo + self._act_hi) / 2.0
            self._act_half = (self._act_hi - self._act_lo) / 2.0

        memory_dim = 2 if memory_obs else 0
        if vision:
            vis_dim = VISION_DIM_STEREO if stereo else VISION_DIM_MONO
        else:
            vis_dim = 0
        target_dim = 3 if target_obs else 0
        n_obs = PROPRIO_DIM + memory_dim + target_dim + vis_dim
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
        self._root_body_id = self.model.jnt_bodyid[self._root_joint_id]
        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom"
        )
        # Second ball (Phase C): present in XML but may be hidden off-platform
        # when only one ball is active. We always look up both ids so contact
        # checking works regardless of mode.
        self._target2_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target2_geom"
        )
        # MIMo body geom ids (anything not world/platform/rails/targets)
        self._mimo_body_ids = self._get_mimo_body_ids()

        # Prism / ghost setup: vision sees a GHOST (mocap, non-physical) displaced from the
        # REAL solid target by prism_offset_deg (bearing rotation). The real target is hidden
        # from the head-cam (moved to geom group 3, disabled in the head-cam scene option) but
        # stays visible in debug renders and stays solid+rewarding. Reward is real contact only.
        self.prism_offset_deg = float(prism_offset_deg)
        self._headcam_vopt = None
        self._ghost_mocap_id = -1
        if self.prism_offset_deg != 0.0:
            self.model.geom_group[self._target_geom_id] = 3
            self._headcam_vopt = mujoco.MjvOption()
            self._headcam_vopt.geomgroup[3] = 0   # real ball invisible to the agent's cameras
            gbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ghost")
            self._ghost_mocap_id = int(self.model.body_mocapid[gbody])

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

        # Place MIMo prone (face down) on platform.
        # Default _PRONE_QUAT puts spine along world +Y. If random_start_orientation,
        # rotate that quat around world Z by a random angle so the creature faces a
        # random world direction at spawn.
        root_qadr = self.model.jnt_qposadr[self._root_joint_id]
        root_dadr = self.model.jnt_dofadr[self._root_joint_id]
        self.data.qpos[root_qadr:root_qadr + 3] = [0.0, 0.0, _PRONE_Z]
        if self.random_start_orientation:
            yaw = self.np_random.uniform(-np.pi, np.pi)
            qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # rot about +Z
            # Quaternion multiplication qz * _PRONE_QUAT
            w0, x0, y0, z0 = qz
            w1, x1, y1, z1 = _PRONE_QUAT
            q = np.array([
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ])
            self.data.qpos[root_qadr + 3:root_qadr + 7] = q
        else:
            self.data.qpos[root_qadr + 3:root_qadr + 7] = _PRONE_QUAT
        # Small random joint noise so episodes differ
        for adr in self._jpos_adr:
            self.data.qpos[adr] += self.np_random.uniform(-0.05, 0.05)
        # Crawl-ready pose: start the selected joints at their re-centred angle so the
        # body begins in the locomotion-adjacent pose (not flat-neutral).
        for i, c in self.crawl_pose.items():
            jid = self.model.actuator_trnid[i, 0]
            self.data.qpos[self.model.jnt_qposadr[jid]] = c

        # Place balls. If fixed_ball_positions is set, override random spawn.
        # If only 1 fixed position, ball2 is parked far off-platform.
        # If 2 fixed positions, both balls are active.
        # If None, default: ball1 random in spawn cone, ball2 parked off-platform.
        tgt1_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt1_qadr = self.model.jnt_qposadr[tgt1_jid]
        tgt2_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_free")
        tgt2_qadr = self.model.jnt_qposadr[tgt2_jid]

        if self.fixed_ball_positions:
            b1x, b1y = self.fixed_ball_positions[0]
            self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [b1x, b1y, PLATFORM_TOP_Z + 0.053]
            self.data.qpos[tgt1_qadr + 3:tgt1_qadr + 7] = [1, 0, 0, 0]
            if len(self.fixed_ball_positions) >= 2:
                b2x, b2y = self.fixed_ball_positions[1]
                self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [b2x, b2y, PLATFORM_TOP_Z + 0.053]
                self._ball2_active = True
            else:
                # Park ball2 well outside platform
                self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [10.0, 10.0, 0.0]
                self._ball2_active = False
            self.data.qpos[tgt2_qadr + 3:tgt2_qadr + 7] = [1, 0, 0, 0]
        else:
            # Original random spawn behavior for ball 1
            angle_rad = np.deg2rad(self.spawn_cone_deg / 2)
            if self.spawn_cone_min_deg > 0.0:
                # Lateral-band: |theta| in [min/2, max/2], random sign.
                lo = np.deg2rad(self.spawn_cone_min_deg / 2)
                mag = self.np_random.uniform(lo, angle_rad)
                theta = mag if self.np_random.uniform() < 0.5 else -mag
            else:
                theta = self.np_random.uniform(-angle_rad, angle_rad)
            r     = self.np_random.uniform(*self.spawn_radius)
            bx, by = r * np.sin(theta), r * np.cos(theta)
            self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [bx, by, PLATFORM_TOP_Z + 0.053]
            self.data.qpos[tgt1_qadr + 3:tgt1_qadr + 7] = [1, 0, 0, 0]
            self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [10.0, 10.0, 0.0]
            self.data.qpos[tgt2_qadr + 3:tgt2_qadr + 7] = [1, 0, 0, 0]
            self._ball2_active = False

        # Prism: place the GHOST at the real ball's bearing rotated by prism_offset_deg
        # (same radius, same height). Vision sees the ghost; the real ball is hidden + solid.
        if self.prism_offset_deg != 0.0 and self._ghost_mocap_id >= 0:
            bx, by, bz = self.data.qpos[tgt1_qadr:tgt1_qadr + 3]
            r = float(np.hypot(bx, by))
            th = np.arctan2(bx, by) + np.deg2rad(self.prism_offset_deg)
            self.data.mocap_pos[self._ghost_mocap_id] = [r * np.sin(th), r * np.cos(th), bz]

        mujoco.mj_forward(self.model, self.data)
        # Crawl track: let the body drop the few cm onto the platform and settle into
        # the crawl-ready pose BEFORE the episode starts, holding the neutral targets.
        # Otherwise the settling rotation would be misread as a tip-over. Gated on the
        # crawl config so all other runs' reset is bit-identical.
        if self.crawl_pose or self.terminate_tilt_deg is not None:
            for _ in range(40):
                self.data.ctrl[:] = self._act_mid
                mujoco.mj_step(self.model, self.data)
        # Reference dorsal(up) axis in body frame, for tip detection: the body-frame
        # vector that points to world-up in the settled starting pose.
        R0 = self.data.xmat[self._root_body_id].reshape(3, 3)
        self._up_local = R0.T @ np.array([0.0, 0.0, 1.0])
        self._step = 0
        self._contact_rewarded = False
        self._ball1_touched = False
        self._ball2_touched = False
        self._prev_ball_dist = self._ball_dist()
        return self._get_obs(), {}

    def _up_tilt_deg(self):
        """Degrees the body's dorsal axis has tilted from world-up (tip-over detector)."""
        R = self.data.xmat[self._root_body_id].reshape(3, 3)
        up_world = R @ self._up_local
        c = float(np.clip(up_world[2] / (np.linalg.norm(up_world) + 1e-9), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    # ------------------------------------------------------------------
    def step(self, action):
        # Apply action based on mode:
        #   torque: ctrl = clip(action, -1, 1) * strength_scale  (legacy)
        #   position_offset: ctrl = mid + clip(action, -1, 1) * half_range
        #     (strength_scale intentionally unused in position mode — kp/kv
        #      in the XML set stiffness; the action selects a target position)
        a = np.clip(action, -1.0, 1.0)
        if self.action_mode == "position_offset":
            self.data.ctrl[:] = self._act_mid + a * self._act_half
        else:
            self.data.ctrl[:] = a * self.strength_scale
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        obs = self._get_obs()

        # Horizontal speed of hip (root body)
        root_dadr = self.model.jnt_dofadr[self._root_joint_id]
        torso_speed = float(np.linalg.norm(
            self.data.qvel[root_dadr:root_dadr + 2]  # vx, vy
        ))

        ball1_hit, ball2_hit = self._check_ball_contact()

        curr_dist = self._ball_dist()
        approach  = self._prev_ball_dist - curr_dist   # positive = got closer
        self._prev_ball_dist = curr_dist

        reward = self.step_cost + self.velocity_bonus_scale * torso_speed + self.approach_reward_scale * approach
        terminated = False
        truncated  = self._step >= self.max_steps

        # Per-ball first-touch rewards
        if ball1_hit and not self._ball1_touched:
            reward += CONTACT_REWARD
            self._ball1_touched = True
        if self._ball2_active and ball2_hit and not self._ball2_touched:
            reward += CONTACT_REWARD
            self._ball2_touched = True

        # Termination rule:
        #   - Single-ball mode (ball2 inactive): terminate on first ball1 touch (legacy behavior)
        #   - Two-ball mode (ball2 active): terminate only when BOTH balls touched
        if self._ball2_active:
            if self._ball1_touched and self._ball2_touched:
                terminated = True
        else:
            if self._ball1_touched:
                terminated = True

        # Keep the old `_contact_rewarded` flag for backward compat
        self._contact_rewarded = self._ball1_touched

        # Tip-over termination: if the body rolls/tips past the allowed dorsal tilt,
        # end the episode with a penalty. Removes the "roll onto side and slide" and
        # "flip over" exploits that game a raw-displacement reward (see DeepMimic:
        # early termination eliminates the on-the-ground local optima).
        if self.terminate_tilt_deg is not None and not terminated:
            tilt = self._up_tilt_deg()
            if self.tilt_cost > 0.0 and tilt > self.tilt_cost_deg:
                reward -= self.tilt_cost * (tilt - self.tilt_cost_deg) / 90.0
            if tilt > self.terminate_tilt_deg:
                reward += self.tip_penalty
                terminated = True

        # Liveness signal for the "realm of possibility" gate: how much the
        # creature is actually doing = hip translation speed + mean actuated
        # joint speed. A frozen body -> ~0; crawling/flailing -> clearly > 0.
        body_motion = torso_speed + float(np.abs(self.data.qvel[self._jvel_adr]).mean())

        return obs, reward, terminated, truncated, {
            "touched": self._ball1_touched,
            "touched_ball1": self._ball1_touched,
            "touched_ball2": self._ball2_touched,
            "step": self._step,
            "strength_scale": self.strength_scale,
            "body_motion": body_motion,
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
        """Returns (ball1_hit, ball2_hit) bools for this physics step."""
        ball1_hit = False
        ball2_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            for target_id, flag_name in [(self._target_geom_id, "b1"),
                                          (self._target2_geom_id, "b2")]:
                if g1 == target_id or g2 == target_id:
                    other = g2 if g1 == target_id else g1
                    if other in self._mimo_body_ids:
                        if flag_name == "b1":
                            ball1_hit = True
                        else:
                            ball2_hit = True
        return ball1_hit, ball2_hit

    def _get_mimo_body_ids(self):
        """All geom ids that belong to MIMo (not world/platform/rails/targets)."""
        world_geoms = {"ground", "platform", "rail_N", "rail_S", "rail_E", "rail_W",
                       "target_geom", "target2_geom"}
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
        ])  # = 69

        if self.memory_obs:
            mem = np.array([
                1.0 if getattr(self, "_ball1_touched", False) else 0.0,
                1.0 if getattr(self, "_ball2_touched", False) else 0.0,
            ], dtype=np.float32)
            proprio = np.concatenate([proprio, mem])

        if self.target_obs:
            # Ball-1 position relative to the body, rotated into the body frame so the
            # signal is heading-invariant (the policy can read "ball is forward-left").
            tgt_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
            tgt_qadr = self.model.jnt_qposadr[tgt_jid]
            ball_w = self.data.qpos[tgt_qadr:tgt_qadr + 3]
            R = self.data.xmat[self._root_body_id].reshape(3, 3)
            ball_ego = (R.T @ (ball_w - root_pos)).astype(np.float32)
            proprio = np.concatenate([proprio, ball_ego])

        if not self.vision:
            return proprio

        _vopt = self._headcam_vopt   # hides the real ball (group 3) in prism mode; None otherwise
        if _vopt is not None:
            self._cam_renderer.update_scene(self.data, camera="left_eye", scene_option=_vopt)
        else:
            self._cam_renderer.update_scene(self.data, camera="left_eye")
        left_img = self._cam_renderer.render().copy().astype(np.float32) / 255.0

        if not self.stereo:
            return np.concatenate([proprio, left_img.ravel()])

        if _vopt is not None:
            self._cam_renderer.update_scene(self.data, camera="right_eye", scene_option=_vopt)
        else:
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
