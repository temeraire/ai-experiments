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
XML_PATH_V9 = str(pathlib.Path(__file__).parent / "platform_creature_v9.xml")

# v9: ball starts with this speed along a random horizontal direction.
# Rolling friction decays it over time — half-life ~5 sim-seconds.
V9_BALL_INITIAL_SPEED = 0.0  # v9.0-static: isolate steering from prediction
# If the ball drops this far below the platform top, it's "lost" and the
# episode ends with no touch credit.
V9_BALL_LOST_DZ = 0.3
# Velocity decay half-life for the ball's linear motion. Longer = harder
# predictions (fast ball, less time to intercept); shorter = easier (ball
# slows before reaching edge). 5s gives a natural "race against time" window.
V9_BALL_HALF_LIFE_SEC = 5.0

CAM_HEIGHT = 32
CAM_WIDTH = 32

PLATFORM_HALF = 1.0
PLATFORM_Z = 2.0
FALL_PENALTY = -500.0
HUNGER_PENALTY = -0.05
EDGE_WARN_DIST = 0.3
EDGE_WARN_PENALTY = -0.5
CONTACT_REWARD = 200.0
ATTRACT_SCALE = 0.3
# "Smelling the food" zone. At d=0 reward is ATTRACT_SCALE (0.3); at
# d=ATTRACT_MAX_DIST (0.25m) it decays to 0; beyond that it is 0. Previously
# 0.15m, which created a too-thin contact cliff that a 32×32 vision + frozen
# proprio couldn't bridge from a cold start (creature hovered at ~0.35m with
# nothing to pull it closer). 0.25m widens the proximity gradient without
# reintroducing the mid-platform bribery of the original 2.0m setting.
ATTRACT_MAX_DIST = 0.25

# Potential-based reward shaping (Ng, Harada & Russell 1999): Φ(s) = -α·d
# where d is body-ball distance. Per-step shaping = γ·Φ(s') - Φ(s), which
# telescopes over an episode to a constant that does not change the optimal
# policy, but provides a dense approach-gradient for SAC during cold-start.
# On terminal transitions Φ(s_terminal) = 0 by convention. Enable via
# constructor kwarg `pbrs_alpha` (0.0 disables, 0.3 = default for experiments).
PBRS_GAMMA = 0.99  # matches SAC's default discount

# Standard MuJoCo locomotion practice: discourage high-frequency torque
# oscillation by paying a per-step cost proportional to action magnitude.
# Scale chosen so worst-case per-step cost (||a||² = 8 with full ±1 across
# 8 dims) ≈ 0.008 — smaller than HUNGER_PENALTY but large enough to bias
# toward low-frequency periodic solutions.
CTRL_COST_SCALE = 0.001

# Small reward for torso speed regardless of direction. Breaks the stillness
# local optimum: a stationary creature pays only hunger; a moving creature
# recovers some of that cost just by being in motion. Scale chosen so
# worst-case (max observed speed ~0.21 m/s) earns ~0.004/step — small enough
# that the creature cannot satisfy hunger by spinning in place, large enough
# to make "move at all" strictly better than "stand still."
VELOCITY_BONUS_SCALE = 0.05

# Tilt-based health termination: cos(60°) = 0.5. Terminate when torso
# z-axis tilts more than 60° from world up. The v8 body rarely actually
# falls (`0/20` in PROJECT_STATUS), but degenerate tilted-but-not-fallen
# states burn the rest of the truncation window with useless transitions.
# Cutting them short keeps the replay buffer clean.
TILT_COS_THRESHOLD = 0.5


class PlatformCreatureEnv(gym.Env):
    """Creature on a platform. Fall off = death. Touch target = reward."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vision=False, render_mode=None, stage=1, v9=False,
                 target_radius_override=None, pbrs_alpha=0.0,
                 closure_bonus_scale=0.0,
                 hand_only_contact=False, spawn_cone_deg=None,
                 xor_color_random=False, max_steps_override=None):
        super().__init__()
        self.vision = vision
        self.render_mode = render_mode
        self.stage = stage
        self.v9 = v9
        # Idea A: only hand-target contacts count as "touched". Default off
        # preserves existing behavior (any creature geom counts).
        self._hand_only_contact = bool(hand_only_contact)
        # Idea A: restrict ball spawn to a body-frame cone of width
        # `spawn_cone_deg` centered straight ahead. None = no override (use
        # whatever the stage/v9 default is). 180.0 = full front hemisphere
        # (matches v9's default). 90.0 = ±45° forward cone — forces the
        # creature to *aim* the hand to make contact.
        self._spawn_cone_deg = (None if spawn_cone_deg is None
                                else float(spawn_cone_deg))
        # Idea B: 50/50 red-good / blue-bad. Blue contact = -CONTACT_REWARD
        # and terminate. Red contact = +CONTACT_REWARD and terminate. Pure
        # visual category — proprio cannot distinguish.
        self._xor_color_random = bool(xor_color_random)
        self._is_decoy = False
        self._target_geom_rgba_addr = None  # set later if XOR enabled

        # v9 = moving target: ball has free joint, initial velocity, can roll
        # off the platform. Taylor-style pressure to make vision load-bearing.
        xml_path = XML_PATH_V9 if v9 else XML_PATH

        # Stage 0 = locomotion-only scaffold: huge platform (no edge to fall
        # off), no fall/edge penalties, target close and in front of the head
        # cam. Stage 1 = full v8 survival env. Scaffolds removed as AB
        # graduates.
        if stage == 0:
            xml_text = pathlib.Path(xml_path).read_text()
            xml_text = xml_text.replace(
                'size="1.0 1.0 0.025"', 'size="10 10 0.025"'
            )
            self.model = mujoco.MjModel.from_xml_string(xml_text)
            self._fall_penalty = 0.0
            self._edge_warn_dist = 0.0
            self._edge_warn_penalty = 0.0
            # Min 0.4 keeps the ball outside the rest-reach of either hand
            # (right hand at rest sits ~0.32 from torso center). AB has to
            # move the body to touch — no freebie contact at spawn.
            self._target_radius_lo = 0.4
            self._target_radius_hi = 0.55
            self._use_forward_cone = False
            # 400 steps = 20 sim-seconds. Longer search window per episode so
            # stochastic exploration has time to stumble into touches.
            self._max_steps = 400
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self._fall_penalty = FALL_PENALTY
            self._edge_warn_dist = EDGE_WARN_DIST
            self._edge_warn_penalty = EDGE_WARN_PENALTY
            self._target_radius_lo = 0.2
            self._target_radius_hi = 0.7
            self._use_forward_cone = True
            self._max_steps = 300

        # v9 bootstrap: fix spawn radius at 0.5m so ball trajectory is
        # predictable relative to AB's kinematics. Gives AB ~3-4 sec of slack
        # to correct errors before ball reaches platform edge.
        if v9:
            self._target_radius_lo = 0.5
            self._target_radius_hi = 0.5

        # Curriculum override — e.g. (0.25, 0.25) for warm-start bootstrapping
        # under the reshaped (last-mile) reward. Applied AFTER stage/v9
        # defaults so it clobbers whichever default the other flags chose.
        if target_radius_override is not None:
            lo, hi = target_radius_override
            self._target_radius_lo = float(lo)
            self._target_radius_hi = float(hi)

        if max_steps_override is not None:
            self._max_steps = int(max_steps_override)

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

        # All creature-body geoms (torso box, wheels, arms, hands). A contact
        # between any of these and the target counts as "AB found the ball" —
        # not just hand-target contact. Walking into the ball is still finding
        # it.
        self._creature_geoms = self._collect_creature_geoms()

        # Target joint addresses differ between v8 (two slide joints) and v9
        # (one free joint — 7 qpos: xyz + quaternion; 6 qvel: linear + angular).
        if self.v9:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
            self._target_qpos_addr = self.model.jnt_qposadr[jid]
            self._target_qvel_addr = self.model.jnt_dofadr[jid]
            self._target_x_addr = None
            self._target_y_addr = None
        else:
            self._target_x_addr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
            ]
            self._target_y_addr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
            ]
            self._target_qpos_addr = None
            self._target_qvel_addr = None

        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        # Cache the original target rgba so we can restore it across resets
        # when xor_color_random toggles colors.
        self._target_rgba_default = self.model.geom_rgba[self._target_geom].copy()

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
        self._prev_dist = None
        self._touched = False
        self._hunger_sum = 0.0
        self._attract_sum = 0.0
        self._dist_sum = 0.0
        self._spawn_angle = 0.0  # body-frame angle at reset; 0=right, π/2=fwd, π=left
        self._pbrs_alpha = float(pbrs_alpha)
        self._prev_body_dist = 0.0
        self._shaping_sum = 0.0
        # Closure bonus (Stage 1c): direct, Markovian reward for per-step
        # body→ball distance reduction, paid as `scale * max(0, Δd)`. Replaces
        # the abstract v_body_forward bonus — the creature no longer has to
        # discover that forward-velocity correlates with approach; it is paid
        # directly for any step that closes distance. Clipping at 0 means
        # receding is not penalized (just not rewarded), matching hunger as
        # the push toward staying near the ball.
        self._closure_bonus_scale = float(closure_bonus_scale)
        self._closure_sum = 0.0
        self._ctrl_cost_sum = 0.0
        self._velocity_bonus_sum = 0.0

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

    def _collect_creature_geoms(self):
        creature_geoms = set()
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            while bid != 0:
                if bid == self._torso_id:
                    creature_geoms.add(gid)
                    break
                bid = self.model.body_parentid[bid]
        return creature_geoms

    def _check_target_contact(self):
        # When hand_only_contact is enabled, only left/right hand geoms count
        # as creature-side contact. Forces aiming — torso lumbering over the
        # ball no longer pays. The set is computed once.
        if self._hand_only_contact:
            allowed = {self._left_hand_geom, self._right_hand_geom}
        else:
            allowed = self._creature_geoms
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in allowed and g2 == self._target_geom) or \
               (g2 in allowed and g1 == self._target_geom):
                return True
        return False

    def _has_fallen(self):
        torso_z = self.data.xpos[self._torso_id][2]
        return torso_z < PLATFORM_Z - 0.2

    def _is_unhealthy_tilt(self):
        """True if torso z-axis is tilted > 60° from world up. Quaternion
        (w,x,y,z) at qpos[3:7]; body-z·world-z = 1 - 2·(x²+y²)."""
        x, y = self.data.qpos[4], self.data.qpos[5]
        cos_tilt = 1.0 - 2.0 * (x * x + y * y)
        return cos_tilt < TILT_COS_THRESHOLD

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

    def _body_ball_dist(self):
        """Horizontal (xy) distance from torso center to target. Used as the
        Ng 1999 potential Φ(s) = -α·d for approach-gradient shaping."""
        body_xy = self.data.xpos[self._torso_id][:2]
        target_xy = self.data.xpos[self._target_id][:2]
        return float(np.linalg.norm(body_xy - target_xy))

    def _pbrs_shaping(self, curr_body_dist, is_terminal):
        """Compute Φ-based reward shaping for this transition.

        Non-terminal: γ·Φ(s') - Φ(s) = α·(prev_d - γ·curr_d)
        Terminal:     0      - Φ(s) = α·prev_d   (convention: Φ(terminal)=0)

        Returns 0 if PBRS is disabled (alpha = 0)."""
        if self._pbrs_alpha <= 0.0:
            return 0.0
        if is_terminal:
            return self._pbrs_alpha * self._prev_body_dist
        return self._pbrs_alpha * (self._prev_body_dist - PBRS_GAMMA * curr_body_dist)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._touched = False
        self._hunger_sum = 0.0
        self._attract_sum = 0.0
        self._dist_sum = 0.0
        self._shaping_sum = 0.0
        self._closure_sum = 0.0
        self._ctrl_cost_sum = 0.0
        self._velocity_bonus_sum = 0.0
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        # Randomize target position. In stage 1 we exclude a ±25° cone in
        # front of AB's head cam so finding the target requires turning the
        # head (gaze pressure via FOV). In stage 0 that scaffold is off —
        # target spawns anywhere in the close radius, often in front, so
        # locomotion can be learned without also needing to look.
        if self.np_random is not None:
            if self.v9:
                # v9.2-mixed: full front hemisphere. ~1/3 center (easy,
                # head alignment suffices), ~1/3 medium, ~1/3 far-lateral
                # (forward paddle insufficient, arm-steering required).
                # Keeps both head and arm vision-mappings useful, prevents
                # the over-specialization collapse seen in v9.1-pure-lateral.
                if self._spawn_cone_deg is not None:
                    # Idea A: narrow body-frame spawn cone, centered straight
                    # ahead (body-frame angle π/2). spawn_cone_deg = full
                    # cone width; e.g. 90° = ±45° around fwd direction.
                    half = np.radians(self._spawn_cone_deg) / 2.0
                    angle = self.np_random.uniform(np.pi / 2 - half,
                                                    np.pi / 2 + half)
                else:
                    angle = self.np_random.uniform(0, np.pi)
            elif self._spawn_cone_deg is not None:
                # Idea A: spawn ball within ±cone/2 of straight ahead. Forces
                # the creature to *aim* — vision becomes instrumental for
                # hand placement when paired with hand_only_contact.
                half = np.radians(self._spawn_cone_deg) / 2.0
                angle = self.np_random.uniform(np.pi / 2 - half,
                                                np.pi / 2 + half)
            elif self._use_forward_cone:
                forward_cone = np.radians(25)  # matches head_cam fovy
                while True:
                    angle = self.np_random.uniform(0, 2 * np.pi)
                    dev = abs(angle - np.pi / 2)
                    dev = min(dev, 2 * np.pi - dev)
                    if dev > forward_cone:
                        break
            else:
                angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(
                self._target_radius_lo, self._target_radius_hi
            )
            # Record body-frame spawn angle for downstream hemisphere analysis
            # (left vs right diagnostic). 0 = purely right, π/2 = straight
            # ahead, π = purely left. Captured before world-frame rotation so
            # the value is independent of the random yaw applied to the torso.
            self._spawn_angle = float(angle)
            if self.v9:
                # Randomize torso yaw from {0, +π/2, π, -π/2} so "arm action →
                # world rotation direction" isn't a reliable default. Vision
                # becomes the only signal that tells the policy which arm to
                # push. Root free-joint quaternion at qpos[3:7] = (w, x, y, z).
                yaw_choices = [0.0, np.pi / 2, np.pi, -np.pi / 2]
                body_yaw = float(self.np_random.choice(yaw_choices))
                self.data.qpos[3] = np.cos(body_yaw / 2)
                self.data.qpos[4] = 0.0
                self.data.qpos[5] = 0.0
                self.data.qpos[6] = np.sin(body_yaw / 2)

                # Target in AB's body-frame front hemisphere, rotated to world.
                # Keeps difficulty relative to AB the same as before; just
                # decouples body-frame actions from world-frame rotation.
                x_body = radius * np.cos(angle)
                y_body = radius * np.sin(angle)
                cos_y, sin_y = np.cos(body_yaw), np.sin(body_yaw)
                x_world = cos_y * x_body - sin_y * y_body
                y_world = sin_y * x_body + cos_y * y_body

                # Free-joint target: set world-space xy, sit on platform top.
                a = self._target_qpos_addr
                self.data.qpos[a + 0] = x_world
                self.data.qpos[a + 1] = y_world
                # Spawn with ball bottom exactly at platform top (no drop, no
                # bounce-and-lose-velocity). ball_center_z = platform_top + r.
                self.data.qpos[a + 2] = PLATFORM_Z + 0.025 + 0.04
                self.data.qpos[a + 3] = 1.0  # quat w
                self.data.qpos[a + 4 : a + 7] = 0.0  # quat xyz

                # Random horizontal initial velocity; zero vertical + angular.
                v_angle = self.np_random.uniform(0, 2 * np.pi)
                v = self._target_qvel_addr
                self.data.qvel[v + 0] = V9_BALL_INITIAL_SPEED * np.cos(v_angle)
                self.data.qvel[v + 1] = V9_BALL_INITIAL_SPEED * np.sin(v_angle)
                self.data.qvel[v + 2 : v + 6] = 0.0
            else:
                self.data.qpos[self._target_x_addr] = radius * np.cos(angle)
                self.data.qpos[self._target_y_addr] = radius * np.sin(angle)

        # Idea B: 50/50 red-good / blue-bad. Color is the only signal
        # discriminating the two; proprio cannot tell. Vision must learn it
        # to avoid the -CONTACT_REWARD penalty on blue contact.
        if self._xor_color_random and self.np_random is not None:
            self._is_decoy = bool(self.np_random.random() < 0.5)
            if self._is_decoy:
                # Distinct blue. Alpha matches default (1.0).
                self.model.geom_rgba[self._target_geom] = (
                    np.array([0.15, 0.30, 0.90, 1.0], dtype=np.float32)
                )
            else:
                self.model.geom_rgba[self._target_geom] = self._target_rgba_default
        else:
            self._is_decoy = False
            # Restore default in case prior episode set it to blue.
            self.model.geom_rgba[self._target_geom] = self._target_rgba_default

        mujoco.mj_forward(self.model, self.data)
        self._prev_dist = self._nearest_hand_dist()
        self._prev_body_dist = self._body_ball_dist()
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # Apply action
        clipped_action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = clipped_action

        # Step physics (5 substeps for stability)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # v9: apply linear velocity decay to ball only (not angular — rolling
        # physics keeps ω coupled to v at contact, so multiplying ω too would
        # double-count decay). One env step = 5 * 0.01s = 0.05s of sim time.
        if self.v9:
            decay = np.exp(-np.log(2) / V9_BALL_HALF_LIFE_SEC * 0.05)
            v_addr = self._target_qvel_addr
            self.data.qvel[v_addr + 0] *= decay
            self.data.qvel[v_addr + 1] *= decay

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False

        curr_body_dist = self._body_ball_dist()

        # Check if fallen off platform
        if self._has_fallen():
            reward = self._fall_penalty
            shaping = self._pbrs_shaping(curr_body_dist, is_terminal=True)
            reward += shaping
            self._shaping_sum += shaping
            terminated = True
            self._prev_body_dist = curr_body_dist
            return obs, reward, terminated, truncated, self._info(
                fell=True, touched=False
            )

        # Tilt-based health termination: cut degenerate tilted episodes short
        # so they don't fill the replay buffer with useless transitions.
        if self._is_unhealthy_tilt():
            shaping = self._pbrs_shaping(curr_body_dist, is_terminal=True)
            reward += shaping
            self._shaping_sum += shaping
            terminated = True
            self._prev_body_dist = curr_body_dist
            return obs, reward, terminated, truncated, self._info(
                fell=False, touched=False, tilted=True
            )

        # v9: ball-lost termination. If the ball has rolled off and dropped
        # below the platform top by more than V9_BALL_LOST_DZ, the critical
        # state is unresolvable and the episode ends with no touch reward.
        # No explicit penalty — the loss of future reward IS the pressure.
        if self.v9:
            ball_z = self.data.xpos[self._target_id][2]
            if ball_z < PLATFORM_Z - V9_BALL_LOST_DZ:
                shaping = self._pbrs_shaping(curr_body_dist, is_terminal=True)
                reward += shaping
                self._shaping_sum += shaping
                terminated = True
                self._prev_body_dist = curr_body_dist
                return obs, reward, terminated, truncated, self._info(
                    fell=False, touched=False, ball_lost=True
                )

        # Hunger: every step without food hurts
        reward += HUNGER_PENALTY
        self._hunger_sum += HUNGER_PENALTY

        # Control cost: discourage jittery torque oscillation. Standard
        # MuJoCo-locomotion-style action² penalty.
        ctrl_cost = -CTRL_COST_SCALE * float(np.sum(clipped_action ** 2))
        reward += ctrl_cost
        self._ctrl_cost_sum += ctrl_cost

        # Velocity bonus: small reward for any torso movement. Breaks the
        # stillness local optimum where the creature discovers that not moving
        # avoids ctrl_cost and pays only hunger. Scale is small enough that
        # the creature cannot satisfy hunger by spinning in place.
        torso_speed = float(np.linalg.norm(self.data.qvel[0:3]))
        velocity_bonus = VELOCITY_BONUS_SCALE * torso_speed
        reward += velocity_bonus
        self._velocity_bonus_sum += velocity_bonus

        # Closure bonus: per-step body→ball distance reduction, clipped at 0.
        # self._prev_body_dist still holds the previous step's distance (it
        # won't be updated until the end of this step, after PBRS shaping).
        # Pays `scale * max(0, Δd)` for approach; receding gets 0 (not
        # penalized — hunger already pushes against stagnation).
        if self._closure_bonus_scale > 0.0:
            delta_closed = self._prev_body_dist - curr_body_dist
            bonus = self._closure_bonus_scale * max(0.0, delta_closed)
            reward += bonus
            self._closure_sum += bonus

        # Edge proximity warning (stage 1 only; stage 0 has _edge_warn_dist=0)
        edge_dist = self._edge_distance()
        if self._edge_warn_dist > 0 and edge_dist < self._edge_warn_dist:
            reward += self._edge_warn_penalty * (
                1.0 - edge_dist / self._edge_warn_dist
            )

        # Absolute attract: stronger signal the closer the hand is to target.
        # Last-mile only — ATTRACT_MAX_DIST caps at hand-reach radius so the
        # signal cannot be harvested from a stable mid-platform grope.
        curr_dist = self._nearest_hand_dist()
        attract = ATTRACT_SCALE * max(0.0, 1.0 - curr_dist / ATTRACT_MAX_DIST)
        reward += attract
        self._attract_sum += attract
        self._dist_sum += curr_dist
        self._prev_dist = curr_dist

        # Contact: eat the target, episode ends. Idea B: in XOR mode, blue
        # contact pays -CONTACT_REWARD (touched=False so the run is recorded
        # as a wrong-action, not a success).
        if self._check_target_contact():
            if self._xor_color_random and self._is_decoy:
                reward += -CONTACT_REWARD
                self._touched = False
                xor_decoy_touch = True
            else:
                reward += CONTACT_REWARD
                self._touched = True
                xor_decoy_touch = False
            terminated = True
            shaping = self._pbrs_shaping(curr_body_dist, is_terminal=True)
            reward += shaping
            self._shaping_sum += shaping
            self._prev_body_dist = curr_body_dist
            return obs, reward, terminated, truncated, self._info(
                fell=False, touched=self._touched, edge_dist=edge_dist,
                xor_decoy_touch=xor_decoy_touch,
                xor_decoy=self._is_decoy,
            )

        # Truncate if max steps
        if self._step_count >= self._max_steps:
            truncated = True

        shaping = self._pbrs_shaping(curr_body_dist, is_terminal=truncated)
        reward += shaping
        self._shaping_sum += shaping
        self._prev_body_dist = curr_body_dist

        return obs, reward, terminated, truncated, self._info(
            fell=False, touched=self._touched, edge_dist=edge_dist
        )

    def set_target_radius(self, radius_range):
        """Mutate spawn-radius bounds at runtime. Driven by a curriculum
        callback during training; changes take effect at the next reset()."""
        lo, hi = radius_range
        self._target_radius_lo = float(lo)
        self._target_radius_hi = float(hi)

    def _info(self, **extras):
        """Build the step-info dict. Always includes episode-level component
        sums (hunger_sum, attract_sum, mean_dist, ep_len) so downstream
        callbacks can verify the reward landscape per episode. mean_dist is
        the secondary signal for the smoke-test abort rule — a downward trend
        across training means exploration is orienting toward the ball even
        when contact hasn't happened yet. Termination flags (touched, fell,
        ball_lost) are seeded with False so every info dict contains all keys
        Monitor's info_keywords lookup needs; callers override via extras."""
        steps_counted = max(self._step_count, 1)
        info = {
            "hunger_sum": self._hunger_sum,
            "attract_sum": self._attract_sum,
            "mean_dist": self._dist_sum / steps_counted,
            "ep_len": self._step_count,
            "touched": False,
            "fell": False,
            "ball_lost": False,
            "tilted": False,
            "spawn_angle": self._spawn_angle,
            "spawn_left": bool(self._spawn_angle > np.pi / 2),
            "shaping_sum": self._shaping_sum,
            "closure_sum": self._closure_sum,
            "ctrl_cost_sum": self._ctrl_cost_sum,
            "mirrored": False,
            "xor_decoy": self._is_decoy,
            "xor_decoy_touch": False,
            "velocity_bonus_sum": self._velocity_bonus_sum,
        }
        info.update(extras)
        return info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
