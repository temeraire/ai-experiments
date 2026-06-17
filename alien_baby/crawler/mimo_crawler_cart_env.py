"""
MimoCrawlerCartEnv — Phase G cart substrate.

AB is rigidly mounted on a constant-velocity 2D bouncer cart.
Locomotion is given, not learned. The policy controls only arms and head
(and trunk joints if --hip-actuation is not off).

Cart kinematics
---------------
Two slide joints (cart_x at qadr=0, cart_y at qadr=1) give the cart
unconstrained XY translation. The cart is driven kinematically: each step,
env.step() writes (cart_vx, cart_vy) directly to data.qvel[0:2].
The 4-line edge-reverse bouncer logic reflects vx/vy when the cart center
approaches a platform edge minus margin.

Hip actuation off
-----------------
When hip_actuation=False, the env zeros actuator indices 0-4 (trunk) and
15-24 (legs) before writing to data.ctrl. The policy still produces a
25-dim action vector, but those slots are silently clamped to zero so they
have no effect on physics. The policy cannot locomote; only the cart moves AB.

Observation
-----------
Proprio dim = 69, matching mimo_crawler_env.PROPRIO_DIM so StereoCrawlerCNN
works without any change:
  cart xy position (2)    — replaces "root pos xy" from the free-joint env
  cart xy velocity (2)    — replaces "root linvel xy"
  dummy 3 zeros           — occupies the 3-slot root pos; we keep 3 values
  root quat from _PRONE_QUAT (4) — AB is rigidly mounted, orientation fixed
  root vel 6 (vx,vy,0,0,0,0)    — cart vel + zeros for angular
  25 jpos + 25 jvel
  vestibular acc (3) + gyro (3)
  = 2 + 2 + 1 + 4 + 6 + 25 + 25 + 3 + 3 = 71... wait —

Actually we keep EXACTLY the same structure as MimoCrawlerEnv to preserve
PROPRIO_DIM = 69 and make StereoCrawlerCNN compatible:

  root_pos  (3): cart_x, cart_y, hip_z (from xpos)
  root_quat (4): from hip body xquat (reflects prone orientation)
  root_vel  (6): cart_vx, cart_vy, 0, 0, 0, 0
  jpos     (25): same as before
  jvel     (25): same as before
  vest_acc  (3): vestibular accelerometer
  vest_gyro (3): vestibular gyro
  = 3+4+6+25+25+3+3 = 69  ✓

Reward (Phase G)
----------------
reward = CONTACT_REWARD if touched_this_step else 0
       + quadratic_hunger_cost(steps_since_last_contact)
       + approach_reward if approach_reward_scale > 0

Quadratic hunger:
  per_step_cost = -(BASE_COST + HUNGER_RATE * (steps_since_last_contact / SCALE)^2)
  defaults: BASE_COST=0.05, HUNGER_RATE=0.20, SCALE=500
  at step 0: -0.05/step
  at step 500: -0.05 - 0.20 = -0.25/step
  at step 1000: -0.05 - 0.80 = -0.85/step

No HER. No fall/tilt termination (AB is mounted, can't fall off).
Two fixed balls; episode ends when both touched OR max_steps reached.

Contact detection uses the same geom-id lookup as MimoCrawlerEnv.
"""

import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_CART_PATH = pathlib.Path(__file__).parent / "mimo_crawler_cart.xml"
XML_CART_HANDS_PATH = pathlib.Path(__file__).parent / "mimo_crawler_cart_hands.xml"
# Rung-0 embodiment fix: the previously-passive wrist + finger joints, actuated
# in the _hands body. Appended to the proprio joint obs so the policy can both
# sense and drive them.
HAND_JOINTS = [
    "robot:right_hand1", "robot:right_hand2", "robot:right_hand3", "robot:right_fingers",
    "robot:left_hand1", "robot:left_hand2", "robot:left_hand3", "robot:left_fingers",
]

PLATFORM_TOP_Z  = 2.025
DEFAULT_MAX_STEPS   = 2000
DEFAULT_SPAWN_RADIUS = (0.5, 1.2)

CONTACT_REWARD      = 200.0
DEFAULT_HUNGER_BASE  = 0.05
DEFAULT_HUNGER_RATE  = 0.20
DEFAULT_HUNGER_SCALE = 500.0
DEFAULT_CART_SPEED   = 0.15

# Cart bouncer bounds. In prone orientation AB's head extends ~0.54m forward
# of the cart center (+y) and the legs extend ~0.27m back (-y); the north and
# south guardrails sit at y=±1.0. If the cart's range allows AB to reach the
# rails, AB's body geoms collide with them and the contact friction pushes
# the cart sideways (cart drifts in x) and decelerates it (cart stalls at the
# rail). Shrinking the cart's y-range to ±0.40 keeps AB clear of both rails
# with margin. X-range is also tightened since the cart now moves y-axis only.
CART_X_MAX  =  0.40
CART_X_MIN  = -0.40
CART_Y_MAX  =  0.40
CART_Y_MIN  = -0.40

# Platform XY bounds for ball bouncing (Phase H).
# Platform geom: size="1.0 1.0 ..." → extends from -1.0 to +1.0 in XY.
# Margin: 0.05 m (50 mm) keeps balls away from the guardrails. Ball bouncing
# uses BALL_*_MAX/MIN, NOT the tighter CART_*_MAX/MIN (those are cart bounds only).
BALL_X_MAX =  0.95
BALL_X_MIN = -0.95
BALL_Y_MAX =  0.95
BALL_Y_MIN = -0.95

CAM_H = 32
CAM_W = 32

PROPRIO_DIM = 69  # same as MimoCrawlerEnv — keeps StereoCrawlerCNN compatible
VISION_DIM_STEREO = CAM_H * CAM_W * 3 * 2
VISION_DIM_MONO   = CAM_H * CAM_W * 3

# Actuator index ranges for hip-actuation masking
# Trunk: indices 0-4 (hip_bend, hip_twist, hip_lean, chest_twist, chest_lean)
# Legs:  indices 15-24 (right hip/knee/ankle + left hip/knee/ankle)
HIP_ACT_INDICES = list(range(0, 5)) + list(range(15, 25))

# Prone quaternion — same as MimoCrawlerEnv
_PRONE_QUAT = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float64)


class MimoCrawlerCartEnv(gym.Env):
    """Phase G: AB on a constant-velocity 2D bouncer cart.

    Phase XVII extension: cart_mode="steer" gives AB control over the cart.
    cart_mode="bouncer" (default) is byte-identical to all prior runs.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        vision=False,
        max_steps=DEFAULT_MAX_STEPS,
        strength_scale=1.0,
        n_substeps=4,
        render_mode=None,
        fixed_ball_positions=None,
        memory_obs=False,
        stereo=True,
        # Cart params
        cart_speed=DEFAULT_CART_SPEED,
        # Phase XVII: cart steering mode.
        # "bouncer" (default) = current autonomous-drift code, byte-identical.
        # "steer" = AB commands cart velocity via 2 appended action dims.
        cart_mode="bouncer",
        # Hunger params
        hunger_base=DEFAULT_HUNGER_BASE,
        hunger_rate=DEFAULT_HUNGER_RATE,
        hunger_scale=DEFAULT_HUNGER_SCALE,
        # Hip actuation
        hip_actuation=True,
        # Approach reward (set to 0.0 per Phase G proposal)
        approach_reward_scale=0.0,
        velocity_bonus_scale=0.0,
        # Optional ball-radius override (default = use XML value, ~0.053).
        # Larger balls increase the contact cross-section, easing the sparse-
        # reward exploration problem at offsets the body footprint barely reaches.
        ball_radius=None,
        # If set, each reset() samples ball positions inside a box around the
        # anchor positions in fixed_ball_positions. Format: (dx, dy) half-ranges.
        # Ball1 sampled from anchor1 + Uniform(-dx,+dx) in x and Uniform(-dy,+dy)
        # in y; same for ball2. This makes ball position unpredictable per
        # episode — proprio cannot memorize it, so vision should become useful.
        random_ball_box=None,
        # If set (e.g. 300), each ball disappears (moves offscreen) after this
        # many env-steps if not touched. Forces AB to act within a deadline:
        # random arm-flailing can't catch a ball before it expires. Vision
        # should be load-bearing here because seeing the ball position lets
        # AB orient correctly within the time budget.
        ball_timeout_steps=None,
        # Phase H: constant-velocity bouncing balls.
        # ball_speed=0.0 (default) triggers the EXACT stationary-ball code path —
        # no integration, no edge check, no velocity write. Backward-compatible
        # with all Phase G results. Non-zero: each ball gets a per-reset heading
        # drawn from Uniform[0, 2π) and integrates pos += vel * dt each env step,
        # reflecting off platform edges (margin ≥ 5 mm). Velocity is NOT exposed
        # in proprio — obs["proprio"] length is identical at 0.0 and 0.08.
        ball_speed=0.0,
        # Phase X: train on multiple ball sizes per episode so the policy
        # generalizes to held-out sizes.  When set (e.g. [0.040, 0.053, 0.075]),
        # reset() samples one radius uniformly and writes it to both ball geoms.
        # When None, behavior is bit-identical to today (uses XML default 0.053
        # or any static ball_radius override).
        random_ball_radius=None,
        # Phase X2: train on multiple MuJoCo PRIMITIVE shapes per episode.
        # Accepted values: "sphere", "box", "cylinder", "ellipsoid", "capsule".
        # When set (e.g. ["sphere", "box", "cylinder"]), reset() samples one
        # shape, updates geom_type for both ball geoms, AND updates geom_size to
        # keep the bounding contact radius ~0.053 m across all shapes.
        # When None, bit-identical to today (always sphere from the XML).
        random_ball_shape=None,
        # Rung 0 (reach-to): give AB a SENSE of the target. When True, _get_obs
        # appends the hand->target error vector for BOTH hands (6 dims:
        # target - right_hand, target - left_hand). This is the most generous
        # target signal (a "feasibility floor") so we can test whether the
        # perceive->reach loop is learnable at all. Default False = unchanged.
        target_obs=False,
        # Rung 0: score/reward a HAND reach, not incidental body contact. When
        # True, contact reward AND termination use the hand-only touch flags
        # instead of the broad any-body-geom flags. Default False = unchanged.
        hand_success=False,
        # Rung 0: pin the target in place so a bump can't roll it away (the
        # landmark must stay put). When True, each step restores the active
        # ball(s) to their episode home pose and zeros their velocity. Default
        # False = unchanged (free-rolling ball).
        pin_targets=False,
        # Rung 0 endgame fix: dense shaping bonus that ramps up as the nearest
        # hand enters contact range, to give the last few cm a gradient (the
        # sparse CONTACT_REWARD alone leaves the policy stalling at the margin).
        # Per-step bonus = scale * max(0, near_contact_range - nearest_hand_dist).
        # Default scale 0.0 = unchanged.
        near_contact_bonus_scale=0.0,
        near_contact_range=0.12,
        # Rung 0 embodiment fix: load the body whose wrist + finger joints are
        # actuated (action dim 25 -> 33), and append those 8 joints to proprio
        # (dim 69 -> 85). Default False = the original passive-hand body.
        actuate_hands=False,
        # Collapse fix: override the sparse contact reward. The default 200 is a
        # huge spike vs ~0 otherwise, which destabilises the SAC critic and drives
        # the actor off the touching policy (the 200->~7 collapse seen in every
        # rung-0 run). A smaller value (e.g. 10) keeps the value range sane.
        contact_reward=None,
        # Phase XVII steer: disc spawn radius range [lo, hi] in meters.
        # Only used when cart_mode=="steer". Ball1 is placed at a random angle
        # and radius drawn from U(spawn_disc_lo, spawn_disc_hi) from platform
        # center. Ball2 is inactive (moved offscreen). Default matches spec.
        spawn_disc_lo=0.30,
        spawn_disc_hi=0.55,
    ):
        super().__init__()
        self.actuate_hands = bool(actuate_hands)
        self._contact_reward = float(contact_reward) if contact_reward is not None else CONTACT_REWARD
        self.vision = vision
        self.max_steps = max_steps
        self.strength_scale = strength_scale
        self.n_substeps = n_substeps
        self.render_mode = render_mode
        self.fixed_ball_positions = fixed_ball_positions
        self.memory_obs = memory_obs
        self.stereo = stereo
        self.cart_speed = float(cart_speed)
        if cart_mode not in ("bouncer", "steer"):
            raise ValueError(f"cart_mode must be 'bouncer' or 'steer', got {cart_mode!r}")
        self.cart_mode = str(cart_mode)
        self.spawn_disc_lo = float(spawn_disc_lo)
        self.spawn_disc_hi = float(spawn_disc_hi)
        self.hunger_base = float(hunger_base)
        self.hunger_rate = float(hunger_rate)
        self.hunger_scale = float(hunger_scale)
        self.hip_actuation = bool(hip_actuation)
        self.target_obs = bool(target_obs)
        self.hand_success = bool(hand_success)
        self.pin_targets = bool(pin_targets)
        self.near_contact_bonus_scale = float(near_contact_bonus_scale)
        self.near_contact_range = float(near_contact_range)
        self._ball1_home = None
        self._ball2_home = None
        self._current_ball_radius = float(ball_radius) if ball_radius is not None else 0.053
        self.approach_reward_scale = float(approach_reward_scale)
        self.velocity_bonus_scale = float(velocity_bonus_scale)
        self.random_ball_box = (None if random_ball_box is None
                                else (float(random_ball_box[0]), float(random_ball_box[1])))
        self.ball_timeout_steps = (None if ball_timeout_steps is None
                                   else int(ball_timeout_steps))
        self.ball_speed = float(ball_speed)
        # Phase H: per-episode ball velocity vectors (2D, shape [2,2]).
        # _ball_vel[0] = [vx, vy] for ball1; _ball_vel[1] = [vx, vy] for ball2.
        # Allocated here; populated in reset() when ball_speed > 0.
        self._ball_vel = np.zeros((2, 2), dtype=np.float64)
        # Phase X: random ball radius list (or None).
        self.random_ball_radius = (None if random_ball_radius is None
                                   else [float(r) for r in random_ball_radius])
        # Phase X2: random ball shape list (or None). Maps name -> mjtGeom int.
        # geom_size layout per shape (3-element row, unused slots = 0):
        #   sphere:    [radius, 0, 0]
        #   capsule:   [radius, half_length, 0]
        #   cylinder:  [radius, half_length, 0]
        #   ellipsoid: [rx, ry, rz]
        #   box:       [hx, hy, hz]  (half-extents)
        _SHAPE_MAP = {
            "sphere":    (int(mujoco.mjtGeom.mjGEOM_SPHERE),    [0.053, 0.0,   0.0]),
            "capsule":   (int(mujoco.mjtGeom.mjGEOM_CAPSULE),   [0.038, 0.038, 0.0]),
            "cylinder":  (int(mujoco.mjtGeom.mjGEOM_CYLINDER),  [0.045, 0.030, 0.0]),
            "ellipsoid": (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), [0.060, 0.045, 0.038]),
            "box":       (int(mujoco.mjtGeom.mjGEOM_BOX),       [0.045, 0.045, 0.045]),
        }
        if random_ball_shape is not None:
            for s in random_ball_shape:
                if s not in _SHAPE_MAP:
                    raise ValueError(f"Unknown ball shape '{s}'. Valid: {list(_SHAPE_MAP)}")
            self.random_ball_shape = list(random_ball_shape)
        else:
            self.random_ball_shape = None
        self._SHAPE_MAP = _SHAPE_MAP

        self.model = mujoco.MjModel.from_xml_path(
            str(XML_CART_HANDS_PATH if self.actuate_hands else XML_CART_PATH))
        self.data  = mujoco.MjData(self.model)

        # Optional ball-radius override (set both geoms' size[0]).
        if ball_radius is not None:
            for gname in ("target_geom", "target2_geom"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    self.model.geom_size[gid, 0] = float(ball_radius)

        # Cart joint addresses
        cart_x_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")
        cart_y_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_y")
        self._cart_x_qadr = self.model.jnt_qposadr[cart_x_jid]  # = 0
        self._cart_y_qadr = self.model.jnt_qposadr[cart_y_jid]  # = 1
        self._cart_x_dadr = self.model.jnt_dofadr[cart_x_jid]   # = 0
        self._cart_y_dadr = self.model.jnt_dofadr[cart_y_jid]   # = 1

        # Cart and hip body ids
        self._cart_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
        self._hip_bid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hip")

        # AB joint addresses — same as MimoCrawlerEnv but offset by 2 (no free joint)
        from alien_baby.crawler.mimo_crawler_env import ACTUATED_JOINTS
        obs_joints = list(ACTUATED_JOINTS) + (HAND_JOINTS if self.actuate_hands else [])
        self._jpos_adr = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ] for j in obs_joints
        ], dtype=np.int32)
        self._jvel_adr = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ] for j in obs_joints
        ], dtype=np.int32)

        # Vestibular sensors
        self._vest_acc_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "vestibular_acc")
        self._vest_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "vestibular_gyro")

        # Ball geom ids
        self._target_geom_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self._target2_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target2_geom")

        # Hand body ids (used by approach reward — distance from nearest hand
        # to nearest active ball). Falling back to None for legacy compatibility.
        self._right_hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
        self._left_hand_bid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand")

        # Joint addresses for the ball free joints (used to read ball positions
        # cheaply each step from qpos rather than re-resolving by name).
        tgt1_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt2_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_free")
        self._tgt1_qadr = self.model.jnt_qposadr[tgt1_jid]
        self._tgt2_qadr = self.model.jnt_qposadr[tgt2_jid]
        # Phase H: ball velocity DOF addresses (for kinematic XY override in _step_balls).
        # Free joint has 6 velocity DOFs: [vx, vy, vz, wx, wy, wz].
        self._tgt1_dadr = self.model.jnt_dofadr[tgt1_jid]
        self._tgt2_dadr = self.model.jnt_dofadr[tgt2_jid]

        # MIMo body geom ids (for contact detection)
        self._mimo_body_ids = self._get_mimo_body_ids()
        # Hand-only geom ids — for the stricter `hand_touched_ball*` metric
        # that disambiguates "reach with a hand" from "ball rolls into a leg
        # while the cart sweeps past". Includes right_hand, left_hand and
        # their child fingers bodies (8 geoms total in the stock MIMo XML).
        self._hand_geom_ids = self._get_hand_geom_ids()

        # Observation space
        memory_dim = 2 if memory_obs else 0
        target_dim = 6 if target_obs else 0   # hand->target vector for both hands
        # proprio = root_pos(3) + root_quat(4) + root_vel(6) + jpos + jvel + vest(6).
        # = 69 with the 25-joint body, 85 when the 8 hand joints are actuated.
        proprio_dim = 3 + 4 + 6 + len(self._jpos_adr) + len(self._jvel_adr) + 6
        if vision:
            vis_dim = VISION_DIM_STEREO if stereo else VISION_DIM_MONO
        else:
            vis_dim = 0
        n_obs = proprio_dim + memory_dim + target_dim + vis_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        # Phase XVII: steer mode adds 2 cart-steering dims after the nu limb dims.
        # Bouncer mode is byte-identical (shape=(nu,)).
        _action_dim = self.model.nu + (2 if self.cart_mode == "steer" else 0)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_action_dim,), dtype=np.float32
        )

        if vision:
            self._cam_renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        else:
            self._cam_renderer = None
        self._render_renderer = None

        # Episode state
        self._step = 0
        self._ball1_touched = False
        self._ball2_touched = False
        self._ball1_hand_touched = False
        self._ball2_hand_touched = False
        self._ball2_active  = False
        self._cart_vx = 0.0
        self._cart_vy = 0.0
        self._steps_since_contact = 0  # hunger counter
        self._prev_ball_dist = 0.0
        # Disappearing-ball state: each ball tracks whether it has expired
        # (vanished due to timeout). Once expired, treated like touched for
        # termination but no contact reward. Set in reset() / step().
        self._ball1_expired = False
        self._ball2_expired = False

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Zero cart position (cart starts at platform center)
        self.data.qpos[self._cart_x_qadr] = 0.0
        self.data.qpos[self._cart_y_qadr] = 0.0
        self.data.qvel[self._cart_x_dadr] = 0.0
        self.data.qvel[self._cart_y_dadr] = 0.0

        # Phase XVII: steer mode starts cart stationary; policy drives it.
        # Bouncer mode: Y-axis-only motion, random sign per episode.
        if self.cart_mode == "steer":
            self._cart_vx = 0.0
            self._cart_vy = 0.0
        else:
            # Y-axis-only cart motion: constrains cart trajectory to a fixed line
            # so balls placed at (0, ±ball_y) are guaranteed to be swept. Random
            # sign on cart_vy so AB sees both directions of travel.
            self._cart_vx = 0.0
            self._cart_vy = self.cart_speed * float(self.np_random.choice([-1.0, 1.0]))

        # Small joint noise so episodes differ
        for adr in self._jpos_adr:
            self.data.qpos[adr] += self.np_random.uniform(-0.05, 0.05)

        # Phase X: per-episode ball radius sampling.
        # Samples one radius from the list and applies it to both ball geoms.
        # When random_ball_radius is None, this block is skipped entirely
        # (bit-identical to previous behavior).
        if self.random_ball_radius is not None:
            idx = int(self.np_random.integers(0, len(self.random_ball_radius)))
            r = self.random_ball_radius[idx]
            self._current_ball_radius = r
            for gid in (self._target_geom_id, self._target2_geom_id):
                self.model.geom_size[gid, 0] = r
        else:
            self._current_ball_radius = float(self.model.geom_size[self._target_geom_id, 0])

        # Phase X2: per-episode ball shape sampling.
        # Samples one shape from the list, updates geom_type and geom_size for
        # both ball geoms. When random_ball_shape is None, skipped entirely.
        if self.random_ball_shape is not None:
            idx = int(self.np_random.integers(0, len(self.random_ball_shape)))
            shape_name = self.random_ball_shape[idx]
            geom_type_int, geom_size = self._SHAPE_MAP[shape_name]
            self._current_ball_shape = shape_name
            for gid in (self._target_geom_id, self._target2_geom_id):
                self.model.geom_type[gid] = geom_type_int
                self.model.geom_size[gid, :] = geom_size
        else:
            self._current_ball_shape = "sphere"

        # Place balls
        tgt1_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt1_qadr = self.model.jnt_qposadr[tgt1_jid]
        tgt2_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_free")
        tgt2_qadr = self.model.jnt_qposadr[tgt2_jid]

        if self.fixed_ball_positions:
            b1x, b1y = self.fixed_ball_positions[0]
            if self.random_ball_box is not None:
                dx, dy = self.random_ball_box
                b1x += float(self.np_random.uniform(-dx, dx))
                b1y += float(self.np_random.uniform(-dy, dy))
            self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [b1x, b1y, PLATFORM_TOP_Z + self._current_ball_radius]
            self.data.qpos[tgt1_qadr + 3:tgt1_qadr + 7] = [1, 0, 0, 0]
            if len(self.fixed_ball_positions) >= 2:
                b2x, b2y = self.fixed_ball_positions[1]
                if self.random_ball_box is not None:
                    dx, dy = self.random_ball_box
                    b2x += float(self.np_random.uniform(-dx, dx))
                    b2y += float(self.np_random.uniform(-dy, dy))
                self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [b2x, b2y, PLATFORM_TOP_Z + self._current_ball_radius]
                self._ball2_active = True
            else:
                self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [10.0, 10.0, 0.0]
                self._ball2_active = False
            self.data.qpos[tgt2_qadr + 3:tgt2_qadr + 7] = [1, 0, 0, 0]
        elif self.cart_mode == "steer":
            # Phase XVII steer: single-ball winnable spawn.
            # Ball1 at r~U(spawn_disc_lo, spawn_disc_hi) from platform center,
            # random angle, on-platform. Ball2 inactive (offscreen).
            # Rejection loop: resample if the ball spawns in contact with AB at
            # rest (which happens when the ball lands inside the body footprint
            # e.g. the head extends ~0.54m in +y from the cart center).
            self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [10.0, 10.0, 0.0]
            self.data.qpos[tgt2_qadr + 3:tgt2_qadr + 7] = [1, 0, 0, 0]
            self._ball2_active = False
            for _spawn_attempt in range(50):
                r = self.np_random.uniform(self.spawn_disc_lo, self.spawn_disc_hi)
                a = self.np_random.uniform(0.0, 2.0 * np.pi)
                bx, by = r * np.cos(a), r * np.sin(a)
                # Clamp to platform interior so ball is never off-edge at spawn
                bx = float(np.clip(bx, BALL_X_MIN + 0.05, BALL_X_MAX - 0.05))
                by = float(np.clip(by, BALL_Y_MIN + 0.05, BALL_Y_MAX - 0.05))
                self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [bx, by, PLATFORM_TOP_Z + self._current_ball_radius]
                self.data.qpos[tgt1_qadr + 3:tgt1_qadr + 7] = [1, 0, 0, 0]
                mujoco.mj_forward(self.model, self.data)
                # Check for step-0 contact between ball and AB body geoms
                _hit = False
                for _ci in range(self.data.ncon):
                    _c = self.data.contact[_ci]
                    _g1, _g2 = _c.geom1, _c.geom2
                    if _g1 == self._target_geom_id or _g2 == self._target_geom_id:
                        _other = _g2 if _g1 == self._target_geom_id else _g1
                        if _other in self._mimo_body_ids:
                            _hit = True
                            break
                if not _hit:
                    break  # clean spawn found
            # (If all 50 attempts hit, accept the last sample — rare edge case)
        else:
            # Random ball1 placement (bouncer default)
            r = self.np_random.uniform(0.5, 1.2)
            a = self.np_random.uniform(0.0, 2.0 * np.pi)
            self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [r * np.cos(a), r * np.sin(a), PLATFORM_TOP_Z + self._current_ball_radius]
            self.data.qpos[tgt1_qadr + 3:tgt1_qadr + 7] = [1, 0, 0, 0]
            self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [10.0, 10.0, 0.0]
            self.data.qpos[tgt2_qadr + 3:tgt2_qadr + 7] = [1, 0, 0, 0]
            self._ball2_active = False

        mujoco.mj_forward(self.model, self.data)

        # Rung 0: remember each ball's start pose so pin_targets can hold it there.
        self._ball1_home = self.data.qpos[tgt1_qadr:tgt1_qadr + 3].copy()
        self._ball2_home = self.data.qpos[tgt2_qadr:tgt2_qadr + 3].copy()
        self._step = 0
        self._ball1_touched = False
        self._ball2_touched = False
        self._ball1_hand_touched = False
        self._ball2_hand_touched = False
        self._ball1_expired = False
        self._ball2_expired = False
        self._steps_since_contact = 0
        self._prev_ball_dist = self._ball_dist()

        # Phase H: draw fresh ball velocities for this episode.
        # ONLY when ball_speed > 0 — the stationary-ball code path (0.0) must
        # be bit-identical to all Phase G runs. _ball_vel is reset to zeros
        # so that even if the gate check is ever skipped, no stale velocity
        # propagates.
        self._ball_vel[:] = 0.0
        if self.ball_speed > 0.0:
            for i in range(2):
                heading = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                self._ball_vel[i, 0] = self.ball_speed * np.cos(heading)
                self._ball_vel[i, 1] = self.ball_speed * np.sin(heading)
            # Write initial ball velocities to MuJoCo qvel so physics is seeded
            # with our kinematic velocity from the start of the episode.
            for i, dadr in enumerate([self._tgt1_dadr, self._tgt2_dadr]):
                self.data.qvel[dadr]     = self._ball_vel[i, 0]
                self.data.qvel[dadr + 1] = self._ball_vel[i, 1]
                self.data.qvel[dadr + 2] = 0.0
            mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        # Read current cart position
        cart_x = float(self.data.qpos[self._cart_x_qadr])
        cart_y = float(self.data.qpos[self._cart_y_qadr])

        if self.cart_mode == "steer":
            # Phase XVII STEER branch:
            # Split action into limb dims and 2 steering dims.
            nu = self.model.nu
            limb_action = action[:nu]
            steer = action[nu:nu + 2]  # [steer_x, steer_y] in [-1, 1]
            # Map steer dims to cart velocity
            desired_vx = float(np.clip(steer[0], -1.0, 1.0)) * self.cart_speed
            desired_vy = float(np.clip(steer[1], -1.0, 1.0)) * self.cart_speed
            # EDGE-CLAMP: if cart is at/over a platform edge and steer still
            # pushes outward, zero that velocity component. AB can never drive
            # off-platform (winnability guarantee).
            if cart_x >= CART_X_MAX and desired_vx > 0.0:
                desired_vx = 0.0
            if cart_x <= CART_X_MIN and desired_vx < 0.0:
                desired_vx = 0.0
            if cart_y >= CART_Y_MAX and desired_vy > 0.0:
                desired_vy = 0.0
            if cart_y <= CART_Y_MIN and desired_vy < 0.0:
                desired_vy = 0.0
            self._cart_vx = desired_vx
            self._cart_vy = desired_vy
            # Limb control (same as bouncer path)
            ctrl = np.clip(limb_action, -1.0, 1.0) * self.strength_scale
            if not self.hip_actuation:
                ctrl[HIP_ACT_INDICES] = 0.0
            self.data.ctrl[:] = ctrl
        else:
            # BOUNCER branch (byte-identical to all prior runs):
            # Check position and reverse velocity components at edges
            if cart_x > CART_X_MAX:
                self._cart_vx = -abs(self._cart_vx)
            if cart_x < CART_X_MIN:
                self._cart_vx =  abs(self._cart_vx)
            if cart_y > CART_Y_MAX:
                self._cart_vy = -abs(self._cart_vy)
            if cart_y < CART_Y_MIN:
                self._cart_vy =  abs(self._cart_vy)
            # Mask hip/leg actuators if disabled
            ctrl = np.clip(action, -1.0, 1.0) * self.strength_scale
            if not self.hip_actuation:
                ctrl[HIP_ACT_INDICES] = 0.0
            self.data.ctrl[:] = ctrl

        # Write cart velocity kinematically (overrides any physics-derived vel)
        self.data.qvel[self._cart_x_dadr] = self._cart_vx
        self.data.qvel[self._cart_y_dadr] = self._cart_vy

        # Step physics
        _dt_sub = float(self.model.opt.timestep)  # 0.005 s per substep
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            # Re-enforce cart velocity each substep so it doesn't drift
            self.data.qvel[self._cart_x_dadr] = self._cart_vx
            self.data.qvel[self._cart_y_dadr] = self._cart_vy
            # Rung 0: pin the target(s) in place so a bump can't roll them away.
            if self.pin_targets:
                if self._ball1_home is not None:
                    self.data.qpos[self._tgt1_qadr:self._tgt1_qadr + 3] = self._ball1_home
                    self.data.qvel[self._tgt1_dadr:self._tgt1_dadr + 6] = 0.0
                if self._ball2_active and self._ball2_home is not None:
                    self.data.qpos[self._tgt2_qadr:self._tgt2_qadr + 3] = self._ball2_home
                    self.data.qvel[self._tgt2_dadr:self._tgt2_dadr + 6] = 0.0
            # Phase H: integrate ball positions (only when ball_speed > 0).
            # Reflection check is inside the substep loop per the proposal —
            # at 0.08 m/s per-substep travel is 0.4 mm, well inside the 5 mm
            # edge margin, so tunneling is not a risk.
            if self.ball_speed > 0.0:
                self._step_balls(_dt_sub)

        self._step += 1
        self._steps_since_contact += 1

        obs = self._get_obs()

        ball1_hit, ball2_hit, ball1_hand_hit, ball2_hand_hit = self._check_ball_contact()
        # Latch hand-only "touched at any point during the episode" flags
        # alongside the existing broad flags. These are diagnostic only —
        # episode termination still uses the broad metric for backward compat.
        if ball1_hand_hit:
            self._ball1_hand_touched = True
        if ball2_hand_hit and self._ball2_active:
            self._ball2_hand_touched = True

        # Rung 0: score/reward a HAND reach. Swap the broad hit flags for the
        # hand-only ones so the contact reward AND termination below fire only
        # on a hand touch, not on incidental body/leg contact.
        if self.hand_success:
            ball1_hit, ball2_hit = ball1_hand_hit, ball2_hand_hit

        curr_dist = self._ball_dist()
        approach  = self._prev_ball_dist - curr_dist
        self._prev_ball_dist = curr_dist

        # Quadratic hunger cost
        s = float(self._steps_since_contact)
        hunger_cost = -(self.hunger_base + self.hunger_rate * (s / self.hunger_scale) ** 2)
        reward = hunger_cost

        # Approach bonus (0.0 per Phase G proposal, included for completeness)
        reward += self.approach_reward_scale * approach

        # Rung 0 endgame shaping: ramp a dense bonus as the nearest hand enters
        # contact range, so the last few cm have a gradient pulling into contact.
        if self.near_contact_bonus_scale > 0.0 and curr_dist < self.near_contact_range:
            reward += self.near_contact_bonus_scale * (self.near_contact_range - curr_dist)

        # Joint-velocity bonus: rewards motion of the actuated joints.
        # In the cart setup AB can't move its body (cart is kinematic), so
        # this specifically rewards arm/head joint movement — i.e., the
        # anti-stillness pressure the strategist's R5 was designed to test.
        # Using mean(|jvel|) over actuated joints so scale=0.2 gives a
        # per-step bonus comparable in magnitude to hunger (~0.05-0.85).
        if self.velocity_bonus_scale > 0:
            jvel = np.abs(self.data.qvel[self._jvel_adr])
            reward += self.velocity_bonus_scale * float(jvel.mean())

        # Contact rewards
        if ball1_hit and not self._ball1_touched:
            reward += self._contact_reward
            self._ball1_touched = True
            self._steps_since_contact = 0
            # Phase H: zero ball1 velocity on touch so it stops moving
            if self.ball_speed > 0.0:
                self._ball_vel[0, :] = 0.0

        if self._ball2_active and ball2_hit and not self._ball2_touched:
            reward += self._contact_reward
            self._ball2_touched = True
            self._steps_since_contact = 0
            # Phase H: zero ball2 velocity on touch so it stops moving
            if self.ball_speed > 0.0:
                self._ball_vel[1, :] = 0.0

        # Disappearing-ball timeout: vanish balls that have been alive too long.
        # An expired ball is moved offscreen so contact detection won't fire
        # and approach_reward stops being pulled toward it; treated like touched
        # for termination purposes (so the episode can still end normally).
        if self.ball_timeout_steps is not None:
            if (not self._ball1_touched and not self._ball1_expired
                    and self._step >= self.ball_timeout_steps):
                self._ball1_expired = True
                self.data.qpos[self._tgt1_qadr:self._tgt1_qadr + 3] = [10.0, 10.0, 0.0]
                mujoco.mj_forward(self.model, self.data)
            if (self._ball2_active and not self._ball2_touched and not self._ball2_expired
                    and self._step >= self.ball_timeout_steps):
                self._ball2_expired = True
                self.data.qpos[self._tgt2_qadr:self._tgt2_qadr + 3] = [10.0, 10.0, 0.0]
                mujoco.mj_forward(self.model, self.data)

        # Termination — a ball counts as "done" if touched OR expired.
        terminated = False
        truncated  = self._step >= self.max_steps
        b1_done = self._ball1_touched or self._ball1_expired
        if self._ball2_active:
            b2_done = self._ball2_touched or self._ball2_expired
            if b1_done and b2_done:
                terminated = True
        else:
            if b1_done:
                terminated = True

        # Phase XVI: compute ego_xy = [ball_x - cart_x, ball_y - cart_y] for
        # the first active ball. Matches probe_vision_latent.py definition so
        # training target == eval target (no metric drift). Stored in info so
        # EgoTargetReplayBuffer can persist it alongside the transition.
        _cart_x = float(self.data.qpos[self._cart_x_qadr])
        _cart_y = float(self.data.qpos[self._cart_y_qadr])
        _b1x = float(self.data.qpos[self._tgt1_qadr])
        _b1y = float(self.data.qpos[self._tgt1_qadr + 1])
        _ego_xy = (_b1x - _cart_x, _b1y - _cart_y)

        return obs, reward, terminated, truncated, {
            "touched":      self._ball1_touched,
            "touched_ball1": self._ball1_touched,
            "touched_ball2": self._ball2_touched,
            "hand_touched_ball1": self._ball1_hand_touched,
            "hand_touched_ball2": self._ball2_hand_touched,
            "expired_ball1": self._ball1_expired,
            "expired_ball2": self._ball2_expired,
            "step":          self._step,
            "steps_since_contact": self._steps_since_contact,
            "hunger_cost":   hunger_cost,
            "cart_x":        _cart_x,
            "cart_y":        _cart_y,
            "cart_vx":       self._cart_vx,
            "cart_vy":       self._cart_vy,
            "strength_scale": self.strength_scale,
            "ego_xy":        _ego_xy,
        }

    # ------------------------------------------------------------------
    def _step_balls(self, dt: float) -> None:
        """Phase H: integrate ball positions by vel*dt and reflect off platform edges.

        Called once per physics substep (dt = model.opt.timestep = 0.005 s) so
        that the edge-reflection check runs at the finest time resolution,
        preventing any tunneling at the platform boundary.

        CRITICAL: this method is ONLY called when self.ball_speed > 0.  The
        stationary-ball (ball_speed=0.0) code path NEVER calls this — no side
        effects on Phase G results.

        Platform XY bounds: BALL_X_MIN/MAX and BALL_Y_MIN/MAX (±0.95 m).
        These are the platform bounds (platform half-size = 1.0 m) minus a
        50 mm margin — NOT the tighter CART_*_MAX/MIN which are cart travel
        bounds only. At ball_speed=0.08, per-substep travel is 0.4 mm, well
        inside the 50 mm margin, so tunneling is not a risk.
        """
        dadr_list = [self._tgt1_dadr, self._tgt2_dadr]
        qadr_list = [self._tgt1_qadr, self._tgt2_qadr]
        touched_list = [self._ball1_touched, self._ball2_touched]

        for i in range(2):
            qadr   = qadr_list[i]
            dadr   = dadr_list[i]
            touched = touched_list[i]

            # Skip touched balls (velocity already zeroed) and ball2 if inactive
            if i == 1 and not self._ball2_active:
                continue
            if touched:
                continue
            vx, vy = self._ball_vel[i, 0], self._ball_vel[i, 1]
            if vx == 0.0 and vy == 0.0:
                continue
            # Integrate kinematically
            x = self.data.qpos[qadr]     + vx * dt
            y = self.data.qpos[qadr + 1] + vy * dt
            # Reflect off X edges (platform bounds ±0.95 m)
            if x > BALL_X_MAX:
                x = 2.0 * BALL_X_MAX - x
                self._ball_vel[i, 0] = -abs(vx)
            elif x < BALL_X_MIN:
                x = 2.0 * BALL_X_MIN - x
                self._ball_vel[i, 0] = abs(vx)
            # Reflect off Y edges (platform bounds ±0.95 m)
            if y > BALL_Y_MAX:
                y = 2.0 * BALL_Y_MAX - y
                self._ball_vel[i, 1] = -abs(vy)
            elif y < BALL_Y_MIN:
                y = 2.0 * BALL_Y_MIN - y
                self._ball_vel[i, 1] = abs(vy)
            # Write updated position (Z and quaternion unchanged)
            self.data.qpos[qadr]     = x
            self.data.qpos[qadr + 1] = y
            # Override physics velocity so MuJoCo doesn't fight our kinematic
            # update. Write [vx, vy, 0, 0, 0, 0] to the ball's free-joint DOFs.
            # Without this, mj_step() uses the ball's physics-derived velocity
            # (from inertia, gravity, contact forces) and competes with qpos writes,
            # causing apparent position drift vs. the kinematic target.
            self.data.qvel[dadr]     = self._ball_vel[i, 0]
            self.data.qvel[dadr + 1] = self._ball_vel[i, 1]
            self.data.qvel[dadr + 2] = 0.0  # vz = 0 (ball stays on platform)
            # Angular velocities (dadr+3, dadr+4, dadr+5) left as-is (physics handles)

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Build 69-dim proprio matching PROPRIO_DIM from MimoCrawlerEnv."""
        # Root position: use cart XY + hip body Z
        cart_x = float(self.data.qpos[self._cart_x_qadr])
        cart_y = float(self.data.qpos[self._cart_y_qadr])
        hip_z  = float(self.data.xpos[self._hip_bid][2])
        root_pos = np.array([cart_x, cart_y, hip_z], dtype=np.float32)

        # Root quaternion: from hip body (reflects AB's orientation on cart)
        root_quat = self.data.xquat[self._hip_bid].astype(np.float32)

        # Root velocity: cart XY vel + zeros for Z + zeros for angular (6 total)
        root_vel = np.array([
            self._cart_vx, self._cart_vy, 0.0,   # linear vx, vy, vz
            0.0, 0.0, 0.0,                         # angular (cart doesn't rotate)
        ], dtype=np.float32)

        # Joint positions and velocities (25 each)
        jpos = self.data.qpos[self._jpos_adr].astype(np.float32)
        jvel = self.data.qvel[self._jvel_adr].astype(np.float32)

        # Vestibular
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
                1.0 if self._ball1_touched else 0.0,
                1.0 if self._ball2_touched else 0.0,
            ], dtype=np.float32)
            proprio = np.concatenate([proprio, mem])

        if self.target_obs:
            # Rung 0 "sense": hand->target error vector for both hands.
            # Target is ball1 (the single active target in reach-to runs).
            tgt = self.data.qpos[self._tgt1_qadr:self._tgt1_qadr + 3]
            r_hand = self.data.xpos[self._right_hand_bid]
            l_hand = self.data.xpos[self._left_hand_bid]
            tvec = np.concatenate([tgt - r_hand, tgt - l_hand]).astype(np.float32)
            proprio = np.concatenate([proprio, tvec])

        if not self.vision:
            return proprio

        self._cam_renderer.update_scene(self.data, camera="left_eye")
        left_img = self._cam_renderer.render().copy().astype(np.float32) / 255.0

        if not self.stereo:
            return np.concatenate([proprio, left_img.ravel()])

        self._cam_renderer.update_scene(self.data, camera="right_eye")
        right_img = self._cam_renderer.render().copy().astype(np.float32) / 255.0
        return np.concatenate([proprio, left_img.ravel(), right_img.ravel()])

    # ------------------------------------------------------------------
    def _ball_dist(self):
        """Minimum 3D distance from either untouched-hand to any active,
        untouched ball. Used by the approach reward — distance reduces only
        when arm motion brings a hand closer to a ball, so the reward
        gradient actually rewards arm extension (not just cart kinematics).

        Touched balls are excluded so the agent isn't pulled toward an
        already-collected ball. If all balls are touched, returns 0.0.
        """
        ball_positions = []
        if not self._ball1_touched:
            ball_positions.append(self.data.qpos[self._tgt1_qadr:self._tgt1_qadr + 3])
        if self._ball2_active and not self._ball2_touched:
            ball_positions.append(self.data.qpos[self._tgt2_qadr:self._tgt2_qadr + 3])
        if not ball_positions:
            return 0.0

        hand_positions = [
            self.data.xpos[self._right_hand_bid],
            self.data.xpos[self._left_hand_bid],
        ]
        best = float("inf")
        for h in hand_positions:
            for b in ball_positions:
                d = float(np.linalg.norm(np.asarray(h) - np.asarray(b)))
                if d < best:
                    best = d
        return best

    # ------------------------------------------------------------------
    def _check_ball_contact(self):
        """Returns (ball1_hit, ball2_hit, ball1_hand_hit, ball2_hand_hit).

        The first two are the broad metric (any MIMo body part — feet, legs,
        torso, head, hands). The last two are the strict hand-only metric
        (only the hand and fingers geoms count). Hand-only disambiguates
        "AB reached with a hand" from "ball rolled into AB's leg while the
        cart swept past it" — important in cart-mode where the cart can
        push a ball into a static body part without any reach happening.
        """
        ball1_hit = False
        ball2_hit = False
        ball1_hand_hit = False
        ball2_hand_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            for target_id, flag in [(self._target_geom_id, "b1"),
                                     (self._target2_geom_id, "b2")]:
                if g1 == target_id or g2 == target_id:
                    other = g2 if g1 == target_id else g1
                    if other in self._mimo_body_ids:
                        if flag == "b1":
                            ball1_hit = True
                            if other in self._hand_geom_ids:
                                ball1_hand_hit = True
                        else:
                            ball2_hit = True
                            if other in self._hand_geom_ids:
                                ball2_hand_hit = True
        return ball1_hit, ball2_hit, ball1_hand_hit, ball2_hand_hit

    def _get_mimo_body_ids(self):
        """All geom ids that belong to MIMo (not world/platform/rails/targets/cart)."""
        exclude = {"ground", "platform", "rail_N", "rail_S", "rail_E", "rail_W",
                   "target_geom", "target2_geom", "cart_geom"}
        ids = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name not in exclude:
                ids.add(i)
        return ids

    def _get_hand_geom_ids(self):
        """Geom ids belonging to the right_hand and left_hand bodies plus their
        descendant bodies (right_fingers, left_fingers). Used for the strict
        hand-only touch metric."""
        # Walk the body hierarchy from each hand outward to include child bodies
        hand_bodies = set()
        for root in (self._right_hand_bid, self._left_hand_bid):
            hand_bodies.add(root)
            changed = True
            while changed:
                changed = False
                for i in range(self.model.nbody):
                    if (self.model.body_parentid[i] in hand_bodies
                            and i not in hand_bodies):
                        hand_bodies.add(i)
                        changed = True
        ids = set()
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) in hand_bodies:
                ids.add(g)
        return ids

    # ------------------------------------------------------------------
    def set_ball_positions(self, b1_xy, b2_xy):
        """Update ball positions on the live env. Used by the curriculum
        callback to ramp ball x-offset over training. Updates both
        self.fixed_ball_positions (so future resets use new positions) and the
        current data.qpos (so the change is visible mid-episode without
        waiting for the next reset)."""
        b1 = (float(b1_xy[0]), float(b1_xy[1]))
        b2 = (float(b2_xy[0]), float(b2_xy[1]))
        self.fixed_ball_positions = [b1, b2]
        self._ball2_active = True
        tgt1_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        tgt1_qadr = self.model.jnt_qposadr[tgt1_jid]
        tgt2_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_free")
        tgt2_qadr = self.model.jnt_qposadr[tgt2_jid]
        self.data.qpos[tgt1_qadr:tgt1_qadr + 3] = [b1[0], b1[1], PLATFORM_TOP_Z + self._current_ball_radius]
        self.data.qpos[tgt2_qadr:tgt2_qadr + 3] = [b2[0], b2[1], PLATFORM_TOP_Z + self._current_ball_radius]
        mujoco.mj_forward(self.model, self.data)

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
