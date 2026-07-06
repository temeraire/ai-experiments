"""
Per-episode left-right mirror augmentation for PlatformCreatureEnv.

Why: stage-1 runs collapse into a lateralization attractor (left-turret vs.
right-turret) where the policy only solves one hemisphere. The creature + env
are mirror-symmetric across the sagittal plane (x=0), so any trajectory has
a valid mirror-image trajectory under the same env. Wrapping the env with a
per-episode coin-flip mirror doubles the effective data on the under-visited
hemisphere and breaks the attractor.

Transforms (reflection across x=0):
    Positions:        x → -x, y,z unchanged
    Linear velocity:  (polar vector) vx → -vx, vy,vz unchanged
    Angular velocity: (pseudo-vector) ωx unchanged, ωy,ωz → -ωy,-ωz
    Quaternion (body→world): (w,x,y,z) → (w,x,-y,-z)
        Derived via R' = M_x R M_x with M_x = diag(-1,1,1).
    Joints: both shoulders/elbows share axis="1 0 0" in XML; mirror is
        swap-L/R with no sign flip. head_pan (axis Z) negates;
        head_tilt (axis X) unchanged.
    Touch: swap L/R hand, torso unchanged.
    Pixels: horizontal column flip.

Obs layout matches PlatformCreatureEnv._get_proprio() (29 dims) + optional
32*32*3 head_cam pixels concatenated. Action layout matches the XML actuator
order: [LSP, LSR, LE, RSP, RSR, RE, head_pan, head_tilt].

Architecture: obs/action transform only — no env-state modification. The
wrapper mirrors obs on the way out and un-mirrors actions on the way in,
relying on the env's mirror symmetry to produce a consistent "world B"
trajectory from the policy's perspective. State mirroring would be
redundant: obs is equivariant under state mirror, so doing both cancels
(obs_returned_to_policy = mirror(obs(mirror(s))) = mirror(mirror(obs(s))) =
obs(s), giving the policy the un-mirrored view). v9's spawn distribution is
already mirror-symmetric (angle ∈ [0, π), invariant under θ → π-θ), so no
task-distribution skew arises from the obs-only approach.
"""

import numpy as np
import gymnasium as gym


PROPRIO_DIM = 29
CAM_HEIGHT = 32
CAM_WIDTH = 32


def mirror_action(action):
    """Mirror-invert an action vector. Involution: mirror(mirror(a)) == a."""
    a = np.asarray(action, dtype=np.float32).copy()
    left = a[0:3].copy()
    a[0:3] = a[3:6]
    a[3:6] = left
    a[6] = -a[6]
    return a


def mirror_obs(obs, has_vision=False):
    """Mirror-invert an obs vector. Involution: mirror(mirror(o)) == o."""
    o = np.asarray(obs, dtype=np.float32).copy()
    # Arm pos: swap L[0:3] ↔ R[3:6]
    left_pos = o[0:3].copy()
    o[0:3] = o[3:6]
    o[3:6] = left_pos
    # Arm vel: swap L[6:9] ↔ R[9:12]
    left_vel = o[6:9].copy()
    o[6:9] = o[9:12]
    o[9:12] = left_vel
    # head_pan negate, head_tilt unchanged
    o[12] = -o[12]
    # Quaternion (w,x,y,z) at [14:18] → (w,x,-y,-z)
    o[16] = -o[16]
    o[17] = -o[17]
    # Linear velocity [18:21]: vx flips
    o[18] = -o[18]
    # Angular velocity [21:24]: ωy, ωz flip
    o[22] = -o[22]
    o[23] = -o[23]
    # Touch [24:27]: swap left_hand/right_hand, torso unchanged
    lh = float(o[24])
    o[24] = o[25]
    o[25] = lh
    # Torso xy [27:29]: x flips
    o[27] = -o[27]
    if has_vision:
        pixels = o[PROPRIO_DIM:].reshape(CAM_HEIGHT, CAM_WIDTH, 3)
        o[PROPRIO_DIM:] = pixels[:, ::-1, :].copy().flatten()
    return o


class MirrorWrapper(gym.Wrapper):
    """Per-episode coin-flip mirror augmentation.

    At reset, flips a coin (probability `mirror_prob`). If heads, the episode
    is played in the mirrored frame: obs returned to the policy are
    left-right reflected, and incoming actions are un-reflected before
    reaching the env. The env's own spawn distribution is already
    mirror-symmetric so no physical state modification is needed.

    `info["mirrored"]` is populated on every reset/step for telemetry. The
    env's `spawn_angle` and `spawn_left` fields are flipped in info when
    mirrored so hemisphere statistics reflect the policy's perceived side,
    not the underlying physical spawn.
    """

    def __init__(self, env, mirror_prob=0.5):
        super().__init__(env)
        self.mirror_prob = float(mirror_prob)
        self._mirrored = False
        self._has_vision = env.observation_space.shape[0] > PROPRIO_DIM

    def reset(self, *, seed=None, options=None):
        # `options["force_mirror"]` is a test hook — lets unit tests pin the
        # coin outcome deterministically without touching internal state.
        force = None
        if options is not None and "force_mirror" in options:
            force = bool(options["force_mirror"])
            options = {k: v for k, v in options.items() if k != "force_mirror"}
            if not options:
                options = None
        obs, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped
        if force is not None:
            self._mirrored = force
        else:
            rng = base.np_random
            coin = rng.random() if rng is not None else np.random.random()
            self._mirrored = bool(coin < self.mirror_prob)
        info = self._augment_info(info)
        if self._mirrored:
            obs = mirror_obs(obs, has_vision=self._has_vision)
        return obs, info

    def step(self, action):
        if self._mirrored:
            action = mirror_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._augment_info(info)
        if self._mirrored:
            obs = mirror_obs(obs, has_vision=self._has_vision)
        return obs, reward, terminated, truncated, info

    def _augment_info(self, info):
        """Flip spawn_angle / spawn_left to the policy's perceived hemisphere
        so downstream hemisphere-balance telemetry aligns with what the
        policy actually saw this episode."""
        info = dict(info) if info else {}
        info["mirrored"] = self._mirrored
        if self._mirrored:
            if "spawn_angle" in info:
                # Convention: 0=right, π/2=forward, π=left. Mirror → π-θ.
                info["spawn_angle"] = float(np.pi - info["spawn_angle"])
            if "spawn_left" in info:
                info["spawn_left"] = not bool(info["spawn_left"])
        return info
