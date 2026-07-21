"""prism_bearing_env.py — the REGIONAL DISSOCIATION apparatus (2026-07-20).

THE QUESTION. When a LEARNED visual encoder realigns under a prism, does the realignment stay
CONFINED to the region of the visual field the creature acted in, or does it spread across the whole
field? Taylor's Ch.9 Experiment II says local: the narrow strip of ground he actually WALKED ON came
back into correct alignment while the rest of the field stayed wrong ("the distortion has to be ironed
out bit by bit"). A convolutional trunk shares weights across the image and might well realign
globally. Our architecture does NOT force either answer, which is what makes this worth running --
unlike "does adaptation need movement", which our own aux-loss `if` statement would decide for us.

DESIGN. One ball per episode. The BAND it spawns in decides whether the creature can act on it:
    ACTED band     -- contact ENABLED. Touching fires the seen-vs-touched aux signal.
    SEEN-ONLY band -- contact DISABLED. Equally visible, for an equally long episode, never actable.
Episodes are FIXED LENGTH in both cases so retinal exposure is matched by construction (verify by
counting frames per band; do not assume it).

WHY ONE BALL AND NOT TWO. The earlier two-ball design was discarded: with both balls visible and only
one ever labelled, the encoder is under direct pressure to IGNORE the unlabelled one. "No realignment
in the seen-only band" would then be DISTRACTOR SUPPRESSION rather than Taylor's regional locality --
indistinguishable in the measurement, and pointing the same way as the hypothesis. That is the
dangerous kind of confound, so the distractor is gone.

THE LENS. Gaze-relative, ported from mimo_crawler_env._update_prism_ghost() per the GAZE rule: the
real ball's geom is hidden and a visual-only ghost is placed each step at the real ball's CAMERA-frame
azimuth shifted by +offset. That makes the displacement a constant RETINAL shift regardless of where
the creature looks or walks -- a real prism is fixed to the EYES, not to the world. The ghost cannot
be touched (contype=0), so contact always resolves against the REAL ball: seen and touched disagree by
exactly the offset, which is the whole mechanism.

THE PRISM-ON CUE. A binary flag is appended to the proprio block whenever the lens is active, per the
standing CLAUDE.md rule -- AB must know the glasses are on, the way a real creature always does.
Binary, never the offset angle: Taylor's cue signals THAT the lens is on, not how much it shifts;
handing over the degrees would be telling AB the answer instead of making it learn the correction.

HONEST LIMITS, both recorded because they bound what may be claimed:
  1. "Contact disabled" in the seen-only band is an artificial manipulation. It is our
     operationalisation of "seen but not acted upon", and must be described as such.
  2. The aux label (in the existing MismatchAuxCallback) is contact-GATED but NOT contact-DERIVED --
     it reads the ball's true bearing from qpos. We may say learning is restricted to episodes where
     the body actually touched; we may NOT say the creature learns only what its touching revealed.
"""
import numpy as np, mujoco
from alien_baby.crawler.train_vision_steer import VisionSteerEnv, CAM, PROP


class PrismBearingEnv(VisionSteerEnv):
    """VisionSteerEnv + gaze-relative prism + acted/seen-only bands + fixed-length episodes."""

    def __init__(self, prism_offset_deg=0.0, acted_band=+1, band_lo=0.25, band_hi=0.9,
                 fixed_len=60, prism_cue=True, seed=0, **kw):
        super().__init__(seed=seed, single_ball=True, cone=band_hi, **kw)
        self.prism_offset_deg = float(prism_offset_deg)
        self.acted_band = int(acted_band)      # +1 => LEFT bearings are the acted band, -1 => RIGHT
        self.band_lo, self.band_hi = float(band_lo), float(band_hi)
        self.fixed_len = int(fixed_len)
        self.prism_cue = bool(prism_cue)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ghost")
        self._ghost_mocap = int(self.model.body_mocapid[gid]) if gid >= 0 else -1
        self._target_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self._ghost_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ghost_geom")
        self._lens_on = abs(self.prism_offset_deg) > 1e-9
        # Lens ON: hide the real ball, show the ghost. Lens OFF: show the real ball, park the ghost
        # far below the floor so it is never in shot.
        self.model.geom_group[self._target_geom] = 3 if self._lens_on else 0
        self.model.geom_group[self._ghost_geom] = 0 if self._lens_on else 3
        if self.prism_cue:
            from gymnasium import spaces
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (self.observation_space.shape[0] + 1,), np.float32)

    # ---------------------------------------------------------------- the lens
    def _update_ghost(self):
        """Place the ghost at the real ball's CAMERA-frame azimuth + offset, preserving elevation and
        range. Called every step so the shift is a constant RETINAL displacement, not a world one."""
        if self._ghost_mocap < 0 or not self._lens_on:
            return
        cl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
        cr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")
        cam = 0.5 * (self.data.cam_xpos[cl] + self.data.cam_xpos[cr])
        M = self.data.cam_xmat[cl].reshape(3, 3)
        right, up, fwd = M[:, 0], M[:, 1], -M[:, 2]
        # SIGN: camera-frame azimuth grows toward the camera's RIGHT, while gaze_bearing() (our
        # readout frame, and the ruler's) is positive to the LEFT. Negate so that a POSITIVE
        # prism_offset_deg produces a POSITIVE measured seen-minus-true shift. Caught by gate P2,
        # which measured -16.7 deg for a nominal +20 (2026-07-20).
        off = -np.deg2rad(self.prism_offset_deg)
        v = self.data.qpos[15:18] - cam
        vf, vr, vu = v @ fwd, v @ right, v @ up
        h = np.hypot(vf, vr)
        az = np.arctan2(vr, vf) + off
        self.data.mocap_pos[self._ghost_mocap] = (
            cam + (h * np.cos(az)) * fwd + (h * np.sin(az)) * right + vu * up)

    def _obs(self):
        o = super()._obs()
        if self.prism_cue:
            o = np.concatenate([o, [1.0 if self._lens_on else 0.0]]).astype(np.float32)
        return o

    # ---------------------------------------------------------------- episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Re-place the ball inside a BAND (an annulus of bearings), not the full cone, and record
        # whether this episode is actable. Band sign is drawn per episode so both bands get equal
        # numbers of episodes and equal retinal exposure.
        M = np.zeros(9); mujoco.mju_quat2Mat(M, self.data.qpos[3:7]); M = M.reshape(3, 3)
        fwd = float(np.arctan2(M[1, 0], M[0, 0]))
        sign = 1 if self.rng.random() < 0.5 else -1
        a = sign * self.rng.uniform(self.band_lo, self.band_hi)
        r = self.rng.uniform(2.5, 4.0)
        self.data.qpos[15:22] = [r * np.cos(fwd + a), r * np.sin(fwd + a), 0.5, 1, 0, 0, 0]
        self.data.qpos[22:29] = [50, -50, 0.5, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self._update_ghost()
        mujoco.mj_forward(self.model, self.data)
        self.band = sign                                  # +1 left, -1 right
        self.actable = (sign == self.acted_band)          # contact ENABLED only in the acted band
        self.t = 0; self.prev = self._dist()
        return self._obs(), {}

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        self._update_ghost()
        # FIXED-LENGTH episodes: never terminate early, so retinal exposure is matched across bands
        # by construction. In the SEEN-ONLY band contact is disabled outright -- the ball is visible
        # for exactly as long, and simply cannot be acted upon.
        reached = bool(info["red"]) and self.actable
        info = {**info, "red": reached,
                "touched_ball1": reached,
                "ball1_bearing": self.gaze_bearing_true(),
                "band": self.band, "actable": self.actable, "lens_on": self._lens_on}
        term = False
        trunc = self.t >= self.fixed_len
        return self._obs(), rew, term, trunc, info

    # ---------------------------------------------------------------- measurement
    def gaze_bearing_true(self):
        """Bearing of the REAL ball (what touching reveals)."""
        return self.gaze_bearing()

    def gaze_bearing_seen(self):
        """Bearing of what the EYE sees -- the ghost when the lens is on, else the real ball. The
        difference between this and gaze_bearing_true() IS the prism offset, and realignment means
        the encoder's readout moving from the seen value toward the true one."""
        if not self._lens_on or self._ghost_mocap < 0:
            return self.gaze_bearing()
        cl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
        cr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")
        eye = 0.5 * (self.data.cam_xpos[cl] + self.data.cam_xpos[cr])
        f = -self.data.cam_xmat[cl].reshape(3, 3)[:, 2]
        f = f[:2] / (np.linalg.norm(f[:2]) + 1e-9)
        v = self.data.mocap_pos[self._ghost_mocap][:2] - eye[:2]
        v = v / (np.linalg.norm(v) + 1e-9)
        return float(np.arctan2(f[0] * v[1] - f[1] * v[0], f[0] * v[0] + f[1] * v[1]))

    def ghost_in_view(self):
        return self.red_in_view()
