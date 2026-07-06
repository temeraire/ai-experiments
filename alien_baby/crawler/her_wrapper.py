"""
her_wrapper.py — Goal-conditioned wrapper around MimoCrawlerEnv for HER training.

Hindsight Experience Replay (Andrychowicz et al. 2017) requires:
  - Dict observation space with keys: observation, achieved_goal, desired_goal
  - A compute_reward(achieved_goal, desired_goal, info) method that produces the
    reward as a pure function of state. This lets HER relabel transitions: take
    a failed episode and pretend the goal was wherever the agent ended up.

For our task:
  - desired_goal = the ball's XY world position
  - achieved_goal = the hip's XY world position
  - reward = +1 if |achieved - desired| < threshold (~ball touch radius), else 0
  - We use sparse positive reward (1/0) per HER convention. SAC handles it fine.

The "observation" key carries everything else (proprio joints, vestibular,
memory flags, optionally pixels). The original Box observation is split: hip
position goes into achieved_goal, ball position into desired_goal, everything
else into observation.

Two-ball mode: HER's standard formulation is for a single goal. For two balls,
we use desired_goal = the FIRST untouched ball's position (switches to ball2
once ball1 is touched), and achieved_goal = hip XY. This naturally creates a
curriculum: first learn to reach ball1, then ball2.

Crucially, the wrapped env returns a Box-shaped observation key, NOT a flat
vector — SB3's HerReplayBuffer needs the Dict structure.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, PLATFORM_TOP_Z


# Distance under which the "achieved" hip is close enough to count as success.
# Slightly larger than ball touch radius (0.053m) to give a more forgiving
# learning signal — the goal is reached when the agent gets close.
GOAL_THRESHOLD = 0.12  # metres


class HERCrawlerWrapper(gym.Env):
    """Goal-conditioned wrapper. Inner env is a MimoCrawlerEnv.

    Observation becomes a Dict:
        observation   : everything except hip_xy and ball positions
                        (full proprio minus 2 dims, plus memory flags,
                         optionally plus pixel obs)
        achieved_goal : hip XY world position (2 dims)
        desired_goal  : XY of the currently-pursued ball (2 dims)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, **inner_kwargs):
        super().__init__()
        # Force fixed_ball_positions: HER goal is meaningful only when the ball
        # has a stable target position. Random spawn invalidates the goal idea.
        if inner_kwargs.get("fixed_ball_positions") is None:
            raise ValueError(
                "HERCrawlerWrapper requires fixed_ball_positions; HER needs "
                "a stable desired_goal across episodes."
            )
        self.inner = MimoCrawlerEnv(**inner_kwargs)

        # Determine the layout of the inner observation
        self._inner_obs_dim = int(self.inner.observation_space.shape[0])
        # First 2 of proprio are hip XY (root_pos x, y). z is dim 2.
        # We strip hip_xy from observation and expose it as achieved_goal.
        # That leaves inner_obs_dim - 2 dims in the "observation" key.
        observation_dim = self._inner_obs_dim - 2

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(observation_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32),
            "desired_goal":  spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32),
        })
        self.action_space = self.inner.action_space

        # Bookkeeping
        self._desired_goal = np.zeros(2, dtype=np.float32)
        self._step = 0

    # ------------------------------------------------------------------
    def _hip_xy(self) -> np.ndarray:
        root_qadr = self.inner.model.jnt_qposadr[self.inner._root_joint_id]
        return self.inner.data.qpos[root_qadr:root_qadr + 2].astype(np.float32).copy()

    def _ball_xy(self, idx: int) -> np.ndarray:
        name = "target_free" if idx == 0 else "target2_free"
        jid = mujoco.mj_name2id(self.inner.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = self.inner.model.jnt_qposadr[jid]
        return self.inner.data.qpos[qadr:qadr + 2].astype(np.float32).copy()

    def _current_desired_goal(self) -> np.ndarray:
        """Whichever ball is currently the active target.
        Single-ball mode → always ball1.
        Two-ball mode → ball1 until touched, then ball2.
        """
        if not self.inner._ball2_active:
            return self._ball_xy(0)
        if not self.inner._ball1_touched:
            return self._ball_xy(0)
        return self._ball_xy(1)

    def _strip_hip_xy(self, inner_obs: np.ndarray) -> np.ndarray:
        # Inner obs starts with [root_pos (3), ...]. We drop indices 0 and 1
        # (hip x, y) but keep z and everything else.
        return np.concatenate([inner_obs[2:3], inner_obs[3:]]).astype(np.float32)

    def _build_obs(self, inner_obs: np.ndarray) -> dict:
        return {
            "observation":   self._strip_hip_xy(inner_obs),
            "achieved_goal": self._hip_xy(),
            "desired_goal":  self._current_desired_goal(),
        }

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        inner_obs, info = self.inner.reset(seed=seed, options=options)
        self._step = 0
        self._desired_goal = self._current_desired_goal()
        return self._build_obs(inner_obs), info

    def step(self, action):
        inner_obs, inner_reward, terminated, truncated, info = self.inner.step(action)
        self._step += 1

        achieved = self._hip_xy()
        desired = self._current_desired_goal()
        self._desired_goal = desired

        # Reward = HER sparse goal signal PLUS inner env's enabling-pressure
        # rewards (velocity bonus, step cost). The HER signal is the canonical
        # 0/-1 "are you there yet" Andrychowicz et al. relabel-able term; the
        # inner reward carries velocity_bonus_scale × hip_speed so stillness
        # is costly. Without this composition the inner env's velocity bonus
        # is silently discarded by HER, which is why Phases D and E flatlined
        # despite --velocity-bonus-scale > 0.
        #
        # CAVEAT: HerReplayBuffer recomputes the reward via compute_reward()
        # for the 4/5 of samples it relabels with hindsight goals, so those
        # samples see only the sparse term — the velocity bonus only applies
        # to the 1/5 of samples that retain the original goal. If the bonus
        # turns out to be insufficiently strong with this dilution, the next
        # iteration is to bake hip_speed into info and have compute_reward
        # read it back so relabeled samples carry the bonus too.
        reward = self.compute_reward(achieved, desired, info) + inner_reward

        obs = self._build_obs(inner_obs)
        info["achieved_goal"] = achieved
        info["desired_goal"]  = desired
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse goal-conditioned reward. Works on single or batched goals.

        Returns 0.0 when within GOAL_THRESHOLD of the desired goal, else -1.0.
        SB3's HerReplayBuffer relies on this method being PURE — given any
        (achieved, desired) pair it must produce the same reward regardless
        of when in the episode it was actually achieved.
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired  = np.asarray(desired_goal,  dtype=np.float32)
        d = np.linalg.norm(achieved - desired, axis=-1)
        return np.where(d < GOAL_THRESHOLD, 0.0, -1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def render(self):
        return self.inner.render()

    def close(self):
        return self.inner.close()
