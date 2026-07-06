"""
Unit tests for MirrorWrapper. Covers:
  - Action/obs mirror involution (mirror∘mirror == identity).
  - Specific-value correctness on the transform (spot checks per component).
  - Shape preservation for both proprio-only and proprio+vision obs.
  - End-to-end rollout with the wrapper produces env-consistent info.

Run: `python -m alien_baby.tests.test_mirror_wrapper`.
"""

import numpy as np

from alien_baby.envs.mirror_wrapper import (
    MirrorWrapper,
    mirror_action,
    mirror_obs,
    PROPRIO_DIM,
    CAM_HEIGHT,
    CAM_WIDTH,
)
from alien_baby.envs.platform_creature_env import PlatformCreatureEnv


def _build_numbered_obs(include_vision=False):
    """Return a 29-vector where each entry is its own index, so permutations
    and sign-flips are obvious when inspected."""
    proprio = np.arange(PROPRIO_DIM, dtype=np.float32)
    if include_vision:
        pixels = np.arange(
            CAM_HEIGHT * CAM_WIDTH * 3, dtype=np.float32
        )
        return np.concatenate([proprio, pixels])
    return proprio


def test_action_involution():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
        assert np.allclose(a, mirror_action(mirror_action(a)))


def test_obs_involution_proprio():
    rng = np.random.default_rng(1)
    for _ in range(50):
        o = rng.uniform(-1.0, 1.0, size=PROPRIO_DIM).astype(np.float32)
        assert np.allclose(o, mirror_obs(mirror_obs(o)))


def test_obs_involution_with_vision():
    rng = np.random.default_rng(2)
    total = PROPRIO_DIM + CAM_HEIGHT * CAM_WIDTH * 3
    for _ in range(5):
        o = rng.uniform(0.0, 1.0, size=total).astype(np.float32)
        assert np.allclose(
            o, mirror_obs(mirror_obs(o, has_vision=True), has_vision=True)
        )


def test_action_specific_values():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    m = mirror_action(a)
    # Arms swap L/R; head_pan negates; head_tilt unchanged.
    assert np.allclose(m, [4, 5, 6, 1, 2, 3, -7, 8])


def test_obs_specific_values():
    o = _build_numbered_obs()
    m = mirror_obs(o)
    # Arm pos [0:6]: L=(0,1,2), R=(3,4,5) → swapped
    assert np.allclose(m[0:3], [3, 4, 5])
    assert np.allclose(m[3:6], [0, 1, 2])
    # Arm vel [6:12]: L=(6,7,8), R=(9,10,11) → swapped
    assert np.allclose(m[6:9], [9, 10, 11])
    assert np.allclose(m[9:12], [6, 7, 8])
    # head_pan (12) negated, head_tilt (13) unchanged
    assert m[12] == -12
    assert m[13] == 13
    # Quaternion (14,15,16,17) = (w,x,y,z) → (w,x,-y,-z)
    assert m[14] == 14 and m[15] == 15
    assert m[16] == -16 and m[17] == -17
    # Linear velocity (18,19,20) → (-18, 19, 20)
    assert m[18] == -18 and m[19] == 19 and m[20] == 20
    # Angular velocity (21,22,23) → (21, -22, -23)
    assert m[21] == 21 and m[22] == -22 and m[23] == -23
    # Touch (24,25,26): swap hands (24↔25), torso (26) unchanged
    assert m[24] == 25 and m[25] == 24 and m[26] == 26
    # Torso xy (27,28): x flips, y unchanged
    assert m[27] == -27 and m[28] == 28


def test_obs_pixel_column_flip():
    """Each row of pixels should be reversed along the width axis."""
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    pixels = np.arange(
        CAM_HEIGHT * CAM_WIDTH * 3, dtype=np.float32
    ).reshape(CAM_HEIGHT, CAM_WIDTH, 3)
    obs = np.concatenate([proprio, pixels.flatten()])
    m = mirror_obs(obs, has_vision=True)
    m_pixels = m[PROPRIO_DIM:].reshape(CAM_HEIGHT, CAM_WIDTH, 3)
    assert np.allclose(m_pixels, pixels[:, ::-1, :])


def test_wrapper_spaces_unchanged():
    env = PlatformCreatureEnv(vision=False, stage=1, v9=True)
    wrapped = MirrorWrapper(env)
    assert wrapped.observation_space.shape == env.observation_space.shape
    assert wrapped.action_space.shape == env.action_space.shape
    env.close()


def test_wrapper_reset_populates_info():
    env = PlatformCreatureEnv(vision=False, stage=1, v9=True)
    wrapped = MirrorWrapper(env)
    obs, info = wrapped.reset(seed=42)
    assert "mirrored" in info
    assert isinstance(info["mirrored"], bool)
    obs, r, term, trunc, info = wrapped.step(
        np.zeros(8, dtype=np.float32)
    )
    assert "mirrored" in info
    env.close()


def test_force_mirror_option():
    """options={'force_mirror': True/False} deterministically pins coin."""
    env = PlatformCreatureEnv(vision=False, stage=1, v9=True)
    wrapped = MirrorWrapper(env)
    _, info = wrapped.reset(seed=7, options={"force_mirror": True})
    assert info["mirrored"] is True
    _, info = wrapped.reset(seed=7, options={"force_mirror": False})
    assert info["mirrored"] is False
    env.close()


def test_mirrored_rollout_matches_mirrored_plain_rollout():
    """Physics-level sanity: for a mirror-symmetric env, running
    forced-mirror=True with zero actions should produce obs that equal the
    mirror of forced-mirror=False with zero actions from the same seed.

    This verifies: (a) _mirror_env_state correctly reflects initial qpos/qvel,
    (b) the env dynamics respect the symmetry (zero action → no asymmetric
    forcing), (c) mirror_obs on the env's evolved state equals what the
    wrapper returns in the mirrored episode.
    """
    plain_env = PlatformCreatureEnv(vision=False, stage=1, v9=True)
    wrapped_env = MirrorWrapper(
        PlatformCreatureEnv(vision=False, stage=1, v9=True)
    )

    obs_plain, _ = plain_env.reset(seed=123)
    obs_mirror, info_m = wrapped_env.reset(
        seed=123, options={"force_mirror": True}
    )
    assert info_m["mirrored"] is True

    # After reset, the wrapped env's obs should equal mirror(plain obs)
    # because both seeded the RNG identically (same spawn angle, same yaw)
    # and the wrapper reflected qpos/qvel.
    assert np.allclose(obs_mirror, mirror_obs(obs_plain), atol=1e-5), (
        "Mirrored reset obs != mirror(plain reset obs)"
    )

    zero_action = np.zeros(8, dtype=np.float32)
    for step_i in range(10):
        obs_plain, _, term_p, trunc_p, _ = plain_env.step(zero_action)
        obs_mirror, _, term_m, trunc_m, _ = wrapped_env.step(zero_action)
        assert term_p == term_m and trunc_p == trunc_m, (
            f"Termination mismatch at step {step_i}"
        )
        assert np.allclose(obs_mirror, mirror_obs(obs_plain), atol=1e-4), (
            f"Mirrored rollout diverges at step {step_i}: "
            f"max|Δ|={np.abs(obs_mirror - mirror_obs(obs_plain)).max():.2e}"
        )
        if term_p or trunc_p:
            break

    plain_env.close()
    wrapped_env.close()


def _run(test_fn):
    name = test_fn.__name__
    try:
        test_fn()
    except AssertionError as e:
        print(f"  FAIL {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return False
    print(f"  ok   {name}")
    return True


if __name__ == "__main__":
    tests = [
        test_action_involution,
        test_obs_involution_proprio,
        test_obs_involution_with_vision,
        test_action_specific_values,
        test_obs_specific_values,
        test_obs_pixel_column_flip,
        test_wrapper_spaces_unchanged,
        test_wrapper_reset_populates_info,
        test_force_mirror_option,
        test_mirrored_rollout_matches_mirrored_plain_rollout,
    ]
    results = [_run(t) for t in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(tests)} passed")
    if passed != len(tests):
        raise SystemExit(1)
