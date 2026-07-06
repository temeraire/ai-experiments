#!/usr/bin/env python3
"""Phase V evaluation foundation + eccentricity (distance/direction) generalization.

Built on the proven eval_phase_i.py loading pattern (DummyVecEnv + VecNormalize +
MICOASAC/SAC). This module is ALSO the shared foundation imported by
eval_generalization_battery.py — the tricky bits (model loading, obs
normalization, hand->ball distance, vision ablation) live here in ONE place.

Why a hand-rolled distance metric: the env's info dict only exposes
`min_dist_ball2` (not ball1), so we compute the minimum hand->ball distance
directly from MuJoCo state each step and track the episode minimum. This is
robust and symmetric across both balls.

Metrics, per eccentricity bin:
  - both / one / neither touched, mean reward
  - reach: mean episode-min hand->ball distance ("how close did it get?")
  - got_close: fraction of episodes whose min-dist < GOT_CLOSE_RADIUS
  - both|close: touch rate CONDITIONED on having gotten close (separates
    "couldn't get near the ball" from "got near and whiffed" — so a floor of
    unreachable placements never masquerades as a vision failure)
  - abl_L2: per-bin vision ablation (vision/micoa runs only); the
    generalization hypothesis predicts this RISES with eccentricity.

Usage:
    python -m alien_baby.visualization.eval_phase_v \
        --run-tag phase_v_R43_micoa_vision_static_randbox
"""
import argparse
import pathlib
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv, PROPRIO_DIM
from alien_baby.agents.micoa_architecture import MICOASAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

MAX_STEPS = 2000
# "got close" = hand came within this radius of a ball at some point. Touch
# contact is ~0.05-0.06; 0.12 means "brought a hand into striking range".
GOT_CLOSE_RADIUS = 0.12


def detect_modality(run_tag):
    vision = ("vision" in run_tag) or ("micoa" in run_tag)
    micoa = "micoa" in run_tag
    return vision, micoa


def _find_checkpoint(run_tag):
    cand = [
        RESULTS_DIR / run_tag / "final_model.zip",
        RESULTS_DIR / f"{run_tag}_best" / "best_model.zip",
        RESULTS_DIR / run_tag / "best_model" / "best_model.zip",
    ]
    for p in cand:
        if p.exists():
            return p
    raise SystemExit(f"No checkpoint found for {run_tag}. Tried:\n  " +
                     "\n  ".join(str(c) for c in cand))


def build_env_and_model(run_tag, eccentricity, ball_speed=0.0, ball_radius=None,
                        ball_y=0.35, ball_shape=None):
    """Fresh vec-normalized env + loaded model. Ball1 at (+ecc, ball_y),
    ball2 at (-ecc, -ball_y), FIXED so the swept dimension is controlled.

    ball_shape: if set (e.g. 'ellipsoid'), forces every episode to use that
    primitive by passing a single-element random_ball_shape list. Lets us
    eval zero-shot transfer to held-out shapes (Phase X2)."""
    vision, micoa = detect_modality(run_tag)

    # NB: the cart env does not take `cart_mode`/`micoa` kwargs (mirrors
    # eval_phase_i.py). `micoa` only selects the loader below.
    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=vision,
            strength_scale=1.0,
            max_steps=MAX_STEPS,
            cart_speed=0.15,
            ball_speed=ball_speed,
            ball_radius=ball_radius,
            random_ball_shape=[ball_shape] if ball_shape else None,
            hip_actuation=False,
            memory_obs=True,
            hunger_base=0.05,
            hunger_rate=0.20,
            hunger_scale=500.0,
            fixed_ball_positions=[(eccentricity, ball_y), (-eccentricity, -ball_y)],
        )

    raw_env = DummyVecEnv([_env_fn])
    vn_path = RESULTS_DIR / run_tag / "vec_normalize.pkl"
    vec_env = VecNormalize.load(str(vn_path), raw_env)
    vec_env.training = False
    vec_env.norm_reward = False

    ckpt = _find_checkpoint(run_tag)
    loader = MICOASAC if micoa else SAC
    model = loader.load(str(ckpt), env=vec_env, device="cpu")
    return vec_env, model, vision


def _hand_ball_min_dist(env):
    """Minimum 3D distance from either hand to either ACTIVE ball, read straight
    from MuJoCo state. Independent of touch state (unlike env._ball_dist)."""
    hands = []
    if getattr(env, "_right_hand_bid", None) is not None:
        hands.append(env.data.xpos[env._right_hand_bid])
    if getattr(env, "_left_hand_bid", None) is not None:
        hands.append(env.data.xpos[env._left_hand_bid])
    balls = [env.data.qpos[env._tgt1_qadr:env._tgt1_qadr + 3]]
    if getattr(env, "_ball2_active", False):
        balls.append(env.data.qpos[env._tgt2_qadr:env._tgt2_qadr + 3])
    if not hands or not balls:
        return np.inf
    return min(float(np.linalg.norm(h - b)) for h in hands for b in balls)


def run_episode(seed, vec_env, model):
    """One deterministic episode. Returns reward, touched1/2, episode-min
    hand->ball distance, and contact_step (first touch step, or None)."""
    raw = vec_env.venv.envs[0]
    obs, _ = raw.reset(seed=seed)
    obs = vec_env.normalize_obs(obs.reshape(1, -1))

    total_r = 0.0
    touched1 = touched2 = False
    min_d = np.inf
    contact_step = None

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs_raw, rew, term, trunc, info = raw.step(action[0])
        obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
        total_r += float(rew)
        min_d = min(min_d, _hand_ball_min_dist(raw))
        b1 = info.get("touched_ball1", False)
        b2 = info.get("touched_ball2", False)
        # Ignore step-0 contacts: a ball overlapping a body part at reset is a
        # spawn artifact, not a reach. (These produced spurious contact_step=0
        # bins that polluted the time-to-contact-vs-distance fit.)
        if (b1 or b2) and contact_step is None and step > 0:
            contact_step = step
        touched1 = touched1 or b1
        touched2 = touched2 or b2
        if term or trunc:
            break
    return total_r, touched1, touched2, min_d, contact_step


def ablation_delta(vec_env, model, n_steps=200, seed=9999):
    """Mean L2 action change when pixel columns are zeroed, over a rollout."""
    raw = vec_env.venv.envs[0]
    obs, _ = raw.reset(seed=seed)
    obs = vec_env.normalize_obs(obs.reshape(1, -1))
    deltas = []
    for _ in range(n_steps):
        a_full, _ = model.predict(obs, deterministic=True)
        blind = obs.copy()
        blind[:, PROPRIO_DIM + 2:] = 0.0      # +2 for memory_obs flags
        a_blind, _ = model.predict(blind, deterministic=True)
        deltas.append(float(np.linalg.norm(a_full - a_blind)))
        obs_raw, _, term, trunc, _ = raw.step(a_full[0])
        obs = vec_env.normalize_obs(obs_raw.reshape(1, -1))
        if term or trunc:
            obs, _ = raw.reset(seed=seed + 1)
            obs = vec_env.normalize_obs(obs.reshape(1, -1))
    return float(np.mean(deltas)) if deltas else float("nan")


def summarize_bin(results, episodes):
    """results: list of (reward, t1, t2, min_d, contact_step)."""
    b1 = np.array([r[1] for r in results])
    b2 = np.array([r[2] for r in results])
    rewards = np.array([r[0] for r in results])
    mind = np.array([r[3] for r in results])
    contact = [r[4] for r in results if r[4] is not None]

    both = int(np.sum(b1 & b2))
    one = int(np.sum(b1 ^ b2))
    neither = int(np.sum(~(b1 | b2)))
    close = mind < GOT_CLOSE_RADIUS
    n_close = int(close.sum())
    both_given_close = (int(np.sum((b1 & b2) & close)) / n_close
                        if n_close else float("nan"))
    return dict(
        both=both, one=one, neither=neither,
        mean_R=float(np.mean(rewards)),
        reach=float(np.mean(mind[np.isfinite(mind)])) if np.any(np.isfinite(mind)) else float("nan"),
        got_close=n_close / episodes,
        both_given_close=both_given_close,
        mean_contact_step=float(np.mean(contact)) if contact else float("nan"),
        n_touched=int(len(contact)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--eccentricities", default="0.0,0.05,0.10,0.15,0.20,0.25")
    ap.add_argument("--episodes-per-bin", type=int, default=20)
    ap.add_argument("--ball-speed", type=float, default=0.0)
    ap.add_argument("--ball-shape", default=None,
                    help="Force a single primitive shape for every episode "
                         "(sphere/box/cylinder/ellipsoid/capsule). Used to eval "
                         "zero-shot transfer to held-out shapes (Phase X2).")
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()

    eccs = [float(x) for x in args.eccentricities.split(",")]
    vision, micoa = detect_modality(args.run_tag)
    print(f"\n{'='*72}\nPHASE V EVAL (distance/direction): {args.run_tag}")
    print(f"  vision={vision} micoa={micoa} ball_speed={args.ball_speed} "
          f"ball_shape={args.ball_shape} eps/bin={args.episodes_per_bin}\n{'='*72}")

    header = (f"{'ecc':>5} | {'both':>4} {'one':>4} {'neith':>5} | {'mean_R':>8} | "
              f"{'reach':>6} | {'got_cl':>6} | {'both|cl':>7} | {'abl_L2':>7}")
    print("\n" + header + "\n" + "-" * len(header))

    rows = []
    for ecc in eccs:
        # Build env+model ONCE per bin (config is constant within a bin); reset
        # between episodes. Avoids reloading the checkpoint from disk per episode.
        vec_env, model, _ = build_env_and_model(args.run_tag, ecc, args.ball_speed,
                                                ball_shape=args.ball_shape)
        results = [run_episode(args.seed + i, vec_env, model)
                   for i in range(args.episodes_per_bin)]
        s = summarize_bin(results, args.episodes_per_bin)
        s["abl"] = ablation_delta(vec_env, model) if vision else float("nan")
        vec_env.close()
        s["ecc"] = ecc
        rows.append(s)
        print(f"{ecc:>5.2f} | {s['both']:>4} {s['one']:>4} {s['neither']:>5} | "
              f"{s['mean_R']:>8.1f} | {s['reach']:>6.3f} | {s['got_close']:>6.2f} | "
              f"{s['both_given_close']:>7.2f} | {s['abl']:>7.4f}")

    print("\n=== Hypothesis read-out ===")
    print("Predict: got_close stays high across bins (proprio approach generalizes);")
    print("         abl_L2 RISES with eccentricity (vision recruited at the margin).")
    if vision:
        abls = [r["abl"] for r in rows if np.isfinite(r["abl"])]
        if len(abls) >= 2:
            print(f"  abl: {abls[0]:.4f}@ecc{rows[0]['ecc']} -> {abls[-1]:.4f}@ecc{rows[-1]['ecc']}"
                  f"  => {'RISES' if abls[-1] > abls[0] else 'does NOT rise'}")
    print("Done.")


if __name__ == "__main__":
    main()
