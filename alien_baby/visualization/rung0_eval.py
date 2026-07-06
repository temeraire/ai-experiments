#!/usr/bin/env python3
"""Rung 0 eval: steps-to-touch + hit-rate@horizon for a reach-to policy.

Primary metric is STEPS-TO-TOUCH (how directly the hand reaches the target) with
hit-rate-within-budget as the headline. Use on TEST (--target-obs) vs CONTROL
(blind) checkpoints to see whether a SENSED target produces a fast directed reach
vs an undirected flail.

Usage:
    python -m alien_baby.visualization.rung0_eval --run-tag rung0_TEST_targetobs
    python -m alien_baby.visualization.rung0_eval --run-tag rung0_TEST_targetobs \
        --checkpoint alien_baby/results/rung0_TEST_targetobs/rl_model_50000_steps.zip
"""
import argparse
import pathlib
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv

RESULTS = pathlib.Path(__file__).parent.parent / "results"
# Must match the rung-0 training config exactly.
ANCHOR = (0.0, 0.30)
BOX = (0.18, 0.08)
HORIZON = 50   # default; overridable with --horizon to match the training horizon


def _detect_target_obs(run_tag):
    t = run_tag.lower()
    if "control" in t or "blind" in t:
        return False
    return True  # TEST / targetobs / default


def _find_ckpt(run_tag, explicit):
    if explicit:
        return pathlib.Path(explicit)
    for p in (RESULTS / f"{run_tag}_best" / "best_model.zip",
              RESULTS / run_tag / "final_model.zip",
              RESULTS / run_tag / "best_model" / "best_model.zip"):
        if p.exists():
            return p
    raise SystemExit(f"No checkpoint for {run_tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--target-obs", dest="target_obs", action="store_true", default=None)
    ap.add_argument("--no-target-obs", dest="target_obs", action="store_false")
    ap.add_argument("--seed", type=int, default=30_000)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--actuate-hands", action="store_true")
    args = ap.parse_args()
    horizon = args.horizon

    target_obs = args.target_obs if args.target_obs is not None else _detect_target_obs(args.run_tag)

    def _mk():
        return MimoCrawlerCartEnv(
            vision=False, max_steps=horizon, strength_scale=1.0, n_substeps=4,
            cart_speed=0.0, fixed_ball_positions=[ANCHOR], random_ball_box=BOX,
            hip_actuation=False, approach_reward_scale=2.0, velocity_bonus_scale=0.05,
            target_obs=target_obs, hand_success=True, pin_targets=True,
            actuate_hands=args.actuate_hands)

    ckpt = _find_ckpt(args.run_tag, args.checkpoint)

    vec = DummyVecEnv([_mk])
    # Pair the checkpoint with its OWN vecnormalize when evaluating a mid-run
    # crawler_ckpt_<N>_steps.zip (so the learning curve uses matched obs stats).
    paired = ckpt.parent / ckpt.name.replace("crawler_ckpt_", "crawler_ckpt_vecnormalize_").replace(".zip", ".pkl")
    vn = paired if paired.exists() else RESULTS / args.run_tag / "vec_normalize.pkl"
    if vn.exists():
        vec = VecNormalize.load(str(vn), vec)
        vec.training = False
        vec.norm_reward = False
    model = SAC.load(str(ckpt), env=vec, device="cpu")
    raw = vec.venv.envs[0]

    # Contact-range = nearest hand body-center within this of the ball center.
    # A touch (geom overlap) registers around ~0.10 m body-center, so this is the
    # flail-resistant "did the hand get to the target" metric.
    CONTACT_RANGE = 0.10

    def _nearest_hand_dist():
        tgt = raw.data.qpos[raw._tgt1_qadr:raw._tgt1_qadr + 3][:2]
        rh = raw.data.xpos[raw._right_hand_bid][:2]
        lh = raw.data.xpos[raw._left_hand_bid][:2]
        return float(min(np.linalg.norm(tgt - rh), np.linalg.norm(tgt - lh)))

    steps_to_touch, steps_to_range = [], []
    hits, range_hits = 0, 0
    for i in range(args.episodes):
        obs, _ = raw.reset(seed=args.seed + i)
        obs = vec.normalize_obs(obs.reshape(1, -1))
        touched_step = range_step = None
        for t in range(horizon):
            a, _ = model.predict(obs, deterministic=True)
            o, r, term, trunc, info = raw.step(a[0])
            obs = vec.normalize_obs(o.reshape(1, -1))
            if range_step is None and _nearest_hand_dist() <= CONTACT_RANGE:
                range_step = t + 1
            if info["hand_touched_ball1"] and touched_step is None:
                touched_step = t + 1
            if term or trunc:
                break
        if touched_step is not None:
            hits += 1
            steps_to_touch.append(touched_step)
        if range_step is not None:
            range_hits += 1
            steps_to_range.append(range_step)

    n = args.episodes
    def _ms(x):
        return (float(np.mean(x)), float(np.median(x))) if x else (float("nan"), float("nan"))
    mst, medt = _ms(steps_to_touch)
    msr, medr = _ms(steps_to_range)
    print(f"\n=== RUNG 0 EVAL: {args.run_tag} ===")
    print(f"  checkpoint   : {ckpt.name}")
    print(f"  target_obs   : {target_obs}  (TEST=sees target / CONTROL=blind)")
    print(f"  episodes     : {n}  horizon={horizon}")
    print(f"  TOUCH@{horizon} (strict geom) : {hits}/{n} = {100*hits/n:.0f}%   steps mean {mst:.1f} median {medt:.1f}")
    print(f"  RANGE@{horizon} (<= {CONTACT_RANGE} m)  : {range_hits}/{n} = {100*range_hits/n:.0f}%   steps mean {msr:.1f} median {medr:.1f}")
    print(f"  (random baseline ~50% strict / ~31 steps; RANGE is the flail-resistant aim metric)")


if __name__ == "__main__":
    main()
