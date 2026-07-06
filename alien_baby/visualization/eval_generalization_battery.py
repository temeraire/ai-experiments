#!/usr/bin/env python3
"""Generalization battery (cheap, zero-shot, no-XML dimensions).

Tests the "generalization is the primary state" hypothesis (2026-05-30 discussion)
by sweeping object/relational dimensions AT TEST TIME on a frozen policy and asking
whether the response is INVARIANT (unchanged) or EQUIVARIANT (changes lawfully):

  1. SIZE  (--ball-radius)        -> INVARIANCE: touch rate should be ~flat.
  2. DISTANCE (forward ball-y)    -> EQUIVARIANCE: time-to-contact should scale
                                     ~linearly with distance and EXTRAPOLATE past
                                     the trained range (the "longer time prediction
                                     the farther away" claim, rendered as a number).
  3. RELATIVE SPEED (--ball-speed)-> EQUIVARIANCE/robustness: graceful degradation,
                                     not the cliff Phase H found at >=0.05.

All metrics are reach-conditional (got_close, both|close) so unreachable placements
are reported separately rather than averaged into a fake "generalization failure".
XML-dependent dimensions (color/shape/texture) are deliberately NOT here: those test
whether VISION generalized, and are only meaningful once vision is load-bearing.

Foundation (model loading, VecNormalize, hand->ball distance, ablation) is imported
from eval_phase_v so there is one source of truth.

Usage:
    python -m alien_baby.visualization.eval_generalization_battery \
        --run-tag phase_v_R43_micoa_vision_static_randbox
"""
import argparse
import numpy as np

from alien_baby.visualization.eval_phase_v import (
    build_env_and_model, run_episode, ablation_delta, summarize_bin,
    detect_modality,
)

DEFAULT_ECC = 0.10            # modest, reachable eccentricity for size/speed sweeps


def _print_table(title, rows, key, extra_cols):
    print(f"\n--- {title} ---")
    head = (f"{key:>8} | {'both':>4} {'one':>4} {'neith':>5} | {'mean_R':>8} | "
            f"{'reach':>6} | {'got_cl':>6} | {'both|cl':>7} | " +
            " ".join(f"{c:>8}" for c in extra_cols))
    print(head + "\n" + "-" * len(head))
    for r in rows:
        line = (f"{r[key]:>8.3f} | {r['both']:>4} {r['one']:>4} {r['neither']:>5} | "
                f"{r['mean_R']:>8.1f} | {r['reach']:>6.3f} | {r['got_close']:>6.2f} | "
                f"{r['both_given_close']:>7.2f} | " +
                " ".join(f"{r.get(c, float('nan')):>8.3f}" for c in extra_cols))
        print(line)


def sweep(run_tag, values, episodes, seed, *, kind, vision):
    """kind in {'size','distance','speed'}; returns list of summary dicts."""
    rows = []
    for v in values:
        kw = dict(eccentricity=DEFAULT_ECC, ball_speed=0.0, ball_radius=None, ball_y=0.35)
        if kind == "size":
            kw["ball_radius"] = v
        elif kind == "distance":
            kw["ball_y"] = v
            kw["eccentricity"] = 0.0       # pure forward distance
        elif kind == "speed":
            kw["ball_speed"] = v

        # Build env+model ONCE per swept value (config constant within a bin).
        vec_env, model, _ = build_env_and_model(run_tag, **kw)
        results = [run_episode(seed + i, vec_env, model) for i in range(episodes)]
        s = summarize_bin(results, episodes)
        s["abl"] = ablation_delta(vec_env, model) if vision else float("nan")
        vec_env.close()
        s["val"] = v
        rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--episodes-per-bin", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--sizes", default="0.035,0.053,0.070,0.090")
    ap.add_argument("--distances", default="0.25,0.35,0.45,0.55,0.65",
                    help="forward ball-y; trained range was ~[0.27,0.43] so >=0.55 is extrapolation")
    ap.add_argument("--speeds", default="0.0,0.01,0.02,0.03")
    args = ap.parse_args()

    vision, micoa = detect_modality(args.run_tag)
    eps = args.episodes_per_bin
    print(f"\n{'='*72}\nGENERALIZATION BATTERY: {args.run_tag}")
    print(f"  vision={vision} micoa={micoa}  eps/bin={eps}  ecc(size/speed)={DEFAULT_ECC}\n{'='*72}")

    # 1) SIZE — invariance
    sizes = [float(x) for x in args.sizes.split(",")]
    size_rows = sweep(args.run_tag, sizes, eps, args.seed, kind="size", vision=vision)
    _print_table("SIZE sweep (ball_radius) — expect INVARIANCE (flat both|cl)",
                 size_rows, "val", ["abl"])
    bc = [r["both_given_close"] for r in size_rows if np.isfinite(r["both_given_close"])]
    if len(bc) >= 2:
        spread = max(bc) - min(bc)
        print(f"  invariance check: both|close range = {spread:.2f}  "
              f"({'INVARIANT' if spread <= 0.25 else 'size-SENSITIVE'})")

    # 2) DISTANCE — equivariance (time-to-contact scales)
    dists = [float(x) for x in args.distances.split(",")]
    dist_rows = sweep(args.run_tag, dists, eps, args.seed + 7, kind="distance", vision=vision)
    _print_table("DISTANCE sweep (forward ball_y) — expect EQUIVARIANCE (contact_step rises)",
                 dist_rows, "val", ["mean_contact_step", "n_touched", "abl"])
    # quantify the time-to-contact-vs-distance law on bins that actually got touches
    pts = [(r["val"], r["mean_contact_step"]) for r in dist_rows
           if r["n_touched"] >= max(3, eps // 4) and np.isfinite(r["mean_contact_step"])]
    if len(pts) >= 3:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        corr = float(np.corrcoef(xs, ys)[0, 1])
        print(f"  time-to-contact law: steps ~= {slope:.0f}*dist + {intercept:.0f}  "
              f"(r={corr:+.2f})  over {len(pts)} reachable bins")
        print(f"  => {'LAWFUL (monotone, well-correlated)' if (slope > 0 and corr > 0.7) else 'NOT clearly lawful'}"
              f" : the farther the ball, the longer to reach it = distance generalization")
    else:
        print("  (too few reachable distance bins to fit a time-to-contact law)")

    # 3) RELATIVE SPEED — graceful degradation, not Phase H's cliff
    speeds = [float(x) for x in args.speeds.split(",")]
    speed_rows = sweep(args.run_tag, speeds, eps, args.seed + 13, kind="speed", vision=vision)
    _print_table("SPEED sweep (ball_speed) — expect graceful, not a cliff",
                 speed_rows, "val", ["abl"])
    bcs = [r["both_given_close"] for r in speed_rows if np.isfinite(r["both_given_close"])]
    if len(bcs) >= 2 and bcs[0] > 0:
        retention = bcs[-1] / bcs[0]
        print(f"  speed retention (fastest/slowest both|close) = {retention:.2f}  "
              f"({'graceful' if retention >= 0.5 else 'cliff'})")

    print("\n=== Battery read-out ===")
    print("Invariances (size): same response regardless of object size.")
    print("Equivariances (distance, speed): response changes LAWFULLY, not randomly.")
    if vision:
        print("abl columns show WHERE vision is recruited across each dimension.")
    print("Done.")


if __name__ == "__main__":
    main()
