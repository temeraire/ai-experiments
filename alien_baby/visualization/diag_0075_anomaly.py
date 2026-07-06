#!/usr/bin/env python3
"""Diagnostic for the Phase XV "0.075 anomaly".

R46 and R48 (independently trained) both showed a reproducible dip in both-touched
success at the TRAINED ball radius 0.075 (10/20 and 6/20), while smaller (0.053 ->
20/20) and LARGER (0.090 -> 15-16/20) sizes do fine. Non-monotonic, so not just
"big balls are hard". Both original evals used the SAME seed base (20000), so the
reproducibility might be a shared-seed geometry artifact rather than a real hole.

This script disambiguates by re-running the size neighbourhood at the ORIGINAL seed
and a FRESH seed, with more episodes:
  - if 0.075 recovers under the fresh seed  -> shared-seed EVAL ARTIFACT
  - if 0.075 stays low across both seeds     -> REAL policy hole (then render a video)

It also adds neighbour sizes 0.070 / 0.080 to test whether the dip is a knife-edge
at exactly 0.075 or a broad basin. Reuses the one-source-of-truth eval functions.

Usage:
    python -m alien_baby.visualization.diag_0075_anomaly --episodes 30
"""
import argparse
import numpy as np

from alien_baby.visualization.eval_phase_v import (
    build_env_and_model, run_episode, summarize_bin,
)

RUNS = {
    "R46": "phase_x_R46_droq_proprio_sizevariety",
    "R48": "phase_x3_R48_droq_proprio_size_and_shape",
}
SIZES = [0.053, 0.070, 0.075, 0.080, 0.090]
SEEDS = {"orig(20k)": 20_000, "fresh(70k)": 70_000}
ECC = 0.10           # matches the battery's DEFAULT_ECC for the size sweep
BALL_Y = 0.35


def _row(tag, size, label, s):
    return (f"  {tag} r={size:.3f} {label:>10} | both {s['both']:>3} "
            f"one {s['one']:>3} neith {s['neither']:>3} | "
            f"got_cl {s['got_close']:.2f} | both|cl {s['both_given_close']:>4.2f} | "
            f"mean_R {s['mean_R']:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()
    eps = args.episodes

    print(f"\n{'='*78}\n0.075-ANOMALY DIAGNOSTIC  (eps/bin={eps}, ecc={ECC}, ball_y={BALL_Y})")
    print("trained sizes: {0.040, 0.053, 0.075}; 0.075 is the largest TRAINED size")
    print(f"{'='*78}")

    summary = []  # (run, size, seedlabel, both, both_per_eps)
    for tag, run_tag in RUNS.items():
        print(f"\n#### {tag}  ({run_tag}) ####")
        for size in SIZES:
            # env/model build is seed-independent -> build once, run both seed batches
            vec_env, model, _ = build_env_and_model(
                run_tag, eccentricity=ECC, ball_speed=0.0,
                ball_radius=size, ball_y=BALL_Y)
            for slabel, sbase in SEEDS.items():
                results = [run_episode(sbase + i, vec_env, model) for i in range(eps)]
                s = summarize_bin(results, eps)
                print(_row(tag, size, slabel, s))
                summary.append((tag, size, slabel, s["both"], s["both"] / eps))
            vec_env.close()

    # verdict: compare 0.075 across seeds vs its neighbours
    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    for tag in RUNS:
        def rate(size, slabel):
            for t, sz, sl, _, r in summary:
                if t == tag and abs(sz - size) < 1e-9 and sl == slabel:
                    return r
            return float("nan")
        o75, f75 = rate(0.075, "orig(20k)"), rate(0.075, "fresh(70k)")
        nbr = np.nanmean([rate(s, sl) for s in (0.053, 0.090)
                          for sl in SEEDS])
        print(f"{tag}: 0.075 orig={o75:.2f} fresh={f75:.2f}  | "
              f"neighbours(0.053/0.090) avg={nbr:.2f}")
        if f75 >= nbr - 0.15:
            print(f"   -> fresh seed RECOVERS 0.075 ~to neighbour level "
                  f"=> shared-seed EVAL ARTIFACT (not a real policy hole)")
        elif min(o75, f75) < nbr - 0.20:
            print(f"   -> 0.075 stays low across BOTH seeds "
                  f"=> REAL reproducible hole at this size (render a video next)")
        else:
            print(f"   -> mixed/partial; treat as noisy, recommend more episodes")
    print("\nDone.")


if __name__ == "__main__":
    main()
