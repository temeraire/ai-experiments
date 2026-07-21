"""ruler_check.py — can our bearing probe TELL ENCODERS APART? (make-or-break preflight)

THE PROBLEM. The probe we have been using is nearly saturated. A hand-coded 32-number
redness-per-column statistic with NO network reads gaze bearing at R2=0.919 on our 32x32 images, and
untrained trunks score 0.762-0.841. Against that, mildhead's 0.997 looked spectacular and R43's 0.828
looked like a null, but the whole comparison is squeezed into the top 8% of the scale. If we score a
matched-pair training experiment on this ruler, every arm lands near 0.9 and nine training runs teach
us nothing.

THE DIAGNOSIS (experiment-strategist, 2026-07-20). With 2000 probe-training samples and a
3072-dimensional input, the probe is not measuring "is bearing present" -- it is measuring "was
bearing DESTROYED". An identity function passes. What we actually care about is whether bearing is
EXPLICIT AND COMPACTLY READABLE, because that is what a policy has to read out.

THE FIX -- change the ruler, not the stimulus (a stimulus change would move representation difficulty,
task difficulty and winnability all at once, and we already lost time this morning to an unwinnable
contact threshold):
  1. EQUALISE READOUT CAPACITY. PCA every representation to the same 32 dims before the ridge.
     Otherwise the spread partly reflects input dimensionality, not information: mildhead's latent, a
     random trunk, and the 3072-d raw pixels currently get very different readout capacity.
  2. STARVE THE PROBE. Sweep n_train. A representation where bearing is EXPLICIT needs ~25 examples
     to expose it; one where it is merely RECOVERABLE needs hundreds. At n=2000 everything looks
     alike; at n=50 they separate. This is a sample-efficiency probe, and it measures the construct.
  3. REPORT DEGREES, not R2. Same information, non-saturating, and it is the plain-English form:
     "8.5 degrees versus 1.6 degrees" is legible; "0.919 versus 0.997" is not.

PASS (ruler is usable): mildhead separates cleanly from the random floors at small n_train -- we have
independent BEHAVIOURAL evidence its eyes carry direction (100% sighted vs ~1% blind, 3 seeds), so a
ruler that cannot reproduce that known separation is broken. Look for the mildhead-vs-raw-pixel gap to
WIDEN as n_train shrinks; that widening is the signature of measuring explicitness rather than mere
preservation.
FAIL: everything within ~2 deg at every n_train -> the ruler cannot discriminate, and the stimulus
must change (textured floor carrying red-ish content) before any compute is committed.
BROKEN: ordering inverts at low n (a random trunk beating mildhead) -> PCA is destroying the relevant
subspace, probe misconfigured.

  python -m alien_baby.crawler.ruler_check --frames 2500
"""
import argparse
import numpy as np, torch
from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP
from alien_baby.crawler.micoa_frame_control import (
    load_mildhead, load_micoa, load_random_crawler, CAM)

MILDHEAD = "alien_baby/results/mildhead_vis_s0_best/best_model.zip"
R43 = "alien_baby/results/phase_v_R43_micoa_vision_static_randbox_best/best_model.zip"
R49 = "alien_baby/results/phase_xvi_R49_micoa_vision_auxdecode_best/best_model.zip"


def redness_stat(obs_np):
    """The no-network baseline: per-COLUMN redness of the left eye, 32 numbers. This is the thing to
    beat -- if a trained encoder cannot beat a hand-coded statistic, it has learned nothing useful
    about where things are."""
    half = CAM * CAM * 3
    im = obs_np[:, PROP:PROP + half].reshape(-1, CAM, CAM, 3)
    red = im[..., 0] - 0.5 * (im[..., 1] + im[..., 2])      # redness per pixel
    return red.mean(axis=1)                                  # average down rows -> per column


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD; k components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k].T


def probe_degrees(X, b, n_train, k_pca=32, seed=0, n_test=400):
    """PCA to k dims on the TRAIN split only, ridge -> (sin,cos), report MEDIAN ABSOLUTE ANGULAR
    ERROR in degrees on held-out data. Returns (median_deg, r2_sin)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
    if len(te) < 50 or n_train < 10:
        return float("nan"), float("nan")
    Xtr_raw, Xte_raw = X[tr], X[te]
    mu, P = pca_fit(Xtr_raw, min(k_pca, Xtr_raw.shape[0] - 1, Xtr_raw.shape[1]))
    Xtr = (Xtr_raw - mu) @ P
    Xte = (Xte_raw - mu) @ P
    s, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - s) / sd, (Xte - s) / sd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
    Ytr = np.stack([np.sin(b[tr]), np.cos(b[tr])], 1)
    Yte = np.stack([np.sin(b[te]), np.cos(b[te])], 1)
    W = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    Pd = Xte @ W
    pred = np.arctan2(Pd[:, 0], Pd[:, 1])
    err = np.degrees(np.abs(np.arctan2(np.sin(pred - b[te]), np.cos(pred - b[te]))))
    r2s = 1.0 - ((Yte[:, 0] - Pd[:, 0]) ** 2).sum() / (((Yte[:, 0] - Yte[:, 0].mean()) ** 2).sum() + 1e-12)
    return float(np.median(err)), float(r2s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=2500)
    p.add_argument("--cone", type=float, default=0.9)
    p.add_argument("--pca-dims", type=int, default=32)
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sweep = [25, 50, 100, 200, 500, 2000]
    sweep = [n for n in sweep if n <= args.frames - 400]

    env = make_bearing_env(seed=args.seed, cone=args.cone, reach=1.0)
    print(f"collecting {args.frames} frames (cone +-{args.cone}, reach 1.0)...")
    obs_list, bears = [], []
    for i in range(args.frames):
        env.rng = np.random.default_rng(args.seed + i)
        o, _ = env.reset()
        obs_list.append(o); bears.append(env.gaze_bearing())
    O = np.array(obs_list, np.float32); b = np.array(bears)
    OT = torch.as_tensor(O)
    print(f"  bearing spread: std {b.std():.2f} rad, range {b.min():+.2f}..{b.max():+.2f}")

    reps = {}
    fn, n, exp, note = load_mildhead(MILDHEAD, env); reps["mildhead (LEARNED, seen-vs-touched)"] = (fn(OT), f"{n}/{exp} {note}")
    fn, n, exp, note = load_micoa(R43, which="mu");  reps["R43 mu_v (reward+MICOA)"] = (fn(OT), f"{n}/{exp} {note}")
    fn, n, exp, note = load_micoa(R49, which="mu");  reps["R49 mu_v (reward+MICOA+aux)"] = (fn(OT), f"{n}/{exp} {note}")
    reps["RAW PIXELS redness/col (no network)"] = (redness_stat(O), "hand-coded, 32 numbers")
    for s in range(args.random_seeds):
        torch.manual_seed(1000 + s)
        fn, _, _, _ = load_random_crawler(env)
        reps[f"__rand_crawler_{s}"] = (fn(OT), "random")
        torch.manual_seed(2000 + s)
        fn, _, _, _ = load_micoa(None, which="mu")
        reps[f"__rand_micoa_{s}"] = (fn(OT), "random")

    print(f"\n=== MEDIAN ABSOLUTE ANGULAR ERROR (degrees, lower=better), PCA-{args.pca_dims}, "
          f"held-out ===")
    hdr = "n_train:".ljust(38) + "".join(f"{n:>8}" for n in sweep)
    print(hdr); print("-" * len(hdr))
    named = [k for k in reps if not k.startswith("__")]
    out = {}
    for k in named:
        X, note = reps[k]
        row = [probe_degrees(X, b, n, args.pca_dims, seed=7)[0] for n in sweep]
        out[k] = row
        print(k.ljust(38) + "".join(f"{v:8.1f}" for v in row) + f"   [{note}]")
    for fam, pre in [("RANDOM crawler trunk", "__rand_crawler_"), ("RANDOM MICOA trunk", "__rand_micoa_")]:
        rows = np.array([[probe_degrees(reps[f"{pre}{s}"][0], b, n, args.pca_dims, seed=7)[0]
                          for n in sweep] for s in range(args.random_seeds)])
        out[fam] = rows.mean(0)
        print(f"{fam} (mean of {args.random_seeds})".ljust(38)
              + "".join(f"{v:8.1f}" for v in rows.mean(0)))
        print("   +- sd".ljust(38) + "".join(f"{v:8.1f}" for v in rows.std(0)))

    mh = np.array(out["mildhead (LEARNED, seen-vs-touched)"])
    rawp = np.array(out["RAW PIXELS redness/col (no network)"])
    floor = np.minimum(np.array(out["RANDOM crawler trunk"]), np.array(out["RANDOM MICOA trunk"]))
    best_floor = np.minimum(rawp, floor)
    print(f"\n=== VERDICT ===")
    for i, n in enumerate(sweep):
        print(f"  n_train={n:5d}: mildhead {mh[i]:5.1f}deg vs best floor {best_floor[i]:5.1f}deg  "
              f"-> gap {best_floor[i] - mh[i]:+5.1f}deg ({best_floor[i]/max(mh[i],1e-6):.1f}x)")
    small = sweep.index(50) if 50 in sweep else 0
    big = len(sweep) - 1
    widens = (best_floor[small] / max(mh[small], 1e-6)) > (best_floor[big] / max(mh[big], 1e-6))
    ok = (best_floor[small] - mh[small]) >= 3.0
    print(f"\n  separation at n_train={sweep[small]}: {best_floor[small]-mh[small]:+.1f} deg "
          f"(need >=3.0)   gap WIDENS as n shrinks: {widens}")
    print(f"  --> RULER {'USABLE - proceed to the matched-pair experiment' if ok else 'CANNOT DISCRIMINATE - change the stimulus before committing compute'}")


if __name__ == "__main__":
    main()
