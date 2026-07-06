"""Print one-line eval summary for a finished run (or all phase G runs)."""
import sys
import pathlib
import numpy as np

RESULTS = pathlib.Path(__file__).parent.parent / "results"


def summarize(tag):
    p = RESULTS / tag / "evaluations.npz"
    if not p.exists():
        print(f"  {tag}: no evaluations.npz")
        return
    d = np.load(p)
    ts, results, lengths = d["timesteps"], d["results"], d["ep_lengths"]
    print(f"\n=== {tag} ===")
    for i, t in enumerate(ts):
        n = len(results[i])
        floor = int((results[i] < -1500).sum())
        success = int((results[i] > 200).sum())
        print(f"  t={t:>6d}  reward={results[i].mean():>+8.1f}±{results[i].std():>6.1f}  "
              f"ep_len={lengths[i].mean():>7.1f}  floor={floor:>2d}/{n}  hit={success:>2d}/{n}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for tag in sys.argv[1:]:
            summarize(tag)
    else:
        for run_dir in sorted(RESULTS.glob("mimo_phase_g_cart_v*"), key=lambda p: p.stat().st_mtime):
            if (run_dir / "evaluations.npz").exists():
                summarize(run_dir.name)
