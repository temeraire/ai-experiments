"""
probe_meaning.py — "meaning, not words" test (David, 2026-07-09).

Hypothesis: phrasing shouldn't matter; MEANING should. Represent each grounded bearing
concept not by ONE phrase but by a CLOUD of meaning-equivalent phrasings from several
lexical families (intensity words / clock / degrees / casual), and use the cloud CENTROID
as the concept's "meaning vector". If the earlier phrasing-dependence (-0.37 .. +0.58) was
lexical noise, averaging over families should (a) cancel it and (b) recover the meaning
geometry -> a stable positive RSA with AB's grounded bearing structure.

We report, per backend, RSA for each single family AND for the centroid-of-all-families,
so we can see whether the centroid is more stable and more positive than any single phrasing.

Usage: PYTHONPATH=<repo> python -u -m alien_baby.grounding.probe_meaning
"""
import numpy as np
from alien_baby.grounding.probe_align import (
    embed_mpnet, embed_openai, embed_qwen, embed_clip, cosine_rdm, perm_p, BEARING_ISO)

RESULTS = "alien_baby/results/grounding/ab_latents.npz"

# 6 bearing bins that have AB data (far-right was empty). Order = BEARING_ISO[0:6].
# Each family gives ONE phrasing per bin; the cloud is the union across families.
FAMILIES = {
 "intensity": ["a ball far to my left", "a ball to my left", "a ball slightly left",
               "a ball straight ahead", "a ball slightly right", "a ball to my right"],
 "clock":     ["a ball at my 9 o'clock", "a ball at my 10 o'clock", "a ball at my 11 o'clock",
               "a ball at my 12 o'clock", "a ball at my 1 o'clock", "a ball at my 2 o'clock"],
 "degrees":   ["a ball at -68 degrees", "a ball at -40 degrees", "a ball at -20 degrees",
               "a ball at 0 degrees", "a ball at 20 degrees", "a ball at 40 degrees"],
 "casual":    ["a ball way over on my left", "a ball off to the left", "a ball a little to the left",
               "a ball right in front of me", "a ball a little to the right", "a ball off to the right"],
}
NBIN = 6


def main():
    d = np.load(RESULTS)
    lat, th = d["latent128"], d["theta_deg"]
    keep = np.abs(th) < 68; lat, th = lat[keep], th[keep]
    ab = np.array([lat[(th >= lo) & (th < hi)].mean(0)
                   for _, lo, hi in BEARING_ISO[:NBIN]])
    ab_rdm = cosine_rdm(ab)

    backends = {"mpnet": embed_mpnet, "openai": embed_openai,
                "qwen": embed_qwen, "clip": embed_clip}
    print(f"AB bearing bins: {NBIN} (far-right empty, excluded)\n")
    print(f"{'backend':8s} " + "".join(f"{k:>11s}" for k in FAMILIES) + f"{'CENTROID':>13s}")
    for name, fn in backends.items():
        try:
            # embed every family's 6 phrases; stack -> [n_fam, NBIN, dim]
            fam_vecs = {k: fn(v) for k, v in FAMILIES.items()}
            cells = []
            for k, r, p in [(k, *perm_p(ab_rdm, cosine_rdm(fam_vecs[k]))) for k in FAMILIES]:
                cells.append(f"{r:+.2f}(p{p:.2f})")
            centroid = np.mean([fam_vecs[k] for k in FAMILIES], axis=0)  # meaning vector per bin
            rc, pc = perm_p(ab_rdm, cosine_rdm(centroid))
            print(f"{name:8s} " + "".join(f"{c:>11s}" for c in cells)
                  + f"{f'{rc:+.2f}(p{pc:.2f})':>13s}")
        except Exception as e:
            print(f"{name:8s} FAILED: {type(e).__name__}: {e}")

    print("\nRead: if the CENTROID column is consistently positive and more stable than the")
    print("spread across single families, 'meaning (averaged over phrasings)' aligns with AB")
    print("even where individual phrasings scatter — phrasing was lexical noise, meaning is the signal.")


if __name__ == "__main__":
    main()
