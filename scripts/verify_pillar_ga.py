"""Re-run the best searched designs at higher N with more slices.

Usage: python -m scripts.verify_pillar_ga
"""
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from data_generation.pillar_bilayer import simulate_pillars
from scripts.pillar_ga import ROOT, profile_of

RAW = os.path.join(ROOT, "datasets", "raw")
OUT = os.path.join(RAW, "pillar_verify")


def load(mode):
    return [json.loads(s) for s in open(os.path.join(RAW, f"pillar_{mode}.jsonl"))]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ga = load("ga")
    jobs = [("ga", ga[k], (2, 3, 4) if k == 179 else (2, 3)) for k in (179, 276, 86)]
    jobs += [("random", load("random")[42], (2, 3)), ("ga", ga[0], (2, 3))]
    for mode, rec, Ns in jobs:
        x, c = rec["x"], rec["lam"]
        lam = np.unique(np.round(np.r_[np.arange(c - 15, c + 15.1, 1), np.arange(c - 3, c + 3.01, 0.25)], 3))
        for N in Ns:
            path = os.path.join(OUT, f"{mode}{rec['k']}_N{N}.npz")
            if not os.path.exists(path):
                r = simulate_pillars(lam, n_tio2=rec["n_tio2"], gap_nm=x[4], twist_deg=x[5], period_nm=x[0],
                                     profile=profile_of(x), N=N, n_cap=6, n_body=3)
                np.savez(path, lam=lam, T0=r.T0, T=r.T, R=r.R, residual=r.residual)
            d = np.load(path)
            s = d["T0"][:, 0] - d["T0"][:, 1]
            i = int(np.argmax(np.abs(s)))
            print(f"{mode}{rec['k']} N={N} |dT0| {abs(s[i]):.3f} CD {s[i] / d['T0'][i].sum():+.3f} at {d['lam'][i]:.2f} "
                  f"(edge {1.46 * x[0]:.1f}) T0 {d['T0'][i].round(3)} residual {d['residual'].max():.1e}", flush=True)
