"""Shape test for the He et al. reproduction: 220 nm pillars (SI diameter), index 2.50, N=3.

Usage: python -m scripts.moire_shape
"""
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from data_generation.pillar_bilayer import PAPER_PROFILE, simulate_pillars
from scripts.reproduce_moire import PILLAR_H, ROOT, TARGETS, TW, find_lobes, misfit

SHAPES = {"cyl220": ((0, 110.0), (150, 110.0), (230, 110.0)),
          "paper_x1.1": tuple((h, r * 1.1) for h, r in PAPER_PROFILE)}

if __name__ == "__main__":
    lam = np.unique(np.r_[np.arange(530, 590, 1.), np.arange(590, 640, .5), np.arange(640, 651, 1.)])
    for name, prof in SHAPES.items():
        for t in (350, 410):
            path = os.path.join(ROOT, "datasets", "raw", "moire_repro", f"shape_{name}_t{t}_N3.npz")
            t0 = time.time()
            if not os.path.exists(path):
                r = simulate_pillars(lam, n_tio2=2.5, gap_nm=t - PILLAR_H, twist_deg=TW, profile=prof, N=3,
                                     n_cap=6, n_body=3)
                np.savez(path, lam=lam, T0=r.T0, T=r.T, R=r.R, residual=r.residual)
            d = np.load(path)
            cd = (d["T0"][:, 0] - d["T0"][:, 1]) / d["T0"].sum(1)
            lb = find_lobes(lam, cd)
            print(f"{name} t={t} misfit {misfit(lb, TARGETS[t])} lobes {lb} residual {d['residual'].max():.1e} "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)
