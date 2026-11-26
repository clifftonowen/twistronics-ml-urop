"""He et al. t = 350 and 410 nm with published wavelength-dependent TiO2 indices (no fitting).

Index files from refractiveindex.info (CC0) in datasets/raw/tio2_nk/. Absorption is
negligible in the band and ignored. Bodurov is fitted to 635 nm; 635-650 nm is extrapolated.

Usage: python -m scripts.moire_dispersive [--smoke]
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from data_generation.pillar_bilayer import simulate_pillars
from scripts.reproduce_moire import PILLAR_H, ROOT, TARGETS, TW, find_lobes, misfit

RAW = os.path.join(ROOT, "datasets", "raw")


def siefke(um):
    rows = []
    for s in open(os.path.join(RAW, "tio2_nk", "Siefke.yml"), encoding="utf8"):
        try:
            v = [float(q) for q in s.split()]
        except ValueError:
            continue
        if len(v) == 3:
            rows.append(v)
    a = np.array(rows)
    return np.interp(um, a[:, 0], a[:, 1])


INDEX = {"devore": lambda u: np.sqrt(5.913 + 0.2441 / (u ** 2 - 0.0803)),
         "siefke": siefke,
         "bodurov": lambda u: np.sqrt(1 + 4.6796 * u ** 2 / (u ** 2 - 0.2002148 ** 2))}

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    lam = (np.arange(600, 630, 10.) if smoke else
           np.unique(np.r_[np.arange(530, 590, 1.), np.arange(590, 640, .5), np.arange(640, 651, 1.)]))
    N = 1 if smoke else 3
    for name, f in INDEX.items():
        n = f(lam / 1000)
        print(name, "n at first / 610 / last:", n[0].round(3), f(0.61).round(3), n[-1].round(3), flush=True)
        for t in (350, 410):
            path = os.path.join(RAW, "moire_repro", f"disp_{name}_t{t}_N{N}.npz")
            t0 = time.time()
            if smoke or not os.path.exists(path):
                rs = [simulate_pillars([w], n_tio2=ni, gap_nm=t - PILLAR_H, twist_deg=TW, N=N, n_cap=6, n_body=3)
                      for w, ni in zip(lam, n)]
                d = {k: np.concatenate([getattr(r, k) for r in rs]) for k in ("T0", "T", "R", "residual")}
                if not smoke:
                    np.savez(path, lam=lam, n=n, **d)
            else:
                d = np.load(path)
            cd = (d["T0"][:, 0] - d["T0"][:, 1]) / d["T0"].sum(1)
            lb = find_lobes(lam, cd)
            print(f"  t={t} misfit {misfit(lb, TARGETS[t])} lobes {lb} residual {d['residual'].max():.1e} "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)
