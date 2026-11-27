"""Dataset of TiO2 pillar bilayers (He et al. 2025 geometry family), spread over the wide box.

Each design: Latin hypercube sample of (period, base radius, taper, height, gap, twist)
plus one incidence tilt (theta 0-8 deg in the n=1.46 medium, phi 0-90 deg). Spectra at
N=2 over 550-850 nm at 0.5 nm, at normal incidence and at the sampled tilt. Every
`check_every`-th design is also run at N=3 (normal incidence, 1 nm grid) to measure
the N=2 error. One .npz per design and per run, skipped if present, so a stopped run
resumes. Saved: zeroth-order and total T, R for RCP/LCP, energy residual, inputs.

Usage: python -m scripts.pillar_dataset --name pillar_pilot40 --n 40 [--smoke]
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.stats import qmc

from data_generation.pillar_bilayer import simulate_pillars
from scripts.pillar_ga import HI_WIDE, LO_WIDE, NAMES, ROOT, profile_of

N_TIO2 = 2.50          # fitted to He et al. at t = 350; equals anatase (Bodurov) at 610 nm
ANG_LO, ANG_HI = np.array([0.0, 0.0]), np.array([8.0, 90.0])


def sample(n, seed):
    u = qmc.LatinHypercube(8, rng=np.random.default_rng(seed)).random(n)
    geo = LO_WIDE + (HI_WIDE - LO_WIDE) * u[:, :6]
    ang = ANG_LO + (ANG_HI - ANG_LO) * u[:, 6:]
    return geo, ang


def run(path, x, lam, N, theta, phi, meta):
    if os.path.exists(path):
        return False
    t0 = time.time()
    r = simulate_pillars(lam, n_tio2=N_TIO2, gap_nm=x[4], twist_deg=x[5], period_nm=x[0], profile=profile_of(x),
                         N=N, n_cap=6, n_body=3, theta_deg=theta, phi_deg=phi)
    np.savez(path + ".tmp.npz", lam=lam, T0=r.T0, T=r.T, R=r.R, residual=r.residual, x=x,
             theta=theta, phi=phi, N=N, seconds=time.time() - t0, meta=json.dumps(meta))
    os.replace(path + ".tmp.npz", path)
    print(f"{os.path.basename(path)}: {time.time() - t0:.0f}s residual {np.nanmax(r.residual):.1e} "
          f"max|dT0| {np.abs(r.T0[:, 0] - r.T0[:, 1]).max():.3f}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "datasets", "raw", a.name)
    os.makedirs(out, exist_ok=True)
    lam2 = np.arange(550, 850.01, 25 if a.smoke else 0.5)
    lam3 = np.arange(550, 850.01, 50 if a.smoke else 1.0)
    geo, ang = sample(a.n, a.seed)
    meta = {"n_tio2": N_TIO2, "n_bg": 1.46, "names": list(NAMES), "box_lo": LO_WIDE.tolist(),
            "box_hi": HI_WIDE.tolist(), "angle_box": [ANG_LO.tolist(), ANG_HI.tolist()], "seed": a.seed,
            "theta_note": "theta in the n=1.46 medium; sin(theta_air) = 1.46 sin(theta)", "slices": "n_cap 6, n_body 3"}
    json.dump({"meta": meta, "geo": geo.tolist(), "angles": ang.tolist()}, open(os.path.join(out, "designs.json"), "w"),
              indent=1)
    for k in range(a.n):                      # all N=2 first, so a partial run is still usable
        run(os.path.join(out, f"d{k:03d}_N2_normal.npz"), geo[k], lam2, 2, 0.0, 0.0, meta)
        run(os.path.join(out, f"d{k:03d}_N2_tilt.npz"), geo[k], lam2, 2, *ang[k], meta)
    for k in range(0, a.n, a.check_every):
        run(os.path.join(out, f"d{k:03d}_N3_normal.npz"), geo[k], lam3, 3, 0.0, 0.0, meta)
    print("DONE", a.name, flush=True)


if __name__ == "__main__":
    main()
