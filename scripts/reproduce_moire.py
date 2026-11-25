"""Reproduce the simulated CD of He et al. (ACS Photonics 2025, Fig. 5b and 6a).

Cases are saved one by one to datasets/raw/moire_repro/ and skipped if present.
The TiO2 index is not given in the main text: it is fitted at t = 350 nm only, and
t = 410 nm is then predicted with that index held fixed.

Usage: python -m scripts.reproduce_moire [--smoke]
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from data_generation.pillar_bilayer import PAPER_PROFILE, simulate_pillars

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW = float(np.degrees(np.arctan2(5, 12)))
PILLAR_H = 230.0
# (wavelength nm, CD) of the four main lobes, read off Fig. 5b by eye
TARGETS = {350: ((602, 0.99), (608, -0.87), (623, 0.98), (629, -0.65)),
           410: ((606, 0.96), (610, -0.60), (622, 0.96), (627, -0.83))}


def find_lobes(lam, cd, floor=0.1):
    """Best ordered (+, -, +, -) run of local extrema in 585-650 nm, or None."""
    best, score = None, -1
    m = (lam >= 585) & (lam <= 650)
    l, c = lam[m], cd[m]
    ext = [i for i in range(1, len(c) - 1)
           if abs(c[i]) >= floor and (c[i] - c[i - 1]) * (c[i + 1] - c[i]) <= 0
           and c[i] * (2 * c[i] - c[i - 1] - c[i + 1]) > 0]
    # strongest extremum of each same-sign run
    runs = []
    for i in ext:
        if runs and np.sign(c[runs[-1]]) == np.sign(c[i]):
            if abs(c[i]) > abs(c[runs[-1]]):
                runs[-1] = i
        else:
            runs.append(i)
    for k in range(len(runs) - 3):
        q = runs[k:k + 4]
        if c[q[0]] > 0 and sum(abs(c[j]) for j in q) > score:
            best, score = q, sum(abs(c[j]) for j in q)
    return None if best is None else [(float(l[j]), float(c[j])) for j in best]


def misfit(lobes, target):
    """Mean of position error in units of 5 nm and height error in units of 0.15."""
    if lobes is None:
        return None
    return float(np.mean([abs(a[0] - b[0]) / 5 + abs(a[1] - b[1]) / 0.15
                          for a, b in zip(lobes, target)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", nargs="+", default=["fit", "ladder", "controls"])
    a = ap.parse_args()
    out = os.path.join(ROOT, "datasets", "raw", "moire_smoke" if a.smoke else "moire_repro")
    os.makedirs(out, exist_ok=True)
    if a.smoke:
        wide = np.arange(590, 640, 5.0)
        ns, Ns, Nfit = (2.4, 2.5), (1, 2), 1
    else:
        wide = np.unique(np.r_[np.arange(530, 590, 1.0), np.arange(590, 640, 0.5), np.arange(640, 651, 1.0)])
        ns, Ns, Nfit = (2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6), (1, 2, 3, 4), 3
    fine = wide[(wide >= 590) & (wide < 640)]

    def run(name, lam, **kw):
        path = os.path.join(out, name + ".npz")
        if not os.path.exists(path):
            t0 = time.time()
            kw = {"n_cap": 6, "n_body": 3, **kw}
            r = simulate_pillars(lam, **kw)
            np.savez(path + ".tmp.npz", lam=r.wavelengths_nm, T0=r.T0, T=r.T, R=r.R, residual=r.residual,
                     meta=json.dumps({k: v for k, v in kw.items()}))
            os.replace(path + ".tmp.npz", path)
            print(f"{name}: {time.time() - t0:.0f}s  residual {np.nanmax(r.residual):.1e}", flush=True)
        d = np.load(path)
        cd0 = (d["T0"][:, 0] - d["T0"][:, 1]) / d["T0"].sum(1)
        return d["lam"], cd0

    summary = {}
    sign = 1.0
    if "fit" in a.stages:
        scores = {}
        for n in ns:
            lam, cd = run(f"t350_n{n:.2f}_N{Nfit}", wide, n_tio2=n, gap_nm=350 - PILLAR_H, twist_deg=TW, N=Nfit)
            # handedness label is a convention: try both, fixed afterwards by the best fit only
            for sg in (1.0, -1.0):
                lb = find_lobes(lam, sg * cd)
                m = misfit(lb, TARGETS[350])
                print(f"  n={n:.2f} sign {sg:+.0f}: misfit {m}  lobes {lb}", flush=True)
                if m is not None:
                    scores[(n, sg)] = m
        if scores:
            n_fit, sign = min(scores, key=scores.get)
        else:
            n_fit, sign = 2.5, 1.0
        summary["fit"] = {"n": n_fit, "sign": sign, "misfit": scores.get((n_fit, sign))}
        print("FIT", summary["fit"], flush=True)
        lam, cd = run(f"t410_n{n_fit:.2f}_N{Nfit}", wide, n_tio2=n_fit, gap_nm=410 - PILLAR_H, twist_deg=TW, N=Nfit)
        lb = find_lobes(lam, sign * cd)
        summary["t410_prediction"] = {"lobes": lb, "misfit": misfit(lb, TARGETS[410])}
        print("t410 (not fitted)", summary["t410_prediction"], flush=True)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    summary = json.load(open(os.path.join(out, "summary.json")))
    n_fit, sign = summary["fit"]["n"], summary["fit"]["sign"]

    if "ladder" in a.stages:
        for t in (350, 410):
            for N in Ns:
                lam, cd = run(f"ladder_t{t}_n{n_fit:.2f}_N{N}", fine, n_tio2=n_fit, gap_nm=t - PILLAR_H,
                              twist_deg=TW, N=N)
                print(f"  ladder t={t} N={N}: {find_lobes(lam, sign * cd)}", flush=True)
    if "controls" in a.stages:
        base = dict(n_tio2=n_fit, N=min(Nfit, 3))
        cyl = tuple((h, 90.0) for h in (0, 150, 230))
        for name, kw in {"t1130": dict(gap_nm=900.0, twist_deg=TW),
                         "twist0": dict(gap_nm=120.0, twist_deg=0.0),
                         "twist45": dict(gap_nm=120.0, twist_deg=45.0),
                         "twist_neg": dict(gap_nm=120.0, twist_deg=-TW),
                         "cylinder": dict(gap_nm=120.0, twist_deg=TW, profile=cyl)}.items():
            lam, cd = run(f"control_{name}", wide, **base, **kw)
            print(f"  control {name}: max {sign * cd[np.argmax(np.abs(cd))]:+.3f} at {lam[np.argmax(np.abs(cd))]:.1f} nm", flush=True)
    print("DONE", a.stages, flush=True)


if __name__ == "__main__":
    main()
