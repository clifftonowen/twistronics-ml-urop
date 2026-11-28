"""Summary of a pillar dataset: N=2 vs N=3 error, normal vs tilt, zeroth vs total CD, cost.

Usage: python -m scripts.pilot_report --name pillar_pilot40
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

from scripts.pillar_ga import ROOT


def dT(d, key="T0"):
    return d[key][:, 0] - d[key][:, 1]


def cd(d, key="T0"):
    t = d[key]
    return (t[:, 0] - t[:, 1]) / np.maximum(t.sum(1), 1e-12)


def win(lam, y, w=5.0):
    """Largest |mean over a w-nm window|."""
    h = int(round(w / (lam[1] - lam[0]) / 2))
    return max(abs(y[i - h:i + h + 1].mean()) for i in range(h, len(y) - h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pillar_pilot40")
    a = ap.parse_args()
    out = os.path.join(ROOT, "datasets", "raw", a.name)
    D = json.load(open(os.path.join(out, "designs.json")))
    geo, ang = np.array(D["geo"]), np.array(D["angles"])
    rows = []
    for k in range(len(geo)):
        n, t = (np.load(os.path.join(out, f"d{k:03d}_N2_{s}.npz")) for s in ("normal", "tilt"))
        lam = n["lam"]
        rows.append(dict(k=k, res=max(n["residual"].max(), t["residual"].max()), sec=n["seconds"] + t["seconds"],
                         dT_n=np.abs(dT(n)).max(), dT_t=np.abs(dT(t)).max(), w_n=win(lam, dT(n)), w_t=win(lam, dT(t)),
                         cd0_n=np.abs(cd(n)).max(), cdT_n=np.abs(cd(n, "T")).max(), theta=ang[k, 0],
                         shape_corr=np.corrcoef(dT(n), dT(t))[0, 1]))
    r = {k: np.array([q[k] for q in rows]) for k in rows[0]}
    print(f"{len(rows)} designs, max residual {r['res'].max():.1e}, mean {r['sec'].mean() / 60:.1f} min/design (2 spectra)")
    for k in ("dT_n", "dT_t", "w_n", "w_t", "cd0_n", "cdT_n"):
        print(f"  {k:6s} median {np.median(r[k]):.3f}  p90 {np.percentile(r[k], 90):.3f}  max {r[k].max():.3f}")
    print(f"  normal vs tilt 5nm-window ratio t/n: median {np.median(r['w_t'] / r['w_n']):.2f}; "
          f"spectrum correlation normal vs tilt median {np.median(r['shape_corr']):.2f}")
    rho = spearmanr(r["theta"], r["w_t"] / r["w_n"])
    print(f"  ratio vs theta: Spearman {rho.statistic:+.2f} (p {rho.pvalue:.2f})")
    for j, nm in enumerate(D["meta"]["names"]):
        s = spearmanr(geo[:, j], r["w_n"])
        print(f"  w_n vs {nm:7s}: Spearman {s.statistic:+.2f} (p {s.pvalue:.3f})")
    print("N=2 vs N=3 (normal incidence, on the N=3 1 nm grid):")
    for f in sorted(glob.glob(os.path.join(out, "*_N3_normal.npz"))):
        k = int(os.path.basename(f)[1:4])
        d3, d2 = np.load(f), np.load(os.path.join(out, f"d{k:03d}_N2_normal.npz"))
        m = np.isin(np.round(d2["lam"], 3), np.round(d3["lam"], 3))
        e = np.abs(dT(d2)[m] - dT(d3))
        e0 = np.abs(d2["T0"][m] - d3["T0"]).max()
        print(f"  d{k:03d} gap {geo[k, 4]:5.1f} twist {geo[k, 5]:4.1f}: max|dT diff| {e.max():.3f}, max|T0 diff| {e0:.3f}, "
              f"5nm score N2 {win(d3['lam'], dT(d2)[m]):.3f} N3 {win(d3['lam'], dT(d3)):.3f}, {d3['seconds'] / 60:.0f} min")
