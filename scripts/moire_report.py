"""Figures for the He et al. reproduction and the genetic search."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.reproduce_moire import ROOT, TARGETS

RAW = os.path.join(ROOT, "datasets", "raw")
FIG = os.path.join(ROOT, "docs", "stage_c", "figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    d = np.load(os.path.join(RAW, "moire_repro", name + ".npz"))
    return d["lam"], (d["T0"][:, 0] - d["T0"][:, 1]) / d["T0"].sum(1)


fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
for a, t in zip(ax, (350, 410)):
    a.axhline(0, color="#999", lw=0.8)
    a.plot(*load(f"t{t}_n2.50_N3"), color="#2a78d6", lw=2, label="RCWA, N=3")
    a.plot(*load(f"ladder_t{t}_n2.50_N4"), color="#eb6834", lw=1.5, ls="--", label="RCWA, N=4 (590-640 nm only)")
    x, y = zip(*TARGETS[t])
    a.plot(x, y, "o", mfc="white", mec="#222", mew=1.8, ms=8, label="paper's simulation (read by eye)")
    note = "index 2.50 fitted here" if t == 350 else "not fitted, index fixed at 2.50"
    a.set(ylim=(-1.1, 1.1), ylabel="CD (zeroth order)", title=f"t = {t} nm: {note}")
ax[0].legend(frameon=False, loc="lower left", fontsize=9)
ax[1].set_xlabel("wavelength (nm)")
fig.savefig(os.path.join(FIG, "moire_repro.png"), dpi=180)

fig, a = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
for m, c, lb in (("random", "#eb6834", "random search"), ("ga", "#2a78d6", "genetic search")):
    f = np.array([json.loads(s)["fitness"] for s in open(os.path.join(RAW, f"pillar_{m}.jsonl"))])
    k = np.arange(1, len(f) + 1)
    a.plot(k, f, ".", color=c, alpha=0.35, ms=5)
    a.step(k, np.maximum.accumulate(f), where="post", color=c, lw=2, label=f"{lb}: best so far ({f.max():.2f})")
a.axhline(f[0], color="#777", ls=":", lw=1.5, label=f"paper's design ({f[0]:.2f})")
a.set(xlabel="designs simulated (N=2)", ylabel="largest |T_RCP - T_LCP|", ylim=(0, 1), title="Search over pillar bilayers, 550-700 nm")
a.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
fig.savefig(os.path.join(FIG, "pillar_search.png"), dpi=180)
