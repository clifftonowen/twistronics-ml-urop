"""Compare a quarter-wave-gap shard against the sampled-gap peak-window set.

Gap d is in units of a and the gap layer is vacuum, so d/lambda at a design's
CD peak is gap * f_peak (f_peak = a/lambda). A quarter-wave gap is d/lambda = 0.25.

Usage:
    python -m scripts.compare_gap_qw --new peak_merged_12 --base peak_merged_71
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from data_generation.parameter_sampler import PARAM_NAMES
from data_generation.resonances import fit_spectrum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAP_COL = PARAM_NAMES.index("gap")


def load(name: str) -> dict:
    d = np.load(os.path.join(ROOT, "datasets", "processed", f"{name}.npz"), allow_pickle=True)
    f_peak = np.asarray(d["f_peak"], float)
    u = np.asarray(d["u"], float)
    freqs = f_peak[:, None] * (1.0 + u[None, :])
    return {
        "gap": np.asarray(d["X_raw"], float)[:, GAP_COL],
        "f_peak": f_peak,
        "peak_cd": np.asarray(d["peak_abs_cd"], float),
        "freqs": freqs,
        "T_RCP": np.asarray(d["T_RCP"], float),
    }


def fitted_q(D: dict, min_r2: float = 0.9) -> np.ndarray:
    qs = np.full(D["f_peak"].size, np.nan)
    for i, (f, t) in enumerate(zip(D["freqs"], D["T_RCP"])):
        order = np.argsort(500.0 / f)
        fits = [x for x in fit_spectrum((500.0 / f)[order], t[order]) if x.success and x.r2 > min_r2]
        if fits:
            qs[i] = max(fits, key=lambda x: x.r2).Q
    return qs


def summarize(label: str, D: dict) -> None:
    dl = D["gap"] * D["f_peak"]
    q = D["q"]
    print(f"{label}: n={dl.size}")
    print(f"  d/lambda      : median {np.median(dl):.3f}  range {dl.min():.3f}-{dl.max():.3f}")
    print(f"  peak|CD|      : median {np.median(D['peak_cd']):.3f}  IQR "
          f"{np.quantile(D['peak_cd'], .25):.3f}-{np.quantile(D['peak_cd'], .75):.3f}")
    if np.isfinite(q).any():
        print(f"  Q (fitted)    : median {np.nanmedian(q):.0f}  p90 {np.nanquantile(q, .9):.0f}  "
              f"fitted {np.isfinite(q).sum()}/{q.size}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new", default="peak_merged_12")
    p.add_argument("--base", default="peak_merged_71")
    p.add_argument("--fig", default=os.path.join(ROOT, "docs", "stage_c", "figures", "gap_quarter_wave.png"))
    args = p.parse_args(argv)

    new, base = load(args.new), load(args.base)
    new["q"], base["q"] = fitted_q(new), fitted_q(base)
    summarize("quarter-wave box", new)
    summarize("sampled gap (base)", base)

    u_cd = mannwhitneyu(new["peak_cd"], base["peak_cd"])
    print(f"\npeak|CD| new vs base: Mann-Whitney p={u_cd.pvalue:.3f}  "
          f"(median diff {np.median(new['peak_cd']) - np.median(base['peak_cd']):+.3f})")
    qn, qb = new["q"][np.isfinite(new["q"])], base["q"][np.isfinite(base["q"])]
    if qn.size > 2 and qb.size > 2:
        u_q = mannwhitneyu(qn, qb)
        print(f"Q new vs base:        Mann-Whitney p={u_q.pvalue:.3f}  "
              f"(median diff {np.median(qn) - np.median(qb):+.0f})")
    print("(n=12 vs 71: effect sizes matter more than p-values)")

    dl_all = np.concatenate([base["gap"] * base["f_peak"], new["gap"] * new["f_peak"]])
    cd_all = np.concatenate([base["peak_cd"], new["peak_cd"]])
    q_all = np.concatenate([base["q"], new["q"]])
    ok = np.isfinite(q_all)
    r1 = spearmanr(dl_all, cd_all)
    r2 = spearmanr(dl_all[ok], q_all[ok])
    print(f"\nacross all {dl_all.size}: Spearman(d/lambda, peak|CD|)={r1.statistic:+.2f} (p={r1.pvalue:.3f}); "
          f"Spearman(d/lambda, Q)={r2.statistic:+.2f} (p={r2.pvalue:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for a, yb, yn, lab in ((ax[0], base["peak_cd"], new["peak_cd"], "peak |CD|"),
                           (ax[1], base["q"], new["q"], "fitted Q")):
        a.scatter(base["gap"] * base["f_peak"], yb, s=18, alpha=.6, label="sampled gap (n=71)")
        a.scatter(new["gap"] * new["f_peak"], yn, s=28, color="tab:red", label="quarter-wave box (n=12)")
        a.axvline(0.25, ls="--", color="gray", lw=1)
        a.set_xlabel("d / lambda at CD peak")
        a.set_ylabel(lab)
    ax[0].legend(fontsize=8)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
