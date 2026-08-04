"""Post-campaign report for peak-centred shards: did the representation do what
it was built to do, and where does it still fall short?

Six questions, in the order they can invalidate the campaign:

  1. WINDOW CENTRING. Every design's target is a window centred on where pass 1
     said the peak was. If the resolved peak lands near the window EDGE, the
     true peak may lie outside it entirely -- and nothing in the data would say
     so, because the shard records the best in-window value either way. This is
     the campaign's one silent failure mode, so it is checked first.
  2. DE-ALIASING GAIN. window peak|CD| / locate peak|CD|. The reason for
     spending N=3 on a narrow grid is that the coarse grid under-reads; this
     measures by how much, per design.
  3. Q DISTRIBUTION vs window coverage. Fits each window and classifies the
     resonance against what the window can resolve. A large "marginal" or
     "unresolved" share means the window is repeating the original aliasing
     problem at a finer scale.
  4. SCREEN behaviour: accept rate, and whether the fuzzy (+-14%) locate
     magnitude put anything near the threshold on the wrong side.
  5. LABEL-SWAP RISK: competing-lobe counts.
  6. PHYSICS SANITY: energy residuals.

Also merges several peak shards into one (X, Y) file for `scripts/cv_eval.py
--features-file`, since the learnability test wants every available design.

Usage:
    python -m scripts.peak_report --shards peak_pilot25 peak_known8
    python -m scripts.peak_report --shards peak_pilot25 --no-fit   # skip Fano
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_generation.dataset_compiler import _standardizer
from data_generation.parameter_sampler import PARAM_NAMES
from data_generation.peak_window import peak_recovery, resolvable_q_range
from data_generation.resonances import fit_spectrum

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
PROC = os.path.join(ROOT, "datasets", "processed")
ARTIFACTS = os.path.join(ROOT, "models", "artifacts")


def _pct(x) -> str:
    return f"{100 * x:.0f}%"


def load_shards(shards: list[str]) -> dict:
    """Concatenate the accepted designs of several peak shards."""
    acc: dict[str, list] = {}
    u = None
    for s in shards:
        d = np.load(os.path.join(RAW, f"{s}.npz"))
        if "window_cd" not in d.files:
            print(f"  (skipping {s}: no accepted designs yet)")
            continue
        if u is None:
            u = d["u"]
        elif not np.allclose(u, d["u"]):
            raise SystemExit(
                f"{s} uses a different relative axis; shards can only be merged "
                "when they share the window half-width and point count"
            )
        n = d["window_cd"].shape[0]
        keep = d["accepted"]
        for k in ("X", "f_peak", "window_cd", "window_T_RCP", "window_T_LCP",
                  "window_freqs", "window_energy_residual", "window_peak_abs_cd"):
            acc.setdefault(k, []).append(d[k])
        # Per-design pass-1 scalars exist only for designs that ran pass 1
        # (reprocess shards skip it); align them to the accepted rows.
        for k in ("locate_peak_abs_cd", "n_competing_lobes", "q_estimate", "origin"):
            v = d[k][keep] if d[k].shape[0] == keep.shape[0] else np.full(n, np.nan)
            acc.setdefault(k, []).append(v)
        acc.setdefault("shard", []).append(np.array([s] * n))
    if not acc:
        raise SystemExit("no accepted designs in any given shard")
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["u"] = u
    return out


def report_centring(D: dict, half_width: float) -> None:
    print("\n1. WINDOW CENTRING (the silent failure mode)")
    u = D["u"]
    at = np.array([u[int(np.argmax(np.abs(cd)))] for cd in D["window_cd"]])
    frac = np.abs(at) / half_width
    print(f"   |u_peak| / half_width: median {np.median(frac):.2f}, "
          f"p90 {np.quantile(frac, 0.9):.2f}, max {frac.max():.2f}")
    edge = int((frac >= 0.95).sum())
    print(f"   at the window EDGE (>=0.95): {edge}/{len(frac)}"
          + ("  <-- true peak may lie OUTSIDE the window for these" if edge else "  (none)"))
    far = int((frac >= 0.5).sum())
    print(f"   beyond half-way (>=0.50):    {far}/{len(frac)} ({_pct(far / len(frac))})")
    if edge:
        print("   ACTION: widen --half-width, or make the locate grid finer, before scaling up.")


def report_gain(D: dict) -> None:
    print("\n2. DE-ALIASING GAIN (window vs coarse locate pass)")
    loc, win = D["locate_peak_abs_cd"], D["window_peak_abs_cd"]
    ok = np.isfinite(loc) & (loc > 1e-9)
    if not ok.any():
        print("   (no pass-1 magnitudes in these shards)")
    else:
        g = win[ok] / loc[ok]
        print(f"   ratio over {ok.sum()} designs: median {np.median(g):.2f}, "
              f"p90 {np.quantile(g, 0.9):.2f}, max {g.max():.2f}, "
              f"under-read in {_pct((g > 1.05).mean())} of designs")
    print(f"   window peak|CD|: median {np.median(win):.3f}, p90 {np.quantile(win, 0.9):.3f}, "
          f"max {win.max():.3f}, min {win.min():.3f}")
    swing = np.array([cd.max() - cd.min() for cd in D["window_cd"]])
    print(f"   in-window CD swing (mode-pair signature): median {np.median(swing):.3f}, "
          f"max {swing.max():.3f}")


def report_q(D: dict, half_width: float, min_r2: float) -> None:
    n_pts = D["u"].size
    qr = resolvable_q_range(half_width, n_pts)
    print(f"\n3. Q vs WINDOW COVERAGE  (resolves [{qr.q_min:.0f}, {qr.q_max_resolved:.0f}], "
          f"detects to {qr.q_max_detectable:.0f})")
    classes: dict[str, int] = {}
    qs = []
    nofit = 0
    for freqs, trcp in zip(D["window_freqs"], D["window_T_RCP"]):
        lam = np.sort(500.0 / freqs)   # a_nm cancels in Q = lam0/gamma
        y = trcp[np.argsort(500.0 / freqs)]
        fits = [f for f in fit_spectrum(lam, y) if f.success and f.r2 > min_r2]
        if not fits:
            nofit += 1
            continue
        b = max(fits, key=lambda f: f.r2)
        qs.append(b.Q)
        classes[qr.classify(b.Q)] = classes.get(qr.classify(b.Q), 0) + 1
    qs = np.array(qs)
    if qs.size:
        print(f"   fitted Q (T_RCP, best fit per design): median {np.median(qs):.0f}, "
              f"p90 {np.quantile(qs, 0.9):.0f}, max {qs.max():.0f}")
    print(f"   classification: " + ", ".join(f"{k}={v}" for k, v in sorted(classes.items()))
          + f", no_fit={nofit}")

    # The classification is a conservative label; peak-height recovery is the
    # number that decides whether the window is actually adequate.
    if qs.size:
        rec = np.array([peak_recovery(q, half_width, n_pts) for q in qs])
        alt = np.array([peak_recovery(q, 2 * half_width / 3, n_pts) for q in qs])
        print(f"   worst-case peak-height recovery: median {np.median(rec):.2f}, "
              f"p10 {np.quantile(rec, 0.1):.2f}, min {rec.min():.2f}")
        print(f"   same Q at a +-{100 * 2 * half_width / 3:.0f}% window (same cost): "
              f"median {np.median(alt):.2f}, min {alt.min():.2f}")
        if np.median(rec) < 0.9:
            print("   -> consider narrowing --half-width for production: it costs nothing, "
                  "but check section 1's partner-lobe offsets first, since a narrower "
                  "window can cut the opposite-handedness mode that generates the CD.")


def report_screen_and_lobes(D: dict, shards: list[str]) -> None:
    print("\n4. SCREEN")
    for s in shards:
        path = os.path.join(RAW, f"{s}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        a = d["accepted"]
        print(f"   {s}: accepted {int(a.sum())}/{len(a)} ({_pct(a.mean())})")
        loc = d["locate_peak_abs_cd"]
        near = np.isfinite(loc) & (np.abs(loc - 0.20) < 0.03)
        if near.any():
            print(f"     {int(near.sum())} design(s) within +-0.03 of the threshold -- the "
                  "locate magnitude is only accurate to ~+-14%, so these could have gone "
                  "either way")

    print("\n5. LABEL-SWAP RISK")
    lb = D["n_competing_lobes"]
    ok = np.isfinite(lb) & (lb >= 0)
    if ok.any():
        print(f"   competing lobes: median {np.median(lb[ok]):.0f}, "
              f">=4 lobes in {int((lb[ok] >= 4).sum())}/{int(ok.sum())} designs")
        print("   (counts are grid-dependent -- a rough risk flag, not a measured property)")

    print("\n6. PHYSICS SANITY")
    res = D["window_energy_residual"]
    print(f"   max energy residual {res.max():.2e}, median {np.median(res.max(axis=1)):.2e}")
    if res.max() > 1e-3:
        print("   WARNING: >1e-3 -- check truncation (small gaps amplify fp32 error on GPU)")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--half-width", type=float, default=0.03)
    p.add_argument("--min-r2", type=float, default=0.9)
    p.add_argument("--no-fit", action="store_true", help="skip the Fano/Q section")
    p.add_argument("--out", default=None, help="write a merged (X, Y) npz for cv_eval --features-file")
    args = p.parse_args(argv)

    D = load_shards(args.shards)
    n = D["window_cd"].shape[0]
    print(f"=== Peak-campaign report: {n} accepted designs from {', '.join(args.shards)} ===")

    report_centring(D, args.half_width)
    report_gain(D)
    if not args.no_fit:
        report_q(D, args.half_width, args.min_r2)
    report_screen_and_lobes(D, args.shards)

    # Standardize X here, not upstream: the scalers on the individual shards
    # were each fitted to that shard's own designs, so they disagree. The merged
    # set needs one scaler fitted across all of it, matching the convention in
    # dataset_compiler.compile_dataset.
    x_mean, x_std = _standardizer(D["X"])
    out = args.out or os.path.join(PROC, f"peak_merged_{n}.npz")
    np.savez_compressed(
        out, X=(D["X"] - x_mean) / x_std, X_raw=D["X"], Y=D["window_cd"],
        u=D["u"], f_peak=D["f_peak"], peak_abs_cd=D["window_peak_abs_cd"],
        T_RCP=D["window_T_RCP"], T_LCP=D["window_T_LCP"], shard=D["shard"],
    )
    np.savez_compressed(
        out.replace(".npz", "_xscaler.npz"),
        x_mean=x_mean, x_std=x_std, param_names=np.array(PARAM_NAMES),
    )
    print(f"\nmerged (X, Y) -> {out}")
    print(f"  cv_eval:  python -m scripts.cv_eval --shard {os.path.basename(out)[:-4]} "
          f"--targets rel_cd peak_cd f_peak window_both_t --model gp")
    return {"n": n, "merged": out}


if __name__ == "__main__":
    main()
