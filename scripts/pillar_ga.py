"""Genetic search over twisted pillar bilayers, with a random-search comparison.

Fitness is the largest |T_RCP - T_LCP| (zeroth order, N=2) in 550-700 nm: CD alone
can reach 1 where almost no light gets through. Both searches start from the same
first generation. Every design is appended to a JSONL file; a rerun replays that
file through the same seeded generator, so a stopped search resumes exactly.

Usage: python -m scripts.pillar_ga --n-tio2 2.5 [--mode ga|random] [--hours 3.5]
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.stats import qmc

from data_generation.pillar_bilayer import PAPER_PROFILE, simulate_pillars

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ("period", "base_r", "taper", "height", "gap", "twist")
LO = np.array([360, 70, 0.80, 150, 50, 15.0])
HI = np.array([440, 130, 1.00, 300, 250, 30.0])
PAPER = np.array([400, 100, 0.90, 230, 120, 22.6199])


def profile_of(x):
    """Paper shape with base radius, body taper and height rescaled."""
    _, base, taper, height, _, _ = x
    rows = [(0.0, base)]
    for h, r in PAPER_PROFILE[1:]:
        rows.append((h / 230 * height, base * taper * r / 90))
    return tuple(rows)


LO_WIDE = np.array([340, 60, 0.70, 120, 20, 10.0])
HI_WIDE = np.array([480, 150, 1.00, 320, 300, 40.0])
ANGLES = ((0.0, 0.0), (5.0, 0.0), (5.0, 45.0))   # (theta in the n=1.46 medium, phi), degrees


def evaluate_robust(x, n_tio2, N, coarse, window, angles=ANGLES, margin=5.0, n_cand=2):
    """Worst case over incidence angles of the window-averaged signed T_RCP - T_LCP.

    A 2 nm scan at normal incidence picks the n_cand strongest windows (at least
    `window` apart, `margin` nm clear of the normal-incidence diffraction edge). Each
    is re-scanned at 0.5 nm at every angle; a window scores the smallest |mean| over
    angles, and 0 if the sign changes between angles. 5 deg in the medium is about
    7.3 deg in air.
    """
    kw = dict(n_tio2=n_tio2, gap_nm=x[4], twist_deg=x[5], period_nm=x[0], profile=profile_of(x), N=N)
    edge = 1.46 * x[0]
    lam = np.arange(550, 700.1, coarse)
    r = simulate_pillars(lam, **kw)
    d = r.T0[:, 0] - r.T0[:, 1]
    m = max(1, int(round(window / coarse / 2)))
    idx = np.arange(m, len(lam) - m)
    idx = idx[np.abs(lam[idx] - edge) >= window / 2 + margin + 4]
    mean = np.array([d[i - m:i + m + 1].mean() for i in idx])
    cands = []
    for j in np.argsort(-np.abs(mean)):
        if all(abs(lam[idx[j]] - c) >= window for c in cands):
            cands.append(lam[idx[j]])
        if len(cands) == n_cand:
            break
    best = {"fitness": 0.0, "lam": float(cands[0]) if cands else 0.0, "cd0": 0.0, "per_angle": [], "T0": [0.0, 0.0]}
    h = int(round(window / 0.5 / 2))
    for c in cands:
        fl = c + np.arange(-window / 2 - 4, window / 2 + 4.01, 0.5)
        runs = [simulate_pillars(fl, theta_deg=t, phi_deg=p, **kw) for t, p in angles]
        res = max(float(q.residual.max()) for q in runs)
        ds = np.array([q.T0[:, 0] - q.T0[:, 1] for q in runs])                # (angles, wavelengths)
        for i in range(h, len(fl) - h):
            if abs(fl[i] - edge) < window / 2 + margin:
                continue
            mu = ds[:, i - h:i + h + 1].mean(1)
            score = 0.0 if (np.sign(mu) != np.sign(mu[0])).any() or res > 1e-3 else float(np.abs(mu).min())
            if score > best["fitness"]:
                t = runs[0].T0[i - h:i + h + 1].mean(0)
                best = {"fitness": score, "lam": float(fl[i]), "cd0": float((t[0] - t[1]) / t.sum()),
                        "per_angle": mu.tolist(), "T0": t.tolist(), "residual": res}
    best.update(edge=float(edge), coarse_T0=r.T0.round(5).tolist(), angles=[list(a) for a in angles])
    return best


def evaluate_window(x, n_tio2, N, coarse, window, margin=5.0):
    """Signed zeroth-order T_RCP - T_LCP averaged over `window` nm, best window wins.

    Windows closer than `margin` nm to the lattice diffraction edge (1.46 * period)
    are not allowed, because responses there are narrow cusps.
    """
    kw = dict(n_tio2=n_tio2, gap_nm=x[4], twist_deg=x[5], period_nm=x[0], profile=profile_of(x), N=N)
    edge = 1.46 * x[0]

    def best(lam, step):
        r = simulate_pillars(lam, **kw)
        ok = np.isfinite(r.residual) & (r.residual < 1e-3)
        d = np.where(ok, r.T0[:, 0] - r.T0[:, 1], 0.0)
        m = max(1, int(round(window / step / 2)))
        idx = np.arange(m, len(lam) - m)
        half = max(m * step, window / 2)
        idx = idx[np.abs(lam[idx] - edge) >= half + margin]
        if len(idx) == 0:
            return r, None, 0.0
        mean = np.array([d[i - m:i + m + 1].mean() for i in idx])
        j = int(np.argmax(np.abs(mean)))
        return r, (idx[j], m), float(abs(mean[j]))

    r, hit, _ = best(np.arange(550, 700.1, coarse), coarse)
    half = window / 2
    c = r.wavelengths_nm[hit[0]]
    f, hit2, fit = best(c + np.arange(-half - 4, half + 4.01, 0.5), 0.5)
    if hit2 is None:
        return {"fitness": 0.0, "lam": float(c), "cd0": 0.0, "T0": [0.0, 0.0], "residual": float(f.residual.max()),
                "edge": float(edge), "coarse_T0": r.T0.round(5).tolist()}
    i, m = hit2
    t = f.T0[i - m:i + m + 1].mean(0)
    return {"fitness": fit, "lam": float(f.wavelengths_nm[i]), "cd0": float((t[0] - t[1]) / t.sum()),
            "T0": t.tolist(), "residual": float(f.residual[i - m:i + m + 1].max()), "edge": float(edge),
            "coarse_T0": r.T0.round(5).tolist()}


def evaluate(x, n_tio2, N, coarse):
    kw = dict(n_tio2=n_tio2, gap_nm=x[4], twist_deg=x[5], period_nm=x[0], profile=profile_of(x), N=N)

    def score(lam):
        r = simulate_pillars(lam, **kw)
        ok = np.isfinite(r.residual) & (r.residual < 1e-3)
        return r, np.where(ok, np.abs(r.T0[:, 0] - r.T0[:, 1]), 0.0)

    r, s = score(np.arange(550, 700.1, coarse))
    c = r.wavelengths_nm[int(np.argmax(s))]
    r2, s2 = score(c + np.arange(-coarse, coarse + 0.01, coarse / 4))
    r, s = (r2, s2) if s2.max() >= s.max() else (r, s)
    i = int(np.argmax(s))
    return {"fitness": float(s[i]), "lam": float(r.wavelengths_nm[i]), "cd0": float(r.cd0[i]),
            "T0": r.T0[i].tolist(), "residual": float(r.residual[i])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tio2", type=float, required=True)
    ap.add_argument("--mode", choices=("ga", "random"), default="ga")
    ap.add_argument("--hours", type=float, default=3.5)
    ap.add_argument("--max-evals", type=int, default=10 ** 6)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--coarse", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=float, default=0.0, help="average over this many nm (0 = single wavelength)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = os.path.join(ROOT, "datasets", "raw", f"pillar_{a.mode}{a.tag}.jsonl")
    old = [json.loads(s) for s in open(path)] if os.path.exists(path) else []
    t0, log = time.time(), []

    def run(x, gen):
        """Evaluate, or replay from the log. None when the budget is spent."""
        k = len(log)
        if k < len(old):
            assert old[k].get("window", 0.0) == a.window, "log was made with another window"
            assert np.allclose(old[k]["x"], x), "log does not match this seed/configuration"
            rec = old[k]
        else:
            if time.time() - t0 > a.hours * 3600 or k >= a.max_evals:
                return None
            rec = {"k": k, "gen": gen, "x": [float(v) for v in x], "n_tio2": a.n_tio2, "N": a.N,
                   "seed": a.seed, "window": a.window,
                   **(evaluate_window(x, a.n_tio2, a.N, a.coarse, a.window) if a.window
                      else evaluate(x, a.n_tio2, a.N, a.coarse))}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{a.mode} {k:4d} gen {gen:2d} fit {rec['fitness']:.3f} CD {rec['cd0']:+.3f} at {rec['lam']:.1f} "
                  f"best {max([r['fitness'] for r in log] + [rec['fitness']]):.3f}  {time.time() - t0:.0f}s", flush=True)
        log.append(rec)
        return rec

    rng = np.random.default_rng(a.seed)
    init = LO + (HI - LO) * qmc.LatinHypercube(6, rng=np.random.default_rng(a.seed)).random(a.pop - 1)
    pop = [PAPER, *init]
    gen, done = 0, False
    while not done:
        recs = []
        for x in pop:
            rec = run(x, gen)
            if rec is None:
                done = True
                break
            recs.append(rec)
        if done:
            break
        gen += 1
        if a.mode == "random":
            pop = list(LO + (HI - LO) * rng.random((a.pop, 6)))
            continue
        # survivors keep their scores (elitism), so only children are re-run
        parents = recs if gen == 1 else sorted(elite + recs, key=lambda r: -r["fitness"])[:a.pop]
        elite = sorted(parents, key=lambda r: -r["fitness"])[:2]
        parents = sorted(parents, key=lambda r: -r["fitness"])

        def pick():
            return np.array(parents[min(rng.integers(len(parents), size=3))]["x"])   # tournament of 3

        pop = []
        for _ in range(a.pop - 2):
            p, q = pick(), pick()
            w = rng.uniform(-0.25, 1.25, 6)                                          # blend crossover
            child = w * p + (1 - w) * q
            hit = rng.random(6) < 0.3
            child = np.clip(child + hit * rng.normal(0, 0.1, 6) * (HI - LO), LO, HI)
            pop.append(child)
    print(f"DONE {a.mode}: {len(log)} designs, best {max(r['fitness'] for r in log):.3f}", flush=True)


if __name__ == "__main__":
    main()
