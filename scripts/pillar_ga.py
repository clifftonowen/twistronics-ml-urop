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
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = os.path.join(ROOT, "datasets", "raw", f"pillar_{a.mode}{a.tag}.jsonl")
    old = [json.loads(s) for s in open(path)] if os.path.exists(path) else []
    t0, log = time.time(), []

    def run(x, gen):
        """Evaluate, or replay from the log. None when the budget is spent."""
        k = len(log)
        if k < len(old):
            assert np.allclose(old[k]["x"], x), "log does not match this seed/configuration"
            rec = old[k]
        else:
            if time.time() - t0 > a.hours * 3600 or k >= a.max_evals:
                return None
            rec = {"k": k, "gen": gen, "x": [float(v) for v in x], "n_tio2": a.n_tio2, "N": a.N,
                   "seed": a.seed, **evaluate(x, a.n_tio2, a.N, a.coarse)}
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
