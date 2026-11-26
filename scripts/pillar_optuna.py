"""Optuna search over pillar bilayers, same score and start as scripts.pillar_ga.

Uses the Gaussian-process sampler. The first generation of the genetic search
(paper design + 15 Latin hypercube designs, same seed) is queued first, so the two
methods start from identical designs. Every design is appended to a JSONL file;
on restart the file is loaded back into the study, so a stopped run resumes.

Usage: python -m scripts.pillar_optuna --n-tio2 2.5 --seed 7 [--max-evals 250] [--window 5]
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from scipy.stats import qmc

from scripts.pillar_ga import HI, LO, NAMES, PAPER, ROOT, evaluate_window

DIST = {k: FloatDistribution(float(lo), float(hi)) for k, lo, hi in zip(NAMES, LO, HI)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tio2", type=float, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-evals", type=int, default=250)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--coarse", type=float, default=2.0)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = os.path.join(ROOT, "datasets", "raw", f"pillar_optuna{a.tag}.jsonl")
    old = [json.loads(s) for s in open(path)] if os.path.exists(path) else []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(seed=a.seed))
    for rec in old:
        study.add_trial(optuna.trial.create_trial(params=dict(zip(NAMES, rec["x"])), distributions=DIST,
                                                  value=rec["fitness"]))
    init = LO + (HI - LO) * qmc.LatinHypercube(6, rng=np.random.default_rng(a.seed)).random(a.pop - 1)
    for x in [PAPER, *init][len(old):]:
        study.enqueue_trial(dict(zip(NAMES, map(float, x))))

    t0, k = time.time(), len(old)
    best = max([r["fitness"] for r in old], default=0.0)
    while k < a.max_evals and time.time() - t0 < a.hours * 3600:
        trial = study.ask(DIST)
        x = np.array([trial.params[n] for n in NAMES])
        rec = {"k": k, "x": [float(v) for v in x], "n_tio2": a.n_tio2, "N": a.N, "seed": a.seed,
               "window": a.window, "sampler": "GPSampler",
               **evaluate_window(x, a.n_tio2, a.N, a.coarse, a.window)}
        study.tell(trial, rec["fitness"])
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        best = max(best, rec["fitness"])
        print(f"optuna {k:4d} fit {rec['fitness']:.3f} CD {rec['cd0']:+.3f} at {rec['lam']:.1f} best {best:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        k += 1
    print(f"DONE optuna: {k} designs, best {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
