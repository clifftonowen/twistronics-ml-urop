"""Genetic vs Optuna (window-averaged score): progress table, and best designs re-scored at N=3.

Usage: python -m scripts.compare_searches
"""
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from scripts.pillar_ga import ROOT, evaluate_window

RAW = os.path.join(ROOT, "datasets", "raw")
out = {}
for m in ("ga", "optuna"):
    for s in (7, 8, 9):
        r = [json.loads(q) for q in open(os.path.join(RAW, f"pillar_{m}_w5_s{s}.jsonl"))]
        f = np.array([q["fitness"] for q in r])
        b = r[int(f.argmax())]
        path = os.path.join(RAW, "pillar_verify", f"w5_{m}_s{s}_N3.json")
        if not os.path.exists(path):
            json.dump(evaluate_window(np.array(b["x"]), 2.5, 3, 2.0, 5.0), open(path, "w"))
        v = json.load(open(path))
        print(f"{m:6s} s{s} by 50/100/150/250: " + " ".join(f"{f[:k].max():.3f}" for k in (50, 100, 150, 250))
              + f" | best k {b['k']} N2 {b['fitness']:.3f} @ {b['lam']:.1f} (edge {b['edge']:.1f}) "
              f"N3 {v['fitness']:.3f} @ {v['lam']:.1f} CD {v['cd0']:+.3f} | x {np.round(b['x'], 1).tolist()}", flush=True)
