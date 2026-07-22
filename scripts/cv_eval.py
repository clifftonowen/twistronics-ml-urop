"""Repeated k-fold CV re-evaluation of the forward surrogate.

`models/forward_surrogate.py` reports metrics on a single 70/15/15 split. At
n=60 that's a 9-design test set, and the mean-predictor baseline itself scored
R2=-0.229 there in one recorded run -- for a baseline that should score ~0 at
scale, that is the size of the measurement noise, not a property of the model.
A single-split R2 near 0 is therefore uninformative: it could be masking a
genuinely learnable signal or confirming a genuinely absent one. This script
repeats K-fold CV R times and reports the resulting spread, so "R2 <= 0" can be
told apart from "R2 <= 0 within noise of a positive result".

Reuses (does not reimplement) `models/forward_surrogate.py`'s
`load_dataset`/`metrics`/`baseline_metrics`/`make_fit`.

Usage:
    python -m scripts.cv_eval --shard highcd_n3_gpu_merged60 \
        --targets cd both_t delta_t --model gp --k 5 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from models.forward_surrogate import (
    ARTIFACTS,
    baseline_metrics,
    load_dataset,
    make_fit,
    metrics,
)


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, test_idx) splits of range(n), shuffled by seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


def cv_once(fit_fn, X, Y, k, seed):
    """One K-fold pass: returns per-fold model R2 and baseline R2 lists."""
    model_r2, base_r2 = [], []
    for train_idx, test_idx in kfold_indices(X.shape[0], k, seed):
        sur = fit_fn(X[train_idx], Y[train_idx], X[test_idx], Y[test_idx], seed=seed)
        model_r2.append(metrics(sur.predict(X[test_idx]), Y[test_idx])["r2"])
        base_r2.append(baseline_metrics(Y[train_idx], Y[test_idx])["r2"])
    return model_r2, base_r2


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", required=True, help="shard name (used for --targets, and as the report label)")
    p.add_argument("--targets", nargs="*", default=[], help="load_dataset-compatible targets (cd, both_t, delta_t, ...)")
    p.add_argument(
        "--features-file", default=None,
        help="path to a pre-built (X, Y) .npz (e.g. from build_resonance_features.py) to evaluate "
        "in addition to/instead of --targets. Must contain 'X' and 'Y' arrays.",
    )
    p.add_argument("--features-label", default="resonance_features", help="label for --features-file in the report")
    p.add_argument("--model", choices=("mlp", "gp"), default="gp")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=ARTIFACTS)
    args = p.parse_args(argv)

    fit_fn = make_fit(args.model)
    rows = []

    sources: list[tuple[str, np.ndarray, np.ndarray]] = []
    for target in args.targets:
        X, Y, _wl = load_dataset(args.shard, target)
        sources.append((target, X, Y))
    if args.features_file:
        d = np.load(args.features_file)
        sources.append((args.features_label, d["X"].astype(np.float32), d["Y"].astype(np.float32)))

    for target, X, Y in sources:
        all_model_r2, all_base_r2 = [], []
        for r in range(args.repeats):
            m_r2, b_r2 = cv_once(fit_fn, X, Y, args.k, seed=args.seed + r)
            all_model_r2.extend(m_r2)
            all_base_r2.extend(b_r2)
        all_model_r2 = np.array(all_model_r2)
        all_base_r2 = np.array(all_base_r2)
        row = {
            "target": target, "model": args.model, "n": int(X.shape[0]),
            "k": args.k, "repeats": args.repeats, "n_trials": int(all_model_r2.size),
            "model_r2_mean": float(all_model_r2.mean()), "model_r2_std": float(all_model_r2.std()),
            "baseline_r2_mean": float(all_base_r2.mean()), "baseline_r2_std": float(all_base_r2.std()),
        }
        rows.append(row)
        print(
            f"target={target:8s} model={args.model} | model R2 {row['model_r2_mean']:+.3f} "
            f"+/- {row['model_r2_std']:.3f}  |  baseline R2 {row['baseline_r2_mean']:+.3f} "
            f"+/- {row['baseline_r2_std']:.3f}  (n_trials={row['n_trials']})"
        )

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"cv_report_{args.shard}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"shard": args.shard, "rows": rows}, fh, indent=2)
    print(f"\nsaved -> {out_path}")
    return {"shard": args.shard, "rows": rows}


if __name__ == "__main__":
    main()
