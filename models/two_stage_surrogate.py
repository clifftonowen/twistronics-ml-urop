"""Two-stage forward surrogate for the peak-centred representation.

Motivation (EXPERIMENTS.md Sec 6h.8). On peak-centred shards each design's
window sits at its OWN absolute frequency, so the representation deliberately
splits "where the resonance is" (`f_peak`) from "what it looks like" (the
lineshape on the shared axis u). A single flat regression X -> lineshape throws
the first half away, and that is measurably costly: the transmission control,
robustly learnable everywhere else, drops to R2 = -0.07 on this representation,
but recovers to +0.30 the moment `f_peak` is supplied as an input.

So the model is structured to match the representation:

    stage A:  X            ->  f_peak        (where)
    stage B:  (X, f_peak)  ->  lineshape     (shape, given where)

At training time stage B sees the TRUE `f_peak`; at prediction time it sees
stage A's estimate, because `f_peak` is an output of the forward problem and is
unknown before simulating. Evaluating with the true value instead would be
leakage -- so `predict` uses the estimate and `predict_oracle` exists only to
measure the ceiling the two-stage design could reach if stage A were perfect.
The gap between them is exactly the cost of stage A's error.

Usage:
    python -m models.two_stage_surrogate --shard peak_merged_31 --target rel_cd
    python -m models.two_stage_surrogate --shard peak_merged_31 --target both_t
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from models.forward_surrogate import ARTIFACTS, baseline_metrics, make_fit, metrics
from scripts.cv_eval import kfold_indices

ROOT = os.path.dirname(os.path.dirname(__file__))
PROC = os.path.join(ROOT, "datasets", "processed")


class TwoStageSurrogate:
    """Fitted (stage A, stage B) pair. See module docstring for the split."""

    def __init__(self, stage_a, stage_b, f_mean: float, f_std: float):
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.f_mean = f_mean
        self.f_std = f_std

    def _augment(self, X: np.ndarray, f_peak: np.ndarray) -> np.ndarray:
        """Append standardized f_peak as an extra input column.

        Standardized with the TRAINING set's statistics so that a fold's test
        designs get exactly the transform its training designs defined.
        """
        z = (np.asarray(f_peak).ravel() - self.f_mean) / self.f_std
        return np.column_stack([X, z]).astype(np.float32)

    def predict_f_peak(self, X: np.ndarray) -> np.ndarray:
        return self.stage_a.predict(X).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Deployable path: stage A's estimate feeds stage B."""
        return self.stage_b.predict(self._augment(X, self.predict_f_peak(X)))

    def predict_oracle(self, X: np.ndarray, f_peak_true: np.ndarray) -> np.ndarray:
        """Diagnostic only -- the ceiling if stage A were perfect. Not deployable."""
        return self.stage_b.predict(self._augment(X, f_peak_true))


def fit_two_stage(X, f_peak, Y, fit_fn, seed: int = 0) -> TwoStageSurrogate:
    f_peak = np.asarray(f_peak).ravel()
    f_mean, f_std = float(f_peak.mean()), float(f_peak.std())
    if f_std < 1e-12:
        f_std = 1.0

    stage_a = fit_fn(X, f_peak[:, None].astype(np.float32), X,
                     f_peak[:, None].astype(np.float32), seed=seed)
    z = ((f_peak - f_mean) / f_std)[:, None]
    Xa = np.column_stack([X, z]).astype(np.float32)
    stage_b = fit_fn(Xa, Y, Xa, Y, seed=seed)
    return TwoStageSurrogate(stage_a, stage_b, f_mean, f_std)


def cv_two_stage(X, f_peak, Y, fit_fn, k: int, repeats: int, seed: int = 0) -> dict:
    """Repeated k-fold CV of the two-stage model, plus its oracle ceiling.

    Reports flat/one-stage on the same folds so the comparison is paired --
    at n~31 the fold-to-fold spread dwarfs the differences between models, so
    comparing across separately-drawn splits would be meaningless.
    """
    rows = {"two_stage": [], "oracle": [], "flat": [], "baseline": [], "f_peak": []}
    for r in range(repeats):
        for tr, te in kfold_indices(X.shape[0], k, seed + r):
            sur = fit_two_stage(X[tr], f_peak[tr], Y[tr], fit_fn, seed=seed + r)
            rows["two_stage"].append(metrics(sur.predict(X[te]), Y[te])["r2"])
            rows["oracle"].append(metrics(sur.predict_oracle(X[te], f_peak[te]), Y[te])["r2"])
            rows["f_peak"].append(
                metrics(sur.predict_f_peak(X[te])[:, None], f_peak[te][:, None])["r2"]
            )
            flat = fit_fn(X[tr], Y[tr], X[te], Y[te], seed=seed + r)
            rows["flat"].append(metrics(flat.predict(X[te]), Y[te])["r2"])
            rows["baseline"].append(baseline_metrics(Y[tr], Y[te])["r2"])
    return {k_: np.array(v) for k_, v in rows.items()}


def load_peak_merged(shard: str, target: str, proc_dir: str = PROC):
    d = np.load(os.path.join(proc_dir, f"{shard}.npz"))
    X = d["X"].astype(np.float32)
    if target == "rel_cd":
        Y = d["Y"]
    elif target == "both_t":
        Y = np.concatenate([d["T_RCP"], d["T_LCP"]], axis=1)
    elif target == "delta_t":
        Y = d["T_RCP"] - d["T_LCP"]
    elif target == "peak_cd":
        Y = d["peak_abs_cd"][:, None]
    else:
        raise ValueError(f"unknown target {target!r}")
    return X, d["f_peak"].astype(np.float32), Y.astype(np.float32)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", default="peak_merged_31")
    p.add_argument("--target", default="rel_cd",
                   choices=("rel_cd", "both_t", "delta_t", "peak_cd"))
    p.add_argument("--model", choices=("gp", "mlp"), default="gp")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=ARTIFACTS)
    args = p.parse_args(argv)

    X, f_peak, Y = load_peak_merged(args.shard, args.target)
    res = cv_two_stage(X, f_peak, Y, make_fit(args.model), args.k, args.repeats, args.seed)

    print(f"\n{args.shard} | target={args.target} | model={args.model} | "
          f"n={X.shape[0]} | {len(res['flat'])} CV trials\n")
    print(f"  {'variant':<26s} {'R2 mean':>9s} {'sd':>7s}")
    for key, label in [
        ("flat", "flat  X -> Y"),
        ("two_stage", "two-stage (deployable)"),
        ("oracle", "two-stage w/ true f_peak"),
        ("baseline", "mean-predictor baseline"),
    ]:
        print(f"  {label:<26s} {res[key].mean():>+9.3f} {res[key].std():>7.3f}")
    print(f"\n  stage A alone (X -> f_peak):  R2 {res['f_peak'].mean():+.3f} "
          f"+/- {res['f_peak'].std():.3f}")

    gain = res["two_stage"].mean() - res["flat"].mean()
    ceiling = res["oracle"].mean() - res["flat"].mean()
    print(f"\n  two-stage vs flat:            {gain:+.3f}")
    print(f"  ceiling if stage A perfect:   {ceiling:+.3f}")
    if ceiling <= 0.05:
        print("  -> the split cannot help here: even a perfect stage A gains nothing,")
        print("     so the limit is stage B (lineshape), not the missing frequency.")
    elif gain < 0.5 * ceiling:
        print("  -> the split helps in principle but stage A is too weak to realize it;")
        print("     improving f_peak prediction is the lever.")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"two_stage_{args.shard}_{args.target}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "shard": args.shard, "target": args.target, "model": args.model,
            "n": int(X.shape[0]), "k": args.k, "repeats": args.repeats,
            **{f"{k}_mean": float(v.mean()) for k, v in res.items()},
            **{f"{k}_std": float(v.std()) for k, v in res.items()},
        }, fh, indent=2)
    print(f"\nsaved -> {out_path}")
    return {k: float(v.mean()) for k, v in res.items()}


if __name__ == "__main__":
    main()
