"""Merge several dataset shards (same grid/config, different sampling seeds)
into one combined shard, in the same raw/processed/scaler/metadata layout
`dataset_compiler.compile_dataset` produces, so downstream tools (e.g.
`models/forward_surrogate.py`) can load the merge like any other shard.

Use case: a single generation campaign that had to be split across several
`generate_dataset.py` runs (different --seed each time, e.g. because a run
was interrupted and the remainder was requested under a new seed) but shares
the same box, N_m, wavelength grid, and materials.

Usage:
    python -m scripts.merge_shards --shards pilot_highcd_n3_gpu highcd_n3_gpu_seed2 \
        highcd_n3_gpu_seed3 highcd_n3_gpu_seed4 highcd_n3_gpu_seed5 \
        --out highcd_n3_gpu_merged60
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
PROC = os.path.join(ROOT, "datasets", "processed")


def _standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", nargs="+", required=True, help="shard names to merge, in datasets/raw/")
    p.add_argument("--out", required=True, help="merged shard name")
    args = p.parse_args()

    raws = [np.load(os.path.join(RAW, f"{s}.npz")) for s in args.shards]
    metas = [json.load(open(os.path.join(PROC, f"{s}_metadata.json"), encoding="utf-8")) for s in args.shards]

    wl = raws[0]["wavelengths_nm"]
    freqs = raws[0]["freqs"]
    for s, r in zip(args.shards, raws):
        if not np.allclose(r["wavelengths_nm"], wl) or not np.allclose(r["freqs"], freqs):
            raise ValueError(f"shard {s!r} uses a different wavelength/frequency grid; cannot merge")

    incomplete = [s for s, m in zip(args.shards, metas) if m.get("complete") is False]
    if incomplete:
        print(f"NOTE: merging partial shards (not yet 'complete'): {incomplete}")

    X = np.concatenate([r["X"] for r in raws], axis=0)
    CD = np.concatenate([r["CD"] for r in raws], axis=0)
    T_RCP = np.concatenate([r["T_RCP"] for r in raws], axis=0)
    T_LCP = np.concatenate([r["T_LCP"] for r in raws], axis=0)
    R_RCP = np.concatenate([r["R_RCP"] for r in raws], axis=0)
    R_LCP = np.concatenate([r["R_LCP"] for r in raws], axis=0)
    energy_res = np.concatenate([r["energy_residual"] for r in raws], axis=0)

    x_mean, x_std = _standardizer(X)
    X_norm = (X - x_mean) / x_std

    raw_path = os.path.join(RAW, f"{args.out}.npz")
    proc_path = os.path.join(PROC, f"{args.out}.npz")
    scaler_path = os.path.join(PROC, f"{args.out}_xscaler.npz")
    meta_path = os.path.join(PROC, f"{args.out}_metadata.json")

    np.savez_compressed(
        raw_path, X=X, wavelengths_nm=wl, freqs=freqs, CD=CD, T_RCP=T_RCP, T_LCP=T_LCP,
        R_RCP=R_RCP, R_LCP=R_LCP, energy_residual=energy_res,
    )
    np.savez_compressed(proc_path, X=X_norm, Y=CD, wavelengths_nm=wl)
    param_names = metas[0].get("param_names")
    np.savez_compressed(
        scaler_path, x_mean=x_mean, x_std=x_std,
        param_names=np.array(param_names) if param_names else np.array([]),
    )

    merged_meta = dict(metas[0])  # base: N_m, box, device, grids, materials, etc. (shared across shards)
    merged_meta.update({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_designs": int(X.shape[0]),
        "max_energy_residual": float(energy_res.max()),
        "mean_energy_residual": float(energy_res.mean()),
        "merged_from_shards": args.shards,
        "merged_from_seeds": [m.get("sampling_seed") for m in metas],
        "complete": True,
        "n_requested": int(X.shape[0]),
    })
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(merged_meta, fh, indent=2)

    print(f"Merged {len(args.shards)} shards -> {X.shape[0]} designs.")
    print(f"Max energy residual |1-(R+T)| = {energy_res.max():.2e}")
    for k, v in {"raw": raw_path, "processed": proc_path, "scaler": scaler_path, "metadata": meta_path}.items():
        print(f"  {k:10s}: {v}")


if __name__ == "__main__":
    main()
