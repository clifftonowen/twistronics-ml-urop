"""Build a fixed-length resonance-feature table from densely-scanned designs.

Turns each design's raw dense T_RCP/T_LCP spectra into a small, fixed-length
feature vector -- the top-K RCP/LCP mode pairs (by predicted |CD|), each
contributing (rcp.Q, lcp.Q, splitting_nm, predicted_peak_abs_cd) -- instead of
a dense pointwise CD curve. This is the target representation Step 4a
(EXPERIMENTS.md Sec 6d) validated: fitting T_RCP/T_LCP as Fano resonances is
accurate, and derived mode-pair CD reproduces true peak|CD| almost exactly in
the common case. Designs with fewer than K matched pairs are zero-padded.

Combines two dense-scan sources: the original 3-design file from Step 3
(`dense_rescan_<shard>.npz`) and the 17-design pilot from Step B
(`dense_rescan_<shard>_pilot17.npz`), giving n=20 total.

Usage:
    python -m scripts.build_resonance_features --shard highcd_n3_gpu_merged60 \
        --pilot-suffix pilot17 --k 3 --out resonance_features_pilot20
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_generation.resonances import fit_spectrum, match_rcp_lcp_modes

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
PROC = os.path.join(ROOT, "datasets", "processed")

FEATURE_NAMES_PER_PAIR = ("rcp_Q", "lcp_Q", "splitting_nm", "predicted_peak_abs_cd")


def _load_dense_source(path: str) -> dict[int, dict]:
    """Return {design_index: {"T_RCP":..., "T_LCP":..., "wl":...}} from one
    dense_rescan_*.npz file."""
    d = np.load(path)
    wl = d["dense_wavelengths_nm"]
    out = {}
    for idx in [int(i) for i in d["indices"]]:
        out[idx] = {
            "T_RCP": d[f"design{idx}_dense_T_RCP"],
            "T_LCP": d[f"design{idx}_dense_T_LCP"],
            "wl": wl,
        }
    return out


def build_features(shard: str, pilot_suffix: str | None, k: int, min_r2: float = 0.9,
                    max_split_nm: float = 15.0, window_nm: float = 60.0):
    sources = [os.path.join(RAW, f"dense_rescan_{shard}.npz")]
    if pilot_suffix:
        sources.append(os.path.join(RAW, f"dense_rescan_{shard}_{pilot_suffix}.npz"))

    designs: dict[int, dict] = {}
    for path in sources:
        if os.path.exists(path):
            designs.update(_load_dense_source(path))
        else:
            print(f"WARNING: {path} not found, skipping")

    raw = np.load(os.path.join(RAW, f"{shard}.npz"))
    X_full = raw["X"]

    indices = sorted(designs.keys())
    n = len(indices)
    n_feat_per_pair = len(FEATURE_NAMES_PER_PAIR)
    Y = np.zeros((n, k * n_feat_per_pair), dtype=np.float32)
    n_pairs_found = np.zeros(n, dtype=int)
    X = np.zeros((n, X_full.shape[1]), dtype=np.float32)

    for row, idx in enumerate(indices):
        X[row] = X_full[idx]
        d = designs[idx]
        res_rcp = fit_spectrum(d["wl"], d["T_RCP"], window_nm=window_nm)
        res_lcp = fit_spectrum(d["wl"], d["T_LCP"], window_nm=window_nm)
        pairs = match_rcp_lcp_modes(res_rcp, res_lcp, max_split_nm=max_split_nm, min_r2=min_r2)
        pairs_sorted = sorted(pairs, key=lambda p: -p.predicted_peak_abs_cd)
        n_pairs_found[row] = len(pairs_sorted)

        for j in range(min(k, len(pairs_sorted))):
            p = pairs_sorted[j]
            Y[row, j * n_feat_per_pair:(j + 1) * n_feat_per_pair] = [
                p.rcp.Q, p.lcp.Q, p.splitting_nm, p.predicted_peak_abs_cd,
            ]
        # remaining slots (if len(pairs_sorted) < k) stay zero -- explicit
        # zero-padding, not a silent default.

    return {
        "indices": np.array(indices), "X": X, "Y": Y,
        "n_pairs_found": n_pairs_found, "k": k,
        "feature_names": [f"pair{j}_{name}" for j in range(k) for name in FEATURE_NAMES_PER_PAIR],
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", required=True)
    p.add_argument("--pilot-suffix", default="pilot17")
    p.add_argument("--k", type=int, default=3, help="top-K mode pairs kept per design")
    p.add_argument("--min-r2", type=float, default=0.9)
    p.add_argument("--out", default=None, help="output name (default: resonance_features_<shard>)")
    args = p.parse_args(argv)

    result = build_features(args.shard, args.pilot_suffix, args.k, min_r2=args.min_r2)

    out_name = args.out or f"resonance_features_{args.shard}"
    out_path = os.path.join(PROC, f"{out_name}.npz")
    np.savez_compressed(
        out_path, indices=result["indices"], X=result["X"], Y=result["Y"],
        n_pairs_found=result["n_pairs_found"], k=result["k"],
        feature_names=np.array(result["feature_names"]),
    )

    print(f"n_designs={len(result['indices'])}  k={result['k']}  Y.shape={result['Y'].shape}")
    print("n_pairs_found per design:", dict(zip(result["indices"].tolist(), result["n_pairs_found"].tolist())))
    n_fully_padded = int(np.sum(result["n_pairs_found"] == 0))
    if n_fully_padded:
        print(f"NOTE: {n_fully_padded} design(s) had 0 matched pairs (fully zero-padded row)")
    print(f"saved -> {out_path}")
    return {"out_path": out_path, **result}


if __name__ == "__main__":
    main()
