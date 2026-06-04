"""Assemble per-design solver outputs into ML-ready, normalized tensors.

Responsibilities (CLAUDE.md sections 5-7):
  * stack designs into X (n, n_params) and the response Y (n, n_lambda);
  * persist raw, un-normalized arrays in datasets/raw/ for re-processing;
  * fit and SAVE feature/target scalers alongside datasets/processed/ so that
    inference and the RCWA validation loop apply the identical transform;
  * record full provenance (N_m, grids, seed, materials, a_nm) in metadata so
    a shard is reproducible and we never mix unit conventions.

Normalization choices:
  * X: per-feature standardization (z-score). Scaler = (mean, std) per column.
  * Y = CD: already bounded in [-1, 1]; we DO NOT rescale it (keeps the
    physical range interpretable and the sign meaningful). T_RCP/T_LCP are
    stored raw too, for models that prefer to predict transmission.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from .parameter_sampler import PARAM_NAMES, DesignParams, params_to_matrix
from .simulate_spectra import SpectrumResult


def _standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) per column; std floored to avoid divide-by-zero."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def compile_dataset(
    params: list[DesignParams],
    results: list[SpectrumResult],
    out_dir: str,
    metadata: dict,
    shard_name: str = "shard000",
) -> dict[str, str]:
    """Write raw + processed arrays and metadata for one batch of designs.

    Returns a dict of written file paths.
    """
    if len(params) != len(results):
        raise ValueError("params and results must have equal length")
    if not results:
        raise ValueError("no results to compile")

    raw_dir = os.path.join(out_dir, "raw")
    proc_dir = os.path.join(out_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    wl = results[0].wavelengths_nm
    for r in results:
        if not np.array_equal(r.wavelengths_nm, wl):
            raise ValueError("all results must share one wavelength grid")

    X = params_to_matrix(params)                      # (n, n_params)
    Y_cd = np.vstack([r.cd for r in results])         # (n, n_lambda)
    T_RCP = np.vstack([r.T_RCP for r in results])
    T_LCP = np.vstack([r.T_LCP for r in results])
    energy_res = np.vstack([r.energy_residual() for r in results])

    x_mean, x_std = _standardizer(X)
    X_norm = (X - x_mean) / x_std

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    full_meta = {
        "created_utc": stamp,
        "n_designs": int(X.shape[0]),
        "param_names": PARAM_NAMES,
        "n_wavelengths": int(wl.shape[0]),
        "response": "transmission_circular_dichroism",
        "cd_range": [-1.0, 1.0],
        "max_energy_residual": float(energy_res.max()),
        "mean_energy_residual": float(energy_res.mean()),
        **metadata,  # N_m, a_nm, materials, angle, seed, bounds, wl grid, etc.
    }

    raw_path = os.path.join(raw_dir, f"{shard_name}.npz")
    proc_path = os.path.join(proc_dir, f"{shard_name}.npz")
    scaler_path = os.path.join(proc_dir, f"{shard_name}_xscaler.npz")
    meta_path = os.path.join(proc_dir, f"{shard_name}_metadata.json")

    np.savez_compressed(
        raw_path,
        X=X,
        wavelengths_nm=wl,
        freqs=results[0].freqs,
        CD=Y_cd,
        T_RCP=T_RCP,
        T_LCP=T_LCP,
        R_RCP=np.vstack([r.R_RCP for r in results]),
        R_LCP=np.vstack([r.R_LCP for r in results]),
        energy_residual=energy_res,
    )
    np.savez_compressed(
        proc_path, X=X_norm, Y=Y_cd, wavelengths_nm=wl
    )
    np.savez_compressed(
        scaler_path, x_mean=x_mean, x_std=x_std, param_names=np.array(PARAM_NAMES)
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2)

    return {
        "raw": raw_path,
        "processed": proc_path,
        "scaler": scaler_path,
        "metadata": meta_path,
    }
