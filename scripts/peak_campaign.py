"""Two-pass, peak-centred data campaign: locate each design's CD peak cheaply,
then resolve a narrow window around it at converged truncation.

Replaces the broad-grid campaign that produced `highcd_n3_gpu_merged60`. That
grid (26 points over 600-850nm, ~10nm spacing) was shown to alias the CD
spectrum -- 41-58% of its true variance falls between samples (EXPERIMENTS.md
Sec 6c) -- and CD has now failed to be learnable on it three times while
transmission, which is smooth at that spacing, learns fine. The target was the
problem, not the model or the design count.

Per design:
  1. locate  (N=2, 81 pts over f in [0.60, 0.92])  ~3 min   -> where is the peak
  2. screen  (peak|CD| >= --min-peak-cd)                    -> is it worth resolving
  3. window  (N=3, 41 pts over f_peak*(1 +- 0.03)) ~23 min  -> the training target

Rejected designs are KEPT, not discarded. They cost only the cheap pass and they
are the only negative examples the dataset will ever contain; training solely on
designs that passed a CD threshold would leave a surrogate unable to recognize a
bad region, which is precisely the out-of-distribution failure mode inverse
design walks into.

Output is written after EVERY design. A previous multi-hour run was killed at
design 28 of 45 and lost all of it because results were only written at the end
(EXPERIMENTS.md Sec 6); that must not be repeatable.

Usage:
    python -m scripts.peak_campaign --smoke
    python -m scripts.peak_campaign --n 25 --device gpu --shard peak_pilot25
    python -m scripts.peak_campaign --reprocess-indices 36 38 15 2 14 10 \
        --device gpu --shard peak_known8      # pass 2 only, f_peak already known
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

import data_generation  # noqa: F401 -- ensures rcwa4d is on sys.path
import rcwa4d.backend as backend
from data_generation.dataset_compiler import _standardizer
from data_generation.parameter_sampler import (
    BOX_PRESETS,
    PARAM_NAMES,
    DesignParams,
    params_to_matrix,
    sample_params,
    sample_perturbed,
)
from data_generation.peak_passes import (
    DEFAULT_LOCATE_F_MAX,
    DEFAULT_LOCATE_F_MIN,
    DEFAULT_LOCATE_N,
    DEFAULT_LOCATE_PTS,
    DEFAULT_WINDOW_N,
    locate_pass,
    window_pass,
)
from data_generation.peak_window import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_WINDOW_PTS,
    locate_peak,
    relative_axis,
    resolvable_q_range,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
DATASETS = os.path.join(ROOT, "datasets")
RAW = os.path.join(DATASETS, "raw")

# Designs with theta >= 25, a resolved dense scan, and peak|CD| >= 0.24. Used
# both as perturbation centres and as the --reprocess-indices default.
DEFAULT_SEED_INDICES = [36, 38, 15, 2, 14, 10]
DEFAULT_SEED_SHARD = "highcd_n3_gpu_merged60"
DENSE_SHARDS = [
    "dense_rescan_highcd_n3_gpu_merged60",
    "dense_rescan_highcd_n3_gpu_merged60_pilot17",
]


def load_seeds(shard: str, indices: list[int]) -> list[DesignParams]:
    X = np.load(os.path.join(RAW, f"{shard}.npz"))["X"]
    return [DesignParams(**dict(zip(PARAM_NAMES, X[i]))) for i in indices]


def load_known_peaks(a_nm: float) -> dict[int, float]:
    """f_peak per design from the dense rescans already on disk.

    Lets `--reprocess-indices` skip pass 1 for designs whose peak position was
    already established by an N=3 dense scan -- there is nothing a cheaper pass
    could add.
    """
    known: dict[int, float] = {}
    for shard in DENSE_SHARDS:
        path = os.path.join(RAW, f"{shard}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        freqs = a_nm / d["dense_wavelengths_nm"]
        for idx in d["indices"]:
            known[int(idx)] = locate_peak(freqs, d[f"design{idx}_dense_cd"]).f_peak
    return known


def build_designs(args) -> tuple[list[DesignParams], list[str], list[float | None]]:
    """Return (designs, origin label per design, known f_peak per design)."""
    if args.reprocess_indices:
        known = load_known_peaks(args.a_nm)
        missing = [i for i in args.reprocess_indices if i not in known]
        if missing:
            raise SystemExit(f"no dense-scan f_peak on disk for design(s) {missing}")
        designs = load_seeds(args.seeds_from, args.reprocess_indices)
        return designs, [f"reprocess:{i}" for i in args.reprocess_indices], \
            [known[i] for i in args.reprocess_indices]

    bounds = BOX_PRESETS[args.box]
    n_perturb = int(round(args.n * args.perturb_frac))
    n_random = args.n - n_perturb
    seeds = load_seeds(args.seeds_from, args.seed_indices)
    designs = sample_perturbed(seeds, n_perturb, bounds, args.sigma_frac, seed=args.seed)
    designs += sample_params(n_random, bounds, seed=args.seed)
    origins = ["perturb"] * n_perturb + ["random"] * n_random
    return designs, origins, [None] * len(designs)


def main(argv: list[str] | None = None) -> dict[str, str]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=25, help="designs to attempt (before screening)")
    p.add_argument("--box", choices=sorted(BOX_PRESETS), default="peak_box")
    p.add_argument("--perturb-frac", type=float, default=0.67,
                   help="fraction drawn by jittering the seeds; the rest is LHS over the box")
    p.add_argument("--sigma-frac", type=float, default=0.15, help="jitter sd, as a fraction of box width")
    p.add_argument("--seeds-from", default=DEFAULT_SEED_SHARD)
    p.add_argument("--seed-indices", type=int, nargs="+", default=DEFAULT_SEED_INDICES)
    p.add_argument("--reprocess-indices", type=int, nargs="*", default=None,
                   help="skip sampling and pass 1: run pass 2 on these existing designs, "
                        "whose f_peak comes from their N=3 dense scan")
    p.add_argument("--min-peak-cd", type=float, default=0.20,
                   help="screen: reject designs whose pass-1 peak|CD| is below this. "
                        "Note 0.1 is effectively non-binding here -- all 20 designs with "
                        "resolved dense scans in this region exceed it (85%% exceed 0.2).")
    p.add_argument("--locate-N", type=int, default=DEFAULT_LOCATE_N)
    p.add_argument("--locate-pts", type=int, default=DEFAULT_LOCATE_PTS)
    p.add_argument("--f-min", type=float, default=DEFAULT_LOCATE_F_MIN)
    p.add_argument("--f-max", type=float, default=DEFAULT_LOCATE_F_MAX)
    p.add_argument("--window-N", type=int, default=DEFAULT_WINDOW_N)
    p.add_argument("--window-pts", type=int, default=DEFAULT_WINDOW_PTS)
    p.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH,
                   help="window half-width in relative frequency; see peak_window.resolvable_q_range")
    p.add_argument("--a-nm", type=float, default=500.0)
    p.add_argument("--slab", default="Si3N4")
    p.add_argument("--hole", default="air")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--out", default=DATASETS)
    p.add_argument("--shard", default="peak_pilot")
    p.add_argument("--smoke", action="store_true",
                   help="tiny fast config (n=3, N=1, coarse grids) to validate plumbing")
    args = p.parse_args(argv)

    if args.smoke:
        args.n, args.locate_N, args.window_N = 3, 1, 1
        args.locate_pts, args.window_pts = 15, 9
        args.min_peak_cd = 0.0  # plumbing test: never let the screen empty the shard

    os.environ["RCWA4D_DEVICE"] = args.device
    backend.set_device(args.device)

    designs, origins, known_f = build_designs(args)
    sim_kw = dict(slab_material=args.slab, hole_material=args.hole, resolution=args.resolution)
    qr = resolvable_q_range(args.half_width, args.window_pts)

    print(
        f"Peak campaign: {len(designs)} designs | box={args.box} "
        f"({sum(o == 'perturb' for o in origins)} perturb / "
        f"{sum(o == 'random' for o in origins)} random / "
        f"{sum(o.startswith('reprocess') for o in origins)} reprocess)\n"
        f"  locate: N={args.locate_N}, f in [{args.f_min}, {args.f_max}] x {args.locate_pts}\n"
        f"  screen: peak|CD| >= {args.min_peak_cd}\n"
        f"  window: N={args.window_N}, +-{args.half_width:.1%} x {args.window_pts} pts "
        f"-> resolves Q in [{qr.q_min:.0f}, {qr.q_max_resolved:.0f}], detects to {qr.q_max_detectable:.0f}\n"
        f"  device={args.device}\n"
    )

    records: list[dict] = []
    t0 = time.perf_counter()

    for i, (design, origin, f_known) in enumerate(
        zip(tqdm(designs, desc="designs"), origins, known_f)
    ):
        rec: dict = {"params": design.as_dict(), "origin": origin}

        if f_known is None:
            loc = locate_pass(
                design, a_nm=args.a_nm, f_min=args.f_min, f_max=args.f_max,
                n_pts=args.locate_pts, N=args.locate_N, **sim_kw,
            )
            L = loc.location
            rec.update(
                locate_cd=loc.spectrum.cd,
                locate_T_RCP=loc.spectrum.T_RCP,
                locate_T_LCP=loc.spectrum.T_LCP,
                locate_energy_residual=loc.spectrum.energy_residual(),
                f_peak=L.f_peak, locate_peak_abs_cd=L.peak_abs_cd,
                locate_cd_at_peak=L.cd_at_peak, n_competing_lobes=L.n_competing_lobes,
                q_estimate=L.q_estimate, fwhm_grid_pts=L.fwhm_grid_pts,
                width_grid_limited=L.width_is_grid_limited,
            )
            accepted = L.peak_abs_cd >= args.min_peak_cd
        else:
            # Pass 1 skipped: an N=3 dense scan already established this peak.
            rec.update(f_peak=f_known, locate_peak_abs_cd=np.nan, locate_cd_at_peak=np.nan,
                       n_competing_lobes=-1, q_estimate=np.nan, fwhm_grid_pts=np.nan,
                       width_grid_limited=False)
            accepted = True

        rec["accepted"] = bool(accepted)
        if accepted:
            win = window_pass(
                design, rec["f_peak"], a_nm=args.a_nm, half_width=args.half_width,
                n_pts=args.window_pts, N=args.window_N, **sim_kw,
            )
            rec.update(
                window_cd=win.spectrum.cd,
                window_T_RCP=win.spectrum.T_RCP, window_T_LCP=win.spectrum.T_LCP,
                window_R_RCP=win.spectrum.R_RCP, window_R_LCP=win.spectrum.R_LCP,
                window_wavelengths_nm=win.spectrum.wavelengths_nm,
                window_freqs=win.spectrum.freqs,
                window_energy_residual=win.spectrum.energy_residual(),
                window_peak_abs_cd=win.peak_abs_cd,
            )

        records.append(rec)
        paths = _write(records, args, qr, complete=False, elapsed=time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    paths = _write(records, args, qr, complete=True, elapsed=elapsed)

    n_acc = sum(r["accepted"] for r in records)
    print(f"\nDone in {elapsed / 60:.1f} min ({elapsed / max(len(records), 1) / 60:.1f} min/design).")
    print(f"  accepted {n_acc}/{len(records)} at peak|CD| >= {args.min_peak_cd}")
    if n_acc:
        pk = np.array([r["window_peak_abs_cd"] for r in records if r["accepted"]])
        res = np.array([float(r["window_energy_residual"].max()) for r in records if r["accepted"]])
        print(f"  window peak|CD|: median {np.median(pk):.3f}  min {pk.min():.3f}  max {pk.max():.3f}")
        print(f"  max energy residual {res.max():.2e}  (>1e-3 => check truncation)")
    for k, v in paths.items():
        print(f"  {k:10s}: {v}")
    return paths


def _stack(records: list[dict], key: str, mask=None) -> np.ndarray:
    rows = [r[key] for r in records if (mask is None or r["accepted"])]
    return np.vstack(rows) if rows else np.zeros((0, 0))


def _col(records: list[dict], key: str, mask=None, default=np.nan) -> np.ndarray:
    return np.array(
        [r.get(key, default) for r in records if (mask is None or r["accepted"])],
        dtype=float,
    )


def _write(records: list[dict], args, qr, complete: bool, elapsed: float) -> dict[str, str]:
    """Persist everything simulated so far. Called after every design."""
    raw_dir, proc_dir = os.path.join(args.out, "raw"), os.path.join(args.out, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    X_all = params_to_matrix([DesignParams(**r["params"]) for r in records])
    accepted = np.array([r["accepted"] for r in records], dtype=bool)
    acc = [r for r in records if r["accepted"]]
    u = relative_axis(args.half_width, args.window_pts)
    has_locate = any("locate_cd" in r for r in records)

    raw: dict = {
        "X_all": X_all,
        "accepted": accepted,
        "origin": np.array([r["origin"] for r in records]),
        "f_peak_all": _col(records, "f_peak"),
        "locate_peak_abs_cd": _col(records, "locate_peak_abs_cd"),
        "locate_cd_at_peak": _col(records, "locate_cd_at_peak"),
        "n_competing_lobes": _col(records, "n_competing_lobes"),
        "q_estimate": _col(records, "q_estimate"),
        "fwhm_grid_pts": _col(records, "fwhm_grid_pts"),
        "u": u,
    }
    if has_locate:
        raw["locate_freqs"] = np.linspace(args.f_min, args.f_max, args.locate_pts)
        for key in ("locate_cd", "locate_T_RCP", "locate_T_LCP", "locate_energy_residual"):
            rows = [r[key] for r in records if key in r]
            raw[key] = np.vstack(rows)
        # Which rows of X_all the locate arrays correspond to (pass 1 is skipped
        # entirely in reprocess mode, so the two can differ in length).
        raw["locate_row_index"] = np.array([i for i, r in enumerate(records) if "locate_cd" in r])
    if acc:
        raw.update(
            X=params_to_matrix([DesignParams(**r["params"]) for r in acc]),
            f_peak=_col(records, "f_peak", mask=True),
            window_cd=_stack(records, "window_cd", mask=True),
            window_T_RCP=_stack(records, "window_T_RCP", mask=True),
            window_T_LCP=_stack(records, "window_T_LCP", mask=True),
            window_R_RCP=_stack(records, "window_R_RCP", mask=True),
            window_R_LCP=_stack(records, "window_R_LCP", mask=True),
            window_wavelengths_nm=_stack(records, "window_wavelengths_nm", mask=True),
            window_freqs=_stack(records, "window_freqs", mask=True),
            window_energy_residual=_stack(records, "window_energy_residual", mask=True),
            window_peak_abs_cd=_col(records, "window_peak_abs_cd", mask=True),
        )

    raw_path = os.path.join(raw_dir, f"{args.shard}.npz")
    np.savez_compressed(raw_path, **raw)

    paths = {"raw": raw_path}
    if acc:
        X = raw["X"]
        x_mean, x_std = _standardizer(X)
        proc_path = os.path.join(proc_dir, f"{args.shard}.npz")
        np.savez_compressed(
            proc_path,
            X=(X - x_mean) / x_std,
            X_raw=X,
            Y=raw["window_cd"],            # the primary target, on the common u axis
            u=u,
            f_peak=raw["f_peak"],
            peak_abs_cd=raw["window_peak_abs_cd"],
            T_RCP=raw["window_T_RCP"],
            T_LCP=raw["window_T_LCP"],
        )
        np.savez_compressed(
            os.path.join(proc_dir, f"{args.shard}_xscaler.npz"),
            x_mean=x_mean, x_std=x_std, param_names=np.array(PARAM_NAMES),
        )
        paths["processed"] = proc_path
        paths["scaler"] = os.path.join(proc_dir, f"{args.shard}_xscaler.npz")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "representation": "peak_centred_relative_window",
        "response": "transmission_circular_dichroism",
        "param_names": PARAM_NAMES,
        "n_attempted": len(records),
        "n_accepted": int(accepted.sum()),
        "n_requested": args.n,
        "complete": complete,
        "box": args.box,
        "param_bounds": {k: list(v) for k, v in BOX_PRESETS[args.box].items()},
        "perturb_frac": args.perturb_frac,
        "sigma_frac": args.sigma_frac,
        "seed_shard": args.seeds_from,
        "seed_indices": args.seed_indices,
        "reprocess_indices": args.reprocess_indices,
        "min_peak_cd": args.min_peak_cd,
        "locate": {"N_m": args.locate_N, "f_min": args.f_min, "f_max": args.f_max,
                   "n_pts": args.locate_pts},
        "window": {"N_m": args.window_N, "half_width": args.half_width,
                   "n_pts": args.window_pts,
                   "q_min": qr.q_min, "q_max_resolved": qr.q_max_resolved,
                   "q_max_detectable": qr.q_max_detectable, "step_u": qr.step_u},
        "a_nm": args.a_nm,
        "a_nm_note": "window grids are defined in normalized frequency; while materials "
                     "are non-dispersive a_nm is a free post-hoc relabelling, so f_peak "
                     "is a fabrication output rather than a fixed property of the shard",
        "slab_material": args.slab,
        "hole_material": args.hole,
        "eps_map_resolution": args.resolution,
        "sampling_seed": args.seed,
        "device": args.device,
        "smoke": args.smoke,
        "wall_time_s": round(elapsed, 2),
        "sec_per_design": round(elapsed / max(len(records), 1), 2),
    }
    meta_path = os.path.join(proc_dir, f"{args.shard}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    paths["metadata"] = meta_path
    return paths


if __name__ == "__main__":
    main()
