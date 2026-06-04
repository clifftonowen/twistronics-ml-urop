"""End-to-end environment + solver sanity check.

Run this first (and in CI) to confirm:
  1. the scientific stack imports;
  2. the rcwa4d solver submodule is importable from a pipeline script;
  3. a minimal untwisted solve conserves energy (R + T ~= 1, lossless);
  4. SYMMETRY SANITY (CLAUDE.md section 8): an untwisted C4v bilayer gives
     CD ~= 0 at normal incidence, and twisting it turns CD on.

Usage:
    python -m scripts.check_setup
"""

from __future__ import annotations

import sys

import numpy as np

# Importing the package puts rcwa4d on sys.path (see data_generation/__init__).
import data_generation  # noqa: F401
from data_generation.parameter_sampler import DesignParams
from data_generation.simulate_spectra import simulate_design

PASS, FAIL = "[ PASS ]", "[ FAIL ]"


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL} {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    ok = True

    # 1. stack
    try:
        import matplotlib  # noqa: F401
        import scipy  # noqa: F401
        import tqdm  # noqa: F401

        ok &= _check(
            "scientific stack imports",
            True,
            f"numpy {np.__version__}, scipy {scipy.__version__}",
        )
    except Exception as exc:  # pragma: no cover
        ok &= _check("scientific stack imports", False, repr(exc))

    # 2. solver import
    try:
        from rcwa4d import rcwa  # noqa: F401

        ok &= _check("import rcwa4d solver", True)
    except Exception as exc:
        ok &= _check("import rcwa4d solver", False, repr(exc))
        print("\nCannot continue without the solver. Did you run "
              "`git submodule update --init --recursive`?")
        return 1

    # A coarse, cheap wavelength grid is enough for sanity checks.
    wl = np.linspace(600.0, 850.0, 7)

    # 3 + 4a. untwisted bilayer: energy conservation + CD ~ 0
    flat = DesignParams(theta_deg=0.0, thickness=0.20, gap=0.20, radius=0.25)
    res0 = simulate_design(flat, wl, a_nm=500.0, N=2)
    max_energy_res = float(res0.energy_residual().max())
    ok &= _check(
        "energy conservation (untwisted, lossless)",
        max_energy_res < 1e-3,
        f"max |1-(R+T)| = {max_energy_res:.2e}",
    )
    max_cd_flat = float(np.abs(res0.cd).max())
    ok &= _check(
        "symmetry: untwisted C4v bilayer -> CD ~ 0",
        max_cd_flat < 1e-3,
        f"max |CD| = {max_cd_flat:.2e}",
    )

    # 4b. twisted bilayer: CD turns on
    twisted = DesignParams(theta_deg=15.0, thickness=0.20, gap=0.20, radius=0.25)
    res1 = simulate_design(twisted, wl, a_nm=500.0, N=2)
    max_cd_tw = float(np.abs(res1.cd).max())
    ok &= _check(
        "twist turns CD on (15 deg)",
        max_cd_tw > max_cd_flat,
        f"max |CD| = {max_cd_tw:.2e} (vs {max_cd_flat:.2e} untwisted)",
    )

    print("\n" + ("All checks passed." if ok else "Some checks FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
