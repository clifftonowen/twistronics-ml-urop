"""Locate the RCWA-4D solver submodule and expose it as an importable package.

The fancompute/rcwa4d fork is tracked as a git submodule at:

    <project_root>/rcwa4d/          <- submodule root (NO __init__.py here)
    <project_root>/rcwa4d/rcwa4d/   <- the actual importable package

So `import rcwa4d` only works if the *submodule root* is on sys.path (then
Python finds the inner `rcwa4d/` package by name). Adding the project root
instead would make `rcwa4d` resolve to the submodule root, which is a bare
directory with no __init__.py -> a namespace package that exposes none of the
solver code. We therefore insert the submodule root specifically.

"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "rcwa4d")


def add_rcwa4d_to_path() -> str:
    """Insert the rcwa4d submodule root at the front of sys.path.

    Returns the submodule root path. Idempotent.
    Raises FileNotFoundError with an actionable message if the submodule has
    not been checked out (the classic "forgot to init submodules" case).
    """
    package_init = os.path.join(SUBMODULE_ROOT, "rcwa4d", "__init__.py")
    if not os.path.isfile(package_init):
        raise FileNotFoundError(
            f"rcwa4d solver package not found at {package_init!r}.\n"
            "The solver is a git submodule. From the project root run:\n"
            "    git submodule update --init --recursive"
        )
    if SUBMODULE_ROOT not in sys.path:
        sys.path.insert(0, SUBMODULE_ROOT)
    return SUBMODULE_ROOT
