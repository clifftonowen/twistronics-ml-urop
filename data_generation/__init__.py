"""Data-generation pipeline for the twisted-bilayer photonic-crystal project.

Phase 1 of the plan: drive the RCWA-4D solver over sampled structural
parameters to build an (X -> CD spectrum) dataset for the forward surrogate.

Importing this package automatically puts the `rcwa4d` solver submodule on
`sys.path` so that `from rcwa4d import *` works from any pipeline script,
regardless of the current working directory.
"""

from . import paths as _paths

_paths.add_rcwa4d_to_path()
