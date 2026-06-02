"""Backward-compatibility shim.

The monolithic module was split into focused submodules (see cellflow/cell.py,
cellflow/simulation.py, cellflow/initializers.py, cellflow/kernels/, etc.).
This module re-exports the full public API so existing imports such as
``from cellflow.cellflow_core import CellSimulation`` keep working.
"""
from .cell import Cell
from .simulation import CellSimulation
from .initializers import (
    setup_central_uniform,
    setup_wound_healing,
    INITIALIZER_MAP,
)
from .kernels.diffusion import diffuse_field_numba, advect_scalar_field_numba
from .kernels.fields import (
    absorb_nutrient_numba,
    secrete_over_area_numba,
    sample_field_at_cell_numba,
    _sample_scalar_field_numba,
)
from .kernels.stokeslet import (
    update_fluid_velocity_numba,
    update_fluid_velocity_with_dipoles_numba,
    compute_cell_velocities_numba,
)
from .kernels.forces import (
    calculate_adhesion_forces_numba,
    calculate_repulsion_forces_numba,
    calculate_propulsion_forces_numba,
    resolve_overlaps_numba,
)

__all__ = [
    "Cell",
    "CellSimulation",
    "setup_central_uniform",
    "setup_wound_healing",
    "INITIALIZER_MAP",
    "diffuse_field_numba",
    "advect_scalar_field_numba",
    "absorb_nutrient_numba",
    "secrete_over_area_numba",
    "sample_field_at_cell_numba",
    "_sample_scalar_field_numba",
    "update_fluid_velocity_numba",
    "update_fluid_velocity_with_dipoles_numba",
    "compute_cell_velocities_numba",
    "calculate_adhesion_forces_numba",
    "calculate_repulsion_forces_numba",
    "calculate_propulsion_forces_numba",
    "resolve_overlaps_numba",
]
