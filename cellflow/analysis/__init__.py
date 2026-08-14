"""Analysis utilities: quantitative diagnostics on simulation output.

These are deliberately separate from the simulation engine — they take plain
arrays (positions, radii), not ``CellSimulation`` objects, so they can be unit
tested against analytic shapes without running a simulation.
"""
from .front import (
    main_cluster_mask,
    front_radii,
    front_modes,
    roughness,
    fit_growth_rate,
)

__all__ = [
    'main_cluster_mask',
    'front_radii',
    'front_modes',
    'roughness',
    'fit_growth_rate',
]
