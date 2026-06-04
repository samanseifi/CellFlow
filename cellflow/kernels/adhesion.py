"""Differential (type-dependent) adhesion force kernel.

Generalizes the scalar adhesion in ``forces.py``: the adhesion strength between
two cells is looked up from an adhesion matrix ``D[type_i, type_j]`` rather than
a single constant. Per the Differential Adhesion Hypothesis (Steinberg), the
resulting interfacial tension ``sigma_ab = D_ab - (D_aa + D_bb)/2`` drives cell
sorting, engulfment, and other tissue-scale patterns.

The force law matches the legacy scalar kernel (attraction linear in the
separation beyond contact, within an outer cutoff band), so a uniform matrix
reproduces the original behavior exactly.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def calculate_differential_adhesion_forces_numba(positions, radii, types,
                                                  adhesion_matrix,
                                                  adhesion_cutoff_factor):
    """Pairwise type-dependent adhesion.

    Parameters
    ----------
    positions : (N, 2) float array
    radii : (N,) float array
    types : (N,) int array of cell-type indices.
    adhesion_matrix : (K, K) float array; entry [a, b] is the adhesion strength
        between a cell of type a and a cell of type b (diagonal = cohesion).
    adhesion_cutoff_factor : float; the attraction acts for
        touching_dist < distance < touching_dist * adhesion_cutoff_factor.

    Each row i accumulates forces from all other cells, so the outer prange
    loop is race-free (each thread owns its own row i).
    """
    num_cells = positions.shape[0]
    forces = np.zeros((num_cells, 2))
    for i in prange(num_cells):
        ti = types[i]
        for j in range(num_cells):
            if j == i:
                continue
            dx_ = positions[j, 0] - positions[i, 0]
            dy_ = positions[j, 1] - positions[i, 1]
            distance = np.sqrt(dx_**2 + dy_**2)
            touching_dist = radii[i] + radii[j]
            cutoff_dist = touching_dist * adhesion_cutoff_factor
            if touching_dist < distance < cutoff_dist:
                strength = adhesion_matrix[ti, types[j]]
                force_magnitude = strength * (distance - touching_dist)
                inv_dist = 1.0 / distance
                forces[i, 0] += force_magnitude * dx_ * inv_dist
                forces[i, 1] += force_magnitude * dy_ * inv_dist
    return forces
