"""Inter-cell mechanical and chemotactic force kernels, plus overlap resolution."""
import numpy as np
from numba import njit, prange

from .fields import _sample_scalar_field_numba


@njit(parallel=True, cache=True)
def calculate_adhesion_forces_numba(positions, radii, adhesion_strength, adhesion_cutoff_factor):
    """Compute pairwise adhesion forces. Each row i accumulates forces from all
    other cells, so the outer prange loop is race-free (each thread owns its own i)."""
    num_cells = len(positions)
    adhesion_forces = np.zeros((num_cells, 2))
    for i in prange(num_cells):
        for j in range(num_cells):
            if j == i:
                continue
            dx_ = positions[j, 0] - positions[i, 0]
            dy_ = positions[j, 1] - positions[i, 1]
            distance = np.sqrt(dx_**2 + dy_**2)
            touching_dist = radii[i] + radii[j]
            cutoff_dist = touching_dist * adhesion_cutoff_factor
            if touching_dist < distance < cutoff_dist:
                force_magnitude = adhesion_strength * (distance - touching_dist)
                inv_dist = 1.0 / distance
                adhesion_forces[i, 0] += force_magnitude * dx_ * inv_dist
                adhesion_forces[i, 1] += force_magnitude * dy_ * inv_dist
    return adhesion_forces


@njit(parallel=True, cache=True)
def calculate_repulsion_forces_numba(positions, radii, repulsion_strength):
    """Compute pairwise repulsion forces. Each row i accumulates forces from all
    other cells, so the outer prange loop is race-free (each thread owns its own i)."""
    num_cells = len(positions)
    repulsion_forces = np.zeros((num_cells, 2))
    for i in prange(num_cells):
        for j in range(num_cells):
            if j == i:
                continue
            dx_ = positions[j, 0] - positions[i, 0]
            dy_ = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx_**2 + dy_**2)
            touch = radii[i] + radii[j]
            if 0.0 < dist < touch:
                overlap = touch - dist
                force_mag = repulsion_strength * np.exp(3.0 * overlap / touch)
                inv_dist = 1.0 / dist
                repulsion_forces[i, 0] -= force_mag * dx_ * inv_dist
                repulsion_forces[i, 1] -= force_mag * dy_ * inv_dist
    return repulsion_forces


@njit(parallel=True, cache=True)
def calculate_propulsion_forces_numba(positions, radii, grad_x_field, grad_y_field,
                                       chi_nutrient, walk_speed, max_propulsive_force, dx):
    """Computes chemotactic propulsion forces for all cells in parallel.

    Replaces the sequential Python loop in _calculate_forces so that
    gradient-field sampling for each cell is done concurrently.

    Note: Numba's parallel mode gives each thread its own independent RNG
    state, so np.random.randn() calls here are thread-safe. The simulation
    is intentionally stochastic (random walk), matching the original behaviour.
    """
    num_cells = len(positions)
    forces = np.zeros((num_cells, 2))

    for i in prange(num_cells):
        avg_grad_x = _sample_scalar_field_numba(positions[i], radii[i], grad_x_field, dx)
        avg_grad_y = _sample_scalar_field_numba(positions[i], radii[i], grad_y_field, dx)

        force_dir_x = chi_nutrient * avg_grad_x + walk_speed * np.random.randn()
        force_dir_y = chi_nutrient * avg_grad_y + walk_speed * np.random.randn()

        norm = np.sqrt(force_dir_x**2 + force_dir_y**2)
        if norm > 0.0:
            forces[i, 0] = (force_dir_x / norm) * max_propulsive_force
            forces[i, 1] = (force_dir_y / norm) * max_propulsive_force

    return forces


@njit(cache=True)
def resolve_overlaps_numba(positions, radii):
    """Resolve pairwise overlaps between cells in-place.

    Sequentially processes all pairs; positions are modified directly so the
    caller only needs a simple write-back loop to update Cell objects.
    """
    n = len(positions)
    for a in range(n):
        for b in range(a + 1, n):
            dx_ = positions[b, 0] - positions[a, 0]
            dy_ = positions[b, 1] - positions[a, 1]
            dist = np.sqrt(dx_**2 + dy_**2)
            touch = radii[a] + radii[b]
            if dist < 1e-12:
                shift_x = 0.5 * touch * np.random.randn()
                shift_y = 0.5 * touch * np.random.randn()
                positions[a, 0] -= shift_x
                positions[a, 1] -= shift_y
                positions[b, 0] += shift_x
                positions[b, 1] += shift_y
            elif dist < touch:
                overlap = touch - dist
                inv_dist = 1.0 / dist
                positions[a, 0] -= 0.5 * overlap * dx_ * inv_dist
                positions[a, 1] -= 0.5 * overlap * dy_ * inv_dist
                positions[b, 0] += 0.5 * overlap * dx_ * inv_dist
                positions[b, 1] += 0.5 * overlap * dy_ * inv_dist
