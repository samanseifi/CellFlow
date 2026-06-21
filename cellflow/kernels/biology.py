"""Batched cell-biology kernel.

Replaces the per-cell Python loop (nutrient uptake, secretion, metabolism,
growth, phase, death) with a single compiled kernel that loops over all cells
internally, eliminating the per-cell Python/Numba dispatch overhead. It is the
top remaining cell-side cost once the O(N^2) overlap solver is removed.

The loop is sequential (njit, not parallel): cells read the frozen
``nutrient_read`` copy and subtract uptake from the live ``nutrient_field``, so
overlapping cells writing the same grid points stay deterministic and match the
original per-cell ordering exactly.
"""
import numpy as np
from numba import njit

from .fields import absorb_nutrient_numba, secrete_over_area_numba


@njit(cache=True)
def cell_biology_step_numba(positions, radii, nutrient_acc, consumption_rate,
                            secretion_rate, basal_rate, active,
                            nutrient_field, nutrient_read, attractant_field,
                            dt, dx, area_conserving, min_radius, max_radius,
                            enable_quiescence, quiescence_threshold,
                            uptake_saturation):
    """Advance one biology step for all cells, in place.

    Updates ``nutrient_acc``, ``radii`` and the ``active`` flags in place,
    modifies the nutrient and attractant fields, and returns:
        reached_division[k] : radius reached max_radius (eligible for division)
        alive[k]            : False if the cell's accumulated nutrient went < 0.

    Active<->passive transition (issue: bacterial-colony branching needs a
    motility/growth state switch). When ``enable_quiescence`` is set, a cell
    goes QUIESCENT where the local nutrient falls below ``quiescence_threshold``
    and reactivates above twice that (hysteresis). Quiescent cells still take up
    nutrient (so the interior stays depleted, shadowing the valleys) but do not
    grow, divide, metabolise, or -- via the propulsion gate in the simulation --
    move. This concentrates growth at the nutrient-exposed rim and lets the
    front finger, as in the continuum models.
    """
    n = positions.shape[0]
    ny, nx = nutrient_read.shape
    reached_division = np.zeros(n, dtype=np.bool_)
    alive = np.ones(n, dtype=np.bool_)
    area_floor = np.pi * (0.5 * min_radius) ** 2
    growth_factor = (max_radius - min_radius) / 100.0
    a_max = np.pi * max_radius ** 2

    for k in range(n):
        # state transition from the local (frozen) nutrient at the cell centre
        if enable_quiescence:
            gx = int(positions[k, 0] / dx)
            gy = int(positions[k, 1] / dx)
            local = nutrient_read[gy, gx] if (0 <= gy < ny and 0 <= gx < nx) else 0.0
            if active[k] and local < quiescence_threshold:
                active[k] = False
            elif (not active[k]) and local > 2.0 * quiescence_threshold:
                active[k] = True

        # nutrient uptake over the cell's area (always, so the matrix shadows)
        uptake = absorb_nutrient_numba(positions[k], radii[k], nutrient_field,
                                       nutrient_read, dt, consumption_rate[k], dx,
                                       uptake_saturation[k])
        nutrient_acc[k] += uptake

        if not active[k]:
            continue                       # quiescent: frozen, persists

        # attractant secretion + basal metabolism (active only)
        secrete_over_area_numba(positions[k], radii[k], attractant_field,
                                secretion_rate[k] * dt, dx)
        nutrient_acc[k] -= basal_rate[k] * dt

        # growth: update radius from accumulated nutrient
        if area_conserving:
            frac = nutrient_acc[k] / 100.0
            if frac < 0.0:
                frac = 0.0
            area = a_max * frac
            if area < area_floor:
                area = area_floor
            r = np.sqrt(area / np.pi)
            radii[k] = r if r < max_radius else max_radius
        else:
            r = min_radius + nutrient_acc[k] * growth_factor
            if r < min_radius:
                r = min_radius
            elif r > max_radius:
                r = max_radius
            radii[k] = r

        if radii[k] >= max_radius:
            reached_division[k] = True
        if nutrient_acc[k] < 0.0:
            alive[k] = False

    return reached_division, alive
